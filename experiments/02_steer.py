#!/usr/bin/env python3
"""Phase 5 — Additive activation steering sweep.

For each concept (food, valence, env_social):
  1. Load the concept vectors from cache/<concept>_vectors.npz.
  2. Measure per-block activation norms for ε calibration.
  3. Sweep relative coefficient r ∈ {-4, -2, -1, -0.5, -0.25, 0, +0.25, +0.5, +1, +2, +4, +8}
     with per-block ε_ℓ = r × median_norm_ℓ.
  4. Generate responses to 7 neutral probe prompts at each r.
  5. Report lexicon-hit rates per r.
  6. Identify the coherence-collapse threshold.
  7. Save side-by-side table to results/steer_<concept>.json and results/steer_<concept>.md.

Usage (from project root):
    PYTHONPATH=. python experiments/02_steer.py
"""

import json
import os
import sys
import time

import numpy as np
import torch
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "guppylm"))

from guppylm import GuppyLM, GuppyConfig   # noqa: E402
from rfm_guppy.steering import (            # noqa: E402
    generate_steered,
    measure_activation_norms,
    lexicon_hit_rate,
)
from rfm_guppy.concepts import CONCEPT_LEXICONS  # noqa: E402

CACHE_DIR   = os.path.join(ROOT, "rfm_guppy", "cache")
RESULTS_DIR = os.path.join(ROOT, "results")
CHECKPOINT  = os.path.join(ROOT, "checkpoints", "guppylm-9M", "pytorch_model.bin")
TOKENIZER   = os.path.join(ROOT, "checkpoints", "guppylm-9M", "tokenizer.json")
CFG_JSON    = os.path.join(ROOT, "checkpoints", "guppylm-9M", "config.json")

CONCEPTS = ["food", "valence", "env_social"]

# Neutral probe prompts that don't favour any concept
PROBE_PROMPTS = [
    "How are you doing today?",
    "What are you up to right now?",
    "Tell me something about yourself.",
    "What's on your mind?",
    "How do you feel at this moment?",
    "Describe what's around you.",
    "What would you like to do?",
]

# Relative-coefficient sweep (ε_ℓ = r × median_norm_ℓ)
R_VALUES = [-4.0, -2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

MAX_NEW_TOKENS = 60
TEMPERATURE    = 0.7
TOP_K          = 50
SEED           = 42


# ── model loading ─────────────────────────────────────────────────────────────

def load_model():
    tokenizer = Tokenizer.from_file(TOKENIZER)
    with open(CFG_JSON) as f:
        cfg = json.load(f)
    config = GuppyConfig(
        vocab_size  = cfg.get("vocab_size", 4096),
        max_seq_len = cfg.get("max_position_embeddings", 128),
        d_model     = cfg.get("hidden_size", 384),
        n_layers    = cfg.get("num_hidden_layers", 6),
        n_heads     = cfg.get("num_attention_heads", 6),
        ffn_hidden  = cfg.get("intermediate_size", 768),
        dropout     = cfg.get("hidden_dropout_prob", 0.1),
        pad_id      = cfg.get("pad_token_id", 0),
        bos_id      = cfg.get("bos_token_id", 1),
        eos_id      = cfg.get("eos_token_id", 2),
    )
    ckpt       = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model      = GuppyLM(config)
    filtered   = {k: v for k, v in state_dict.items() if k in model.state_dict()}
    model.load_state_dict(filtered)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[load] GuppyLM {n_params/1e6:.2f}M params, {config.n_layers} blocks, "
          f"d_model={config.d_model}")
    return model, tokenizer


# ── helpers ───────────────────────────────────────────────────────────────────

def detect_collapse(text: str) -> bool:
    """Heuristic: output is collapsed if > 60% of words are the same token,
    or the output is fewer than 3 words."""
    words = text.split()
    if len(words) < 3:
        return True
    most_common = max(set(words), key=words.count)
    return (words.count(most_common) / len(words)) > 0.60


def _r_label(r: float) -> str:
    """Human-readable label for a relative coefficient."""
    if r == 0.0:
        return "baseline (r=0)"
    sign = "+" if r > 0 else ""
    return f"r={sign}{r}"


# ── markdown table ────────────────────────────────────────────────────────────

def _md_table(concept: str, results_per_r: dict, lexicon: list[str],
              block_norms: np.ndarray) -> str:
    lines = [
        f"# Steering results — {concept}",
        "",
        "Per-block median activation norms used for ε calibration:",
        "| " + " | ".join(f"Blk{i}" for i in range(len(block_norms))) + " |",
        "| " + " | ".join("-" * 6 for _ in block_norms) + " |",
        "| " + " | ".join(f"{n:.1f}" for n in block_norms) + " |",
        "",
        "## Lexicon hit rates per r",
        "",
        "| r | ε_Blk0 | ε_Blk5 | hit_rate | collapsed |",
        "|----|--------|--------|----------|-----------|",
    ]
    for r, data in sorted(results_per_r.items()):
        eps0 = r * block_norms[0]
        eps5 = r * block_norms[5]
        hr   = data["lexicon_hit_rate"]
        col  = data["n_collapsed"]
        n    = data["n_total"]
        lines.append(f"| {r:+.2f} | {eps0:+.1f} | {eps5:+.1f} | "
                     f"{hr:.2%} ({int(hr*n)}/{n}) | {col}/{n} |")

    lines += ["", "## Generated outputs"]

    for r, data in sorted(results_per_r.items()):
        lines.append(f"\n### r = {r:+.2f}  (ε_Blk3 = {r*block_norms[3]:+.1f})")
        for entry in data["outputs"]:
            prompt  = entry["prompt"]
            text    = entry["text"]
            hit     = "✓" if lexicon_hit_rate([text], lexicon) > 0 else "·"
            col_tag = " [COLLAPSED]" if entry["collapsed"] else ""
            lines.append(f"\n**Q:** {prompt}")
            lines.append(f"**A** [{hit}]{col_tag}: {text}")

    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("Phase 5 — Steering Sweep")
    print("=" * 60)

    model, tokenizer = load_model()

    # Measure per-block activation norms from cached concept activations
    block_norms = measure_activation_norms(CACHE_DIR)  # [L]
    print("\n[norms] Per-block median activation norms:")
    for i, n in enumerate(block_norms):
        print(f"  Block {i}: {n:.2f}")

    all_summaries = {}

    for concept in CONCEPTS:
        print(f"\n{'─'*60}")
        print(f"Concept: {concept.upper()}")
        print(f"{'─'*60}")

        # Load concept vectors
        npz_path = os.path.join(CACHE_DIR, f"{concept}_vectors.npz")
        npz = np.load(npz_path)
        vectors_per_block = npz["v_per_block"].astype(np.float64)  # [L, k]
        val_abs_rho       = npz["val_abs_rho"]                     # [L]
        best_blk          = int(np.argmax(val_abs_rho))

        print(f"  Loaded vectors, best block={best_blk}, "
              f"val|ρ|_max={val_abs_rho[best_blk]:.3f}")
        print(f"  Lexicon: {CONCEPT_LEXICONS[concept]}")

        lexicon = CONCEPT_LEXICONS[concept]
        results_per_r: dict[float, dict] = {}
        concept_summary = []

        for r in R_VALUES:
            eps_per_block = r * block_norms                # [L]
            r_outputs     = []
            n_collapsed   = 0

            for prompt in PROBE_PROMPTS:
                text = generate_steered(
                    prompt, model, tokenizer,
                    vectors_per_block=vectors_per_block,
                    eps_per_block=eps_per_block,
                    max_new_tokens=MAX_NEW_TOKENS,
                    seed=SEED,
                    temperature=TEMPERATURE,
                    top_k=TOP_K,
                    format_chatml=True,
                )
                collapsed = detect_collapse(text)
                if collapsed:
                    n_collapsed += 1
                r_outputs.append({
                    "prompt":    prompt,
                    "text":      text,
                    "collapsed": collapsed,
                })

            hr = lexicon_hit_rate([e["text"] for e in r_outputs], lexicon)
            results_per_r[r] = {
                "r":                r,
                "eps_per_block":    (r * block_norms).tolist(),
                "lexicon_hit_rate": float(hr),
                "n_collapsed":      n_collapsed,
                "n_total":          len(PROBE_PROMPTS),
                "outputs":          r_outputs,
            }

            flag = " ← collapse" if n_collapsed > len(PROBE_PROMPTS) // 2 else ""
            print(f"  r={r:+.2f}  hit_rate={hr:.0%}  "
                  f"collapsed={n_collapsed}/{len(PROBE_PROMPTS)}{flag}")

        # Identify coherence-collapse threshold
        collapse_r = None
        for r in sorted(r for r in R_VALUES if r > 0):
            d = results_per_r[r]
            if d["n_collapsed"] > len(PROBE_PROMPTS) // 2:
                collapse_r = r
                break

        # Best positive r with good hit rate and no collapse
        best_r = None
        best_hr = 0.0
        for r in sorted(r for r in R_VALUES if r > 0):
            d = results_per_r[r]
            if d["n_collapsed"] <= 1 and d["lexicon_hit_rate"] > best_hr:
                best_hr = d["lexicon_hit_rate"]
                best_r  = r

        print(f"\n  Coherence-collapse threshold: r={collapse_r}")
        print(f"  Best positive r (no collapse): r={best_r}, hit_rate={best_hr:.0%}")
        print(f"  Baseline (r=0) hit_rate: "
              f"{results_per_r[0.0]['lexicon_hit_rate']:.0%}")

        concept_summary = {
            "concept":                concept,
            "best_block":             best_blk,
            "val_abs_rho_per_block":  val_abs_rho.tolist(),
            "block_norms":            block_norms.tolist(),
            "collapse_threshold_r":   collapse_r,
            "best_r_no_collapse":     best_r,
            "lexicon":                lexicon,
            "r_values":               R_VALUES,
            "results_per_r":          results_per_r,
        }

        # Save JSON
        json_path = os.path.join(RESULTS_DIR, f"steer_{concept}.json")
        with open(json_path, "w") as f:
            json.dump(concept_summary, f, indent=2)
        print(f"  Saved JSON → results/steer_{concept}.json")

        # Save Markdown
        md = _md_table(concept, results_per_r, lexicon, block_norms)
        md_path = os.path.join(RESULTS_DIR, f"steer_{concept}.md")
        with open(md_path, "w") as f:
            f.write(md)
        print(f"  Saved MD  → results/steer_{concept}.md")

        all_summaries[concept] = concept_summary

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PHASE 5 SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Concept':12s}  {'baseline':>8s}  {'best_r':>6s}  "
          f"{'hit@best_r':>10s}  {'collapse_r':>10s}")
    print("-" * 55)
    for concept, summ in all_summaries.items():
        base_hr = summ["results_per_r"][0.0]["lexicon_hit_rate"]
        best_r  = summ["best_r_no_collapse"]
        col_r   = summ["collapse_threshold_r"]
        if best_r is not None:
            best_hr = summ["results_per_r"][best_r]["lexicon_hit_rate"]
        else:
            best_hr = 0.0
        print(f"{concept:12s}  {base_hr:>8.0%}  {str(best_r):>6s}  "
              f"{best_hr:>10.0%}  {str(col_r):>10s}")

    # ── Acceptance check ───────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("Acceptance criteria check:")

    # AC1: some ε causes visible shift (hit rate improves over baseline for at least 1 concept)
    any_shift = False
    for concept, summ in all_summaries.items():
        base_hr = summ["results_per_r"][0.0]["lexicon_hit_rate"]
        for r in [r for r in R_VALUES if r > 0]:
            d = summ["results_per_r"][r]
            if d["lexicon_hit_rate"] > base_hr + 0.10 and d["n_collapsed"] <= 1:
                any_shift = True
                print(f"  ✓ AC1: {concept} at r={r:+.2f} → "
                      f"hit_rate {base_hr:.0%} → {d['lexicon_hit_rate']:.0%}")
                break

    if not any_shift:
        print("  ✗ AC1: No concept showed clear lexicon shift — check steering!")

    # AC2: +r and -r push in opposite directions
    for concept, summ in all_summaries.items():
        pos_hits = [summ["results_per_r"][r]["lexicon_hit_rate"]
                    for r in R_VALUES if r > 0 and r <= 2]
        neg_hits = [summ["results_per_r"][r]["lexicon_hit_rate"]
                    for r in R_VALUES if r < 0 and r >= -2]
        if pos_hits and neg_hits and np.mean(pos_hits) > np.mean(neg_hits):
            print(f"  ✓ AC2: {concept} +r mean hit={np.mean(pos_hits):.0%} > "
                  f"-r mean hit={np.mean(neg_hits):.0%}")
            break
    else:
        print("  ✗ AC2: No asymmetry found between +r and -r")

    # AC3: collapse threshold identified
    for concept, summ in all_summaries.items():
        col_r = summ["collapse_threshold_r"]
        if col_r is not None:
            print(f"  ✓ AC3: {concept} coherence collapse at r={col_r}")
            break
    else:
        print("  ~ AC3: No collapse detected within the r sweep range (r ≤ 8)")

    print(f"\n[Phase 5] Complete.\n")


if __name__ == "__main__":
    main()
