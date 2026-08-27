#!/usr/bin/env python3
"""Phase 8 — Baseline comparison: PCA / DiffMeans / LogReg vs RFM.

For each concept × method:
  (a) Extract concept vectors per block using the baseline method.
  (b) Monitoring: project test activations onto those vectors, train an LR
      probe, evaluate AUROC. Compare to RFM's Phase-7 AUROC.
  (c) Steering: steer with baseline vectors over a reduced ε sweep
      {0, ±0.25, ±0.5, ±1, ±2} × median_norms, score with the existing
      Phase-7 RFM probe (same scoring function as Phase 6) plus lexicon hit.

Outputs (saved to results/):
  baselines_monitoring.json   — AUROC table: {method × concept}
  baselines_steering.json     — success-rate table: {method × concept}
  baselines_comparison.md     — human-readable comparison tables

Usage:
    source .venv/bin/activate && PYTHONPATH=. python experiments/04_baselines.py
"""

import json
import os
import pickle
import re
import sys
import time

import numpy as np
import torch
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "guppylm"))

from guppylm import GuppyLM, GuppyConfig           # noqa: E402
from rfm_guppy.baselines import (                   # noqa: E402
    extract_all_baseline_vectors,
    baseline_monitoring_auroc,
)
from rfm_guppy.concepts import load_concept_dataset, CONCEPT_LEXICONS  # noqa: E402
from rfm_guppy.steering import (                    # noqa: E402
    generate_steered,
    measure_activation_norms,
)
from rfm_guppy.monitoring import predict_probe_probs  # noqa: E402

CACHE_DIR   = os.path.join(ROOT, "rfm_guppy", "cache")
RESULTS_DIR = os.path.join(ROOT, "results")
CHECKPOINT  = os.path.join(ROOT, "checkpoints", "guppylm-9M", "pytorch_model.bin")
TOKENIZER   = os.path.join(ROOT, "checkpoints", "guppylm-9M", "tokenizer.json")
CFG_JSON    = os.path.join(ROOT, "checkpoints", "guppylm-9M", "config.json")

CONCEPTS = ["food", "valence", "env_social"]
METHODS  = ["pca", "diffmeans", "logreg"]   # RFM handled separately (already computed)

PROBE_PROMPTS = [
    "How are you doing today?",
    "What are you up to right now?",
    "Tell me something about yourself.",
    "What's on your mind?",
    "How do you feel at this moment?",
    "Describe what's around you.",
    "What would you like to do?",
]

# Reduced ε sweep (relative coefficients; ε_ℓ = r × median_norm_ℓ)
R_VALUES = [-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0]

MAX_NEW_TOKENS = 60
TEMPERATURE    = 0.7
TOP_K          = 50
SEED           = 42


# ── model loading ──────────────────────────────────────────────────────────────

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
    print(f"[load] GuppyLM {n_params/1e6:.2f}M params")
    return model, tokenizer


# ── helpers ────────────────────────────────────────────────────────────────────

def load_activations(concept: str, split: str) -> np.ndarray:
    path = os.path.join(CACHE_DIR, f"{concept}_{split}_activations.npy")
    acts = np.load(path)
    return acts


def load_rfm_vectors(concept: str) -> dict:
    npz = np.load(os.path.join(CACHE_DIR, f"{concept}_vectors.npz"))
    return {
        "v_per_block":   npz["v_per_block"],
        "eigvecs_top3":  npz["eigvecs_top3"],
        "input_mean":    npz["input_mean"],
        "hp_normalize":  npz["hp_normalize"],
        "val_abs_rho":   npz["val_abs_rho"],
    }


def load_probe(concept: str) -> dict:
    path = os.path.join(CACHE_DIR, f"{concept}_probe.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def _lexicon_hit(text: str, keywords: list[str]) -> bool:
    lo = text.lower()
    return any(re.search(r"\b" + re.escape(kw) + r"\b", lo) for kw in keywords)


def _detect_collapse(text: str) -> bool:
    words = text.split()
    if len(words) < 3:
        return True
    most_common = max(set(words), key=words.count)
    return (words.count(most_common) / len(words)) > 0.60


# ── Phase 8a: monitoring AUROC ─────────────────────────────────────────────────

def run_monitoring(concept: str, method_vectors: dict[str, dict],
                   acts_train, y_train, acts_val, y_val, acts_test, y_test,
                   rfm_monitor_auroc: dict) -> dict:
    """AUROC table for all methods on one concept."""
    results = {}

    for method, vdata in method_vectors.items():
        val_auroc, test_auroc = baseline_monitoring_auroc(
            acts_train, y_train,
            acts_val,   y_val,
            acts_test,  y_test,
            vdata["v_per_block"],
            vdata["mean_per_block"],
        )
        results[method] = {"val_auroc": val_auroc, "test_auroc": test_auroc}
        print(f"    {method:10s}: val={val_auroc:.4f}  test={test_auroc:.4f}")

    # RFM from Phase 7
    rfm = rfm_monitor_auroc[concept]
    results["rfm"] = {
        "val_auroc":  rfm["aggregate_val_auroc"],
        "test_auroc": rfm["aggregate_test_auroc"],
    }
    print(f"    {'rfm':10s}: val={rfm['aggregate_val_auroc']:.4f}  "
          f"test={rfm['aggregate_test_auroc']:.4f}  [Phase 7 result]")

    return results


# ── Phase 8b: steering sweep ───────────────────────────────────────────────────

def run_steering_method(
    concept: str,
    method: str,
    v_per_block: np.ndarray,
    block_norms: np.ndarray,
    probe_pkg: dict,
    rfm_vecs: dict,
    model,
    tokenizer,
    lexicon: list[str],
    probe_threshold: float = 0.5,
) -> dict:
    """Steer with baseline vectors, score with RFM probe + lexicon."""
    results_per_r: dict[str, dict] = {}

    for r in R_VALUES:
        eps_per_block = r * block_norms
        outputs = []
        n_collapsed = 0

        for prompt in PROBE_PROMPTS:
            text = generate_steered(
                prompt, model, tokenizer,
                vectors_per_block=v_per_block,
                eps_per_block=eps_per_block,
                max_new_tokens=MAX_NEW_TOKENS,
                seed=SEED,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                format_chatml=True,
            )
            collapsed = _detect_collapse(text)
            if collapsed:
                n_collapsed += 1
            outputs.append({"prompt": prompt, "text": text, "collapsed": collapsed})

        # Score with RFM probe (same as Phase 6)
        full_exchanges = [
            f"<|im_start|>user\n{item['prompt']}<|im_end|>\n"
            f"<|im_start|>assistant\n{item['text']}<|im_end|>"
            for item in outputs
        ]
        probe_probs = predict_probe_probs(
            full_exchanges, model, tokenizer,
            probe_pkg,
            rfm_vecs["eigvecs_top3"],
            rfm_vecs["input_mean"],
            rfm_vecs["hp_normalize"],
            format_chatml=False,
        )

        lex_hits  = [_lexicon_hit(item["text"], lexicon) for item in outputs]
        lex_rate  = float(np.mean(lex_hits))
        success   = [float(p) > probe_threshold or h
                     for p, h in zip(probe_probs.tolist(), lex_hits)]
        suc_rate  = float(np.mean(success))

        results_per_r[str(float(r))] = {
            "r":                    float(r),
            "mean_probe_score":     float(np.mean(probe_probs)),
            "lexicon_hit_rate":     lex_rate,
            "steering_success_rate": suc_rate,
            "n_collapsed":          n_collapsed,
        }
        print(f"      r={float(r):+.2f}: probe={float(np.mean(probe_probs)):.3f}  "
              f"lex={lex_rate:.3f}  success={suc_rate:.3f}  "
              f"collapsed={n_collapsed}/{len(PROBE_PROMPTS)}")

    # Best positive r
    pos_keys = [str(float(r)) for r in R_VALUES if r > 0]
    best_r_str = max(pos_keys, key=lambda k: results_per_r[k]["steering_success_rate"])
    best_success = results_per_r[best_r_str]["steering_success_rate"]
    baseline_success = results_per_r["0.0"]["steering_success_rate"]

    return {
        "concept":               concept,
        "method":                method,
        "results_per_r":         results_per_r,
        "best_r_pos":            float(best_r_str),
        "best_success_rate":     best_success,
        "baseline_success_rate": baseline_success,
    }


# ── comparison tables ──────────────────────────────────────────────────────────

def _monitoring_table_md(monitoring_results: dict) -> str:
    """monitoring_results[concept][method] = {val_auroc, test_auroc}"""
    methods_ordered = ["pca", "diffmeans", "logreg", "rfm"]
    method_labels   = {"pca": "PCA", "diffmeans": "DiffMeans", "logreg": "LogReg", "rfm": "RFM"}

    lines = [
        "## Monitoring AUROC Comparison",
        "",
        "*(Aggregate probe; test split)*",
        "",
        "| Method | food | valence | env_social | Overall |",
        "|--------|------|---------|------------|---------|",
    ]
    overall_by_method = {m: [] for m in methods_ordered}
    for method in methods_ordered:
        row_vals = []
        for concept in CONCEPTS:
            auroc = monitoring_results[concept][method]["test_auroc"]
            row_vals.append(auroc)
            overall_by_method[method].append(auroc)
        overall = float(np.mean(row_vals))
        cells = " | ".join(f"{v:.4f}" for v in row_vals)
        lines.append(f"| {method_labels[method]:10s} | {cells} | {overall:.4f} |")

    return "\n".join(lines) + "\n"


def _steering_table_md(steering_results: dict, phase6_results: dict) -> str:
    """steering_results[concept][method] = steer dict; phase6_results[concept] = {best_success_rate}"""
    methods_ordered = ["pca", "diffmeans", "logreg", "rfm"]
    method_labels   = {"pca": "PCA", "diffmeans": "DiffMeans", "logreg": "LogReg", "rfm": "RFM"}

    lines = [
        "## Steering Success-Rate Comparison",
        "",
        "*(Best-over-positive-ε; probe-score > 0.5 OR lexicon hit)*",
        "",
        "| Method | food | valence | env_social | Overall |",
        "|--------|------|---------|------------|---------|",
    ]

    for method in methods_ordered:
        row_vals = []
        for concept in CONCEPTS:
            if method == "rfm":
                val = phase6_results[concept]
            else:
                val = steering_results[concept][method]["best_success_rate"]
            row_vals.append(val)
        overall = float(np.mean(row_vals))
        cells = " | ".join(f"{v:.3f}" for v in row_vals)
        lines.append(f"| {method_labels[method]:10s} | {cells} | {overall:.3f} |")

    return "\n".join(lines) + "\n"


def _val_rho_table_md(concept_baseline_vecs: dict) -> str:
    """concept_baseline_vecs[concept][method]['val_abs_rho'] = [L]"""
    lines = [
        "## Concept Vector Quality: Val |Pearson| (best block)",
        "",
        "| Method | food | valence | env_social |",
        "|--------|------|---------|------------|",
    ]
    for method in ["pca", "diffmeans", "logreg"]:
        row = []
        for concept in CONCEPTS:
            rho_arr = concept_baseline_vecs[concept][method]["val_abs_rho"]
            row.append(float(np.max(rho_arr)))
        cells = " | ".join(f"{v:.4f}" for v in row)
        lines.append(f"| {method:10s} | {cells} |")
    return "\n".join(lines) + "\n"


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 68)
    print("Phase 8 — Baseline Comparison (PCA / DiffMeans / LogReg vs RFM)")
    print("=" * 68)

    # Load RFM monitoring results from Phase 7
    with open(os.path.join(RESULTS_DIR, "monitor_auroc.json")) as f:
        rfm_monitor_auroc = json.load(f)

    # Load Phase 6 RFM steering-success rates
    rfm_steer_success: dict[str, float] = {}
    for concept in CONCEPTS:
        p = os.path.join(RESULTS_DIR, f"eval6_{concept}.json")
        with open(p) as f:
            d = json.load(f)
        rfm_steer_success[concept] = d["best_success_rate"]

    # ── Part A: Extract baseline vectors + monitoring AUROC ────────────────────
    print("\n[A] Extracting baseline vectors and evaluating monitoring AUROC ...\n")

    monitoring_results: dict[str, dict] = {c: {} for c in CONCEPTS}
    concept_baseline_vecs: dict[str, dict] = {}

    for concept in CONCEPTS:
        print(f"  Concept: {concept.upper()}")

        acts_train = load_activations(concept, "train")
        acts_val   = load_activations(concept, "val")
        acts_test  = load_activations(concept, "test")
        ds         = load_concept_dataset(concept)

        y_train = np.array([item["label"] for item in ds["train"]], dtype=int)
        y_val   = np.array([item["label"] for item in ds["val"]],   dtype=int)
        y_test  = np.array([item["label"] for item in ds["test"]],  dtype=int)

        print(f"    Extracting vectors (train={len(y_train)}, val={len(y_val)}, test={len(y_test)})")
        t0 = time.time()
        bvecs = extract_all_baseline_vectors(acts_train, y_train, acts_val, y_val)
        concept_baseline_vecs[concept] = bvecs
        print(f"    Extracted in {time.time()-t0:.1f}s")

        # Print val |ρ| per method (best block)
        for method, vdata in bvecs.items():
            best_rho = float(np.max(vdata["val_abs_rho"]))
            best_blk = int(np.argmax(vdata["val_abs_rho"]))
            print(f"    {method:10s}: best block={best_blk}, val|ρ|_max={best_rho:.4f}")

        print(f"    Monitoring AUROC:")
        monitoring_results[concept] = run_monitoring(
            concept, bvecs,
            acts_train, y_train,
            acts_val,   y_val,
            acts_test,  y_test,
            rfm_monitor_auroc,
        )

    # Save monitoring results
    mon_path = os.path.join(RESULTS_DIR, "baselines_monitoring.json")
    with open(mon_path, "w") as f:
        json.dump(monitoring_results, f, indent=2)
    print(f"\n  Monitoring results → {os.path.relpath(mon_path, ROOT)}")

    # ── Part B: Steering with baseline vectors ─────────────────────────────────
    print("\n[B] Steering sweep with baseline vectors ...\n")
    print("    Loading model ...")
    model, tokenizer = load_model()

    block_norms = measure_activation_norms(CACHE_DIR)
    print("    Per-block median activation norms:")
    for i, n in enumerate(block_norms):
        print(f"      Block {i}: {n:.2f}")

    steering_results: dict[str, dict] = {c: {} for c in CONCEPTS}

    for concept in CONCEPTS:
        print(f"\n  Concept: {concept.upper()}")
        probe_pkg = load_probe(concept)
        rfm_vecs  = load_rfm_vectors(concept)
        lexicon   = CONCEPT_LEXICONS[concept]

        for method in METHODS:
            print(f"    [{method}] steering sweep r ∈ {R_VALUES} ...")
            t0 = time.time()
            v_per_block = concept_baseline_vecs[concept][method]["v_per_block"]
            steer_result = run_steering_method(
                concept=concept,
                method=method,
                v_per_block=v_per_block,
                block_norms=block_norms,
                probe_pkg=probe_pkg,
                rfm_vecs=rfm_vecs,
                model=model,
                tokenizer=tokenizer,
                lexicon=lexicon,
            )
            steering_results[concept][method] = steer_result
            dt = time.time() - t0
            print(f"      → best success {steer_result['best_success_rate']:.1%} "
                  f"at r={steer_result['best_r_pos']:+.2f}  ({dt:.1f}s)")

    # Save steering results
    steer_path = os.path.join(RESULTS_DIR, "baselines_steering.json")
    with open(steer_path, "w") as f:
        json.dump(steering_results, f, indent=2)
    print(f"\n  Steering results → {os.path.relpath(steer_path, ROOT)}")

    # ── Comparison tables ──────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("MONITORING AUROC TABLE (test split, aggregate probe)")
    print("=" * 68)
    methods_all = ["pca", "diffmeans", "logreg", "rfm"]
    method_labels = {"pca": "PCA", "diffmeans": "DiffMeans", "logreg": "LogReg", "rfm": "RFM"}
    print(f"  {'Method':12s}  {'food':>8}  {'valence':>8}  {'env_social':>10}  {'Overall':>8}")
    print(f"  {'─'*52}")
    for method in methods_all:
        vals = []
        for concept in CONCEPTS:
            auroc = monitoring_results[concept][method]["test_auroc"]
            vals.append(auroc)
        overall = float(np.mean(vals))
        print(f"  {method_labels[method]:12s}  " +
              "  ".join(f"{v:8.4f}" for v in vals) +
              f"  {overall:8.4f}")

    print("\n" + "=" * 68)
    print("STEERING SUCCESS TABLE (best positive r, probe>0.5 OR lexicon)")
    print("=" * 68)
    print(f"  {'Method':12s}  {'food':>8}  {'valence':>8}  {'env_social':>10}  {'Overall':>8}")
    print(f"  {'─'*52}")
    for method in methods_all:
        vals = []
        for concept in CONCEPTS:
            if method == "rfm":
                val = rfm_steer_success[concept]
            else:
                val = steering_results[concept][method]["best_success_rate"]
            vals.append(val)
        overall = float(np.mean(vals))
        print(f"  {method_labels[method]:12s}  " +
              "  ".join(f"{v:8.3f}" for v in vals) +
              f"  {overall:8.3f}")

    # Who wins on monitoring?
    print("\n[Monitoring check] Method with highest overall test AUROC:")
    best_mon_method = max(
        methods_all,
        key=lambda m: float(np.mean([
            monitoring_results[c][m]["test_auroc"] for c in CONCEPTS
        ])),
    )
    print(f"  → {method_labels[best_mon_method]}")

    # Who wins on steering?
    print("[Steering check] Method with highest overall success rate:")
    all_steer_overall = {}
    for method in methods_all:
        vals = []
        for concept in CONCEPTS:
            if method == "rfm":
                vals.append(rfm_steer_success[concept])
            else:
                vals.append(steering_results[concept][method]["best_success_rate"])
        all_steer_overall[method] = float(np.mean(vals))
    best_steer_method = max(all_steer_overall, key=lambda m: all_steer_overall[m])
    print(f"  → {method_labels[best_steer_method]}")

    # ── Markdown report ────────────────────────────────────────────────────────
    md_lines = ["# Phase 8 — Baseline Comparison", ""]
    md_lines.append(_val_rho_table_md(concept_baseline_vecs))
    md_lines.append("")
    md_lines.append(_monitoring_table_md(monitoring_results))
    md_lines.append("")
    md_lines.append(_steering_table_md(steering_results, rfm_steer_success))
    md_lines.append("")
    md_lines.append("### Winner summary")
    md_lines.append("")
    md_lines.append(f"- Best monitoring AUROC: **{method_labels[best_mon_method]}**")
    md_lines.append(f"- Best steering success: **{method_labels[best_steer_method]}**")
    md_lines.append("")
    md_lines.append("*Note: RFM monitoring uses 3 eigenvectors per block (Phase 7); "
                    "baselines use 1 vector per block.*  ")
    md_lines.append("*RFM steering uses vectors from 01_extract.py; "
                    "success rates come from eval6 Phase 6 results.*  ")
    md_lines.append("*All methods evaluated on identical splits, identical ε sweeps, "
                    "identical probe threshold (0.5).*")

    md_path = os.path.join(RESULTS_DIR, "baselines_comparison.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"\n  Markdown comparison → {os.path.relpath(md_path, ROOT)}")

    print("\n[Phase 8] Complete.\n")


if __name__ == "__main__":
    main()
