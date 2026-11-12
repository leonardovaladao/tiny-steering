#!/usr/bin/env python3
"""Phase 7 + Phase 6 — Monitoring probes and steering evaluation.

Runs on the selected concepts (default: every concept in rfm_guppy.concepts, i.e. food, valence,
env_social). --run DIR is required: it is the run folder that 02_steer.py wrote to.

Phase 7 (monitoring):
  1. Load cached activations (train/val/test) and binary labels per concept.
  2. Build probe features: project each block's last-token activation onto
     the top-3 AGOP eigenvectors stored in data/cache/<concept>_vectors.npz.
  3. Train per-block and aggregate logistic-regression probes (C swept on val).
  4. Evaluate on test split by AUROC.
  5. Print AUROC table; save probe packages to data/cache/<concept>_probe.pkl (overwritten)
     and the AUROC summary to monitor_auroc.json in the run folder.

Phase 6 (steering evaluation):
  6. Load Phase 5 steer results (steer_<concept>.json in the run folder).
  7. Feed (prompt, steered_text) exchanges back through GuppyLM.
  8. Score with the aggregate probe; also compute lexicon hit rate.
  9. Report probe-score and steering-success rate vs. ε sweep.
  10. Save Phase 6 eval to eval6_<concept>.json and .md in the run folder.

Existing files in the run folder are never overwritten unless --force is given; run.json in it
records this step.

Usage (from project root):
    .venv/bin/python replication/03_monitor.py --run DIR [--concepts NAME ...] [--force]
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
from tokenizers import Tokenizer

_r = os.path.dirname(os.path.abspath(__file__))   # walk up to the repo root (holds rfm_guppy/paths.py)
while not os.path.isfile(os.path.join(_r, "rfm_guppy", "paths.py")):
    if _r == os.path.dirname(_r): raise RuntimeError("rfm_guppy/paths.py not found above " + __file__)
    _r = os.path.dirname(_r)
sys.path.insert(0, _r)
from rfm_guppy import paths  # noqa: E402
ROOT = paths.ROOT
sys.path.insert(0, paths.GUPPYLM_DIR)

from guppylm import GuppyLM, GuppyConfig          # noqa: E402
from rfm_guppy.concepts import (                   # noqa: E402
    load_concept_dataset,
    resolve_concepts,
    CONCEPT_LEXICONS,
    CONCEPT_NAMES,
)
from rfm_guppy import runs                         # noqa: E402
from rfm_guppy.monitoring import (                 # noqa: E402
    build_probe_features,
    train_probes,
    evaluate_probes_on_test,
)
from rfm_guppy.evaluate import evaluate_steering, eval_to_markdown  # noqa: E402

CACHE_DIR   = paths.CACHE_DIR
CHECKPOINT  = os.path.join(paths.CKPT_DIR, "pytorch_model.bin")
TOKENIZER   = os.path.join(paths.CKPT_DIR, "tokenizer.json")
CFG_JSON    = os.path.join(paths.CKPT_DIR, "config.json")

N_BLOCKS  = 6
P         = 3


# ── model loading ─────────────────────────────────────────────────────────────

def load_model():
    tokenizer = Tokenizer.from_file(TOKENIZER)

    with open(CFG_JSON) as f:
        cfg = json.load(f)

    config = GuppyConfig(
        vocab_size  = cfg.get("vocab_size",              4096),
        max_seq_len = cfg.get("max_position_embeddings", 128),
        d_model     = cfg.get("hidden_size",             384),
        n_layers    = cfg.get("num_hidden_layers",       6),
        n_heads     = cfg.get("num_attention_heads",     6),
        ffn_hidden  = cfg.get("intermediate_size",       768),
        dropout     = cfg.get("hidden_dropout_prob",     0.1),
        pad_id      = cfg.get("pad_token_id",            0),
        bos_id      = cfg.get("bos_token_id",            1),
        eos_id      = cfg.get("eos_token_id",            2),
    )

    ckpt       = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)

    model = GuppyLM(config)
    filtered = {k: v for k, v in state_dict.items() if k in model.state_dict()}
    model.load_state_dict(filtered)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[load] GuppyLM: {n_params/1e6:.2f}M params")
    return model, tokenizer


# ── activation / vector helpers ───────────────────────────────────────────────

def load_activations(concept: str, split: str) -> np.ndarray:
    path = os.path.join(CACHE_DIR, f"{concept}_{split}_activations.npy")
    acts = np.load(path)
    print(f"  [cache] {concept}/{split}: {acts.shape}")
    return acts


def load_vectors(concept: str) -> dict:
    path = os.path.join(CACHE_DIR, f"{concept}_vectors.npz")
    npz  = np.load(path)
    return {
        "eigvecs_top3": npz["eigvecs_top3"],   # [L, P, k]
        "input_mean":   npz["input_mean"],      # [L, k]
        "hp_normalize": npz["hp_normalize"],    # [L] int32
        "v_per_block":  npz["v_per_block"],     # [L, k]
        "val_abs_rho":  npz["val_abs_rho"],     # [L]
    }


# ── probe caching ─────────────────────────────────────────────────────────────

def save_probe(concept: str, probe_pkg: dict):
    path = os.path.join(CACHE_DIR, f"{concept}_probe.pkl")
    with open(path, "wb") as f:
        pickle.dump(probe_pkg, f)
    return path


def load_probe_if_cached(concept: str) -> dict | None:
    path = os.path.join(CACHE_DIR, f"{concept}_probe.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


# ── Phase 7 ───────────────────────────────────────────────────────────────────

def run_phase7_concept(concept: str) -> tuple[dict, dict]:
    """Train and evaluate probes for one concept.

    Returns (probe_pkg, test_eval_dict).
    """
    print(f"\n{'─'*60}")
    print(f"[Phase 7] Concept: {concept.upper()}")
    print(f"{'─'*60}")

    vecs = load_vectors(concept)
    ds   = load_concept_dataset(concept)

    acts_train = load_activations(concept, "train")  # [n_train, L, k]
    acts_val   = load_activations(concept, "val")
    acts_test  = load_activations(concept, "test")

    y_train = np.array([item["label"] for item in ds["train"]], dtype=int)
    y_val   = np.array([item["label"] for item in ds["val"]],   dtype=int)
    y_test  = np.array([item["label"] for item in ds["test"]],  dtype=int)

    print(f"  samples: train={len(y_train)} val={len(y_val)} test={len(y_test)}")
    print(f"  test class balance: pos={y_test.sum()} neg={(1-y_test).sum()}")

    # Build features: [n, L, P]
    feats_train = build_probe_features(
        acts_train, vecs["eigvecs_top3"], vecs["input_mean"], vecs["hp_normalize"]
    )
    feats_val = build_probe_features(
        acts_val, vecs["eigvecs_top3"], vecs["input_mean"], vecs["hp_normalize"]
    )
    feats_test = build_probe_features(
        acts_test, vecs["eigvecs_top3"], vecs["input_mean"], vecs["hp_normalize"]
    )

    print(f"  feature shapes: train={feats_train.shape} val={feats_val.shape} test={feats_test.shape}")
    print("  Training probes (C sweep on val) ...")
    t0 = time.time()
    probe_pkg = train_probes(feats_train, y_train, feats_val, y_val)
    print(f"  Training done in {time.time()-t0:.1f}s")

    print("  Evaluating on test ...")
    test_eval = evaluate_probes_on_test(probe_pkg, feats_test, y_test)

    best_block   = probe_pkg["best_block"]
    val_auroc_bb = probe_pkg["best_block_val_auroc"]
    val_auroc_ag = probe_pkg["aggregate_val_auroc"]
    te_auroc_bb  = test_eval["best_block_test_auroc"]
    te_auroc_ag  = test_eval["aggregate_test_auroc"]

    print(f"\n  {'Strategy':22s}  {'Val AUROC':>10}  {'Test AUROC':>10}")
    print(f"  {'─'*46}")
    print(f"  {'Per-block best (blk '+str(best_block)+')':22s}  "
          f"{val_auroc_bb:10.4f}  {te_auroc_bb:10.4f}")
    print(f"  {'Aggregate (all blocks)':22s}  "
          f"{val_auroc_ag:10.4f}  {te_auroc_ag:10.4f}")

    return probe_pkg, test_eval


# ── Phase 6 ───────────────────────────────────────────────────────────────────

def run_phase6_concept(concept: str, probe_pkg: dict, model, tokenizer,
                       run_dir: str, force: bool = False):
    """Evaluate steering results with probe-score and lexicon metrics."""
    print(f"\n{'─'*60}")
    print(f"[Phase 6] Concept: {concept.upper()}")
    print(f"{'─'*60}")

    steer_json = os.path.join(run_dir, f"steer_{concept}.json")
    if not os.path.exists(steer_json):
        print(f"  [SKIP] {steer_json} not found — run 02_steer.py first")
        return None

    vecs     = load_vectors(concept)
    lexicon  = CONCEPT_LEXICONS[concept]

    print(f"  Lexicon ({len(lexicon)} keywords): {lexicon}")
    print("  Scoring steered generations ...")
    t0 = time.time()

    eval_result = evaluate_steering(
        steer_json_path = steer_json,
        probe_pkg       = probe_pkg,
        eigvecs_top3    = vecs["eigvecs_top3"],
        input_mean      = vecs["input_mean"],
        model           = model,
        tokenizer       = tokenizer,
        lexicon         = lexicon,
        hp_normalize    = vecs["hp_normalize"],
        probe_threshold = 0.5,
    )
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  Baseline success:   {eval_result['baseline_success_rate']:.1%}")
    print(f"  Best (+r) success:  {eval_result['best_success_rate']:.1%}  "
          f"at r={eval_result['best_r_pos']:+.2f}")

    # Save JSON
    json_path = runs.out_path(run_dir, f"eval6_{concept}.json", force)
    with open(json_path, "w") as f:
        json.dump(eval_result, f, indent=2)

    # Save Markdown
    md_path = runs.out_path(run_dir, f"eval6_{concept}.md", force)
    with open(md_path, "w") as f:
        f.write(eval_to_markdown(eval_result))

    print(f"  Saved: {runs.display_path(json_path)}")
    print(f"         {runs.display_path(md_path)}")
    return eval_result


# ── summary tables ────────────────────────────────────────────────────────────

def print_phase7_summary(phase7_results: dict[str, tuple[dict, dict]]):
    sep = "=" * 72
    print(f"\n{sep}")
    print("PHASE 7 SUMMARY — Monitoring AUROC per Concept")
    print(sep)
    print(f"  {'Concept':14s}  {'Best block':>12}  {'Val AUROC':>10}  "
          f"{'Test AUROC':>12}  {'Agg Val':>10}  {'Agg Test':>10}")
    print(f"  {'─'*68}")
    for concept, (probe_pkg, test_eval) in phase7_results.items():
        bb  = probe_pkg["best_block"]
        val = probe_pkg["best_block_val_auroc"]
        te  = test_eval["best_block_test_auroc"]
        av  = probe_pkg["aggregate_val_auroc"]
        at  = test_eval["aggregate_test_auroc"]
        print(f"  {concept:14s}  {f'blk {bb}':>12}  {val:10.4f}  "
              f"{te:12.4f}  {av:10.4f}  {at:10.4f}")
    print(sep)


def print_phase6_summary(phase6_results: dict[str, dict]):
    sep = "=" * 68
    print(f"\n{sep}")
    print("PHASE 6 SUMMARY — Steering Evaluation (probe + lexicon)")
    print(sep)
    print(f"  {'Concept':14s}  {'Baseline success':>18}  "
          f"{'Best r':>8}  {'Best success':>14}")
    print(f"  {'─'*64}")
    for concept, result in phase6_results.items():
        if result is None:
            continue
        print(f"  {concept:14s}  "
              f"{result['baseline_success_rate']:18.1%}  "
              f"{result['best_r_pos']:>+8.2f}  "
              f"{result['best_success_rate']:14.1%}")
    print(sep)


def save_phase7_json(phase7_results: dict[str, tuple[dict, dict]], run_dir: str,
                     force: bool = False):
    """Serialise AUROC results (no sklearn objects) to monitor_auroc.json in the run folder."""
    summary = {}
    for concept, (probe_pkg, test_eval) in phase7_results.items():
        summary[concept] = {
            "best_block":            probe_pkg["best_block"],
            "best_block_val_auroc":  probe_pkg["best_block_val_auroc"],
            "best_block_test_auroc": test_eval["best_block_test_auroc"],
            "aggregate_val_auroc":   probe_pkg["aggregate_val_auroc"],
            "aggregate_test_auroc":  test_eval["aggregate_test_auroc"],
            "per_block_val_auroc":   probe_pkg["per_block_val_auroc"],
        }
    path = runs.out_path(run_dir, "monitor_auroc.json", force)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Phase 7] AUROC summary → {runs.display_path(path)}")
    return path


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Phases 7 + 6: monitoring probes and steering evaluation. Reads "
                    "steer_<c>.json from the run folder; writes eval6_<c>.json, eval6_<c>.md and "
                    "monitor_auroc.json there, and data/cache/<c>_probe.pkl.")
    parser.add_argument("--concepts", nargs="+", metavar="NAME",
                        help=f"concepts to process (default: all, {' '.join(CONCEPT_NAMES)})")
    parser.add_argument("--run", metavar="DIR", required=True,
                        help="run folder written by 02_steer.py (holds steer_<c>.json)")
    parser.add_argument("--force", action="store_true",
                        help="allow overwriting files that already exist in the run folder")
    args = parser.parse_args()
    try:
        args.concepts = resolve_concepts(args.concepts)
    except ValueError as e:
        parser.error(str(e))
    return args


def main():
    args = parse_args()
    concepts, run_dir, force = args.concepts, args.run, args.force

    # Refuse before any work: Phase 5 inputs must be in the run folder, outputs must not be.
    missing = [f"steer_{c}.json" for c in concepts
               if not os.path.exists(os.path.join(run_dir, f"steer_{c}.json"))]
    if missing:
        sys.exit(f"[03_monitor] {runs.display_path(run_dir)} lacks {', '.join(missing)}. "
                 f"Run 02_steer.py --run {run_dir} --concepts {' '.join(concepts)} first.")
    runs.out_path(run_dir, "monitor_auroc.json", force)
    for c in concepts:
        for ext in ("json", "md"):
            runs.out_path(run_dir, f"eval6_{c}.{ext}", force)
    run_dir = runs.open_run(run_dir, __file__, concepts,
                            params={"force": force, "probe_threshold": 0.5})

    print("=" * 60)
    print("Phases 7 + 6 — Monitoring and Steering Evaluation")
    print("=" * 60)
    print(f"[run] {runs.display_path(run_dir)}")
    print(f"[concepts] {', '.join(concepts)}")

    # ── Phase 7: train / evaluate probes ──────────────────────────────────────
    print("\n[PHASE 7] Training and evaluating concept probes ...")
    phase7_results: dict[str, tuple[dict, dict]] = {}

    for concept in concepts:
        probe_pkg, test_eval = run_phase7_concept(concept)
        phase7_results[concept] = (probe_pkg, test_eval)
        path = save_probe(concept, probe_pkg)
        print(f"  Probe saved → {os.path.relpath(path, ROOT)}")

    print_phase7_summary(phase7_results)
    save_phase7_json(phase7_results, run_dir, force)

    # Acceptance check: AUROC > 0.5 (printed only; the original run checked food and env_social)
    for concept in concepts:
        _, test_eval = phase7_results[concept]
        agg_auroc = test_eval["aggregate_test_auroc"]
        status = "PASS ✓" if agg_auroc > 0.5 else "FAIL ✗ (check model/features)"
        print(f"[check] {concept} aggregate test AUROC = {agg_auroc:.4f}  → {status}")

    # ── Phase 6: probe-score steering evaluation ──────────────────────────────
    print("\n\n[PHASE 6] Evaluating steering results with probe scores ...")
    model, tokenizer = load_model()

    phase6_results: dict[str, dict | None] = {}

    for concept in concepts:
        probe_pkg, _ = phase7_results[concept]
        result = run_phase6_concept(concept, probe_pkg, model, tokenizer, run_dir, force)
        phase6_results[concept] = result

    print_phase6_summary(phase6_results)

    # Acceptance check: at least one concept exceeds baseline by clear margin
    any_success = False
    for concept, result in phase6_results.items():
        if result is None:
            continue
        delta = result["best_success_rate"] - result["baseline_success_rate"]
        if delta > 0.2:
            any_success = True
            print(f"[check] {concept}: success delta = {delta:+.1%} ✓")
    if not any_success:
        print("[check] WARNING: no concept showed >20% improvement over baseline")

    print(f"\n[Phases 7 + 6] Complete. Run folder: {runs.display_path(run_dir)}\n")


if __name__ == "__main__":
    main()
