#!/usr/bin/env python3
"""Phase 4 — Extract and store RFM concept vectors for all three concepts.

For each concept (food, valence, env_social):
  1. Extract per-block activations for train / val / test splits (cached as .npy).
  2. Run RFM HP sweep per block → oriented unit concept vector v_ℓ.
  3. Retain top-3 AGOP eigenvectors per block (for Phase 7 monitoring).
  4. Save to cache/<concept>_vectors.npz.
  5. Print per-concept × per-block summary table.

Usage (from project root):
    PYTHONPATH=. python experiments/01_extract.py
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

from guppylm import GuppyLM, GuppyConfig  # noqa: E402
from rfm_guppy.activations import extract_activations  # noqa: E402
from rfm_guppy.concepts import load_concept_dataset, CONCEPTS  # noqa: E402
from rfm_guppy.rfm import extract_concept_vector, rfm_synthetic_test  # noqa: E402

CACHE_DIR   = os.path.join(ROOT, "rfm_guppy", "cache")
CHECKPOINT  = os.path.join(ROOT, "checkpoints", "guppylm-9M", "pytorch_model.bin")
TOKENIZER   = os.path.join(ROOT, "checkpoints", "guppylm-9M", "tokenizer.json")
CFG_JSON    = os.path.join(ROOT, "checkpoints", "guppylm-9M", "config.json")

N_BLOCKS = 6
P = 3  # number of eigenvectors to keep per block for monitoring


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

    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)

    model = GuppyLM(config)
    filtered = {k: v for k, v in state_dict.items() if k in model.state_dict()}
    model.load_state_dict(filtered)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[load] GuppyLM: {n_params/1e6:.2f}M params, "
          f"{N_BLOCKS} blocks, d_model={config.d_model}")
    return model, tokenizer


# ── activation caching ────────────────────────────────────────────────────────

def get_or_extract(concept_name: str, split: str, prompts: list[str],
                   model, tokenizer) -> np.ndarray:
    """Return cached activations or extract and cache them."""
    path = os.path.join(CACHE_DIR, f"{concept_name}_{split}_activations.npy")
    if os.path.exists(path):
        acts = np.load(path)
        print(f"  [cache] {concept_name}/{split}: {acts.shape} loaded")
        return acts
    print(f"  [extract] {concept_name}/{split}: {len(prompts)} prompts ...")
    t0   = time.time()
    acts = extract_activations(prompts, model, tokenizer,
                               batch_size=64, format_chatml=False)
    print(f"  [extract] done in {time.time()-t0:.1f}s  shape={acts.shape}  "
          f"NaN={int(np.isnan(acts).sum())}")
    np.save(path, acts)
    return acts


# ── per-block RFM ─────────────────────────────────────────────────────────────

def rfm_all_blocks(Z_train, y_train, Z_val, y_val):
    """Run extract_concept_vector independently for each block.

    Returns list of (v, info) for blocks 0 … N_BLOCKS-1.
    """
    results = []
    for blk in range(N_BLOCKS):
        v, info = extract_concept_vector(
            Z_train[:, blk, :], y_train,
            Z_val[:,   blk, :], y_val,
        )
        results.append((v, info))
        print(f"    blk {blk}:  val|ρ|={info['val_abs_rho']:.3f}  "
              f"λ1/Σλ={info['lam1_share']:.3f}  "
              f"best=(L={info['L']}, T={info['T']}, norm={info['normalize']})")
    return results


# ── .npz serialisation ────────────────────────────────────────────────────────

def save_vectors(concept_name: str, block_results) -> str:
    """Pack per-block RFM results into a single .npz file."""
    k = block_results[0][0].shape[0]

    v_per_block    = np.zeros((N_BLOCKS, k),       dtype=np.float32)
    eigvecs_top3   = np.zeros((N_BLOCKS, P, k),    dtype=np.float32)
    input_mean     = np.zeros((N_BLOCKS, k),       dtype=np.float32)
    eigval_spectra = np.zeros((N_BLOCKS, k),       dtype=np.float32)
    lam1_share     = np.zeros(N_BLOCKS,            dtype=np.float32)
    val_abs_rho    = np.zeros(N_BLOCKS,            dtype=np.float32)
    hp_L           = np.zeros(N_BLOCKS,            dtype=np.float32)
    hp_T           = np.zeros(N_BLOCKS,            dtype=np.int32)
    hp_normalize   = np.zeros(N_BLOCKS,            dtype=np.int32)

    for blk, (v, info) in enumerate(block_results):
        v_per_block[blk]  = v.astype(np.float32)
        # V_top_p is [k, P] — transpose to [P, k] so axis-0 indexes eigenvectors
        V_top = info["V_top_p"]                            # [k, P]
        eigvecs_top3[blk] = V_top.T[:P].astype(np.float32)  # [P, k]
        input_mean[blk]   = info["mean_vec"].astype(np.float32)
        spec = info["eigenvalue_spectrum_full"]
        eigval_spectra[blk, :len(spec)] = spec.astype(np.float32)
        lam1_share[blk]   = float(info["lam1_share"])
        val_abs_rho[blk]  = float(info["val_abs_rho"])
        hp_L[blk]         = float(info["L"])
        hp_T[blk]         = int(info["T"])
        hp_normalize[blk] = int(bool(info["normalize"]))

    path = os.path.join(CACHE_DIR, f"{concept_name}_vectors.npz")
    np.savez(
        path,
        v_per_block    = v_per_block,
        eigvecs_top3   = eigvecs_top3,
        input_mean     = input_mean,
        eigenvalue_spectra = eigval_spectra,
        lam1_share     = lam1_share,
        val_abs_rho    = val_abs_rho,
        hp_L           = hp_L,
        hp_T           = hp_T,
        hp_normalize   = hp_normalize,
    )
    return path


# ── pretty-print summary ──────────────────────────────────────────────────────

def print_summary(concept_results: dict):
    concepts = list(concept_results.keys())
    sep = "=" * 74

    print(f"\n{sep}")
    print("PHASE 4 SUMMARY  —  val |Pearson| / λ1/Σλ per concept × block")
    print(sep)

    blk_hdr = "  ".join(f"  Blk{i}" for i in range(N_BLOCKS))
    print(f"{'Concept':12s}  {'Metric':8s}  {blk_hdr}")
    print("-" * 74)

    for cname, block_res in concept_results.items():
        rhos   = [info["val_abs_rho"] for _, info in block_res]
        shares = [info["lam1_share"]  for _, info in block_res]
        best_b = int(np.argmax(rhos))

        rho_str   = "  ".join(
            f"[{r:.3f}]" if i == best_b else f" {r:.3f} " for i, r in enumerate(rhos)
        )
        share_str = "  ".join(f" {s:.3f} " for s in shares)

        print(f"{cname:12s}  {'val|ρ|':8s}  {rho_str}   ← best blk {best_b}")
        print(f"{'':12s}  {'λ1/Σλ':8s}  {share_str}")
        print()

    print(sep)


# ── acceptance verification ───────────────────────────────────────────────────

def verify_npz(concept_name: str):
    path = os.path.join(CACHE_DIR, f"{concept_name}_vectors.npz")
    npz  = np.load(path)

    v  = npz["v_per_block"]
    ev = npz["eigvecs_top3"]
    mu = npz["input_mean"]

    assert v.shape  == (N_BLOCKS, 384), f"bad v_per_block shape {v.shape}"
    assert ev.shape == (N_BLOCKS, P, 384), f"bad eigvecs_top3 shape {ev.shape}"
    assert mu.shape == (N_BLOCKS, 384),   f"bad input_mean shape {mu.shape}"

    norms = np.linalg.norm(v, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), f"v not unit norm: {norms}"

    print(f"  {concept_name}: ✓  "
          f"v{v.shape}  eigvecs{ev.shape}  mean{mu.shape}  "
          f"val|ρ|=[{', '.join(f'{x:.3f}' for x in npz['val_abs_rho'])}]")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Phase 4 — RFM Concept Vector Extraction")
    print("=" * 60)

    # Gate: synthetic RFM recovery test must pass
    print("\n[gate] Synthetic RFM recovery test ...")
    passed, cos_val = rfm_synthetic_test(n=1000, k=384, seed=0, cos_threshold=0.90)
    if not passed:
        print("[ABORT] Gate failed — fix rfm.py before proceeding.")
        sys.exit(1)
    print(f"[gate] PASSED (|cos| = {cos_val:.4f})\n")

    model, tokenizer = load_model()

    concept_results = {}

    for concept_name in CONCEPTS:
        print(f"\n{'─'*60}")
        print(f"Concept: {concept_name.upper()}")
        print(f"{'─'*60}")

        ds = load_concept_dataset(concept_name)

        # 1. Extract / load activations for all three splits
        acts = {}
        for split in ("train", "val", "test"):
            prompts = [item["prompt"] for item in ds[split]]
            acts[split] = get_or_extract(concept_name, split, prompts,
                                         model, tokenizer)

        Z_train = acts["train"]   # [n_train, 6, 384]
        Z_val   = acts["val"]     # [n_val,   6, 384]
        y_train = np.array([item["label"] for item in ds["train"]], dtype=np.float64)
        y_val   = np.array([item["label"] for item in ds["val"]],   dtype=np.float64)

        print(f"  Z_train={Z_train.shape}  "
              f"pos={int(y_train.sum())} neg={int((1-y_train).sum())}")
        print(f"  Z_val={Z_val.shape}    "
              f"pos={int(y_val.sum())} neg={int((1-y_val).sum())}")

        # 2. Run RFM per block
        t0 = time.time()
        print(f"  Running RFM (HP sweep: L∈{{1,10,100}}, T=1..10, norm∈{{F,T}}) ...")
        block_results = rfm_all_blocks(Z_train, y_train, Z_val, y_val)
        print(f"  RFM done in {time.time()-t0:.1f}s")

        # 3. Save .npz
        path = save_vectors(concept_name, block_results)
        print(f"  Saved → {os.path.relpath(path, ROOT)}")

        concept_results[concept_name] = block_results

    # 4. Summary table
    print_summary(concept_results)

    # 5. Verify .npz acceptance criteria
    print("[verify] Checking .npz files ...")
    for concept_name in CONCEPTS:
        verify_npz(concept_name)

    print("\n[Phase 4] Complete. All concept vectors extracted and verified.\n")


if __name__ == "__main__":
    main()
