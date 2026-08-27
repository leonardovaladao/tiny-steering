"""Shared boilerplate for the notebooks in this folder.

Only the uninteresting parts live here — locating the project root, loading the
checkpoint, and a consistent plot style.  Every step that is actually part of
the replication is written out in the notebooks themselves.
"""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np


# ── paths ─────────────────────────────────────────────────────────────────────

def find_root(start: str | None = None) -> str:
    """Walk upward from `start` (default: cwd) until the project root is found."""
    p = os.path.abspath(start or os.getcwd())
    while True:
        if (os.path.isdir(os.path.join(p, "rfm_guppy"))
                and os.path.isfile(os.path.join(p, "requirements.txt"))):
            return p
        parent = os.path.dirname(p)
        if parent == p:
            raise RuntimeError(
                "Project root not found — run this notebook from inside the repository."
            )
        p = parent


ROOT        = find_root()
CACHE_DIR   = os.path.join(ROOT, "rfm_guppy", "cache")
RESULTS_DIR = os.path.join(ROOT, "results")
CKPT_DIR    = os.path.join(ROOT, "checkpoints", "guppylm-9M")


# ── model ─────────────────────────────────────────────────────────────────────

def load_guppy(verbose: bool = True):
    """Load the pretrained GuppyLM checkpoint in eval mode. Returns (model, tokenizer)."""
    import torch
    from tokenizers import Tokenizer
    from guppylm import GuppyLM, GuppyConfig

    ckpt_bin = os.path.join(CKPT_DIR, "pytorch_model.bin")
    if not os.path.exists(ckpt_bin):
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_bin}\n"
            "Fetch it first:  hf download arman-bd/guppylm-9M "
            "--local-dir checkpoints/guppylm-9M"
        )

    tokenizer = Tokenizer.from_file(os.path.join(CKPT_DIR, "tokenizer.json"))
    with open(os.path.join(CKPT_DIR, "config.json")) as f:
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

    ckpt       = torch.load(ckpt_bin, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model      = GuppyLM(config)
    model.load_state_dict({k: v for k, v in state_dict.items() if k in model.state_dict()})
    model.eval()

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"GuppyLM loaded: {n_params/1e6:.2f}M params · {config.n_layers} blocks "
              f"· d_model={config.d_model} · vocab={config.vocab_size}")
    return model, tokenizer


def load_activations(concept: str, split: str) -> np.ndarray:
    """Cached last-token activations for one concept/split: [n, 6, 384]."""
    return np.load(os.path.join(CACHE_DIR, f"{concept}_{split}_activations.npy"))


def load_vectors(concept: str) -> dict:
    """Cached RFM concept vectors and AGOP eigenbasis for one concept."""
    npz = np.load(os.path.join(CACHE_DIR, f"{concept}_vectors.npz"))
    return {k: npz[k] for k in npz.files}


def load_result(name: str) -> dict:
    """Load a committed JSON file from results/."""
    with open(os.path.join(RESULTS_DIR, name)) as f:
        return json.load(f)


# ── plotting ──────────────────────────────────────────────────────────────────

CONCEPTS       = ["food", "valence", "env_social"]
CONCEPT_COLORS = {"food": "#4C72B0", "valence": "#DD8452", "env_social": "#55A868"}
METHOD_COLORS  = {"rfm": "#C44E52", "pca": "#4C72B0",
                  "diffmeans": "#DD8452", "logreg": "#8172B3"}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi":        110,
        "figure.facecolor":  "white",
        "savefig.facecolor": "white",
        "axes.grid":         True,
        "grid.alpha":        0.25,
        "grid.linewidth":    0.6,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.titlesize":    11,
        "axes.titleweight":  "bold",
        "font.size":         9.5,
        "legend.frameon":    False,
    })
