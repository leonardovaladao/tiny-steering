"""Python reference generator for the steering demo (Seam A in SPEC.md).

Same additive steering as rfm_guppy/steering.py, with a position set P:

    h_{ℓ,t}  →  h_{ℓ,t} + 1[t ∈ P] · ε_ℓ · v_ℓ ,      ε_ℓ = r · med_ℓ  (0 for ℓ ∉ S)

    P_prompt = {t : t < T0 − 1},   P_gen = {t : t ≥ T0 − 1},   P_all = P_prompt ⊔ P_gen

where T0 is the length of the ChatML-wrapped prompt. P_gen starts at the last prompt token, so
every generated token is predicted from a steered state.

The decoding loop mirrors GuppyLM.generate op for op (same RNG calls), so with P = P_all the text
equals generate_steered character for character. Calibration, export and the JS golden fixtures
all go through generate().
"""

import contextlib
import json
import os
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

_r = os.path.dirname(os.path.abspath(__file__))   # walk up to the repo root (holds rfm_guppy/paths.py)
while not os.path.isfile(os.path.join(_r, "rfm_guppy", "paths.py")):
    if _r == os.path.dirname(_r): raise RuntimeError("rfm_guppy/paths.py not found above " + __file__)
    _r = os.path.dirname(_r)
sys.path.insert(0, _r)
from rfm_guppy.paths import ROOT, GUPPYLM_DIR, CKPT_DIR, CACHE_DIR  # noqa: E402  (re-exported)
sys.path.insert(0, GUPPYLM_DIR)

from guppylm import GuppyLM, GuppyConfig   # noqa: E402
from tokenizers import Tokenizer            # noqa: E402

CONCEPTS      = ("food", "valence", "env_social")
SOURCES       = ("rfm", "dm", "random")
BLOCK_SETS    = ("0", "1", "2", "3", "4", "5", "all")
POSITION_SETS = ("all", "prompt", "gen")

U_SEED        = 20261020                               # the page's random direction u
CONTROL_SEEDS = (20261021, 20261022, 20261023, 20261024, 20261025)   # confirmation controls


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model():
    """GuppyLM + tokenizer, loaded as in Check III."""
    tok = Tokenizer.from_file(os.path.join(CKPT_DIR, "tokenizer.json"))
    with open(os.path.join(CKPT_DIR, "config.json")) as f:
        cfg = json.load(f)
    config = GuppyConfig(
        vocab_size=cfg["vocab_size"], max_seq_len=cfg["max_position_embeddings"],
        d_model=cfg["hidden_size"], n_layers=cfg["num_hidden_layers"],
        n_heads=cfg["num_attention_heads"], ffn_hidden=cfg["intermediate_size"],
        dropout=cfg["hidden_dropout_prob"], pad_id=cfg["pad_token_id"],
        bos_id=cfg["bos_token_id"], eos_id=cfg["eos_token_id"],
    )
    ckpt = torch.load(os.path.join(CKPT_DIR, "pytorch_model.bin"),
                      map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model = GuppyLM(config)
    model.load_state_dict({k: v for k, v in sd.items() if k in model.state_dict()})
    model.eval()
    return model, tok


def chatml(prompt: str) -> str:
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


# ─────────────────────────────────────────────────────────────────────────────
# Vectors, ε and dose for a cell (c, m, S, r)
# ─────────────────────────────────────────────────────────────────────────────

def random_directions(seed: int, L: int = 6, k: int = 384) -> np.ndarray:
    """[L, k] directions, each row uniform on S^{k−1} (normalised Gaussian), fixed seed."""
    g = np.random.default_rng(seed).standard_normal((L, k))
    return g / np.linalg.norm(g, axis=1, keepdims=True)


def concept_vectors(c: str, m: str) -> tuple[np.ndarray, np.ndarray]:
    """(V [L, k] unit, oriented to label 1; μ [L, k] training mean) for concept c, source m.

    rfm     leading AGOP eigenvector, as stored by the pipeline (rfm_guppy/cache)
    dm      rfm_guppy.baselines.diffmeans_vector per block on the cached train/val activations
    random  u = random_directions(U_SEED), independent of c; μ is the concept's training mean
    """
    npz = np.load(os.path.join(CACHE_DIR, f"{c}_vectors.npz"))
    mu = npz["input_mean"].astype(np.float64)
    if m == "rfm":
        return npz["v_per_block"].astype(np.float64), mu
    if m == "random":
        return random_directions(U_SEED, *mu.shape), mu
    if m != "dm":
        raise ValueError(f"source must be one of {SOURCES}, got {m!r}")

    from rfm_guppy.baselines import diffmeans_vector
    with open(os.path.join(CACHE_DIR, f"concept_{c}.json")) as f:
        ds = json.load(f)
    acts = {s: np.load(os.path.join(CACHE_DIR, f"{c}_{s}_activations.npy")) for s in ("train", "val")}
    y = {s: np.array([it["label"] for it in ds[s]], dtype=np.float64) for s in ("train", "val")}
    V, MU = np.zeros_like(mu), np.zeros_like(mu)
    for ell in range(mu.shape[0]):
        V[ell], MU[ell], _ = diffmeans_vector(acts["train"][:, ell].astype(np.float64), y["train"],
                                              acts["val"][:, ell].astype(np.float64), y["val"])
    return V, MU


def block_indices(S: str) -> list[int]:
    return list(range(6)) if S == "all" else [int(S)]


def eps_for(S: str, r: float, med: np.ndarray) -> np.ndarray:
    """ε_ℓ = r · med_ℓ for ℓ ∈ S, 0 elsewhere. With S = all this is Check III's `r * med`."""
    eps = np.zeros(len(med))
    idx = block_indices(S)
    eps[idx] = r * med[idx]
    return eps


def dose(V: np.ndarray, S: str, r: float, med: np.ndarray) -> float:
    """ρ = ‖Σ_{ℓ∈S} r med_ℓ v_ℓ‖₂ / med_5."""
    idx = block_indices(S)
    return float(np.linalg.norm((r * med[idx, None] * V[idx]).sum(0)) / med[-1])


# ─────────────────────────────────────────────────────────────────────────────
# Position-masked steering hooks
# ─────────────────────────────────────────────────────────────────────────────

def position_mask(positions: str, T0: int, T: int) -> torch.Tensor:
    """Boolean [T] mask of 1[t ∈ P] for absolute positions t = 0, …, T−1."""
    t = torch.arange(T)
    if positions == "all":
        return torch.ones(T, dtype=torch.bool)
    if positions == "prompt":
        return t < T0 - 1
    if positions == "gen":
        return t >= T0 - 1
    raise ValueError(f"positions must be one of {POSITION_SETS}, got {positions!r}")


@contextlib.contextmanager
def masked_steering_hooks(model, vectors_per_block: np.ndarray, eps_per_block: np.ndarray,
                          positions: str, T0: int):
    """Forward hooks adding ε_ℓ v_ℓ to block ℓ's output at the positions in P.

    Assumes the block input starts at absolute position 0 (no context crop), which generate()
    guarantees. Masked-in positions are computed as `output + ε·v`, the exact expression of
    rfm_guppy.steering.steering_hooks; the mask is applied with torch.where, so they are
    bitwise equal to it.
    """
    device = next(model.parameters()).device
    handles = []

    for blk_idx, block in enumerate(model.blocks):
        v_t = torch.tensor(vectors_per_block[blk_idx], dtype=torch.float32, device=device)
        eps = float(eps_per_block[blk_idx])

        def _make_hook(vec, e):
            def hook(module, inp, output):
                # output: [B, T, k]
                mask = position_mask(positions, T0, output.shape[1]).to(device)
                return torch.where(mask[None, :, None], output + e * vec, output)
            return hook

        handles.append(block.register_forward_hook(_make_hook(v_t, eps)))

    try:
        yield
    finally:
        for h in handles:
            h.remove()


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate(
    prompt: str,
    model,
    tokenizer,
    vectors_per_block: np.ndarray,
    eps_per_block: np.ndarray,
    positions: str = "all",
    max_new_tokens: int = 60,
    seed: Optional[int] = 42,
    temperature: float = 0.7,
    top_k: int = 50,
    greedy: bool = False,
    return_ids: bool = False,
):
    """Steered generation with a position set. Same output format as generate_steered.

    With return_ids, returns (text, new token ids including a final EOS if one was sampled)."""
    model.eval()
    device = next(model.parameters()).device
    cfg = model.config

    input_ids = tokenizer.encode(chatml(prompt)).ids
    T0 = len(input_ids)
    if T0 + max_new_tokens > cfg.max_seq_len:
        raise ValueError(f"prompt ({T0} tokens) + {max_new_tokens} new tokens exceeds the "
                         f"{cfg.max_seq_len}-token context; positions would shift")
    idx = torch.tensor([input_ids], dtype=torch.long, device=device)

    if seed is not None:
        torch.manual_seed(seed)

    # Mirrors GuppyLM.generate step for step; the crop never fires (checked above).
    with masked_steering_hooks(model, vectors_per_block, eps_per_block, positions, T0):
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits, _ = model(idx)
                logits = logits[:, -1, :]
                if greedy:
                    next_id = logits.argmax(dim=-1, keepdim=True)
                else:
                    logits = logits / temperature
                    if top_k > 0:
                        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                        logits[logits < v[:, [-1]]] = float("-inf")
                    probs = F.softmax(logits, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, next_id], dim=1)
                if next_id.item() == cfg.eos_id:
                    break

    decoded = tokenizer.decode(idx[0, T0:].tolist())
    for ctrl in ("<|im_end|>", "<|im_start|>"):
        if ctrl in decoded:
            decoded = decoded.split(ctrl)[0]
    if return_ids:
        return decoded.strip(), idx[0, T0:].tolist()
    return decoded.strip()
