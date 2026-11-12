"""Phase 5: Additive activation steering for GuppyLM.

Each transformer block's output is modified in-place during the forward pass:
    output_ℓ  →  output_ℓ + ε_ℓ · v_ℓ

where v_ℓ ∈ R^{384} is the oriented concept vector for block ℓ (unit norm, positive
direction = label-1 class) and ε_ℓ is the per-block steering coefficient.

The caller sets  ε_{c,ℓ} = r · m_{c,ℓ}  where r is a relative coefficient (the scripts sweep
r over a grid such as {-4, ..., 8}) and m_{c,ℓ} = median_i ‖a^{(c,train)}_{i,ℓ}‖₂ is the median
per-sample norm of block ℓ's activations over concept c's train split, computed by
concept_activation_norms(). measure_activation_norms() is the legacy pooled rule (all concepts,
all splits in the cache), kept for the 2026-08-27 run and the demo.
"""

import contextlib
import os
from typing import Optional

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Norm measurement (calibration)
# ─────────────────────────────────────────────────────────────────────────────

def measure_activation_norms(cache_dir: str) -> np.ndarray:
    """Legacy pooled norms: per-block median of ‖a_ℓ‖₂ over every activation file in the cache.

    Loads every *_activations.npy in cache_dir, i.e. ALL concepts and ALL splits (train, val
    and test), stacks the per-sample norms and returns their per-block median. The result
    therefore changes whenever a concept is added to the cache, and it reads test activations.
    Used by the pinned 2026-08-27 run (reproduced with --norms pooled-legacy) and by the demo
    (demo-ui-guppy-steer/), whose test_step0.py checks it against the stored block_norms.
    New runs use concept_activation_norms().

    Returns: [L] float array
    """
    all_norms = []
    for fname in sorted(os.listdir(cache_dir)):
        if fname.endswith("_activations.npy"):
            acts = np.load(os.path.join(cache_dir, fname))  # [n, L, k]
            norms = np.linalg.norm(acts, axis=2)             # [n, L]
            all_norms.append(norms)
    if not all_norms:
        raise FileNotFoundError(f"No *_activations.npy files found in {cache_dir}")
    combined = np.concatenate(all_norms, axis=0)   # [N, L]
    return np.median(combined, axis=0)              # [L]


def concept_activation_norms(cache_dir: str, concept: str, split: str = "train") -> np.ndarray:
    """Per-block median activation norm of one concept on one split.

    m_{c,ℓ} = median_i ‖a^{(c,split)}_{i,ℓ}‖₂, read from <concept>_<split>_activations.npy
    in cache_dir (shape [n, L, k]). It depends only on that file, so adding another concept to
    the cache leaves it unchanged, and with the default split="train" no val/test activation
    enters the steering step size.

    Returns: [L] float array
    """
    path = os.path.join(cache_dir, f"{concept}_{split}_activations.npy")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found: the per-concept norms of '{concept}' need its {split} "
            f"activations. Run replication/01_extract.py --concepts {concept} first.")
    acts = np.load(path)                       # [n, L, k]
    norms = np.linalg.norm(acts, axis=2)       # [n, L]
    return np.median(norms, axis=0)            # [L]


# ─────────────────────────────────────────────────────────────────────────────
# Steering hook context manager
# ─────────────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def steering_hooks(model, vectors_per_block: np.ndarray, eps_per_block: np.ndarray):
    """Register modifying forward hooks on all transformer blocks.

    Args:
        model: GuppyLM (model.blocks is the nn.ModuleList)
        vectors_per_block: [L, k] float32 concept vectors (unit norm, oriented)
        eps_per_block:     [L]    float  per-block ε (may be 0 to skip a block)

    The hook returns `output + ε_ℓ * v_ℓ`, replacing the block's original output.
    Returning a tensor from a PyTorch forward hook replaces the module output.
    """
    device = next(model.parameters()).device
    handles = []

    for blk_idx, block in enumerate(model.blocks):
        v_t = torch.tensor(
            vectors_per_block[blk_idx], dtype=torch.float32, device=device
        )
        eps = float(eps_per_block[blk_idx])

        def _make_hook(vec, e):
            def hook(module, inp, output):
                # output: [B, T, k]; vec: [k] broadcasts over B and T
                return output + e * vec
            return hook

        handles.append(block.register_forward_hook(_make_hook(v_t, eps)))

    try:
        yield
    finally:
        for h in handles:
            h.remove()


# ─────────────────────────────────────────────────────────────────────────────
# Main generation function
# ─────────────────────────────────────────────────────────────────────────────

def generate_steered(
    prompt: str,
    model,
    tokenizer,
    vectors_per_block: np.ndarray,
    eps_per_block: np.ndarray,
    max_new_tokens: int = 60,
    seed: Optional[int] = 42,
    temperature: float = 0.7,
    top_k: int = 50,
    format_chatml: bool = True,
) -> str:
    """Generate text with additive activation steering.

    Args:
        prompt:            user-facing text
        model:             GuppyLM in eval mode
        tokenizer:         HuggingFace tokenizer
        vectors_per_block: [L, k] oriented unit concept vectors
        eps_per_block:     [L] absolute ε per block (0 = no steering on that block)
        max_new_tokens:    max tokens to generate
        seed:              RNG seed for reproducibility (same seed = same sampling path)
        temperature, top_k: sampling hyperparameters
        format_chatml:     wrap prompt in ChatML user/assistant tags

    Returns:
        Generated text, decoded and stripped of control tokens.
    """
    model.eval()
    device = next(model.parameters()).device

    if format_chatml:
        text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    else:
        text = prompt

    input_ids = tokenizer.encode(text).ids
    input_t = torch.tensor([input_ids], dtype=torch.long, device=device)
    prompt_len = len(input_ids)

    if seed is not None:
        torch.manual_seed(seed)

    with steering_hooks(model, vectors_per_block, eps_per_block):
        with torch.no_grad():
            output_t, _ = model.generate(
                input_t,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )

    new_ids = output_t[0, prompt_len:].tolist()
    decoded = tokenizer.decode(new_ids)

    for ctrl in ("<|im_end|>", "<|im_start|>"):
        if ctrl in decoded:
            decoded = decoded.split(ctrl)[0]

    return decoded.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Lexicon-based quick evaluation (also used in Phase 6)
# ─────────────────────────────────────────────────────────────────────────────

def lexicon_hit(text: str, keywords: list[str]) -> bool:
    """True if any keyword appears as a whole word in lowercased text."""
    import re
    lo = text.lower()
    return any(re.search(r'\b' + re.escape(kw) + r'\b', lo) for kw in keywords)


def lexicon_hit_rate(texts: list[str], keywords: list[str]) -> float:
    """Fraction of texts that contain at least one keyword."""
    if not texts:
        return 0.0
    hits = sum(1 for t in texts if lexicon_hit(t, keywords))
    return hits / len(texts)
