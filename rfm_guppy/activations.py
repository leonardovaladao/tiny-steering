"""Phase 1: Per-block, last-real-token activation extraction for GuppyLM."""

import os
import numpy as np
import torch


def extract_activations(prompts: list[str], model, tokenizer,
                        batch_size: int = 64, format_chatml: bool = True) -> np.ndarray:
    """Returns array [n, L, k] of last-token block activations.

    For each prompt, tokenizes it (optionally wrapping in ChatML), runs a
    forward pass with hooks on every transformer block, and takes the
    activation at the last non-pad token position.
    """
    model.eval()
    device = next(model.parameters()).device
    pad_id = model.config.pad_id
    max_seq_len = model.config.max_seq_len
    n_blocks = len(model.blocks)
    d_model = model.config.d_model

    block_outputs: dict[int, torch.Tensor] = {}
    hooks = []

    def _make_hook(idx):
        def hook_fn(module, input, output):
            block_outputs[idx] = output.detach()
        return hook_fn

    for i, block in enumerate(model.blocks):
        hooks.append(block.register_forward_hook(_make_hook(i)))

    all_activations = []

    try:
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start:start + batch_size]

            if format_chatml:
                texts = [
                    f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
                    for p in batch
                ]
            else:
                texts = list(batch)

            token_ids = [tokenizer.encode(t).ids for t in texts]
            lengths = [len(ids) for ids in token_ids]
            max_len = min(max(lengths), max_seq_len)

            padded = []
            for ids in token_ids:
                ids = ids[:max_len]
                padded.append(ids + [pad_id] * (max_len - len(ids)))

            input_t = torch.tensor(padded, dtype=torch.long, device=device)

            block_outputs.clear()
            with torch.no_grad():
                model(input_t)

            batch_acts = np.zeros((len(batch), n_blocks, d_model), dtype=np.float32)
            for seq_idx in range(len(batch)):
                last_pos = min(lengths[seq_idx], max_len) - 1
                for blk in range(n_blocks):
                    batch_acts[seq_idx, blk] = (
                        block_outputs[blk][seq_idx, last_pos].cpu().numpy()
                    )

            all_activations.append(batch_acts)
    finally:
        for h in hooks:
            h.remove()

    return np.concatenate(all_activations, axis=0)


def save_activations(acts: np.ndarray, name: str,
                     cache_dir: str = "rfm_guppy/cache") -> str:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{name}_activations.npy")
    np.save(path, acts)
    return path


def load_activations(name: str,
                     cache_dir: str = "rfm_guppy/cache") -> np.ndarray:
    path = os.path.join(cache_dir, f"{name}_activations.npy")
    return np.load(path)
