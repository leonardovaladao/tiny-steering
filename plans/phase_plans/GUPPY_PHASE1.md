# Phase 1 — Activation Extraction Harness

## What was built

A module `rfm_guppy/activations.py` that captures per-block, last-real-token activations from GuppyLM for arbitrary batches of prompts. This is the data-collection foundation for all later phases (concept vector extraction, steering, and monitoring).

### Core function

```python
def extract_activations(prompts: list[str], model, tokenizer,
                        batch_size: int = 64, format_chatml: bool = True) -> np.ndarray
```

Takes a list of text prompts and returns a NumPy array of shape `[n, 6, 384]` — one 384-dimensional activation vector per transformer block (6 blocks) per prompt.

### How it works

1. **Hook registration.** A `forward_hook` is attached to each of the 6 modules in `model.blocks`. Each hook captures the block's output tensor `[B, T, 384]`, which is the full residual stream after attention and FFN.

2. **Tokenization and batching.** Prompts are wrapped in ChatML format (`<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`) to match GuppyLM's training distribution, then tokenized. Within each batch, sequences are right-padded with `pad_id=0` to a uniform length.

3. **Last-token extraction.** After each forward pass, the hook outputs are indexed at the last non-pad position for each sequence. This follows the paper's convention: the per-prompt feature is the activation at the final real token, which aggregates the model's representation of the entire input.

4. **Cleanup.** Hooks are removed in a `finally` block so the model is left unmodified.

Two helper functions (`save_activations`, `load_activations`) handle caching to `rfm_guppy/cache/` as `.npy` files.

## Acceptance criteria results

| Criterion | Result |
|-----------|--------|
| Output shape `[n, 6, 384]`, no NaNs | **Passed.** 3 prompts → shape `(3, 6, 384)`, zero NaN values. |
| Distinct prompts yield distinct activations | **Passed.** Max element-wise difference between two different prompts: 6.78. |
| Same prompt twice yields identical activations | **Passed.** Max element-wise difference: 0.0 (exact match, with `model.eval()` disabling dropout). |
| Padding invariance | **Passed.** The same prompt batched with short vs. long companions (forcing different padding lengths) produced activations differing by at most 7.15×10⁻⁷ — float32 rounding noise from attention softmax on differently-sized tensors, not a real padding dependency. |

## Observations for later phases

- **Activation norm growth across blocks.** Block 0 activations have norms around 3.1; block 5 norms are around 23.0. The steering coefficient ε in Phase 5 will need to be calibrated relative to these per-block scales, not taken from the paper's Llama-specific values.
- **Causal mask is built internally.** The model constructs `torch.tril(ones(T,T))` inside its forward pass from the sequence length. No external mask needs to be provided, but this is why different padding lengths cause negligible float32 differences (the attention kernel operates on tensors of different size T).
- **Hook target confirmed.** `model.blocks[0..5]` are the correct modules. Each `Block.forward` returns the residual stream tensor directly (not a tuple), so the hook's `output` argument is the tensor itself.
