# Phase 0 — Environment Setup & Smoke Test

## What was done

### 1. Cloned GuppyLM

The GuppyLM repository was cloned from `https://github.com/arman-bd/guppylm` into `./guppylm/`. This provides the model architecture (`model.py`), config (`config.py`), and inference code (`inference.py`).

### 2. Created a Python virtual environment

A venv was created at `.venv/` and the following packages were installed:

- **From GuppyLM's requirements:** `torch`, `tokenizers`, `tqdm`, `numpy`, `datasets`
- **Additional (for the replication):** `scipy`, `scikit-learn`, `matplotlib`, `pandas`, `huggingface_hub`

### 3. Downloaded the pretrained model

The pretrained checkpoint was downloaded from HuggingFace (`arman-bd/guppylm-9M`) into `checkpoints/guppylm-9M/`. Key files:

| File | Purpose |
|------|---------|
| `pytorch_model.bin` | Trained weights |
| `tokenizer.json` | BPE tokenizer |
| `config.json` | Model configuration |

### 4. Created project scaffolding

The directory structure for the replication package was created:

```
rfm_guppy/          # Will hold all replication code
rfm_guppy/__init__.py
rfm_guppy/cache/    # For saved activations, vectors, results
experiments/        # Experiment scripts
results/            # Figures, tables, final report
```

## Model details confirmed

| Property | Value |
|----------|-------|
| Parameters | 8,726,016 (8.7M) |
| Transformer blocks | 6 (`model.blocks`, an `nn.ModuleList` of `Block` modules) |
| Hidden dimension | 384 |
| Attention heads | 6 |
| FFN hidden dim | 768 (ReLU activation) |
| Vocab capacity | 4,096 |
| Active BPE tokens | 2,418 |
| Max sequence length | 128 tokens |
| Special tokens | pad=0, bos=1 (`<\|im_start\|>`), eos=2 (`<\|im_end\|>`) |
| Normalization | LayerNorm (pre-norm residual blocks) |
| Position encoding | Learned embeddings |
| LM head | Weight-tied with token embeddings |

The block architecture is a standard pre-norm transformer: `x = x + attn(norm1(x))`, then `x = x + ffn(norm2(x))`. The full residual stream tensor `[B, T, 384]` is the block output — this is the target for activation hooks in later phases.

## Acceptance criteria results

| Criterion | Result |
|-----------|--------|
| `./guppylm` imports; model + tokenizer load in Python | **Passed.** Model loads as 8.7M params, tokenizer loads with 2,418 tokens. |
| Forward pass on a sample prompt returns logits of shape `[B, T, 4096]` | **Passed.** Input `[1, 10]` produced logits `[1, 10, 4096]`. |
| Inference produces coherent in-character output | **Passed.** Prompt "hi guppy" produced: *"oh hello. you look big today. you always look big. i blew some bubbles earlier."* |

## Notes for later phases

- **Hook target for activation extraction (Phase 1):** Register hooks on each module in `model.blocks` (indices 0–5). Each block's output is the residual stream `[B, T, 384]`.
- **Padding handling:** The model uses `pad_id=0`. When extracting last-token activations, the hook must find the last non-pad position per sequence.
- **Prompt format:** The model expects ChatML-style formatting: `<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n`.
- **Checkpoint paths:** `checkpoints/guppylm-9M/pytorch_model.bin` (weights), `checkpoints/guppylm-9M/tokenizer.json` (tokenizer).
- **No causal mask input:** The model builds its own causal mask internally from sequence length — no mask needs to be passed.
