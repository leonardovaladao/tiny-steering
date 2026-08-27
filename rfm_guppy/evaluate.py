"""Phase 6: Steering evaluation — probe-score, lexicon, and success-rate metrics.

Three metrics per r value:
  1. Probe score  — feed the (prompt, steered_text) exchange back through GuppyLM
                    and record the aggregate probe's concept-positive probability.
  2. Lexicon hit rate — fraction of outputs containing a concept keyword.
  3. Steering-success rate — probe_score > threshold OR lexicon hit (mirrors
                              the paper's "any coefficient worked" criterion).
"""

from __future__ import annotations

import json
import re

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _lexicon_hit(text: str, keywords: list[str]) -> bool:
    lo = text.lower()
    return any(re.search(r"\b" + re.escape(kw) + r"\b", lo) for kw in keywords)


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation function
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_steering(
    steer_json_path: str,
    probe_pkg: dict,
    eigvecs_top3: np.ndarray,
    input_mean: np.ndarray,
    model,
    tokenizer,
    lexicon: list[str],
    hp_normalize: np.ndarray | None = None,
    probe_threshold: float = 0.5,
) -> dict:
    """Compute probe-score and lexicon metrics for every r in the steering sweep.

    Steered outputs are re-ingested as full ChatML exchanges
    (original probe prompt as user turn, steered text as assistant turn) so that
    the probe sees the same format as the concept-labelled training data.

    Args:
        steer_json_path:  path to results/steer_<concept>.json (Phase 5 output)
        probe_pkg:        trained probe dict from monitoring.train_probes()
        eigvecs_top3:     [L, P, k] AGOP eigenvectors
        input_mean:       [L, k] per-block training mean
        model, tokenizer: GuppyLM
        lexicon:          keyword list for the concept
        hp_normalize:     [L] int 0/1 (from vectors.npz)
        probe_threshold:  probability cutoff for "steered" (default 0.5)

    Returns dict with per-r metrics and overall summary.
    """
    from rfm_guppy.monitoring import predict_probe_probs

    with open(steer_json_path) as f:
        steer_data = json.load(f)

    r_values = steer_data["r_values"]
    results_per_r = steer_data["results_per_r"]
    concept = steer_data["concept"]

    eval_results: dict[str, dict] = {}

    for r in r_values:
        r_str = str(float(r))
        r_data = results_per_r[r_str]
        items = r_data["outputs"]

        # Format steered outputs as full ChatML exchanges for probe activation
        full_exchanges = [
            f"<|im_start|>user\n{item['prompt']}<|im_end|>\n"
            f"<|im_start|>assistant\n{item['text']}<|im_end|>"
            for item in items
        ]

        probe_probs = predict_probe_probs(
            full_exchanges, model, tokenizer,
            probe_pkg, eigvecs_top3, input_mean, hp_normalize,
            format_chatml=False,
        )

        lex_hits = [_lexicon_hit(item["text"], lexicon) for item in items]
        lex_rate = float(np.mean(lex_hits))

        success = [
            float(p) > probe_threshold or h
            for p, h in zip(probe_probs.tolist(), lex_hits)
        ]
        success_rate = float(np.mean(success))

        eval_results[r_str] = {
            "r":                    float(r),
            "mean_probe_score":     float(np.mean(probe_probs)),
            "lexicon_hit_rate":     lex_rate,
            "steering_success_rate": success_rate,
            "n_collapsed":          int(r_data.get("n_collapsed", 0)),
            "probe_probs":          probe_probs.tolist(),
        }
        print(
            f"  r={float(r):+.2f}: "
            f"probe={float(np.mean(probe_probs)):.3f}  "
            f"lex={lex_rate:.3f}  "
            f"success={success_rate:.3f}"
        )

    # Best over positive ε sweep
    pos_keys = [str(float(r)) for r in r_values if float(r) > 0]
    best_r_str = max(pos_keys, key=lambda k: eval_results[k]["steering_success_rate"])
    best_success = eval_results[best_r_str]["steering_success_rate"]
    baseline_success = eval_results["0.0"]["steering_success_rate"]

    return {
        "concept":               concept,
        "eval_per_r":            eval_results,
        "best_r_pos":            float(best_r_str),
        "best_success_rate":     float(best_success),
        "baseline_success_rate": float(baseline_success),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Markdown table formatter
# ─────────────────────────────────────────────────────────────────────────────

def eval_to_markdown(eval_result: dict) -> str:
    """Render a Phase 6 eval result as a readable Markdown table."""
    concept = eval_result["concept"]
    lines = [
        f"## Phase 6 — Steering Evaluation: `{concept}`",
        "",
        f"Baseline (r=0) success rate: **{eval_result['baseline_success_rate']:.1%}**  ",
        f"Best positive-r success rate: **{eval_result['best_success_rate']:.1%}** "
        f"at r={eval_result['best_r_pos']:+.2f}",
        "",
        "| r | Probe score | Lexicon hit | Success rate | Collapsed |",
        "|---|-------------|-------------|--------------|-----------|",
    ]
    for r_str, row in sorted(eval_result["eval_per_r"].items(), key=lambda x: float(x[0])):
        lines.append(
            f"| {row['r']:+.2f} "
            f"| {row['mean_probe_score']:.3f} "
            f"| {row['lexicon_hit_rate']:.3f} "
            f"| {row['steering_success_rate']:.3f} "
            f"| {row['n_collapsed']} |"
        )
    return "\n".join(lines) + "\n"
