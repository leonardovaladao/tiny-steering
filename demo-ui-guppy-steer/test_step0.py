"""Step 0 gate: the position-masked reference generator reproduces the Check III operation.

Run from the project root:   .venv/bin/python demo-ui-guppy-steer/test_step0.py

Checks
  A1  P = P_all: steer_ref.generate == rfm_guppy.steering.generate_steered, character for
      character, on the 7 Check III prompts × 12 stored r × 3 concepts.
  A2  P = P_all: steer_ref.generate == stored texts in results/steer_<c>.json (same cells).
  A3  r = 0, every P: text equals the unhooked model (G3 of Check III).
  A4  P_prompt and P_gen partition the positions: steering block 5 only, the deltas at its output
      are disjoint in t and add up to the P_all delta.
  A5  greedy decoding does not depend on the seed.
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import steer_ref                                               # noqa: E402
from steer_ref import ROOT, CACHE_DIR, POSITION_SETS          # noqa: E402
from rfm_guppy.steering import generate_steered, measure_activation_norms   # noqa: E402

CONCEPTS = ["food", "valence", "env_social"]
GEN_KW = dict(max_new_tokens=60, seed=42, temperature=0.7, top_k=50)

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


model, tok = steer_ref.load_model()
med = measure_activation_norms(CACHE_DIR)
L, k = model.config.n_layers, model.config.d_model

# ── A1, A2 ───────────────────────────────────────────────────────────────────
for c in CONCEPTS:
    stored = json.load(open(os.path.join(ROOT, "results", f"steer_{c}.json")))
    check(f"{c}: med_ℓ equals the stored block_norms",
          np.allclose(med, stored["block_norms"], rtol=0, atol=1e-9))
    V = np.load(os.path.join(CACHE_DIR, f"{c}_vectors.npz"))["v_per_block"].astype(np.float64)
    n_ref = n_stored = n = 0
    first_bad = None
    for r in stored["r_values"]:
        eps = r * med
        for o in stored["results_per_r"][str(float(r))]["outputs"]:
            new = steer_ref.generate(o["prompt"], model, tok, V, eps, positions="all", **GEN_KW)
            ref = generate_steered(o["prompt"], model, tok, V, eps, **GEN_KW)
            n += 1
            n_ref += new == ref
            n_stored += new == o["text"]
            if (new != ref or new != o["text"]) and first_bad is None:
                first_bad = (r, o["prompt"], new[:60], ref[:60], o["text"][:60])
    check(f"A1 {c}: P_all == generate_steered", n_ref == n, f"{n_ref}/{n}")
    check(f"A2 {c}: P_all == stored Check III text", n_stored == n, f"{n_stored}/{n}")
    if first_bad:
        print("      first mismatch:", first_bad)

# ── A3 ───────────────────────────────────────────────────────────────────────
Z_V, Z_EPS = np.zeros((L, k)), np.zeros(L)
stored0 = json.load(open(os.path.join(ROOT, "results", "steer_food.json")))["results_per_r"]["0.0"]
V_food = np.load(os.path.join(CACHE_DIR, "food_vectors.npz"))["v_per_block"]
for P in POSITION_SETS:
    ok = all(
        steer_ref.generate(o["prompt"], model, tok, V_food, 0.0 * med, positions=P, **GEN_KW)
        == o["text"]
        for o in stored0["outputs"]
    )
    check(f"A3 r=0, P_{P}: equals unhooked model (7 prompts)", ok)

# ── A4 ───────────────────────────────────────────────────────────────────────
prompt = stored0["outputs"][0]["prompt"]
cont = stored0["outputs"][0]["text"]
ids = tok.encode(steer_ref.chatml(prompt)).ids
T0 = len(ids)
x = torch.tensor([(ids + tok.encode(cont).ids)[:model.config.max_seq_len]])
eps5 = np.zeros(L)
eps5[5] = 0.5 * med[5]


def block5_out(P, eps):
    cap = {}
    with steer_ref.masked_steering_hooks(model, V_food, eps, P, T0):
        h = model.blocks[5].register_forward_hook(lambda m, i, o: cap.setdefault("h", o.clone()))
        try:
            with torch.no_grad():
                model(x)
        finally:
            h.remove()
    return cap["h"][0]


base = block5_out("all", np.zeros(L))
d = {P: block5_out(P, eps5) - base for P in POSITION_SETS}
supp = {P: d[P].abs().sum(-1) > 0 for P in POSITION_SETS}
check("A4 supports: P_prompt ∩ P_gen = ∅", not (supp["prompt"] & supp["gen"]).any())
check("A4 supports: P_prompt ∪ P_gen = P_all = every position",
      bool((supp["prompt"] | supp["gen"]).all()) and bool(supp["all"].all()))
check("A4 P_gen starts at T0 − 1", int(supp["gen"].float().argmax()) == T0 - 1,
      f"T0 = {T0}, sequence length {x.shape[1]}")
err = (d["prompt"] + d["gen"] - d["all"]).abs().max().item()
check("A4 deltas add up: Δ_prompt + Δ_gen = Δ_all", err < 1e-5, f"max |err| = {err:.2e}")

# ── A5 ───────────────────────────────────────────────────────────────────────
g = [steer_ref.generate(prompt, model, tok, V_food, 0.1 * med, positions="gen",
                        greedy=True, seed=s) for s in (0, 42, 123)]
check("A5 greedy output independent of seed", len(set(g)) == 1, repr(g[0][:60]))

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
