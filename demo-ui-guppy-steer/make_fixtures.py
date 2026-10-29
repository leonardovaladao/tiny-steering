"""Golden fixtures for the JS engine (Seam B), written by the Python reference (Seam A).

Run from the project root, after export.py:   .venv/bin/python demo-ui-guppy-steer/make_fixtures.py

Writes demo-ui-guppy-steer/fixtures/fixtures.json. Float arrays are base64 of float32-le.
  tokenize      ids of `tokenizers` on the 20 confirmation + 7 exploration prompts, raw and
                ChatML-wrapped, plus edge-case strings; decode of id lists
  logits        last-position logits of the ChatML prompt (20 prompts), and every position for 2
  greedy        64-token greedy generations: unsteered on the 20 prompts; steered on 5 cells
                covering P ∈ {all, prompt, gen} (4 prompts each)
  judge         g_c from sklearn for every concept on every fixture text and 300 calibration texts
  reread        clean re-read p_t = ⟨a_{5,t} − μ_5, v_5⟩ at the positions t ≥ T0 of the generated
                tokens, unsteered model; v_5 = v_rfm for source = random
  dose          ρ for a few cells
"""

import base64
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import steer_ref                                  # noqa: E402
from steer_ref import CACHE_DIR, CONCEPTS         # noqa: E402
import metrics                                    # noqa: E402

OUT = os.path.join(HERE, "fixtures", "fixtures.json")
NEW = 64

STEERED_CELLS = [            # (concept, source, S, P, r)
    ("food", "rfm", "all", "all", 0.25),
    ("food", "dm", "3", "prompt", 0.15),
    ("env_social", "rfm", "5", "gen", 0.1),
    ("valence", "random", "all", "gen", -0.1),
    ("env_social", "dm", "0", "prompt", -0.25),
]
EDGE_STRINGS = [
    "", " ", "hello  world", "Hello, World!!", "  leading and trailing  ", "tabs\tand\nnewlines\n",
    "numbers 12345 and 3.14", "it's don't we'll they're", "café naïve über", "emoji 🐟🐠 fish",
    "<|im_start|>user\n<|im_end|>", "mixed<|im_end|>text<|im_start|>more", "ALL CAPS SHOUTING",
    "punctuation... ?! ;: -- ()[]{}", "über-long-hyphenated-word-with-many-parts",
]


def b64(a):
    return base64.b64encode(np.asarray(a, dtype="<f4").tobytes()).decode()


def main():
    model, tok = steer_ref.load_model()
    from rfm_guppy.steering import measure_activation_norms
    med = measure_activation_norms(CACHE_DIR)
    frozen = json.load(open(os.path.join(HERE, "prompts_frozen.json")))
    prompts = frozen["confirmation"] + frozen["exploration"]
    p20 = frozen["confirmation"]
    L, k = model.config.n_layers, model.config.d_model
    Z_V, Z_EPS = np.zeros((L, k)), np.zeros(L)

    fx = dict(note="written by make_fixtures.py; floats are base64 float32-le", max_new_tokens=NEW)

    # ── tokenizer ───────────────────────────────────────────────────────────
    strings = prompts + [steer_ref.chatml(p) for p in prompts] + EDGE_STRINGS
    fx["tokenize"] = [dict(text=s, ids=tok.encode(s).ids) for s in strings]

    # ── logits ──────────────────────────────────────────────────────────────
    fx["logits"] = []
    with torch.no_grad():
        for i, p in enumerate(p20):
            ids = tok.encode(steer_ref.chatml(p)).ids
            z = model(torch.tensor([ids]))[0][0]
            e = dict(prompt=p, ids=ids, last=b64(z[-1].numpy()))
            if i < 2:
                e["all"] = b64(z.numpy())
            fx["logits"].append(e)

    # ── greedy generations ──────────────────────────────────────────────────
    fx["greedy"] = []
    for p in p20:
        text, ids = steer_ref.generate(p, model, tok, Z_V, Z_EPS, positions="all",
                                       max_new_tokens=NEW, greedy=True, return_ids=True)
        fx["greedy"].append(dict(prompt=p, concept=None, source=None, blocks=None, positions="all",
                                 r=0.0, text=text, ids=ids))
    for (c, m, S, P, r) in STEERED_CELLS:
        V = steer_ref.concept_vectors(c, m)[0]
        for p in p20[:4]:
            text, ids = steer_ref.generate(p, model, tok, V, steer_ref.eps_for(S, r, med),
                                           positions=P, max_new_tokens=NEW, greedy=True,
                                           return_ids=True)
            fx["greedy"].append(dict(prompt=p, concept=c, source=m, blocks=S, positions=P, r=r,
                                     text=text, ids=ids))
    fx["decode"] = [dict(ids=g["ids"], text=tok.decode(g["ids"])) for g in fx["greedy"][:10]]
    fx["decode"] += [dict(ids=[1, 2, 5, 6, 3000, 4095], text=tok.decode([1, 2, 5, 6, 3000, 4095]))]

    # ── judge ───────────────────────────────────────────────────────────────
    judges = metrics.fit_judges()
    calib = pd.read_csv(os.path.join(HERE, "calibration_generations.csv"), keep_default_na=False)
    texts = [g["text"] for g in fx["greedy"]] + list(calib.text.drop_duplicates()[:300]) + EDGE_STRINGS
    fx["judge"] = dict(texts=texts, g={c: metrics.signed(judges[c], texts).tolist() for c in CONCEPTS})

    # ── clean re-read ───────────────────────────────────────────────────────
    fx["reread"] = []
    for g in fx["greedy"][20:]:
        c, m = g["concept"], g["source"]
        V = steer_ref.concept_vectors(c, "rfm" if m == "random" else m)[0]
        mu = steer_ref.concept_vectors(c, "rfm")[1]                      # the exported μ
        ids = tok.encode(steer_ref.chatml(g["prompt"])).ids
        T0 = len(ids)
        x = torch.tensor([(ids + g["ids"])[: model.config.max_seq_len]])
        cap = {}
        h = model.blocks[L - 1].register_forward_hook(lambda mod, i, o: cap.setdefault("h", o))
        try:
            with torch.no_grad():
                model(x)
        finally:
            h.remove()
        a = cap["h"][0, T0:].numpy().astype(np.float64)
        fx["reread"].append(dict(concept=c, source=m, prompt=g["prompt"], ids=g["ids"],
                                 p=((a - mu[L - 1]) @ V[L - 1]).tolist()))

    # ── dose ────────────────────────────────────────────────────────────────
    fx["dose"] = [dict(concept=c, source=m, blocks=S, r=r,
                       rho=steer_ref.dose(steer_ref.concept_vectors(c, m)[0], S, r, med))
                  for (c, m, S, _, r) in STEERED_CELLS + [("food", "rfm", "5", "all", 1.0)]]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(fx, f)
    print(f"written {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB): {len(fx['tokenize'])} tokenize, "
          f"{len(fx['logits'])} logits, {len(fx['greedy'])} greedy, {len(texts)} judge texts")


if __name__ == "__main__":
    main()
