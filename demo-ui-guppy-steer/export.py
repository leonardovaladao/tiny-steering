"""Step 2: export the browser bundle (SPEC.md, "Export").

Run from the project root:   .venv/bin/python demo-ui-guppy-steer/export.py

Writes demo-ui-guppy-steer/web/data/:
  weights_0.wasm … weights_2.wasm  raw fp32, row-major, little-endian (not WebAssembly: .wasm is
                                  the binary type the Artifact host serves); tied lm_head stored once
  weights.json                    manifest: tensor name → {file, offset (bytes), shape}; model config
  tokenizer.json                  copy of the checkpoint tokenizer
  shared.json                     med[6], u[6×384], u_seed
  concept_<c>.json                §5 of PLAN_grilled.md; bands, ticks and presets read from
                                  calibration.json, never typed by hand

Gate (PASS/FAIL): reloaded tensors identical to the checkpoint; exported judge reproduces sklearn's
g_c; exported vectors equal steer_ref.concept_vectors.
"""

import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import steer_ref                                                  # noqa: E402
from steer_ref import ROOT, CKPT_DIR, CACHE_DIR, CONCEPTS         # noqa: E402
import metrics                                                    # noqa: E402
from rfm_guppy.paths import CHECKS_RUN_2026_10_09                 # noqa: E402
from rfm_guppy.concepts import CONCEPTS as CONCEPT_DEFS           # noqa: E402

OUT = os.path.join(HERE, "web", "data")
N_FILES = 3
MAX_BYTES = 15 * 1024 * 1024

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


def judge_numpy(j, texts):
    """g_c from the exported judge parameters, re-implemented without sklearn."""
    import re
    from collections import Counter
    idx = {t: i for i, t in enumerate(j["vocab"])}
    idf, w = np.array(j["idf"]), np.array(j["coef"])
    out = []
    for t in texts:
        toks = re.findall(r"(?u)\b\w+\b", t.lower())
        grams = toks + [f"{a} {b}" for a, b in zip(toks, toks[1:])]
        cnt = Counter(g for g in grams if g in idx)
        x = {idx[g]: (1 + np.log(n)) * idf[idx[g]] for g, n in cnt.items()}
        nrm = np.sqrt(sum(v * v for v in x.values()))
        z = j["intercept"] + (sum(w[i] * v for i, v in x.items()) / nrm if nrm > 0 else 0.0)
        out.append(2 / (1 + np.exp(-z)) - 1)
    return np.array(out)


def main():
    os.makedirs(OUT, exist_ok=True)
    model, tok = steer_ref.load_model()
    cfg = model.config

    # ── weights ─────────────────────────────────────────────────────────────
    sd = {k: v.detach().cpu().numpy().astype("<f4") for k, v in model.state_dict().items()
          if k != "lm_head.weight"}
    total = sum(a.nbytes for a in sd.values())
    target = -(-total // N_FILES)
    manifest, chunks, cur, off = {}, [[]], 0, 0
    for name, a in sd.items():
        if off > 0 and off + a.nbytes > target and len(chunks) < N_FILES:
            chunks.append([])
            off = 0
        manifest[name] = dict(file=f"weights_{len(chunks) - 1}.wasm", offset=off, shape=list(a.shape))
        chunks[-1].append(a)
        off += a.nbytes
    for i, ch in enumerate(chunks):
        with open(os.path.join(OUT, f"weights_{i}.wasm"), "wb") as f:
            for a in ch:
                f.write(np.ascontiguousarray(a).tobytes())
    sizes = [os.path.getsize(os.path.join(OUT, f"weights_{i}.wasm")) for i in range(len(chunks))]
    with open(os.path.join(OUT, "weights.json"), "w") as f:
        json.dump(dict(
            files=[f"weights_{i}.wasm" for i in range(len(chunks))], tensors=manifest,
            tied={"lm_head.weight": "tok_emb.weight"}, dtype="float32-le",
            config=dict(vocab_size=cfg.vocab_size, max_seq_len=cfg.max_seq_len, d_model=cfg.d_model,
                        n_layers=cfg.n_layers, n_heads=cfg.n_heads, ffn_hidden=cfg.ffn_hidden,
                        pad_id=cfg.pad_id, bos_id=cfg.bos_id, eos_id=cfg.eos_id, layer_norm_eps=1e-5),
        ), f, indent=1)
    shutil.copy(os.path.join(CKPT_DIR, "tokenizer.json"), os.path.join(OUT, "tokenizer.json"))

    check(f"weights in {len(chunks)} files of ≤ 15 MB", len(chunks) == N_FILES and max(sizes) <= MAX_BYTES,
          ", ".join(f"{s / 1e6:.2f} MB" for s in sizes))
    ok = True
    ref = model.state_dict()
    for name, e in manifest.items():
        n = int(np.prod(e["shape"]))
        a = np.fromfile(os.path.join(OUT, e["file"]), dtype="<f4", count=n,
                        offset=e["offset"]).reshape(e["shape"])
        ok &= np.array_equal(a, ref[name].numpy())
    ok &= np.array_equal(ref["lm_head.weight"].numpy(), ref["tok_emb.weight"].numpy())
    check("B1 reloaded tensors identical to the checkpoint (lm_head = tok_emb)", ok,
          f"{len(manifest)} tensors")

    # ── shared ──────────────────────────────────────────────────────────────
    from rfm_guppy.steering import measure_activation_norms
    med = measure_activation_norms(CACHE_DIR)
    u = steer_ref.random_directions(steer_ref.U_SEED)
    with open(os.path.join(OUT, "shared.json"), "w") as f:
        json.dump(dict(med=med.tolist(), u=u.tolist(), u_seed=steer_ref.U_SEED,
                       concepts=list(CONCEPTS)), f)

    # ── concept bundles ─────────────────────────────────────────────────────
    cal = json.load(open(os.path.join(HERE, "calibration.json")))
    validity = pd.read_csv(os.path.join(CHECKS_RUN_2026_10_09,
                                        "III-steering-metric_judge_validity.csv")).set_index("concept")
    judges = metrics.fit_judges()
    texts = pd.read_csv(os.path.join(HERE, "calibration_generations.csv"), keep_default_na=False).text
    texts = list(texts.drop_duplicates()[:2000])
    for c in CONCEPTS:
        V = {m: steer_ref.concept_vectors(c, m) for m in ("rfm", "dm")}
        vec, lr = judges[c].named_steps["tfidfvectorizer"], judges[c].named_steps["logisticregression"]
        vocab = [None] * len(vec.vocabulary_)
        for t, i in vec.vocabulary_.items():
            vocab[i] = t
        judge = dict(vocab=vocab, idf=vec.idf_.tolist(), coef=lr.coef_[0].tolist(),
                     intercept=float(lr.intercept_[0]), ngram_range=[1, 2],
                     token_pattern=r"(?u)\b\w+\b", lowercase=True, sublinear_tf=True, norm="l2")
        auroc = float(validity.loc[c, "cv_auroc_full"])
        bundle = dict(
            name=c,
            pos_label=" / ".join(CONCEPT_DEFS[c]["pos"]), neg_label=" / ".join(CONCEPT_DEFS[c]["neg"]),
            validated=bool(validity.loc[c, "G1_pass"]), judge_cv_auroc=auroc,
            badge=None if validity.loc[c, "G1_pass"] else
                  f"judge not validated (G1 failed: CV AUROC {auroc:.2f})",
            blocks=[dict(v_rfm=V["rfm"][0][l].tolist(), v_dm=V["dm"][0][l].tolist(),
                         mu=V["rfm"][1][l].tolist()) for l in range(cfg.n_layers)],
            bands=cal["bands"][c], band_label="exploratory (7 prompts)",
            ticks=cal["ticks"][c], presets=cal["presets"][c],
            any_supported=cal["any_supported"][c], any_supported_global=any(cal["any_supported"].values()),
            judge=judge,
        )
        with open(os.path.join(OUT, f"concept_{c}.json"), "w") as f:
            json.dump(bundle, f)

        err = np.abs(judge_numpy(judge, texts) - metrics.signed(judges[c], texts)).max()
        check(f"B2 {c}: exported judge reproduces sklearn g_c", err < 1e-9,
              f"{len(texts)} texts, max |Δg| = {err:.1e}")
        rt = json.load(open(os.path.join(OUT, f"concept_{c}.json")))
        ok = all(np.array_equal(np.array([b[f"v_{m}"] for b in rt["blocks"]]), V[m][0])
                 for m in ("rfm", "dm"))
        ok &= np.array_equal(np.array([b["mu"] for b in rt["blocks"]]), V["rfm"][1])
        ok &= np.allclose(V["dm"][1], V["rfm"][1], rtol=0, atol=1e-5)
        check(f"B3 {c}: exported v_rfm, v_dm, μ round-trip (μ_dm = μ_rfm)", ok)
        ok = rt["bands"] == cal["bands"][c] and rt["ticks"] == cal["ticks"][c]
        check(f"B4 {c}: bands and ticks equal calibration.json", ok,
              f"{len(rt['bands'])} bands, {len(rt['ticks'])} ticks")

    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
