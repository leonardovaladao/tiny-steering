"""Step 1: pre-registered calibration (§8.1 of PLAN_grilled.md).

Run from the project root:   .venv/bin/python demo-ui-guppy-steer/calibrate.py [--workers 8]

Writes, in demo-ui-guppy-steer/:
  calibration.json              raw grid (per cell and per prompt), bands, selection, confirmation,
                                ticks and presets — the only source the export reads
  calibration_generations.csv   every generated text
  CALIBRATION.md                the write-up, generated from calibration.json

Rules fixed here, before the run, where the plan leaves a choice open:
  - selection ties: most coherent ∧ on-concept, then smallest ρ, then smallest |r|, then r > 0,
    then block-set order 0…5, all, then position-set order all, prompt, gen;
  - "Best regime" preset: the supported cell with the largest k (then smallest p) if any;
    otherwise the (m, S, P) with m ∈ {rfm, dm} and the widest band r⁺ − r⁻, at r = r⁺ (r⁻ if
    r⁺ = 0), ties broken by most coherent ∧ on-concept at that r, then the selection order;
  - a generation whose text is empty has PPL = NaN and is not coherent.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from multiprocessing import get_context

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import steer_ref                                                     # noqa: E402
from steer_ref import ROOT, CACHE_DIR, CONCEPTS, BLOCK_SETS, POSITION_SETS, CONTROL_SEEDS  # noqa: E402
import metrics                                                       # noqa: E402

GEN_KW = dict(max_new_tokens=60, seed=42, temperature=0.7, top_k=50)   # Check III
R_GRID = sorted([s * r for r in (0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.25) for s in (-1, 1)])
SELECTED_SOURCES = ("rfm", "dm")
ON_MARGIN = 0.2
BAND_MIN = 4               # coherent on ≥ 4/7 exploration prompts
X_MIN = 10                 # Σ X_i ≥ 10 of 20
JACC_MAX = 0.5
ALPHA = 0.05 / 6           # Bonferroni over the 6 selected cells
CALIB_CSV = os.path.join(ROOT, "revision_20261002", "run_20261009",
                         "III-steering-metric_calibration.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Workers: one model per process, vectors keyed by (c, m) or a control seed
# ─────────────────────────────────────────────────────────────────────────────

_W = {}


def _init():
    import torch
    torch.set_num_threads(1)
    from rfm_guppy.steering import measure_activation_norms
    _W["model"], _W["tok"] = steer_ref.load_model()
    _W["med"] = measure_activation_norms(CACHE_DIR)
    _W["V"] = {}


def _vectors(vkey):
    if vkey not in _W["V"]:
        if vkey[0] == "control":
            _W["V"][vkey] = steer_ref.random_directions(vkey[1])
        else:
            _W["V"][vkey] = steer_ref.concept_vectors(*vkey)[0]
    return _W["V"][vkey]


def _gen(task):
    prompt, vkey, S, P, r = task
    V = _vectors(vkey)
    eps = steer_ref.eps_for(S, r, _W["med"])
    return steer_ref.generate(prompt, _W["model"], _W["tok"], V, eps, positions=P, **GEN_KW)


def _score(task):
    prompt, text = task
    m, tok = _W["model"], _W["tok"]
    return metrics.uncond_ppl(m, tok, text), metrics.cond_ppl(m, tok, prompt, text)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def check(gates, name, ok, detail=""):
    gates[name] = dict(ok=bool(ok), detail=detail)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        sys.exit(f"gate {name} failed; calibration stopped")


def key(m, S, P):
    return f"{m}|{S}|{P}"


def sel_order(row):
    return (-row["n_joint"], row["rho"], abs(row["r"]), -np.sign(row["r"]),
            BLOCK_SETS.index(row["S"]), POSITION_SETS.index(row["P"]))


def mcnemar_one_sided(X, Y):
    b = int(np.sum((X == 1) & (Y == 0)))
    c2 = int(np.sum((X == 0) & (Y == 1)))
    p = float(stats.binom.sf(b - 1, b + c2, 0.5)) if b + c2 > 0 else 1.0
    return b, c2, p


def fmt_S(S):
    return "all blocks" if S == "all" else f"block {S}"


def fmt_P(P):
    return {"all": "P_all", "prompt": "P_prompt", "gen": "P_gen"}[P]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    gates = {}

    frozen = json.load(open(os.path.join(HERE, "prompts_frozen.json")))
    lists = {k: frozen[k] for k in ("exploration", "calibration", "confirmation")}
    digest = hashlib.sha256(json.dumps(lists, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    check(gates, "F0 prompt lists unchanged since freezing", digest == frozen["sha256"], digest[:16])
    explore, calib_prompts, confirm = lists["exploration"], lists["calibration"], lists["confirmation"]

    from rfm_guppy.steering import measure_activation_norms
    med = measure_activation_norms(CACHE_DIR)
    V = {(c, m): steer_ref.concept_vectors(c, m)[0] for c in CONCEPTS for m in steer_ref.SOURCES}

    print("fitting judges …")
    judges = metrics.fit_judges()
    calib_csv = pd.read_csv(CALIB_CSV, keep_default_na=False)
    err = max(np.abs(metrics.signed(judges[c], calib_csv.text) - calib_csv[f"g_{c}"]).max()
              for c in CONCEPTS)
    check(gates, "J0 judges reproduce Check III g_c on its 100 calibration texts", err < 1e-9,
          f"max |Δg| = {err:.1e}")
    validity = pd.read_csv(os.path.join(ROOT, "revision_20261002", "run_20261009",
                                        "III-steering-metric_judge_validity.csv"))
    validated = {r.concept: bool(r.G1_pass) for r in validity.itertuples()}

    ctx = get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init) as pool:
        # ── thresholds ──────────────────────────────────────────────────────
        zero = ("food", "rfm")
        calib_txt = pool.map(_gen, [(p, zero, "all", "all", 0.0) for p in calib_prompts])
        n_eq = sum(a == b for a, b in zip(calib_txt, calib_csv.text))
        check(gates, "C0 the 100 calibration texts regenerate", n_eq == 100, f"{n_eq}/100")
        calib_ppl = pool.map(_score, list(zip(calib_prompts, calib_txt)))
        ppl_unc_calib = np.array([u for u, _ in calib_ppl])
        PPL_MAX = float(np.nanquantile(ppl_unc_calib, 0.95))
        D2_MIN = float(np.quantile([metrics.distinct2(t) for t in calib_txt], 0.05))
        print(f"thresholds: PPL(y|∅) ≤ {PPL_MAX:.3f},  distinct-2 ≥ {D2_MIN:.3f}")

        def score_rows(prompts, texts, concept, r, y0_g):
            ppl = pool.map(_score, list(zip(prompts, texts)))
            g = metrics.signed(judges[concept], texts)
            rows = []
            for i, (t, (pu, pc), gi) in enumerate(zip(texts, ppl, g)):
                d2 = metrics.distinct2(t)
                coh = bool(pu <= PPL_MAX and d2 >= D2_MIN)            # NaN ⇒ False
                on = bool(np.sign(r) * (gi - y0_g[i]) > ON_MARGIN) if r != 0 else False
                rows.append(dict(ppl_unc=pu, ppl_cond=pc, d2=d2, g=float(gi),
                                 coherent=coh, on=on, joint=coh and on))
            return rows

        # ── exploration: r = 0 references ───────────────────────────────────
        y0_txt = pool.map(_gen, [(p, zero, "all", "all", 0.0) for p in explore])
        stored0 = json.load(open(os.path.join(ROOT, "results", "steer_food.json")))
        stored0 = [o["text"] for o in stored0["results_per_r"]["0.0"]["outputs"]]
        check(gates, "E0 r = 0 exploration texts equal the stored Check III texts",
              y0_txt == stored0, f"{sum(a == b for a, b in zip(y0_txt, stored0))}/7")
        y0_g = {c: metrics.signed(judges[c], y0_txt) for c in CONCEPTS}
        y0_rows = {c: score_rows(explore, y0_txt, c, 0.0, y0_g[c]) for c in CONCEPTS}
        print("r = 0 coherent:", {c: sum(r["coherent"] for r in y0_rows[c]) for c in CONCEPTS})

        # ── exploration grid (random texts do not depend on c: generated once) ──
        cells_spec = [(S, P, r) for S in BLOCK_SETS for P in POSITION_SETS for r in R_GRID]
        vkeys = [(c, m) for c in CONCEPTS for m in SELECTED_SOURCES] + [("food", "random")]
        tasks = [(p, vk, S, P, r) for vk in vkeys for (S, P, r) in cells_spec for p in explore]
        print(f"exploration: {len(tasks)} generations …")
        texts = pool.map(_gen, tasks, chunksize=16)
        T = {}
        for (p, vk, S, P, r), t in zip(tasks, texts):
            T.setdefault((vk[1] if vk[1] == "random" else vk, S, P, r), []).append(t)

        cells, gen_rows = [], []
        for c in CONCEPTS:
            for m in steer_ref.SOURCES:
                tk = (lambda S, P, r: ("random", S, P, r)) if m == "random" else \
                     (lambda S, P, r: ((c, m), S, P, r))
                for (S, P, r) in cells_spec:
                    tx = T[tk(S, P, r)]
                    rows = score_rows(explore, tx, c, r, y0_g[c])
                    cells.append(dict(
                        c=c, m=m, S=S, P=P, r=r, rho=steer_ref.dose(V[(c, m)], S, r, med),
                        n_coherent=sum(x["coherent"] for x in rows),
                        n_on=sum(x["on"] for x in rows),
                        n_joint=sum(x["joint"] for x in rows),
                        mean_jaccard=metrics.mean_jaccard(tx), per_prompt=rows))
                    for i, (t, x) in enumerate(zip(tx, rows)):
                        gen_rows.append(dict(phase="explore", c=c, m=m, S=S, P=P, r=r, i=i,
                                             prompt=explore[i], text=t, **x))
                print(f"  scored {c} / {m}")

        # ── bands ───────────────────────────────────────────────────────────
        by_key = {}
        for x in cells:
            by_key.setdefault((x["c"], key(x["m"], x["S"], x["P"])), {})[x["r"]] = x

        def edge(row_by_r, sign):
            best = 0.0
            for r in sorted((r for r in R_GRID if np.sign(r) == sign), key=abs):
                if row_by_r[r]["n_coherent"] < BAND_MIN:
                    break
                best = r
            return best

        bands = {c: {} for c in CONCEPTS}
        for (c, k), rr in by_key.items():
            bands[c][k] = [edge(rr, -1), edge(rr, +1)]

        # ── selection ───────────────────────────────────────────────────────
        selected = []
        for c in CONCEPTS:
            for m in SELECTED_SOURCES:
                pool_cells = [x for x in cells if x["c"] == c and x["m"] == m]
                s = min(pool_cells, key=sel_order)
                selected.append({k: s[k] for k in ("c", "m", "S", "P", "r", "rho", "n_joint",
                                                    "n_coherent", "n_on", "mean_jaccard")})
                print(f"  selected {c}/{m}: {s['S']} {s['P']} r={s['r']:+} "
                      f"joint {s['n_joint']}/7  ρ={s['rho']:.4f}")

        # ── confirmation ────────────────────────────────────────────────────
        print("confirmation …")
        yc0_txt = pool.map(_gen, [(p, zero, "all", "all", 0.0) for p in confirm])
        yc0_g = {c: metrics.signed(judges[c], yc0_txt) for c in CONCEPTS}
        for i, t in enumerate(yc0_txt):
            gen_rows.append(dict(phase="confirm", c="", m="", S="", P="", r=0.0, i=i,
                                 prompt=confirm[i], text=t))
        confirmation = []
        for s in selected:
            c, m, S, P, r = s["c"], s["m"], s["S"], s["P"], s["r"]
            tx = pool.map(_gen, [(p, (c, m), S, P, r) for p in confirm])
            rows = score_rows(confirm, tx, c, r, yc0_g[c])
            X = np.array([x["joint"] for x in rows], dtype=int)
            ctrl_joint = []
            for seed in CONTROL_SEEDS:
                tr = pool.map(_gen, [(p, ("control", seed), S, P, r) for p in confirm])
                rr = score_rows(confirm, tr, c, r, yc0_g[c])
                ctrl_joint.append([x["joint"] for x in rr])
                for i, (t, x) in enumerate(zip(tr, rr)):
                    gen_rows.append(dict(phase=f"confirm_control_{seed}", c=c, m="random", S=S,
                                         P=P, r=r, i=i, prompt=confirm[i], text=t, **x))
            Y = (np.sum(ctrl_joint, axis=0) >= 3).astype(int)
            b, c2, p = mcnemar_one_sided(X, Y)
            jac = metrics.mean_jaccard(tx)
            crit = dict(k_ge_10=bool(X.sum() >= X_MIN), jaccard_lt_05=bool(jac < JACC_MAX),
                        mcnemar_rejects=bool(p < ALPHA))
            confirmation.append(dict(
                **{k: s[k] for k in ("c", "m", "S", "P", "r", "rho")},
                k=int(X.sum()), n_coherent=sum(x["coherent"] for x in rows),
                n_on=sum(x["on"] for x in rows), X=X.tolist(), Y=Y.tolist(),
                control_joint=ctrl_joint, b=b, c_prime=c2, p_mcnemar=p, mean_jaccard=jac,
                criteria=crit, supported=all(crit.values()), per_prompt=rows))
            for i, (t, x) in enumerate(zip(tx, rows)):
                gen_rows.append(dict(phase="confirm", c=c, m=m, S=S, P=P, r=r, i=i,
                                     prompt=confirm[i], text=t, **x))
            print(f"  {c}/{m}: k={X.sum()}/20  Y={Y.sum()}/20  b={b} c'={c2}  p={p:.4g}  "
                  f"J={jac:.2f}  → {'SUPPORTED' if all(crit.values()) else 'refuted'}")

    # ── ticks and presets ───────────────────────────────────────────────────
    ticks = {c: {} for c in CONCEPTS}
    for x in confirmation:
        if x["supported"]:
            ticks[x["c"]][key(x["m"], x["S"], x["P"])] = dict(
                r_star=x["r"], k_of_20=x["k"], p_mcnemar=x["p_mcnemar"])

    presets = {}
    for c in CONCEPTS:
        sup = [x for x in confirmation if x["c"] == c and x["supported"]]
        if sup:
            b = min(sup, key=lambda x: (-x["k"], x["p_mcnemar"]))
            best = dict(m=b["m"], S=b["S"], P=b["P"], r=b["r"], confirmed=True)
            cap_best = (f"Confirmed cell: {b['m'].upper()}, {fmt_S(b['S'])}, {fmt_P(b['P'])}, "
                        f"r = {b['r']:+g} — coherent and on-concept on {b['k']}/20 new prompts, "
                        f"beating random directions (McNemar p = {b['p_mcnemar']:.2g}).")
        else:
            cand = []
            for m in SELECTED_SOURCES:
                for S in BLOCK_SETS:
                    for P in POSITION_SETS:
                        lo, hi = bands[c][key(m, S, P)]
                        r = hi if hi > 0 else lo
                        row = by_key[(c, key(m, S, P))].get(r) if r != 0 else None
                        nj = row["n_joint"] if row else 0
                        cand.append((-(hi - lo), -nj, sel_order(row) if row else (0,), m, S, P, r))
            w, _, _, m, S, P, r = min(cand, key=lambda t: t[:3])
            best = dict(m=m, S=S, P=P, r=r, confirmed=False)
            cap_best = (f"Widest coherent band found (exploratory, 7 prompts): {m.upper()}, "
                        f"{fmt_S(S)}, {fmt_P(P)}, r = {r:+g}. No cell was confirmed on new prompts.")
        presets[c] = [
            dict(id="check3", label="Check III setup", m="rfm", S="all", P="all", r=0.25,
                 caption="The regime Check III ran: all 6 blocks, every position, r = +0.25. "
                         "It produced no coherent generation."),
            dict(id="best", label="Best regime", **best, caption=cap_best),
            dict(id="random", label="Random control", m="random", S=best["S"], P=best["P"],
                 r=best["r"], caption="Same blocks, positions and r as “Best regime”, along a "
                                      "random unit direction instead of the concept."),
        ]

    out = dict(
        meta=dict(run_on=pd.Timestamp.now().strftime("%Y-%m-%d"), prompts_sha256=digest,
                  gen_kw=GEN_KW, r_grid=R_GRID, block_sets=BLOCK_SETS, position_sets=POSITION_SETS,
                  thresholds=dict(ppl_unc_max=PPL_MAX, d2_min=D2_MIN), on_margin=ON_MARGIN,
                  band_min=BAND_MIN, x_min=X_MIN, jaccard_max=JACC_MAX, alpha=ALPHA,
                  u_seed=steer_ref.U_SEED, control_seeds=CONTROL_SEEDS, med=med.tolist(),
                  validated=validated),
        gates=gates, r0_exploration={c: y0_rows[c] for c in CONCEPTS}, cells=cells,
        bands=bands, selected=selected, confirmation=confirmation, ticks=ticks, presets=presets,
        any_supported={c: bool(ticks[c]) for c in CONCEPTS},
    )
    with open(os.path.join(HERE, "calibration.json"), "w") as f:
        json.dump(out, f, indent=1)
    pd.DataFrame(gen_rows).to_csv(os.path.join(HERE, "calibration_generations.csv"), index=False,
                                  quoting=csv.QUOTE_NONNUMERIC)
    write_md(out, lists)
    print("written calibration.json, calibration_generations.csv, CALIBRATION.md")


# ─────────────────────────────────────────────────────────────────────────────
# Write-up
# ─────────────────────────────────────────────────────────────────────────────

def write_md(out, lists):
    M = out["meta"]
    th = M["thresholds"]
    L = []
    add = L.append
    add("# Calibration — GuppyLM steering demo (step 1)\n")
    add(f"*Generated by `calibrate.py` on {M['run_on']} from `calibration.json`. Pre-registered in "
        "`PLAN_grilled.md` §8.1; do not edit by hand — rerun the script.*\n")
    add("## Verdict\n")
    add("| concept | source | S\\* | P\\* | r\\* | ρ | k/20 | random ≥3/5 | b | c′ | McNemar p | Jaccard | "
        "H-demo |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for x in out["confirmation"]:
        note = "" if M["validated"][x["c"]] else " (not interpreted: G1 failed)"
        add(f"| {x['c']} | {x['m']} | {x['S']} | {x['P']} | {x['r']:+g} | {x['rho']:.4f} | "
            f"{x['k']} | {sum(x['Y'])} | {x['b']} | {x['c_prime']} | {x['p_mcnemar']:.3g} | "
            f"{x['mean_jaccard']:.2f} | **{'supported' if x['supported'] else 'refuted'}**{note} |")
    n_sup = sum(x["supported"] for x in out["confirmation"])
    add(f"\nH-demo is supported for **{n_sup} of 6** (concept, source) pairs at α = 0.05/6.\n")

    add("## Question and criterion\n")
    add("Does some intervention (single block or all, any position set, small |r|) produce "
        "generations that are coherent, on-concept, prompt-dependent, and better than a random "
        "direction? H-demo is supported for (c, m) iff, on the 20 confirmation prompts, "
        f"(1) Σ X_i ≥ {M['x_min']}; (2) mean bigram Jaccard < {M['jaccard_max']}; (3) the exact "
        f"one-sided McNemar test rejects at α = {M['alpha']:.5f}, p = Pr[Bin(b + c′, ½) ≥ b]. "
        "X_i = 1 iff the concept-direction generation is coherent ∧ on-concept; Y_i = 1 iff at "
        "least 3 of the 5 random-direction generations are.\n")

    add("## Metric definitions\n")
    add(f"- **Coherent:** PPL_θ(y | ∅) ≤ {th['ppl_unc_max']:.3f} (q95) and distinct-2 ≥ "
        f"{th['d2_min']:.3f} (q05). y is scored after the ChatML template with an empty user turn; "
        "the quantiles come from the same score on the 100 Check III calibration generations "
        "(regenerated, identical to the stored texts). Empty text is not coherent.")
    add(f"- **On-concept:** sign(r)·[g_c(y_r) − g_c(y_0)] > {M['on_margin']}, y_0 the r = 0 "
        "generation on the same prompt and seed; g_c is the Check III judge (refit, reproduces "
        "Check III's g_c to 1e-9).")
    add("- **Prompt-dependent:** mean pairwise bigram Jaccard across prompts.")
    add("- PPL_θ(y | x) (Check III's metric) is stored per generation as `ppl_cond`.\n")
    add(f"Decoding: {M['gen_kw']}; the seed is set inside every call (paired design). Vectors: "
        f"RFM and DiffMeans as in §3; random u from seed {M['u_seed']}; the confirmation controls "
        f"use seeds {list(M['control_seeds'])}. med_ℓ = {[round(v, 3) for v in M['med']]}.\n")

    add("## Gates\n")
    for k, g in out["gates"].items():
        add(f"- {'PASS' if g['ok'] else 'FAIL'} — {k}" + (f" ({g['detail']})" if g["detail"] else ""))
    add("")

    add("## Exploration grid (7 prompts)\n")
    add(f"{len(out['cells'])} cells: 3 concepts × 3 sources × 7 block sets × 3 position sets × "
        f"{len(M['r_grid'])} values of r. Best cells per (concept, source), ranked by the "
        "selection rule:\n")
    for c in CONCEPTS:
        for m in steer_ref.SOURCES:
            cs = sorted([x for x in out["cells"] if x["c"] == c and x["m"] == m], key=sel_order)[:5]
            add(f"**{c} / {m}**\n")
            add("| S | P | r | ρ | coherent | on | joint | Jaccard |")
            add("|---|---|---|---|---|---|---|---|")
            for x in cs:
                add(f"| {x['S']} | {x['P']} | {x['r']:+g} | {x['rho']:.4f} | {x['n_coherent']}/7 | "
                    f"{x['n_on']}/7 | {x['n_joint']}/7 | {x['mean_jaccard']:.2f} |")
            add("")
    add("**Coherence over the grid** (number of cells with ≥ 4/7 coherent, out of 21 (S, P) per r):\n")
    add("| r | " + " | ".join(f"{c}/{m}" for c in CONCEPTS for m in steer_ref.SOURCES) + " |")
    add("|---|" + "---|" * 9)
    for r in M["r_grid"]:
        vals = [sum(1 for x in out["cells"] if x["c"] == c and x["m"] == m and x["r"] == r
                    and x["n_coherent"] >= M["band_min"]) for c in CONCEPTS for m in steer_ref.SOURCES]
        add(f"| {r:+g} | " + " | ".join(map(str, vals)) + " |")
    add("")

    add("## Bands (exploratory)\n")
    add("[r⁻, r⁺] per (source | blocks | positions); r⁺ is the largest grid r > 0 such that every "
        f"grid r′ ∈ (0, r] is coherent on ≥ {M['band_min']}/7 prompts (0 if none).\n")
    for c in CONCEPTS:
        add(f"**{c}**: " + "; ".join(f"`{k}` [{lo:+g}, {hi:+g}]"
                                     for k, (lo, hi) in sorted(out["bands"][c].items())) + "\n")

    add("## Selected cells and presets\n")
    for s in out["selected"]:
        add(f"- {s['c']} / {s['m']}: S = {s['S']}, P = {s['P']}, r = {s['r']:+g}, ρ = {s['rho']:.4f}, "
            f"coherent ∧ on-concept {s['n_joint']}/7")
    add("")
    for c in CONCEPTS:
        for p in out["presets"][c]:
            add(f"- {c} — *{p['label']}*: {p['caption']}")
    add("")

    add("## Frozen prompt lists\n")
    add(f"SHA-256 `{M['prompts_sha256']}` (`prompts_frozen.json`).\n")
    for name in ("exploration", "confirmation"):
        add(f"**{name}**: " + " · ".join(f"“{p}”" for p in lists[name]) + "\n")
    add("The 100 calibration prompts (Check III's draw) are in `prompts_frozen.json`.\n")

    with open(os.path.join(HERE, "CALIBRATION.md"), "w") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
