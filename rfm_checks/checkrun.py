"""Check runs: a run folder for Checks I-IV and the κ side run, with its configuration in run.json.

A check run is a run folder in the sense of `rfm_guppy/runs.py` (append-only, provenance in
`run.json`) whose `run.json` also holds a `config` block, written once by `new`:

    {"study": "rfm_checks", "created": "<ISO time>",
     "config": {"concepts": [...],                      # Checks I, Ia, II, III and κ
                "replication_run": "replication/runs/2026-08-27",
                "IV": {"concepts": [...], "main": [...]}},  # "main" only when given (--iv-main)
     "steps": [...]}                                    # one per notebook / script invocation

Create a run (from the repo root):

    .venv/bin/python -m rfm_checks.checkrun new [--run DIR] [--concepts NAME ...]
        [--iv-main NAME ...] [--replication-run DIR]

`new` writes `run.json` and copies the five check notebooks into the run folder, with their
stored outputs cleared. Execute each copy in place there (`jupyter nbconvert --to notebook
--execute --inplace`); every notebook calls `load_check_run` in its setup cell.

Concept order fixes the seeds. Check Ia and the κ side run draw permutations sequentially over the
concept list, and Check I and Check IV seed with the concept's index in their list. So each check
keeps its historical order for the default concepts (food, valence, env_social for I, Ia, II, III
and κ; valence, env_social, food for IV) and appends extra concepts in the order given. A default
run therefore reproduces `rfm_checks/runs/2026-10-09/`. Dropping a default concept shifts the
seeds of the concepts after it.

Check IV's MAIN set (the concepts its H3a / H-artigo / H3b verdicts are about) is pre-registered
in `rfm_checks/IV-small-n_design.md`. The default MAIN (valence, env_social) applies only to the
default concept list; any other list needs `--iv-main` at `new` time, which is a design amendment.

Environment variables read by the notebooks:
  RFM_RUN    run folder to use (default: the notebook's current directory).
  RFM_FORCE  "1" allows overwriting an output file that already exists in the run folder.
             Without it every write refuses an existing file (FileExistsError).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

if __package__ in (None, ""):                         # run as a file: put the repo root on sys.path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rfm_guppy import paths, runs                     # noqa: E402
from rfm_guppy.concepts import resolve_concepts       # noqa: E402

NOTEBOOKS = ("I-shuffled-label-control", "Ia-shuffle-label-on-blocks", "II-PCA-baseline",
             "III-steering-metric", "IV-small-n")
KAPPA = "kappa_side_run"

DEFAULT_CONCEPTS = ("food", "valence", "env_social")       # I, Ia, II, III, κ (historical order)
IV_DEFAULT_CONCEPTS = ("valence", "env_social", "food")    # IV (historical order)
IV_DEFAULT_MAIN = ("valence", "env_social")                # pre-registered for the default list
IV_MAIN_SIZE = 2      # Holm k = 10 = 2 main concepts x 5 values of n, hard-coded in IV cells 13, 15
IV_DESIGN = "rfm_checks/IV-small-n_design.md"

# Files each check writes into the run folder (checked before any work starts).
OUTPUTS = {
    "I-shuffled-label-control": [
        "I-shuffled-label-control_results.npz", "I-shuffled-label-control_resultsB.npz",
        "I-shuffled-label-control_resultsC.npz", "I-shuffled-label-control_share_vs_T.png",
        "I-shuffled-label-control_rho_baselines.csv", "I-shuffled-label-control_regimeA.csv",
        "I-shuffled-label-control_regimeB.csv"],
    "Ia-shuffle-label-on-blocks": [
        "Ia-shuffle-label-on-blocks_sT.npz", "Ia-shuffle-label-on-blocks_H0_tests.csv",
        "Ia-shuffle-label-on-blocks_per_cell.csv", "Ia-shuffle-label-on-blocks_stored_hp.csv",
        "Ia-shuffle-label-on-blocks_share_vs_T.png"],
    "II-PCA-baseline": [
        "II-PCA-baseline_perblock.csv", "II-PCA-baseline_headline.csv",
        "II-PCA-baseline_verdict.csv", "II-PCA-baseline_bootstrap.csv"],
    "III-steering-metric": [
        f"III-steering-metric_{s}.csv" for s in (
            "judge_validity", "calibration", "generations", "per_r", "H1", "H2", "H3",
            "old_vs_new", "duplication")],
    "IV-small-n": [
        "IV-small-n_hp.csv", "IV-small-n_draws.csv", "IV-small-n_summary.csv",
        "IV-small-n_T_sensitivity.csv", "IV-small-n_verdicts.csv", "IV-small-n_curves.png",
        "IV-small-n_T_curves.png"],
    KAPPA: ["kappa.csv"],
}

CACHE_FILES = ("concept_{c}.json", "{c}_train_activations.npy", "{c}_val_activations.npy",
               "{c}_test_activations.npy", "{c}_vectors.npz")


# ── small helpers ──────────────────────────────────────────────────────────────

def force() -> bool:
    """True when the environment variable RFM_FORCE is "1"."""
    return os.environ.get("RFM_FORCE", "") == "1"


def _out_path(run_dir: str, name: str) -> str:
    """`runs.out_path` with RFM_FORCE as the force flag, and an error message that names it."""
    try:
        return runs.out_path(run_dir, name, force())
    except FileExistsError:
        raise FileExistsError(
            f"{runs.display_path(os.path.join(run_dir, name))} already exists. Run folders are "
            "append-only: create a new check run (python -m rfm_checks.checkrun new), or set "
            "RFM_FORCE=1 to overwrite it.") from None


def out_file(path: str) -> str:
    """`path` (a file in a run folder), refusing to overwrite it unless RFM_FORCE=1.

    Every notebook write goes through this function; it calls `runs.out_path`, so pinned run
    folders are refused even with RFM_FORCE=1.
    """
    return _out_path(os.path.dirname(os.path.abspath(path)), os.path.basename(path))


def _ordered(default: tuple[str, ...], requested: list[str]) -> list[str]:
    """Default concepts in their historical order, then the extra ones in the order given."""
    return [c for c in default if c in requested] + [c for c in requested if c not in default]


def _abspath(path: str) -> str:
    """A config path: repo-relative paths are resolved against the repo root."""
    return path if os.path.isabs(path) else os.path.join(paths.ROOT, path)


def replication_run(cfg: dict) -> str:
    """Absolute path of the replication run the config reads from (II, III; read only)."""
    return _abspath(cfg["replication_run"])


def concepts_for(cfg: dict, name: str) -> list[str]:
    """Concept list, in seed order, that notebook / script `name` runs on."""
    return list(cfg["IV"]["concepts"] if name == "IV-small-n" else cfg["concepts"])


def iv_main(cfg: dict) -> list[str]:
    """Check IV's MAIN set from the config; raise ValueError if it is not pre-registered.

    The default MAIN (valence, env_social) is valid only for the default IV concept list. For any
    other list, `config.IV.main` must be present (written by `new --iv-main`). MAIN must be a
    subset of IV's concepts with IV_MAIN_SIZE elements (Holm k = 10 in IV cells 13 and 15).
    """
    iv = cfg["IV"]
    concepts, main = list(iv["concepts"]), iv.get("main")
    if main is None:
        if concepts != list(IV_DEFAULT_CONCEPTS):
            raise ValueError(
                f"Check IV: the concept list {concepts} differs from the default "
                f"{list(IV_DEFAULT_CONCEPTS)}, and run.json has no config.IV.main. MAIN is "
                "pre-registered only for the default list. Choosing a new MAIN is a design "
                f"amendment to {IV_DESIGN}: record it there, then create a new run with "
                "`.venv/bin/python -m rfm_checks.checkrun new --concepts ... --iv-main NAME NAME`.")
        return list(IV_DEFAULT_MAIN)
    main = list(main)
    outside = [c for c in main if c not in concepts]
    if outside:
        raise ValueError(f"Check IV: MAIN {main} contains {outside}, which are not in IV's "
                         f"concepts {concepts}.")
    if len(main) != IV_MAIN_SIZE or len(set(main)) != len(main):
        raise ValueError(
            f"Check IV: MAIN must name {IV_MAIN_SIZE} distinct concepts (got {main}). The Holm "
            "family k = 10 = 2 main concepts x 5 values of n is fixed in IV cells 13 and 15 and in "
            f"§3.5 of {IV_DESIGN}; another size needs a design amendment and a code change.")
    return main


# ── creating a run ─────────────────────────────────────────────────────────────

def make_config(concepts: list[str] | None = None, iv_main_names: list[str] | None = None,
                replication: str | None = None) -> dict:
    """Validate the options of `new` and return the `config` block. Raises ValueError."""
    requested = resolve_concepts(concepts) if concepts is not None else list(DEFAULT_CONCEPTS)
    cfg = {"concepts": _ordered(DEFAULT_CONCEPTS, requested)}

    repl = os.path.abspath(replication) if replication is not None else paths.RESULTS_DIR
    cfg["replication_run"] = runs.display_path(repl)

    cfg["IV"] = {"concepts": _ordered(IV_DEFAULT_CONCEPTS, requested)}
    if iv_main_names is not None:
        cfg["IV"]["main"] = list(dict.fromkeys(iv_main_names))
        iv_main(cfg)                    # subset of IV's concepts, right size

    missing = [f.format(c=c) for c in cfg["concepts"] for f in CACHE_FILES
               if not os.path.isfile(os.path.join(paths.CACHE_DIR, f.format(c=c)))]
    if missing:
        raise ValueError(f"missing in {runs.display_path(paths.CACHE_DIR)}: {', '.join(missing)}. "
                         "Build them with the replication pipeline (replication/README.md).")

    missing = [f"{n}_{c}.json" for c in cfg["concepts"] for n in ("steer", "eval6")
               if not os.path.isfile(os.path.join(repl, f"{n}_{c}.json"))]
    mon = os.path.join(repl, "baselines_monitoring.json")
    if not os.path.isfile(mon):
        missing.append("baselines_monitoring.json")
    else:
        with open(mon) as f:
            have = json.load(f)
        missing += [f"baselines_monitoring.json entry '{c}'" for c in cfg["concepts"] if c not in have]
    if missing:
        raise ValueError(f"missing in the replication run {runs.display_path(repl)}: "
                         f"{', '.join(missing)}. Check II reads baselines_monitoring.json and "
                         "Check III reads steer_<c>.json and eval6_<c>.json from it.")
    return cfg


def _copy_notebook(src: str, dst: str) -> None:
    """Copy a notebook with its outputs and execution counts cleared."""
    with open(src, encoding="utf-8") as f:
        nb = json.load(f)
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"], cell["execution_count"] = [], None
    with open(dst, "w", encoding="utf-8") as f:
        f.write(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")


def create_check_run(run_dir: str | None = None, concepts: list[str] | None = None,
                     iv_main_names: list[str] | None = None,
                     replication: str | None = None) -> str:
    """Create a check run folder with run.json (config, no steps) and the five notebooks."""
    run_dir = os.path.abspath(run_dir or runs.default_run_dir(paths.CHECKS_DIR))
    runs.check_writable(run_dir)
    if os.path.realpath(run_dir) == os.path.realpath(paths.CHECKS_DIR):
        raise PermissionError("refusing to use rfm_checks/ itself as a run folder.")
    if os.path.isdir(run_dir) and os.listdir(run_dir):
        raise FileExistsError(f"{runs.display_path(run_dir)} exists and is not empty. A check run "
                              "starts in a new or empty folder.")
    if os.path.exists(run_dir) and not os.path.isdir(run_dir):
        raise FileExistsError(f"{runs.display_path(run_dir)} exists and is not a folder.")
    cfg = make_config(concepts, iv_main_names, replication)

    os.makedirs(run_dir, exist_ok=True)
    record = {"study": runs.display_path(paths.CHECKS_DIR), "created": runs._now(),
              "config": cfg, "steps": []}
    with open(os.path.join(run_dir, runs.RUN_JSON), "w") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    for name in NOTEBOOKS:
        _copy_notebook(os.path.join(paths.CHECKS_DIR, f"{name}.ipynb"),
                       os.path.join(run_dir, f"{name}.ipynb"))
    return run_dir


# ── loading a run (called by every notebook and by kappa_side_run.py) ──────────

def read_check_run(run_dir: str | None = None) -> tuple[str, dict]:
    """Locate a check run and return `(run_dir, cfg)` without writing anything.

    run_dir: the argument, else the environment variable RFM_RUN, else the current directory. It
    must hold a run.json with a `config` block (made by `new`). Refused: rfm_checks/ itself and
    the pinned runs (paths.PINNED_RUNS).
    """
    if run_dir is None:
        run_dir = os.environ.get("RFM_RUN") or os.getcwd()
    run_dir = os.path.abspath(run_dir)
    if os.path.realpath(run_dir) == os.path.realpath(paths.CHECKS_DIR):
        raise PermissionError(
            "refusing to run in rfm_checks/ itself: the top-level notebooks are templates. Create "
            "a run with `.venv/bin/python -m rfm_checks.checkrun new`, then execute the copies in "
            "that folder (or set RFM_RUN to it).")
    runs.check_writable(run_dir)

    record_path = os.path.join(run_dir, runs.RUN_JSON)
    hint = ("Create a check run with `.venv/bin/python -m rfm_checks.checkrun new` and run from "
            "that folder, or set RFM_RUN to it.")
    if not os.path.isfile(record_path):
        raise FileNotFoundError(f"{runs.display_path(run_dir)} has no {runs.RUN_JSON}. {hint}")
    with open(record_path) as f:
        cfg = json.load(f).get("config")
    if not cfg:
        raise ValueError(f"{runs.display_path(record_path)} has no config block. {hint}")
    return run_dir, cfg


def load_check_run(name: str, params: dict | None = None, run_dir: str | None = None,
                   reuse: tuple[str, ...] | list[str] = (), script: str | None = None):
    """Open the check run for notebook (or script) `name`; return `(run_dir, cfg)`.

    The run folder is found and checked by `read_check_run`. Before any work, every output of
    `name` (OUTPUTS) is checked with `runs.out_path`, except the cached files in `reuse` that the
    notebook will reload instead of writing; an existing output raises FileExistsError unless RFM_FORCE=1. Then one
    step is appended to run.json (`runs.open_run`): script = `script` or the notebook path,
    concepts = the list this check runs on, params = `params` plus RFM_FORCE (and MAIN for IV).
    """
    run_dir, cfg = read_check_run(run_dir)
    concepts = concepts_for(cfg, name)
    resolve_concepts(concepts)                          # still registered (run.json may be edited)
    params = dict(params or {})
    if name == "IV-small-n":
        params["MAIN"] = iv_main(cfg)

    for out in OUTPUTS[name]:
        if out not in reuse:
            _out_path(run_dir, out)

    if script is None:
        here = os.path.join(os.getcwd(), f"{name}.ipynb")
        script = here if os.path.isfile(here) else os.path.join(run_dir, f"{name}.ipynb")
    params["RFM_FORCE"] = force()
    runs.open_run(run_dir, script, concepts, params=params)
    print(f"[checkrun] {name}: run folder {runs.display_path(run_dir)}, concepts {concepts}")
    return run_dir, cfg


# ── CLI ────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m rfm_checks.checkrun",
                                     description="Create a check run for Checks I-IV and κ.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("new", help="create a run folder with run.json (config) and the notebooks")
    p.add_argument("--run", metavar="DIR",
                   help="run folder (default: rfm_checks/runs/<today>, or <today>_2, ...); must "
                        "be new or empty")
    p.add_argument("--concepts", nargs="+", metavar="NAME",
                   help=f"concepts (default: {' '.join(DEFAULT_CONCEPTS)}). Each check keeps its "
                        "historical order for the default concepts and appends the others in "
                        "the order given")
    p.add_argument("--iv-main", nargs="+", metavar="NAME",
                   help="Check IV MAIN set, required when IV's concept list is not the default "
                        f"({' '.join(IV_DEFAULT_CONCEPTS)}); a new MAIN is a design amendment to "
                        f"{IV_DESIGN}")
    p.add_argument("--replication-run", metavar="DIR",
                   help="replication run read by II (baselines_monitoring.json) and III "
                        "(steer_<c>.json, eval6_<c>.json); default replication/runs/2026-08-27")
    args = parser.parse_args(argv)

    try:
        run_dir = create_check_run(args.run, args.concepts, args.iv_main, args.replication_run)
    except (ValueError, FileExistsError, PermissionError) as e:
        parser.error(str(e))
    with open(os.path.join(run_dir, runs.RUN_JSON)) as f:
        cfg = json.load(f)["config"]
    shown = runs.display_path(run_dir)
    print(f"created {shown}")
    print(json.dumps(cfg, indent=2))
    try:
        iv_main(cfg)
    except ValueError as e:
        print(f"\nnote: the IV notebook will refuse to run in this folder. {e}")
    inside = runs._is_within(run_dir, paths.ROOT)
    up = os.path.relpath(paths.ROOT, run_dir) if inside else paths.ROOT
    env = "" if inside else f"PYTHONPATH={paths.ROOT} "
    print("\nNext, from the repo root:")
    print(f"  cd {shown}")
    for name in NOTEBOOKS:
        print(f"  {env}{up}/.venv/bin/jupyter nbconvert --to notebook --execute --inplace "
              f"{name}.ipynb")
    print(f"  cd {up}")
    print(f"  .venv/bin/python rfm_checks/kappa_side_run.py --run {shown}")
    print("Run I before Ia (Ia cross-checks Check I's _resultsC.npz). Set RFM_FORCE=1 only to "
          "overwrite outputs on purpose.")


if __name__ == "__main__":
    main()
