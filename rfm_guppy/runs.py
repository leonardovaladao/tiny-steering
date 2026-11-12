"""Run folders: where a script writes, and the provenance record `run.json` that goes with it.

A run folder is `<study>/runs/<name>/` (for example `replication/runs/2026-11-12/`). The rules:

  * Append-only. A script never overwrites a file that already exists in the run folder unless
    it was started with `--force`. Every write goes through `out_path`.
  * Pinned evidence is read-only. The folders in `paths.PINNED_RUNS` (the replication run
    `paths.RESULTS_DIR` of 2026-08-27 and the check runs 2026-10-02 and 2026-10-09), and any
    folder inside or containing one of them, are refused as a run folder, even with `--force`.
    Any other run folder, including a new check run under `rfm_checks/runs/`, is writable and
    protected by the append-only rule.
  * Provenance. `open_run` keeps `run.json` in the run folder:

        {"study": "replication", "created": "<ISO time>",
         "steps": [{"script", "argv", "started", "concepts", "params",
                    "git_commit", "git_dirty", "python", "versions"}, ...]}

    One step is appended per script invocation. `git_commit` / `git_dirty` are null when git is
    unavailable; `git_dirty` is true when `git status --porcelain` prints anything.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import subprocess
import sys
from importlib import metadata

from . import paths

RUN_JSON = "run.json"


def _now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _is_within(path: str, folder: str) -> bool:
    path, folder = os.path.realpath(path), os.path.realpath(folder)
    return path == folder or path.startswith(folder + os.sep)


def check_writable(run_dir: str) -> None:
    """Raise PermissionError if `run_dir` is a pinned evidence folder (`paths.PINNED_RUNS`), lies
    inside one, or contains one (e.g. `rfm_checks/runs/` itself)."""
    for pinned in paths.PINNED_RUNS:
        if _is_within(run_dir, pinned) or _is_within(pinned, run_dir):
            raise PermissionError(
                f"refusing to write to {display_path(run_dir)}: the pinned run "
                f"{display_path(pinned)} is read-only evidence (paths.PINNED_RUNS). Pass a new "
                "run folder (--run DIR).")


def display_path(path: str) -> str:
    """Path relative to the repo root when inside it, else absolute (for log lines)."""
    path = os.path.abspath(path)
    return os.path.relpath(path, paths.ROOT) if _is_within(path, paths.ROOT) else path


def default_run_dir(study_dir: str) -> str:
    """`<study_dir>/runs/<today YYYY-MM-DD>`, or `<date>_2`, `_3`, ... if that one is in use."""
    base = os.path.join(study_dir, "runs", _dt.date.today().isoformat())
    candidate, k = base, 1
    while os.path.isdir(candidate) and os.listdir(candidate):
        k += 1
        candidate = f"{base}_{k}"
    return candidate


def out_path(run_dir: str, name: str, force: bool = False) -> str:
    """Path of output file `name` in `run_dir`, refusing to overwrite an existing file.

    Raises PermissionError for a pinned run folder (even with `force`) and FileExistsError if the
    file exists and `force` is False.
    """
    check_writable(run_dir)
    path = os.path.join(run_dir, name)
    if os.path.exists(path) and not force:
        raise FileExistsError(
            f"{display_path(path)} already exists. Run folders are append-only: choose a new "
            "--run folder, or pass --force to overwrite it.")
    return path


def _git(*args: str) -> str | None:
    try:
        res = subprocess.run(["git", "-C", paths.ROOT, *args], capture_output=True, text=True,
                             timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return res.stdout if res.returncode == 0 else None


def _versions() -> dict[str, str]:
    out = {}
    for key, dist in (("numpy", "numpy"), ("torch", "torch"), ("sklearn", "scikit-learn")):
        try:
            out[key] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            pass
    return out


def open_run(run_dir: str, script: str, concepts: list[str], params: dict | None = None) -> str:
    """Create `run_dir` if needed and append one step for `script` to its `run.json`.

    `script` is the path of the calling script (pass `__file__`); the study is the folder that
    holds it. Returns `run_dir` as an absolute path.
    """
    check_writable(run_dir)
    run_dir = os.path.abspath(run_dir)
    os.makedirs(run_dir, exist_ok=True)

    script = os.path.abspath(script)
    record_path = os.path.join(run_dir, RUN_JSON)
    if os.path.exists(record_path):
        with open(record_path) as f:
            record = json.load(f)
    else:
        record = {"study": display_path(os.path.dirname(script)), "created": _now(), "steps": []}

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    record["steps"].append({
        "script":     display_path(script),
        "argv":       list(sys.argv),
        "started":    _now(),
        "concepts":   list(concepts),
        "params":     params or {},
        "git_commit": commit.strip() if commit is not None else None,
        "git_dirty":  bool(status.strip()) if status is not None else None,
        "python":     platform.python_version(),
        "versions":   _versions(),
    })

    tmp = record_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    os.replace(tmp, record_path)
    return run_dir
