"""
eval_storage.py
───────────────
Read/write evaluation runs as JSON files under evaluations/.

Why files, not a database?
    Streamlit Cloud wipes SQLite on redeploy. JSON files committed to git
    survive redeploys AND become part of the submission — my professor can
    open the evaluations/ folder on GitHub and see every run I did.
    That's more valuable than a database dump I'd have to export separately.

Layout:
    evaluations/
        runs/
            2026-04-20_1432_conflict_strong_<runid>.json    ← one per run
        aggregates/
            latest.json                                      ← written by analytics tab
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

EVAL_DIR = Path(__file__).parent / "evaluations"
RUNS_DIR = EVAL_DIR / "runs"
AGGREGATES_DIR = EVAL_DIR / "aggregates"


def _ensure_dirs():
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    AGGREGATES_DIR.mkdir(parents=True, exist_ok=True)


def save_run(run_result: dict) -> Path:
    """
    Save a single evaluation run to disk.
    Filename: {YYYY-MM-DD_HHMM}_{test_case_id}_{run_id}.json
    Returns the path to the written file.
    """
    _ensure_dirs()
    ts = datetime.utcnow().strftime("%Y-%m-%d_%H%M")
    tc_id = run_result.get("test_case", {}).get("id", "unknown")
    run_id = run_result.get("run_id", "00000000")
    filename = f"{ts}_{tc_id}_{run_id}.json"
    path = RUNS_DIR / filename
    with open(path, "w") as f:
        json.dump(run_result, f, indent=2)
    return path


def load_all_runs() -> list:
    """Load every run in evaluations/runs/ as a list of dicts."""
    _ensure_dirs()
    runs = []
    for p in sorted(RUNS_DIR.glob("*.json")):
        try:
            with open(p) as f:
                data = json.load(f)
            data["_filename"] = p.name
            runs.append(data)
        except Exception as e:
            print(f"[eval_storage] skipping {p.name}: {e}")
    return runs


def load_run(filename: str) -> dict:
    """Load one run by filename."""
    path = RUNS_DIR / filename
    with open(path) as f:
        return json.load(f)


def list_runs() -> list:
    """Return a list of run filenames sorted newest first."""
    _ensure_dirs()
    files = sorted(RUNS_DIR.glob("*.json"), reverse=True)
    return [p.name for p in files]


def delete_run(filename: str) -> bool:
    """Delete a run file. Returns True on success."""
    path = RUNS_DIR / filename
    if path.exists():
        path.unlink()
        return True
    return False
