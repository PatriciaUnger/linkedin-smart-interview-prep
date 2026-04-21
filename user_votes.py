"""
user_votes.py
─────────────
Lightweight user feedback capture on the AI's feedback quality.
Added in V3 in response to professor feedback: "You could let the users vote."

Each vote is saved as a JSON file in evaluations/user_votes/ with:
  - timestamp, question, answer, score
  - helpful (True/False)
  - optional free-text comment (the reflection prompt when the user votes down)

Kept separate from the Eval Harness runs because votes are subjective user
signals during normal app use, while the harness is a controlled algorithmic
comparison. Both feed the Analytics tab.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

VOTES_DIR = Path(__file__).parent / "evaluations" / "user_votes"


def _ensure_dir():
    VOTES_DIR.mkdir(parents=True, exist_ok=True)


def save_vote(question: str, answer: str, score: int,
              helpful: bool, comment: str = "",
              job_title: str = "", q_type: str = "") -> tuple:
    """Persist a single thumbs-up/thumbs-down vote to disk.
    Returns (Path, vote_id) so the caller can later update the comment."""
    _ensure_dir()
    vote_id = uuid.uuid4().hex[:8]
    ts = datetime.utcnow().strftime("%Y-%m-%d_%H%M%S")
    payload = {
        "vote_id": vote_id,
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "job_title": job_title,
        "question": question,
        "question_type": q_type,
        "answer": answer,
        "score": score,
        "helpful": helpful,
        "comment": comment,
    }
    filename = f"{ts}_{'up' if helpful else 'down'}_{vote_id}.json"
    path = VOTES_DIR / filename
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path, vote_id


def update_vote_comment(vote_id: str, comment: str) -> bool:
    """Add a comment to an existing vote (used when the user expands on a down-vote)."""
    _ensure_dir()
    for p in VOTES_DIR.glob("*.json"):
        try:
            with open(p) as f:
                data = json.load(f)
            if data.get("vote_id") == vote_id:
                data["comment"] = comment
                with open(p, "w") as f:
                    json.dump(data, f, indent=2)
                return True
        except Exception:
            continue
    return False


def load_all_votes() -> list:
    _ensure_dir()
    votes = []
    for p in sorted(VOTES_DIR.glob("*.json")):
        try:
            with open(p) as f:
                data = json.load(f)
            votes.append(data)
        except Exception as e:
            print(f"[user_votes] skipping {p.name}: {e}")
    return votes


def vote_summary() -> dict:
    votes = load_all_votes()
    if not votes:
        return {"total": 0, "up": 0, "down": 0, "up_pct": 0.0}
    up = sum(1 for v in votes if v.get("helpful"))
    down = len(votes) - up
    up_pct = round((up / len(votes)) * 100, 1)
    return {"total": len(votes), "up": up, "down": down, "up_pct": up_pct}
