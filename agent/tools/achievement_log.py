"""
Achievement log: records completed agent actions to a local JSON file.

This is intentionally simple — an append-only local log, not a database.
It exists to demonstrate that the agent's actions are auditable and that
completed workflows are durably tracked, which is the "achievements"
concept from the original Nexalith Foreman product framing.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

LOG_PATH = Path(__file__).parent.parent / "data" / "achievements_log.json"


def _load_log() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_log(entries: list[dict]) -> None:
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


def log_achievement(action: str, summary: str) -> dict:
    """Record a completed agent action. Called after a tool-calling task finishes."""
    entries = _load_log()
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "action": action,
        "summary": summary,
    }
    entries.append(entry)
    _save_log(entries)
    return entry


def get_recent_achievements(limit: int = 10) -> list[dict]:
    entries = _load_log()
    return entries[-limit:][::-1]


if __name__ == "__main__":
    log_achievement("crm_followup", "Drafted follow-ups for 2 stale deals")
    print(json.dumps(get_recent_achievements(), indent=2))
