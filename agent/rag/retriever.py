"""
Structured retrieval layer for Nexalith Foreman.

Deliberately NOT embeddings-based. The underlying data (HR records, CRM
deals, CMS drafts) is structured, not free-text prose — so retrieval here
means "run the right filtered query," not "nearest-neighbor over vectors."
The LLM's job (via tool-calling in orchestrator.py) is to decide *which*
query to run and *how to reason* over what comes back, not to do the
retrieval itself.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "seed_corpus.json"


def _load() -> dict:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _days_since(date_str: str, reference: date | None = None) -> int:
    ref = reference or date.today()
    then = datetime.strptime(date_str, "%Y-%m-%d").date()
    return (ref - then).days


def get_stale_deals(min_days_inactive: int = 14) -> list[dict]:
    """Deals with no activity in at least `min_days_inactive` days."""
    data = _load()
    stale = []
    for deal in data["crm_deals"]:
        days = _days_since(deal["last_activity_date"])
        if days >= min_days_inactive:
            stale.append({**deal, "days_since_activity": days})
    return sorted(stale, key=lambda d: -d["days_since_activity"])


def get_deal_by_id(deal_id: str) -> dict | None:
    data = _load()
    return next((d for d in data["crm_deals"] if d["id"] == deal_id), None)


def get_upcoming_starters(within_days: int = 7) -> list[dict]:
    """Employees whose start_date is within the next `within_days` days."""
    data = _load()
    today = date.today()
    upcoming = []
    for emp in data["hr_employees"]:
        start = datetime.strptime(emp["start_date"], "%Y-%m-%d").date()
        delta = (start - today).days
        if 0 <= delta <= within_days:
            upcoming.append({**emp, "days_until_start": delta})
    return sorted(upcoming, key=lambda e: e["days_until_start"])


def get_employee_by_id(emp_id: str) -> dict | None:
    data = _load()
    return next((e for e in data["hr_employees"] if e["id"] == emp_id), None)


def get_draft_content_for_employee(emp_id: str) -> list[dict]:
    """CMS drafts linked to a specific employee (e.g. new-hire announcements)."""
    data = _load()
    return [c for c in data["cms_content"] if c.get("linked_to") == emp_id and c["status"] == "draft"]


def get_all_open_deals() -> list[dict]:
    data = _load()
    return [d for d in data["crm_deals"] if d["stage"] != "closed"]
