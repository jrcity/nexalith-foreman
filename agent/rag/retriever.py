"""
Structured retrieval layer for Nexalith Foreman, backed by SQLite.

Still deliberately NOT embeddings-based, for the same reason as before:
the underlying data is structured (real columns, real types), so retrieval
means "run the right SQL query," not "nearest-neighbor over vectors."

Function signatures are unchanged from the original JSON-backed version,
so orchestrator.py and the tools/ layer required no changes for this
migration — only this file and migrate_to_sqlite.py changed.
"""
from __future__ import annotations

import sqlite3
from datetime import date, datetime
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "foreman.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _days_since(date_str: str, reference: date | None = None) -> int:
    ref = reference or date.today()
    then = datetime.strptime(date_str, "%Y-%m-%d").date()
    return (ref - then).days


def get_stale_deals(min_days_inactive: int = 14) -> list[dict]:
    """Deals with no activity in at least `min_days_inactive` days."""
    conn = _connect()
    rows = conn.execute("SELECT * FROM crm_deals").fetchall()
    conn.close()

    stale = []
    for row in rows:
        days = _days_since(row["last_activity_date"])
        if days >= min_days_inactive:
            stale.append({**dict(row), "days_since_activity": days})
    return sorted(stale, key=lambda d: -d["days_since_activity"])


def get_deal_by_id(deal_id: str) -> dict | None:
    conn = _connect()
    row = conn.execute("SELECT * FROM crm_deals WHERE id = ?", (deal_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_upcoming_starters(within_days: int = 7) -> list[dict]:
    """Employees whose start_date is within the next `within_days` days."""
    conn = _connect()
    employees = conn.execute("SELECT * FROM hr_employees").fetchall()

    today = date.today()
    upcoming = []
    for emp in employees:
        start = datetime.strptime(emp["start_date"], "%Y-%m-%d").date()
        delta = (start - today).days
        if 0 <= delta <= within_days:
            checklist = conn.execute(
                "SELECT item, done FROM onboarding_checklist_items WHERE employee_id = ?",
                (emp["id"],),
            ).fetchall()
            upcoming.append({
                **dict(emp),
                "days_until_start": delta,
                "onboarding_checklist": [
                    {"item": c["item"], "done": bool(c["done"])} for c in checklist
                ],
            })
    conn.close()
    return sorted(upcoming, key=lambda e: e["days_until_start"])


def get_employee_by_id(emp_id: str) -> dict | None:
    conn = _connect()
    emp = conn.execute("SELECT * FROM hr_employees WHERE id = ?", (emp_id,)).fetchone()
    if emp is None:
        conn.close()
        return None
    checklist = conn.execute(
        "SELECT item, done FROM onboarding_checklist_items WHERE employee_id = ?",
        (emp_id,),
    ).fetchall()
    conn.close()
    return {
        **dict(emp),
        "onboarding_checklist": [
            {"item": c["item"], "done": bool(c["done"])} for c in checklist
        ],
    }


def get_draft_content_for_employee(emp_id: str) -> list[dict]:
    """CMS drafts linked to a specific employee (e.g. new-hire announcements)."""
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM cms_content WHERE linked_to = ? AND status = 'draft'",
        (emp_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_open_deals() -> list[dict]:
    conn = _connect()
    rows = conn.execute("SELECT * FROM crm_deals WHERE stage != 'closed'").fetchall()
    conn.close()
    return [dict(r) for r in rows]
