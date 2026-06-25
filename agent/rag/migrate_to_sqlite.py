"""
One-time migration: builds agent/data/foreman.db from agent/data/seed_corpus.json.

Run this once to create the database. retriever.py reads from the .db file
going forward, not the JSON file directly (the JSON file is kept as the
human-readable source of truth for the synthetic dataset).
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
JSON_PATH = DATA_DIR / "seed_corpus.json"
DB_PATH = DATA_DIR / "foreman.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS hr_employees (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    role TEXT NOT NULL,
    start_date TEXT NOT NULL,
    status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS onboarding_checklist_items (
    employee_id TEXT NOT NULL,
    item TEXT NOT NULL,
    done INTEGER NOT NULL,
    FOREIGN KEY (employee_id) REFERENCES hr_employees(id)
);

CREATE TABLE IF NOT EXISTS crm_deals (
    id TEXT PRIMARY KEY,
    client_name TEXT NOT NULL,
    rep TEXT NOT NULL,
    stage TEXT NOT NULL,
    value_usd INTEGER NOT NULL,
    last_activity_date TEXT NOT NULL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS cms_content (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    status TEXT NOT NULL,
    linked_to TEXT,
    body_summary TEXT
);
"""


def migrate() -> None:
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))

    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)

    conn.execute("DELETE FROM hr_employees")
    conn.execute("DELETE FROM onboarding_checklist_items")
    conn.execute("DELETE FROM crm_deals")
    conn.execute("DELETE FROM cms_content")

    for emp in data["hr_employees"]:
        conn.execute(
            "INSERT INTO hr_employees (id, name, role, start_date, status) VALUES (?, ?, ?, ?, ?)",
            (emp["id"], emp["name"], emp["role"], emp["start_date"], emp["status"]),
        )
        for item in emp["onboarding_checklist"]:
            conn.execute(
                "INSERT INTO onboarding_checklist_items (employee_id, item, done) VALUES (?, ?, ?)",
                (emp["id"], item["item"], int(item["done"])),
            )

    for deal in data["crm_deals"]:
        conn.execute(
            """INSERT INTO crm_deals
               (id, client_name, rep, stage, value_usd, last_activity_date, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                deal["id"], deal["client_name"], deal["rep"], deal["stage"],
                deal["value_usd"], deal["last_activity_date"], deal.get("notes", ""),
            ),
        )

    for cms in data["cms_content"]:
        conn.execute(
            "INSERT INTO cms_content (id, title, status, linked_to, body_summary) VALUES (?, ?, ?, ?, ?)",
            (cms["id"], cms["title"], cms["status"], cms.get("linked_to"), cms.get("body_summary", "")),
        )

    conn.commit()
    conn.close()
    print(f"migrated {JSON_PATH.name} -> {DB_PATH}")


if __name__ == "__main__":
    migrate()
