"""HR tool: onboarding checklist lookup for new and upcoming hires."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.retriever import get_upcoming_starters, get_employee_by_id  # noqa: E402


def hr_lookup(within_days: int = 7) -> dict:
    """
    Find employees starting within `within_days` days and return their
    onboarding checklists. The model uses this to tell the user what
    onboarding steps remain incomplete.
    """
    upcoming = get_upcoming_starters(within_days=within_days)
    return {
        "upcoming_starter_count": len(upcoming),
        "employees": [
            {
                "id": e["id"],
                "name": e["name"],
                "role": e["role"],
                "start_date": e["start_date"],
                "days_until_start": e["days_until_start"],
                "checklist": e["onboarding_checklist"],
                "incomplete_items": [
                    item["item"] for item in e["onboarding_checklist"] if not item["done"]
                ],
            }
            for e in upcoming
        ],
    }


def hr_employee_detail(emp_id: str) -> dict | None:
    """Look up a single employee by ID — used after hr_lookup identifies one."""
    emp = get_employee_by_id(emp_id)
    if emp is None:
        return None
    return emp


if __name__ == "__main__":
    import json
    print(json.dumps(hr_lookup(), indent=2))
