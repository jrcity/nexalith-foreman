"""HR tools: onboarding checklist lookup, new-hire onboarding, leave approval."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.retriever import (  # noqa: E402
    get_upcoming_starters,
    get_employee_by_id,
    onboard_employee as _onboard_employee,
    approve_leave_request as _approve_leave_request,
    get_pending_leave_requests,
)


def hr_lookup(within_days: int = 7) -> dict:
    """
    Find employees starting within `within_days` days and return their
    onboarding checklists.
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
    return get_employee_by_id(emp_id)


def onboard_employee(name: str, email: str, role: str, department: str) -> dict:
    """
    Create a new HR profile for a hire and synthetically provision their
    NexID credentials. Writes a new employee record to the local dataset.
    """
    return _onboard_employee(name=name, email=email, role=role, department=department)


def approve_leave_request(request_id: str, manager_id: str) -> dict:
    """Approve a pending leave/PTO request by ID."""
    return _approve_leave_request(request_id=request_id, manager_id=manager_id)


def hr_pending_leave() -> dict:
    """List all leave requests awaiting approval."""
    pending = get_pending_leave_requests()
    return {"pending_count": len(pending), "requests": pending}


if __name__ == "__main__":
    import json
    print(json.dumps(hr_lookup(), indent=2))
