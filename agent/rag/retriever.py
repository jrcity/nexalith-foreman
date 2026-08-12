"""
Structured retrieval layer for Nexalith Foreman.

Deliberately NOT embeddings-based. The underlying data (HR records, CRM
deals/customers, CMS drafts, FMS invoices, IT service/log records) is
structured, not free-text prose -- so retrieval here means "run the right
filtered query, or write the right structured update," not "nearest-
neighbor over vectors." The LLM's job (via tool-calling in
orchestrator.py) is to decide *which* operation to perform and *how to
reason* over the result, not to do the retrieval itself.

IMPORTANT: it_services / it_logs are a SYNTHETIC, LOCAL simulation only.
restart_service() flips a status flag in seed_corpus.json -- it does not
touch any real running process. This is explicit by design: the demo
shows the agent's decision-making and tool-orchestration pattern, not
live infrastructure control.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "seed_corpus.json"


def _load() -> dict:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(data: dict) -> None:
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _days_since(date_str: str, reference: date | None = None) -> int:
    ref = reference or date.today()
    then = datetime.strptime(date_str, "%Y-%m-%d").date()
    return (ref - then).days


# ---------------------------------------------------------------------------
# Existing: HR onboarding
# ---------------------------------------------------------------------------

def get_upcoming_starters(within_days: int = 7) -> list[dict]:
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


def onboard_employee(name: str, email: str, role: str, department: str) -> dict:
    """Create a new HR profile and (synthetically) provision NexID."""
    data = _load()
    new_id = f"emp_{len(data['hr_employees']) + 1:03d}"
    new_emp = {
        "id": new_id,
        "name": name,
        "email": email,
        "role": role,
        "department": department,
        "start_date": date.today().isoformat(),
        "status": "pending_onboarding",
        "onboarding_checklist": [
            {"item": "Provision laptop and accounts", "done": False},
            {"item": "Assign NexID credentials", "done": True},
            {"item": "Schedule first-week 1:1", "done": False},
            {"item": "Add to engineering Slack channels", "done": False},
        ],
    }
    data["hr_employees"].append(new_emp)
    _save(data)
    return {"status": "success", "emp_id": new_id, "nexid_provisioned": True}


# ---------------------------------------------------------------------------
# New: HR leave requests
# ---------------------------------------------------------------------------

def approve_leave_request(request_id: str, manager_id: str) -> dict:
    """Approve a pending leave request."""
    data = _load()
    req = next((r for r in data.get("hr_leave_requests", []) if r["id"] == request_id), None)
    if req is None:
        return {"status": "error", "message": f"no leave request found with id {request_id}"}
    req["status"] = "approved"
    req["approved_by"] = manager_id
    _save(data)
    return {
        "status": "approved",
        "request_id": request_id,
        "employee_name": req["employee_name"],
        "dates": f"{req['start_date']} to {req['end_date']}",
    }


def get_pending_leave_requests() -> list[dict]:
    data = _load()
    return [r for r in data.get("hr_leave_requests", []) if r["status"] == "pending"]


# ---------------------------------------------------------------------------
# Existing: CRM deals
# ---------------------------------------------------------------------------

def get_stale_deals(min_days_inactive: int = 14) -> list[dict]:
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


def get_all_open_deals() -> list[dict]:
    data = _load()
    return [d for d in data["crm_deals"] if d["stage"] != "closed"]


# ---------------------------------------------------------------------------
# New: CRM customer profiles, lead status, interaction logging
# ---------------------------------------------------------------------------

def get_customer_profile(company_name: str) -> dict | None:
    """Look up a CRM customer profile by company name (case-insensitive)."""
    data = _load()
    target = company_name.strip().lower()
    return next(
        (c for c in data.get("crm_customers", []) if c["company_name"].lower() == target),
        None,
    )


def update_lead_status(crm_id: str, status: str) -> dict:
    """Update a customer's pipeline status, e.g. to 'closed_won' or 'closed_lost'."""
    data = _load()
    customer = next((c for c in data.get("crm_customers", []) if c["crm_id"] == crm_id), None)
    if customer is None:
        return {"success": False, "message": f"no customer found with crm_id {crm_id}"}
    previous = customer["current_status"]
    customer["current_status"] = status
    _save(data)
    return {"success": True, "crm_id": crm_id, "previous_status": previous, "new_status": status}


def log_sales_interaction(crm_id: str, summary: str) -> dict:
    """Append a sales interaction note to the customer's interaction log."""
    data = _load()
    data.setdefault("crm_interaction_log", [])
    entry = {
        "crm_id": crm_id,
        "summary": summary,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    data["crm_interaction_log"].append(entry)
    _save(data)
    return {"success": True, "logged": entry}


# ---------------------------------------------------------------------------
# Existing + new: CMS content
# ---------------------------------------------------------------------------

def get_draft_content_for_employee(emp_id: str) -> list[dict]:
    data = _load()
    return [c for c in data["cms_content"] if c.get("linked_to") == emp_id and c["status"] == "draft"]


def publish_announcement(title: str, platform: str) -> dict:
    """Publish a new CMS announcement to a given platform (synthetic push)."""
    data = _load()
    new_id = f"cms_{len(data['cms_content']) + 200:03d}"
    entry = {
        "id": new_id,
        "title": title,
        "status": "published",
        "linked_to": None,
        "platform": platform,
        "body_summary": title,
    }
    data["cms_content"].append(entry)
    _save(data)
    # Synthetic engagement number, deterministic-ish for demo purposes.
    push_notifications_sent = 4500 if platform.lower() == "washy" else 1200
    return {
        "success": True,
        "cms_id": new_id,
        "platform": platform,
        "push_notifications_sent": push_notifications_sent,
    }


# ---------------------------------------------------------------------------
# New: FMS invoicing
# ---------------------------------------------------------------------------

def generate_invoice(client_id: str, amount: float, description: str) -> dict:
    """Generate a new invoice for a client (synthetic FMS record)."""
    data = _load()
    data.setdefault("fms_invoices", [])
    new_id = f"INV-2026-{len(data['fms_invoices']) + 89:03d}"
    invoice = {
        "invoice_id": new_id,
        "client_id": client_id,
        "amount_usd": amount,
        "description": description,
        "status": "sent_to_client",
    }
    data["fms_invoices"].append(invoice)
    _save(data)
    return {"invoice_id": new_id, "status": "sent_to_client"}


# ---------------------------------------------------------------------------
# New: IT infra (synthetic simulation -- see module docstring)
# ---------------------------------------------------------------------------

def query_system_logs(service: str, level: str = "error", minutes_ago: int = 10) -> dict:
    """
    Query synthetic log entries for a service. This reads from the static
    seed_corpus.json log records -- it does not query any real running
    process or log aggregator.
    """
    data = _load()
    logs = data.get("it_logs", [])
    matches = [
        entry["message"]
        for entry in logs
        if entry["service"] == service and entry["level"] == level
    ]
    return {"service": service, "level": level, "logs": matches}


def restart_service(service_name: str) -> dict:
    """
    SYNTHETIC ACTION ONLY. Flips the status of a service record in the
    local seed_corpus.json to simulate a restart. Does not start, stop,
    or signal any real process.
    """
    data = _load()
    services = data.get("it_services", [])
    svc = next((s for s in services if s["service_name"] == service_name), None)
    if svc is None:
        return {"success": False, "message": f"no service found named {service_name}"}
    svc["status"] = "running"
    svc["last_restart"] = datetime.now().isoformat(timespec="seconds")
    _save(data)
    return {"success": True, "service_name": service_name, "simulated": True}
