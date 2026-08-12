"""CRM tools: stale-deal follow-up, customer profile lookup, lead status, interaction log."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.retriever import (  # noqa: E402
    get_stale_deals,
    get_customer_profile as _get_customer_profile,
    update_lead_status as _update_lead_status,
    log_sales_interaction as _log_sales_interaction,
)


def crm_followup(min_days_inactive: int = 14) -> dict:
    """Find CRM deals with no activity for at least `min_days_inactive` days."""
    stale = get_stale_deals(min_days_inactive=min_days_inactive)
    return {
        "stale_deal_count": len(stale),
        "deals": [
            {
                "client_name": d["client_name"],
                "rep": d["rep"],
                "stage": d["stage"],
                "value_usd": d["value_usd"],
                "days_since_activity": d["days_since_activity"],
                "notes": d["notes"],
            }
            for d in stale
        ],
    }


def get_customer_profile(company_name: str) -> dict:
    """Look up a CRM customer profile by company name."""
    profile = _get_customer_profile(company_name)
    if profile is None:
        return {"found": False, "company_name": company_name}
    return {"found": True, **profile}


def update_lead_status(crm_id: str, status: str) -> dict:
    """
    Update a customer's pipeline status (e.g. 'closed_won', 'closed_lost',
    'negotiating'). Writes the change to the local dataset.
    """
    return _update_lead_status(crm_id=crm_id, status=status)


def log_sales_interaction(crm_id: str, summary: str) -> dict:
    """Record a sales interaction note against a customer's CRM record."""
    return _log_sales_interaction(crm_id=crm_id, summary=summary)


if __name__ == "__main__":
    import json
    print(json.dumps(crm_followup(), indent=2))
