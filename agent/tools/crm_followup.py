"""CRM tool: surface stale deals and draft follow-up messages."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.retriever import get_stale_deals  # noqa: E402


def crm_followup(min_days_inactive: int = 14) -> dict:
    """
    Find CRM deals with no activity for at least `min_days_inactive` days.

    Returns structured data for the model to reason over and draft
    follow-ups from — this tool does NOT generate the messages itself,
    it only retrieves the facts. Message drafting is the model's job.
    """
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


if __name__ == "__main__":
    import json
    print(json.dumps(crm_followup(), indent=2))
