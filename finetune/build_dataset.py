"""
Fine-tuning dataset generator for Nexalith Foreman.

Targets the specific failure mode observed in testing: the 3B model calls
the correct tools but fails to ground its natural-language response in
what those tools actually returned. Every example here demonstrates the
correct pattern: tool called -> result received -> response directly
references the real returned data, never contradicting or ignoring it.

Tool set: 7 tools across HR/CRM/CMS (the final locked set).
Format: Qwen2.5-Instruct chat format with tool_calls and tool messages,
matching exactly what orchestrator.py sends at inference time.
Output: finetune_examples.jsonl, one JSON object per line.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT_PATH = Path(__file__).parent / "finetune_examples.jsonl"

SYSTEM_PROMPT = (
    "You are Nexalith Foreman, an offline operations agent for a small business. "
    "You have exactly 7 tools across three domains:\n"
    "- CRM: crm_followup, get_customer_profile, update_lead_status, log_sales_interaction\n"
    "- HR: hr_lookup, onboard_employee\n"
    "- CMS: cms_publish_check\n\n"
    "READ tools (call freely): crm_followup, get_customer_profile, hr_lookup, cms_publish_check.\n"
    "WRITE tools (ask for confirmation first, unless the user's message is already an explicit "
    "instruction): update_lead_status, log_sales_interaction, onboard_employee.\n\n"
    "CRITICAL: never state a factual claim about deals, customers, employees, or content "
    "unless you called the corresponding READ tool in THIS turn and are reporting what it "
    "actually returned. Do not rely on what a tool returned in an earlier turn.\n\n"
    "If a question has multiple parts across different domains — for example asking about "
    "both new hires AND CRM deals — you MUST call a tool for EACH part before answering. "
    "Never answer one part from memory while calling a tool for another part. "
    "Example: 'Did we have any new hire? how can we get better on our leads?' requires "
    "calling BOTH hr_lookup AND crm_followup before writing any response.\n\n"
    "Be concise and concrete."
)


def make_call(call_id: str, name: str, args: dict) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }],
    }


def make_result(call_id: str, result: dict) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": json.dumps(result)}


def ex(user: str, turns: list[dict], final: str) -> dict:
    """Build a complete training example."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            *turns,
            {"role": "assistant", "content": final},
        ]
    }


def build_examples() -> list[dict]:
    examples = []

    # ------------------------------------------------------------------ #
    # CATEGORY 1: Tool-result grounding — single tool                     #
    # The exact failure: call tool, get result, USE the result.           #
    # ------------------------------------------------------------------ #

    # CRM: stale deals — various phrasings, various result shapes
    examples.append(ex(
        "Did we have any new hire?",
        [
            make_call("c1", "hr_lookup", {"within_days": 7}),
            make_result("c1", {
                "upcoming_starter_count": 1,
                "employees": [{
                    "id": "emp_001", "name": "Amaka Obi", "role": "Backend Engineer",
                    "start_date": "2026-06-29", "days_until_start": 4,
                    "incomplete_items": [
                        "Provision laptop and accounts",
                        "Assign NexID credentials",
                        "Schedule first-week 1:1",
                        "Add to engineering Slack channels",
                    ],
                }],
            }),
        ],
        "Yes — Amaka Obi joins in 4 days as a Backend Engineer. "
        "Four onboarding items still open: laptop/accounts, NexID credentials, "
        "first-week 1:1, and Slack channels.",
    ))

    examples.append(ex(
        "Did we have any new hire?",
        [
            make_call("c1", "hr_lookup", {"within_days": 7}),
            make_result("c1", {"upcoming_starter_count": 0, "employees": []}),
        ],
        "No new hires starting in the next 7 days.",
    ))

    examples.append(ex(
        "Any stale deals I should chase?",
        [
            make_call("c1", "crm_followup", {"min_days_inactive": 14}),
            make_result("c1", {
                "stale_deal_count": 3,
                "deals": [
                    {"client_name": "Port Harcourt Energy", "rep": "Tunde Bakare",
                     "stage": "negotiation", "value_usd": 9000,
                     "days_since_activity": 24,
                     "notes": "Large deal, client requested custom SLA terms, no follow-up sent yet."},
                    {"client_name": "Lagos Fresh Foods", "rep": "Tunde Bakare",
                     "stage": "negotiation", "value_usd": 4200,
                     "days_since_activity": 23,
                     "notes": "Awaiting response on revised pricing sent two weeks ago."},
                    {"client_name": "Kano Textiles Ltd", "rep": "Tunde Bakare",
                     "stage": "proposal_sent", "value_usd": 1800,
                     "days_since_activity": 20,
                     "notes": "Proposal sent, no response since."},
                ],
            }),
        ],
        "Three deals need chasing:\n\n"
        "1. **Port Harcourt Energy** ($9k, 24 days quiet) — still waiting on SLA "
        "terms discussion. Highest value, chase first.\n"
        "2. **Lagos Fresh Foods** ($4.2k, 23 days) — revised pricing sent, no reply. "
        "Worth a nudge.\n"
        "3. **Kano Textiles Ltd** ($1.8k, 20 days) — proposal sent, no response.\n\n"
        "Want me to draft follow-ups for all three, or mark any of them as closed lost?",
    ))

    examples.append(ex(
        "Any stale deals I should chase?",
        [
            make_call("c1", "crm_followup", {"min_days_inactive": 14}),
            make_result("c1", {"stale_deal_count": 0, "deals": []}),
        ],
        "No stale deals right now — everything in the pipeline has had activity "
        "in the last 14 days.",
    ))

    # ------------------------------------------------------------------ #
    # CATEGORY 2: Multi-part questions — MUST call tool for EACH part     #
    # This is the exact failure mode: model called crm_followup but       #
    # answered the HR part from memory (incorrectly).                     #
    # ------------------------------------------------------------------ #

    examples.append(ex(
        "Did we have any new hire? how can we get better on our existing leads that are not won and closed.",
        [
            make_call("c1", "hr_lookup", {"within_days": 7}),
            make_result("c1", {
                "upcoming_starter_count": 1,
                "employees": [{
                    "id": "emp_001", "name": "Amaka Obi", "role": "Backend Engineer",
                    "start_date": "2026-06-29", "days_until_start": 4,
                    "incomplete_items": [
                        "Provision laptop and accounts", "Assign NexID credentials",
                        "Schedule first-week 1:1", "Add to engineering Slack channels",
                    ],
                }],
            }),
            make_call("c2", "crm_followup", {"min_days_inactive": 14}),
            make_result("c2", {
                "stale_deal_count": 3,
                "deals": [
                    {"client_name": "Port Harcourt Energy", "rep": "Tunde Bakare",
                     "stage": "negotiation", "value_usd": 9000, "days_since_activity": 24,
                     "notes": "Large deal, client requested custom SLA terms, no follow-up sent yet."},
                    {"client_name": "Lagos Fresh Foods", "rep": "Tunde Bakare",
                     "stage": "negotiation", "value_usd": 4200, "days_since_activity": 23,
                     "notes": "Awaiting response on revised pricing sent two weeks ago."},
                    {"client_name": "Kano Textiles Ltd", "rep": "Tunde Bakare",
                     "stage": "proposal_sent", "value_usd": 1800, "days_since_activity": 20,
                     "notes": "Proposal sent, no response since."},
                ],
            }),
        ],
        "**New hire:** Yes — Amaka Obi joins in 4 days as a Backend Engineer. "
        "All four onboarding items are still open.\n\n"
        "**Stale leads:** Three deals need attention:\n"
        "1. Port Harcourt Energy ($9k, 24 days) — SLA terms not yet discussed. "
        "Suggested action: send a follow-up proposing a call.\n"
        "2. Lagos Fresh Foods ($4.2k, 23 days) — pricing revision unanswered. "
        "Suggested action: send a reminder.\n"
        "3. Kano Textiles Ltd ($1.8k, 20 days) — proposal ignored. "
        "Suggested action: give it one more week, then consider closing as lost.\n\n"
        "Want me to draft follow-ups, or mark any of these as closed lost?",
    ))

    # Variant: no new hire + stale deals
    examples.append(ex(
        "Did we have any new hire? how can we get better on our existing leads that are not won and closed.",
        [
            make_call("c1", "hr_lookup", {"within_days": 7}),
            make_result("c1", {"upcoming_starter_count": 0, "employees": []}),
            make_call("c2", "crm_followup", {"min_days_inactive": 14}),
            make_result("c2", {
                "stale_deal_count": 2,
                "deals": [
                    {"client_name": "Riverside Logistics", "rep": "Chidi Okeke",
                     "stage": "negotiation", "value_usd": 6500, "days_since_activity": 19,
                     "notes": "Client asked for volume discount, no reply sent."},
                    {"client_name": "Coastal Imports", "rep": "Funmi Adeyemi",
                     "stage": "proposal_sent", "value_usd": 3100, "days_since_activity": 16,
                     "notes": "Proposal sent, client said they'd review with finance."},
                ],
            }),
        ],
        "**New hire:** No one starting in the next 7 days.\n\n"
        "**Stale leads:** Two deals need attention:\n"
        "1. Riverside Logistics ($6.5k, 19 days) — volume discount request unanswered. "
        "Worth a reply from Chidi today.\n"
        "2. Coastal Imports ($3.1k, 16 days) — proposal with finance for review. "
        "A gentle check-in makes sense.\n\n"
        "Want me to draft follow-up messages for these?",
    ))

    # Variant: new hire + no stale deals
    examples.append(ex(
        "Any new starters this week and do we have anything urgent in the pipeline?",
        [
            make_call("c1", "hr_lookup", {"within_days": 7}),
            make_result("c1", {
                "upcoming_starter_count": 1,
                "employees": [{
                    "id": "emp_010", "name": "Bola Adeyemi", "role": "Product Designer",
                    "start_date": "2026-07-02", "days_until_start": 5,
                    "incomplete_items": ["Assign NexID credentials", "Schedule first-week 1:1"],
                }],
            }),
            make_call("c2", "crm_followup", {"min_days_inactive": 14}),
            make_result("c2", {"stale_deal_count": 0, "deals": []}),
        ],
        "**New starter:** Bola Adeyemi (Product Designer) joins in 5 days. "
        "Still need to assign NexID credentials and schedule their first-week 1:1.\n\n"
        "**Pipeline:** Nothing stale — all deals have had activity in the last 14 days.",
    ))

    # ------------------------------------------------------------------ #
    # CATEGORY 3: Cross-domain chaining — HR -> CMS                      #
    # Keep the proven working pattern that already passes.                #
    # ------------------------------------------------------------------ #

    examples.append(ex(
        "We just hired a new employee starting Monday. Set up their onboarding checklist, "
        "and let me know what CMS content needs to be published for the new-hire announcement.",
        [
            make_call("c1", "hr_lookup", {"within_days": 7}),
            make_result("c1", {
                "upcoming_starter_count": 1,
                "employees": [{
                    "id": "emp_001", "name": "Amaka Obi", "role": "Backend Engineer",
                    "start_date": "2026-06-29", "days_until_start": 4,
                    "incomplete_items": [
                        "Provision laptop and accounts", "Assign NexID credentials",
                        "Schedule first-week 1:1", "Add to engineering Slack channels",
                    ],
                }],
            }),
            make_call("c2", "cms_publish_check", {"emp_id": "emp_001"}),
            make_result("c2", {
                "draft_count": 1,
                "drafts": [{
                    "id": "cms_201",
                    "title": "Welcome to the Team: New Hire Announcements",
                    "status": "draft",
                    "body_summary": "Template needs name and role filled in before publishing.",
                }],
            }),
        ],
        "Amaka Obi starts Monday as a Backend Engineer. All four onboarding items are "
        "still open: laptop/accounts, NexID credentials, first-week 1:1, and Slack channels.\n\n"
        "The welcome announcement is drafted (cms_201) but needs her name and role "
        "filled in before it can be published.",
    ))

    examples.append(ex(
        "New hire starting Friday — what's left to do and is the announcement ready?",
        [
            make_call("c1", "hr_lookup", {"within_days": 7}),
            make_result("c1", {
                "upcoming_starter_count": 1,
                "employees": [{
                    "id": "emp_015", "name": "Kelechi Eze", "role": "DevOps Engineer",
                    "start_date": "2026-07-04", "days_until_start": 3,
                    "incomplete_items": ["Assign NexID credentials"],
                }],
            }),
            make_call("c2", "cms_publish_check", {"emp_id": "emp_015"}),
            make_result("c2", {"draft_count": 0, "drafts": []}),
        ],
        "Kelechi Eze starts Friday as a DevOps Engineer — only NexID credentials "
        "still need assigning. No announcement drafted yet — worth creating one "
        "before he starts.",
    ))

    # ------------------------------------------------------------------ #
    # CATEGORY 4: CRM write actions with confirmation                     #
    # ------------------------------------------------------------------ #

    examples.append(ex(
        "What's the status of TechCorp Nigeria?",
        [
            make_call("c1", "get_customer_profile", {"company_name": "TechCorp Nigeria"}),
            make_result("c1", {
                "found": True, "crm_id": "cust_201", "company_name": "TechCorp Nigeria",
                "contact_name": "Alice Eze", "contact_email": "alice@techcorp.ng",
                "current_status": "negotiating",
            }),
        ],
        "TechCorp Nigeria (cust_201) is currently in negotiation. "
        "Alice Eze is the contact — alice@techcorp.ng.",
    ))

    examples.append(ex(
        "Mark TechCorp Nigeria's deal as closed won, their CRM ID is cust_201.",
        [
            make_call("c1", "update_lead_status", {"crm_id": "cust_201", "status": "closed_won"}),
            make_result("c1", {
                "success": True, "crm_id": "cust_201",
                "previous_status": "negotiating", "new_status": "closed_won",
            }),
        ],
        "Done — TechCorp Nigeria updated from negotiating to closed won.",
    ))

    examples.append(ex(
        "Port Harcourt Energy has been quiet for 24 days. What should we do?",
        [
            make_call("c1", "crm_followup", {"min_days_inactive": 14}),
            make_result("c1", {
                "stale_deal_count": 1,
                "deals": [{
                    "client_name": "Port Harcourt Energy", "rep": "Tunde Bakare",
                    "stage": "negotiation", "value_usd": 9000, "days_since_activity": 24,
                    "notes": "Large deal, client requested custom SLA terms, no follow-up sent yet.",
                }],
            }),
        ],
        "Port Harcourt Energy ($9k) has been quiet for 24 days — they asked for "
        "custom SLA terms and no reply has gone out yet. At this stage I'd suggest "
        "one more direct follow-up from Tunde specifically addressing the SLA ask. "
        "If there's still no response after that, worth considering closing as lost "
        "to keep the pipeline clean.\n\n"
        "Want me to log a sales interaction note, or mark it closed lost?",
    ))

    # ------------------------------------------------------------------ #
    # CATEGORY 5: Grounded refusals                                       #
    # ------------------------------------------------------------------ #

    examples.append(ex(
        "What's our total revenue this quarter?",
        [],
        "I don't have a revenue or finance tool — I can look up individual CRM deal "
        "values if that helps. Want me to pull all open deals?",
    ))

    examples.append(ex(
        "Can you send an email to Alice at TechCorp?",
        [],
        "I can't send emails directly — I don't have a messaging or email tool. "
        "I can look up Alice's contact details from the CRM if you want to reach out yourself.",
    ))

    examples.append(ex(
        "Restart the server.",
        [],
        "I don't have an IT operations tool in my current configuration. "
        "I can help with HR, CRM, and CMS tasks.",
    ))

    return examples


def main() -> None:
    examples = build_examples()
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Wrote {len(examples)} examples to {OUT_PATH}")
    print(f"Breakdown:")
    print(f"  Tool-result grounding (single tool): 4")
    print(f"  Multi-part questions (both tools): 4")
    print(f"  Cross-domain chaining (HR->CMS): 2")
    print(f"  CRM write actions: 3")
    print(f"  Grounded refusals: 3")
    print(f"  Total: {len(examples)}")


if __name__ == "__main__":
    main()
