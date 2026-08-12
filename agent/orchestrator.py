"""
Orchestrator: the agent's core tool-calling loop.

7 tools across HR, CRM, and CMS. Deliberately scoped to three domains
for reliability on a 3B model — the cut tools (leave, FMS invoicing,
IT ops, CMS publishing) remain in agent/tools/ for future re-activation
after fine-tuning improves multi-tool coherence.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from tools.crm_followup import (
    crm_followup,
    get_customer_profile,
    update_lead_status,
    log_sales_interaction,
)
from tools.hr_lookup import hr_lookup, onboard_employee
from tools.cms_publish import cms_publish_check
from tools.achievement_log import log_achievement

LLAMA_SERVER_URL = "http://localhost:8080/v1/chat/completions"

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "crm_followup",
            "description": "Find CRM deals with no activity for at least N days, to draft follow-up messages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "min_days_inactive": {"type": "integer", "description": "Defaults to 14."}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_customer_profile",
            "description": "Look up a CRM customer profile by company name.",
            "parameters": {
                "type": "object",
                "properties": {"company_name": {"type": "string"}},
                "required": ["company_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_lead_status",
            "description": "Update a customer's CRM pipeline status, e.g. 'closed_won', 'closed_lost', 'negotiating'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "crm_id": {"type": "string"},
                    "status": {"type": "string"},
                },
                "required": ["crm_id", "status"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "log_sales_interaction",
            "description": "Record a sales interaction note against a customer's CRM record.",
            "parameters": {
                "type": "object",
                "properties": {
                    "crm_id": {"type": "string"},
                    "summary": {"type": "string"},
                },
                "required": ["crm_id", "summary"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hr_lookup",
            "description": "Find employees starting within N days and their onboarding checklists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "within_days": {"type": "integer", "description": "Defaults to 7."}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "onboard_employee",
            "description": "Create a new HR profile for a hire and provision NexID credentials.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                    "role": {"type": "string"},
                    "department": {"type": "string"},
                },
                "required": ["name", "email", "role", "department"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cms_publish_check",
            "description": "Find draft CMS content linked to a specific employee ID.",
            "parameters": {
                "type": "object",
                "properties": {"emp_id": {"type": "string"}},
                "required": ["emp_id"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "crm_followup": crm_followup,
    "get_customer_profile": get_customer_profile,
    "update_lead_status": update_lead_status,
    "log_sales_interaction": log_sales_interaction,
    "hr_lookup": hr_lookup,
    "onboard_employee": onboard_employee,
    "cms_publish_check": cms_publish_check,
}

SYSTEM_PROMPT = (
    "You are Nexalith Foreman, an offline operations agent for a small business. "
    "You have exactly 7 tools across three domains:\n"
    "- CRM: crm_followup, get_customer_profile, update_lead_status, log_sales_interaction\n"
    "- HR: hr_lookup, onboard_employee\n"
    "- CMS: cms_publish_check\n\n"
    "READ tools (call freely): crm_followup, get_customer_profile, hr_lookup, cms_publish_check.\n"
    "WRITE tools (ask for confirmation first, unless the user's message is already an explicit instruction): "
    "update_lead_status, log_sales_interaction, onboard_employee.\n\n"
    "CRITICAL: never state a factual claim about deals, customers, employees, or content "
    "unless you called the corresponding READ tool in THIS turn and are reporting what it "
    "actually returned. Do not rely on what a tool returned in an earlier turn.\n\n"
    "If a question has multiple parts across different domains, call a READ tool for EACH part. "
    "When asked open-ended questions like 'how do we improve X', propose a specific, concrete "
    "next action per item with brief reasoning, then ask for confirmation before calling any "
    "WRITE tool.\n\n"
    "Be concise and concrete."
)


def _call_model(messages: list[dict]) -> dict:
    response = requests.post(
        LLAMA_SERVER_URL,
        json={"messages": messages, "tools": TOOL_DEFINITIONS, "temperature": 0.3},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def run_agent(user_request: str, max_tool_rounds: int = 5) -> str:
    """Run the full tool-calling loop for a single user request."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_request},
    ]

    tools_used = []

    for _ in range(max_tool_rounds):
        result = _call_model(messages)
        choice = result["choices"][0]
        message = choice["message"]
        messages.append(message)

        tool_calls = message.get("tool_calls")
        if not tool_calls:
            final_text = message.get("content", "")
            if tools_used:
                log_achievement(
                    action=",".join(tools_used),
                    summary=f"Handled request: {user_request[:80]}",
                )
            return final_text

        for call in tool_calls:
            fn_name = call["function"]["name"]
            try:
                fn_args = json.loads(call["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                fn_args = {}

            tools_used.append(fn_name)
            fn = TOOL_FUNCTIONS.get(fn_name)
            tool_result = (
                {"error": f"unknown tool {fn_name}"} if fn is None else fn(**fn_args)
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(tool_result),
                }
            )

    return "I wasn't able to finish this request within the allowed tool-call steps."


if __name__ == "__main__":
    query = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "What's our biggest deal at risk right now?"
    )
    print(f"> {query}\n")
    print(run_agent(query))
