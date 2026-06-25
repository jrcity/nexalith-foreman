"""
Orchestrator: the agent's core tool-calling loop.

Talks to a running llama-server instance (OpenAI-compatible API) and gives
it access to the four domain tools. The model decides which tool(s) to
call based on the user's natural-language request; this module executes
the actual Python function and feeds the result back to the model for
a final natural-language response.

This is the literal "load-bearing" link in cross_disciplinary_pairing:
remove the model and there is no decision-making about which tool to
call or how to interpret the results — just raw JSON.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from tools.crm_followup import crm_followup
from tools.hr_lookup import hr_lookup, hr_employee_detail
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
                    "min_days_inactive": {
                        "type": "integer",
                        "description": "Minimum days of inactivity to count as stale. Defaults to 14.",
                    }
                },
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
                    "within_days": {
                        "type": "integer",
                        "description": "Look-ahead window in days. Defaults to 7.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cms_publish_check",
            "description": "Find draft CMS content linked to a specific employee ID (e.g. a new-hire announcement still in draft).",
            "parameters": {
                "type": "object",
                "properties": {
                    "emp_id": {
                        "type": "string",
                        "description": "Employee ID returned by hr_lookup, e.g. 'emp_001'.",
                    }
                },
                "required": ["emp_id"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "crm_followup": crm_followup,
    "hr_lookup": hr_lookup,
    "cms_publish_check": cms_publish_check,
}

SYSTEM_PROMPT = (
    "You are Nexalith Foreman, an offline operations agent for a small business. "
    "You have tools to look up CRM deals, HR onboarding status, and CMS draft content. "
    "Use the tools to gather real facts before answering. When asked to draft messages "
    "(e.g. follow-up emails), write them yourself based on the retrieved facts — the tools "
    "only give you data, they do not write messages for you. Be concise and concrete."
)


def _call_model(messages: list[dict]) -> dict:
    response = requests.post(
        LLAMA_SERVER_URL,
        json={
            "messages": messages,
            "tools": TOOL_DEFINITIONS,
            "temperature": 0.3,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def run_agent(user_request: str, max_tool_rounds: int = 4) -> str:
    """
    Run the full tool-calling loop for a single user request.
    Returns the final natural-language response.
    """
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
            # No more tool calls — this is the final answer.
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
            if fn is None:
                tool_result = {"error": f"unknown tool {fn_name}"}
            else:
                tool_result = fn(**fn_args)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(tool_result),
                }
            )

    return "I wasn't able to finish this request within the allowed tool-call steps."


if __name__ == "__main__":
    import sys as _sys

    if len(_sys.argv) > 1:
        query = " ".join(_sys.argv[1:])
    else:
        query = "One of our sales reps has three deals that have had no activity in over two weeks. Find them and draft a short follow-up message for each."

    print(f"> {query}\n")
    print(run_agent(query))
