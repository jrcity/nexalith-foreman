"""
Nexalith Foreman — interactive CLI.

Usage:
    python3 agent/cli.py

Requires llama-server already running locally (see README for the start
command). This CLI is a thin presentation layer over orchestrator.py —
it contains no agent logic of its own, so the web UI (built separately)
can reuse the exact same orchestrator functions without any drift
between the two interfaces.
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from orchestrator import run_agent, LLAMA_SERVER_URL

BANNER = r"""
 _____                            _   _    ______                              
|  ___|                          | | | |   |  ___|                             
| |_ ___  _ __ ___ _ __ ___   __ _| |_| |__ | |_ ___  _ __ ___ _ __ ___   __ _ _ __
|  _/ _ \| '__/ _ \ '_ ` _ \ / _` | __| '_ \|  _/ _ \| '__/ _ \ '_ ` _ \ / _` | '_ \
| || (_) | | |  __/ | | | | | (_| | |_| | | | || (_) | | |  __/ | | | | | (_| | | | |
\_| \___/|_|  \___|_| |_| |_|\__,_|\__|_| |_\_| \___/|_|  \___|_| |_| |_|\__,_|_| |_|

  Offline AI ops agent — HR, CRM & CMS. No cloud. Type 'exit' to quit.
"""


def _check_server() -> bool:
    try:
        requests.get(LLAMA_SERVER_URL.replace("/v1/chat/completions", "/health"), timeout=3)
        return True
    except requests.exceptions.RequestException:
        return False


def main() -> None:
    print(BANNER)

    if not _check_server():
        print(
            "⚠️  Could not reach llama-server at "
            f"{LLAMA_SERVER_URL.rsplit('/v1', 1)[0]}.\n"
            "   Start it first, e.g.:\n"
            "   ./build/bin/llama-server -m model/qwen2.5-3b-instruct-q4_k_m.gguf "
            "--ctx-size 4096 --threads 4 --port 8080\n"
        )
        sys.exit(1)

    print("Connected to local model. Ready.\n")

    while True:
        try:
            user_input = input("foreman> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        print()
        try:
            response = run_agent(user_input)
            print(response)
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Lost connection to the model server: {e}")
        except Exception as e:  # noqa: BLE001 — surface unexpected errors plainly in a CLI
            print(f"⚠️  Something went wrong handling that request: {e}")
        print()


if __name__ == "__main__":
    main()
