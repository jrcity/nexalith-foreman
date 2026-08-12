"""
IT/Infrastructure tools: log inspection and service restart.

SYNTHETIC SIMULATION ONLY. These tools read and write static records in
the local seed_corpus.json -- they do not connect to, query, or control
any real running process, container, or log aggregator. This demo
intentionally simulates the *pattern* of an ops agent reacting to a
detected issue (read logs -> detect a problem -> take a corrective
action), not real infrastructure access.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.retriever import (  # noqa: E402
    query_system_logs as _query_system_logs,
    restart_service as _restart_service,
)


def query_system_logs(service: str, level: str = "error", minutes_ago: int = 10) -> dict:
    """
    Look up synthetic log entries for a service at a given severity level.
    (Simulated data -- see module docstring.)
    """
    return _query_system_logs(service=service, level=level, minutes_ago=minutes_ago)


def restart_service(service_name: str) -> dict:
    """
    Simulate restarting a named service -- flips a status flag in the
    local dataset only. Does not affect any real process.
    """
    return _restart_service(service_name=service_name)


if __name__ == "__main__":
    import json
    print(json.dumps(query_system_logs("nexid-service"), indent=2))
