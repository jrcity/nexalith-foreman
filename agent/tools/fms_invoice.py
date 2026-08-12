"""FMS tool: invoice generation."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.retriever import generate_invoice as _generate_invoice  # noqa: E402


def generate_invoice(client_id: str, amount: float, description: str) -> dict:
    """
    Generate a new invoice for a client and mark it sent. Writes a new
    invoice record to the local dataset. amount is in USD.
    """
    return _generate_invoice(client_id=client_id, amount=amount, description=description)


if __name__ == "__main__":
    import json
    print(json.dumps(generate_invoice("cust_201", 250.0, "Test invoice"), indent=2))
