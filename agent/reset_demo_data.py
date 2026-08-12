"""
Restores agent/data/seed_corpus.json from the committed baseline.

Several tools now write to seed_corpus.json (onboard_employee,
approve_leave_request, update_lead_status, log_sales_interaction,
publish_announcement, generate_invoice, restart_service). Run this
before recording a demo or starting a fresh test session, so leftover
test data (extra synthetic employees, invoices, etc.) doesn't pollute
the next run.

Usage:
    python3 agent/reset_demo_data.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
BASELINE = DATA_DIR / "seed_corpus.baseline.json"
LIVE = DATA_DIR / "seed_corpus.json"


def main() -> None:
    if not BASELINE.exists():
        print(
            f"No baseline found at {BASELINE}.\n"
            f"Create one once, the first time, with:\n"
            f"  cp {LIVE} {BASELINE}"
        )
        return
    shutil.copy(BASELINE, LIVE)
    print(f"Restored {LIVE} from baseline.")


if __name__ == "__main__":
    main()
