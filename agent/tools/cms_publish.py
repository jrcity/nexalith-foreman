"""CMS tools: draft content lookup, publishing new announcements."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.retriever import (  # noqa: E402
    get_draft_content_for_employee,
    publish_announcement as _publish_announcement,
)


def cms_publish_check(emp_id: str) -> dict:
    """Find draft CMS content linked to a specific employee."""
    drafts = get_draft_content_for_employee(emp_id)
    return {
        "draft_count": len(drafts),
        "drafts": [
            {
                "id": d["id"],
                "title": d["title"],
                "status": d["status"],
                "body_summary": d["body_summary"],
            }
            for d in drafts
        ],
    }


def publish_announcement(title: str, platform: str) -> dict:
    """
    Publish a new CMS announcement to a platform (e.g. 'Washy', 'EduiSuite',
    'internal'). Writes a new published content record to the local dataset.
    """
    return _publish_announcement(title=title, platform=platform)


if __name__ == "__main__":
    import json
    print(json.dumps(cms_publish_check("emp_001"), indent=2))
