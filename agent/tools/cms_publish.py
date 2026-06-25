"""CMS tool: find draft content that needs publishing, often linked to HR events."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.retriever import get_draft_content_for_employee  # noqa: E402


def cms_publish_check(emp_id: str) -> dict:
    """
    Find draft CMS content linked to a specific employee — e.g. a
    new-hire announcement post that's still in draft and needs the
    employee's name/role filled in before publishing.

    This is the cross-domain link: an HR event (new hire) surfaces a
    CMS action (publish announcement), which is what makes the agent's
    cross_disciplinary_pairing genuinely load-bearing rather than two
    unrelated tools bolted together.
    """
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


if __name__ == "__main__":
    import json
    print(json.dumps(cms_publish_check("emp_001"), indent=2))
