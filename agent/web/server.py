"""
Nexalith Foreman — web dashboard backend.

Thin FastAPI wrapper over orchestrator.py and the retriever functions.
Contains NO agent logic of its own — every data-fetching or agent-running
call here delegates to the exact same functions the CLI uses, so the two
interfaces can never drift out of sync with each other.
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))
from orchestrator import run_agent
from rag.retriever import get_stale_deals, get_all_open_deals, get_upcoming_starters
from tools.achievement_log import get_recent_achievements

app = FastAPI(title="Nexalith Foreman")

STATIC_DIR = Path(__file__).parent / "static"


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/chat")
def chat(req: ChatRequest):
    response = run_agent(req.message)
    return {"response": response}


@app.get("/api/deals")
def deals():
    open_deals = get_all_open_deals()
    stale_ids = {d["id"] for d in get_stale_deals()}
    return {
        "deals": [
            {**d, "is_stale": d["id"] in stale_ids} for d in open_deals
        ]
    }


@app.get("/api/onboarding")
def onboarding():
    return {"upcoming": get_upcoming_starters(within_days=30)}


@app.get("/api/achievements")
def achievements():
    return {"achievements": get_recent_achievements(limit=20)}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8090)
