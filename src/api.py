from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from src.pipeline import process_chat_turn
from src.state import SessionManager, SessionState


app = FastAPI(title="Drift-Free Customer Service Agent API")


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ResetRequest(BaseModel):
    session_id: str


@app.post("/api/v1/chat")
async def chat_endpoint(request: ChatRequest):
    """Process a chat turn and return the response."""
    result = await process_chat_turn(request.session_id, request.message)
    return result


@app.get("/api/v1/session/{session_id}")
async def get_session_endpoint(session_id: str):
    """Return full SessionState for the given session_id."""
    state = SessionManager.get_session(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return state.model_dump()


@app.post("/api/v1/session/reset")
async def reset_session_endpoint(request: ResetRequest):
    """Reset state for the given session_id."""
    SessionManager.reset_session(request.session_id)
    return {"status": "reset", "session_id": request.session_id}