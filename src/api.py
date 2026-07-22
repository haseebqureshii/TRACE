import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from src.pipeline import process_chat_turn
from src.state import SessionManager, SessionState, KBSchema, KBDocument
from src.guardrails.input_rail import get_kb_retriever


app = FastAPI(title="Drift-Free Customer Service Agent API")


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ResetRequest(BaseModel):
    session_id: str


class InitSessionRequest(BaseModel):
    kb_data: dict


@app.post("/api/v1/session/init")
async def init_session_endpoint(request: InitSessionRequest):
    """Initialize a new session with custom KB data."""
    # Validate kb_data against KBSchema
    try:
        kb_schema = KBSchema(**request.kb_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid KB data: {str(e)}")
    
    # Get KB retriever and compute embeddings
    retriever = get_kb_retriever()
    kb_documents = kb_schema.documents
    kb_embeddings = retriever.embed_kb_documents(kb_documents)
    
    # Generate new session_id (UUID4)
    session_id = str(uuid.uuid4())
    
    # Initialize SessionState
    state = SessionState(
        session_id=session_id,
        business_name=kb_schema.business_name,
        domain_description=kb_schema.domain_description,
        kb_documents=kb_documents,
        kb_embeddings=kb_embeddings,
        chat_history=[],
        drift_strikes=0,
        max_strikes=3,
        is_escalated=False
    )
    
    # Save state
    SessionManager.update_session(state)
    
    return {
        "session_id": session_id,
        "business_name": kb_schema.business_name,
        "message": "Session initialized successfully."
    }


@app.post("/api/v1/chat")
async def chat_endpoint(request: ChatRequest):
    """Process a chat turn and return the response."""
    # Verify session_id exists
    state = SessionManager.get_session(request.session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session Not Found")
    
    result = await process_chat_turn(request.session_id, request.message)
    return result


@app.get("/api/v1/session/{session_id}")
async def get_session_endpoint(session_id: str):
    """Return full SessionState for the given session_id."""
    state = SessionManager.get_session(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Return state without embeddings (not serializable)
    state_dict = state.model_dump()
    state_dict["kb_embeddings"] = None
    return state_dict


@app.post("/api/v1/session/reset")
async def reset_session_endpoint(request: ResetRequest):
    """Reset state for the given session_id."""
    SessionManager.reset_session(request.session_id)
    return {"status": "reset", "session_id": request.session_id}
