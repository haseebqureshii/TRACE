from typing import List, Dict, Any
from pydantic import BaseModel


class KBDocument(BaseModel):
    id: str
    category: str
    title: str
    content: str


class KBSchema(BaseModel):
    business_name: str
    domain_description: str
    documents: List[KBDocument]


class SessionState(BaseModel):
    session_id: str
    business_name: str = "Default Business"
    domain_description: str = "Default domain description"
    kb_documents: List[KBDocument] = []
    kb_embeddings: Any = None
    chat_history: List[Dict[str, str]] = []
    drift_strikes: int = 0
    max_strikes: int = 3
    is_escalated: bool = False


class SessionManager:
    _sessions: Dict[str, SessionState] = {}

    @classmethod
    def get_or_create_session(cls, session_id: str) -> SessionState:
        if session_id not in cls._sessions:
            cls._sessions[session_id] = SessionState(session_id=session_id)
        return cls._sessions[session_id]

    @classmethod
    def update_session(cls, state: SessionState) -> None:
        cls._sessions[state.session_id] = state

    @classmethod
    def reset_session(cls, session_id: str) -> None:
        if session_id in cls._sessions:
            cls._sessions[session_id] = SessionState(session_id=session_id)

    @classmethod
    def get_session(cls, session_id: str) -> SessionState | None:
        return cls._sessions.get(session_id)