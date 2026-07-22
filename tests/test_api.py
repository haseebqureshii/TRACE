import json
import os
import pytest
from fastapi.testclient import TestClient
from src.api import app, ChatRequest, ResetRequest
from src.state import SessionManager


def load_test_conversations() -> dict:
    """Load test conversations from tests/fixtures/test_conversations.json."""
    fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")
    conv_file = os.path.join(fixtures_dir, "test_conversations.json")
    
    with open(conv_file, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(autouse=True)
def cleanup_sessions():
    """Clean up sessions before and after each test."""
    SessionManager._sessions = {}
    yield
    SessionManager._sessions = {}


@pytest.fixture
def client():
    """Create a FastAPI TestClient."""
    return TestClient(app)


def test_chat_endpoint_in_domain_query(client):
    """Test POST /api/v1/chat with an in-domain query."""
    conversations = load_test_conversations()
    in_domain_query = conversations["in_domain_queries"][0]  # "What is your return policy?"
    
    request_data = {
        "session_id": "test_api_session_001",
        "message": in_domain_query
    }
    
    response = client.post("/api/v1/chat", json=request_data)
    
    assert response.status_code == 200
    result = response.json()
    
    assert "response" in result
    assert "is_escalated" in result
    assert "strikes" in result
    assert "rationale" in result
    
    assert result["is_escalated"] is False
    assert result["strikes"] == 0


def test_chat_endpoint_off_topic_query(client):
    """Test POST /api/v1/chat with an off-topic query."""
    conversations = load_test_conversations()
    off_topic_query = conversations["off_topic_queries"][0]  # "What is the capital of France?"
    
    request_data = {
        "session_id": "test_api_session_002",
        "message": off_topic_query
    }
    
    # First off-topic query
    response1 = client.post("/api/v1/chat", json=request_data)
    assert response1.status_code == 200
    result1 = response1.json()
    assert result1["is_escalated"] is False
    assert result1["strikes"] == 1
    
    # Second off-topic query
    response2 = client.post("/api/v1/chat", json=request_data)
    assert response2.status_code == 200
    result2 = response2.json()
    assert result2["is_escalated"] is False
    assert result2["strikes"] == 2
    
    # Third off-topic query should trigger escalation
    response3 = client.post("/api/v1/chat", json=request_data)
    assert response3.status_code == 200
    result3 = response3.json()
    assert result3["is_escalated"] is True
    assert result3["strikes"] == 3


def test_get_session_endpoint_existing_session(client):
    """Test GET /api/v1/session/{session_id} for an existing session."""
    # First create a session via chat
    conversations = load_test_conversations()
    in_domain_query = conversations["in_domain_queries"][0]
    
    chat_request = {
        "session_id": "test_api_session_003",
        "message": in_domain_query
    }
    client.post("/api/v1/chat", json=chat_request)
    
    # Now get the session
    response = client.get("/api/v1/session/test_api_session_003")
    
    assert response.status_code == 200
    session_data = response.json()
    
    assert "session_id" in session_data
    assert session_data["session_id"] == "test_api_session_003"
    assert "chat_history" in session_data
    assert "drift_strikes" in session_data
    assert "max_strikes" in session_data
    assert "is_escalated" in session_data


def test_get_session_endpoint_non_existing_session(client):
    """Test GET /api/v1/session/{session_id} for a non-existing session."""
    response = client.get("/api/v1/session/non_existent_session_12345")
    
    assert response.status_code == 404
    assert response.json()["detail"] == "Session not found"


def test_reset_session_endpoint(client):
    """Test POST /api/v1/session/reset."""
    session_id = "test_api_session_004"
    
    # First create a session via chat with an off-topic query to accumulate strikes
    conversations = load_test_conversations()
    off_topic_query = conversations["off_topic_queries"][0]
    
    chat_request = {
        "session_id": session_id,
        "message": off_topic_query
    }
    client.post("/api/v1/chat", json=chat_request)
    
    # Verify session has strikes
    get_response = client.get(f"/api/v1/session/{session_id}")
    assert get_response.status_code == 200
    session_data = get_response.json()
    assert session_data["drift_strikes"] == 1
    
    # Reset the session
    reset_request = {
        "session_id": session_id
    }
    reset_response = client.post("/api/v1/session/reset", json=reset_request)
    
    assert reset_response.status_code == 200
    reset_data = reset_response.json()
    assert reset_data["status"] == "reset"
    assert reset_data["session_id"] == session_id
    
    # Verify session is reset
    get_response_after = client.get(f"/api/v1/session/{session_id}")
    assert get_response_after.status_code == 200
    session_data_after = get_response_after.json()
    assert session_data_after["drift_strikes"] == 0
    assert session_data_after["is_escalated"] is False
    assert len(session_data_after["chat_history"]) == 0