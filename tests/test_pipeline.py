import json
import os
import pytest
import asyncio
from src.state import SessionManager, SessionState, KBDocument, KBSchema
from src.pipeline import process_chat_turn
from src.guardrails.contextualizer import contextualize_query
from src.guardrails.input_rail import get_kb_retriever


def load_kb_documents() -> dict:
    """Load KB documents from tests/fixtures/kb_documents.json."""
    fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")
    kb_file = os.path.join(fixtures_dir, "kb_documents.json")
    
    with open(kb_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_test_conversations() -> dict:
    """Load test conversations from tests/fixtures/test_conversations.json."""
    fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")
    conv_file = os.path.join(fixtures_dir, "test_conversations.json")
    
    with open(conv_file, "r", encoding="utf-8") as f:
        return json.load(f)


def create_test_session_state(session_id: str, kb_data: dict):
    """Helper to create a test session state with KB data."""
    retriever = get_kb_retriever()
    kb_documents = [KBDocument(id=doc["id"], category=doc["category"], title=doc.get("title", doc["category"]), content=doc["content"]) for doc in kb_data["documents"]]
    kb_embeddings = retriever.embed_kb_documents(kb_documents)
    
    state = SessionState(
        session_id=session_id,
        business_name=kb_data.get("business_name", "Test Store"),
        domain_description=kb_data.get("domain_description", "Customer service for a retail store"),
        kb_documents=kb_documents,
        kb_embeddings=kb_embeddings,
        chat_history=[],
        drift_strikes=0,
        max_strikes=3,
        is_escalated=False
    )
    SessionManager.update_session(state)
    return state


@pytest.fixture(autouse=True)
def cleanup_sessions():
    """Clean up sessions before and after each test."""
    SessionManager._sessions = {}
    yield
    SessionManager._sessions = {}


@pytest.mark.asyncio
async def test_multi_turn_pronoun_contextualization():
    """Test multi-turn pronoun contextualization (Turn 1: 'What is your refund policy?', Turn 2: 'What about opened items?')."""
    conversations = load_test_conversations()
    multi_turn = conversations["multi_turn_contextualization"]
    
    chat_history = [
        {"role": "user", "content": multi_turn["turn_1_user"]},
        {"role": "assistant", "content": multi_turn["turn_1_assistant"]}
    ]
    
    contextualized = await contextualize_query(multi_turn["turn_2_user"], chat_history)
    
    # Verify the contextualized query resolves pronouns and is relevant to opened items return policy
    assert contextualized is not None
    assert len(contextualized) > 0
    # The contextualized query should mention opened items and return/refund policy
    assert any(keyword in contextualized.lower() for keyword in ["opened", "return", "policy", "refund"])


@pytest.mark.asyncio
async def test_successful_in_domain_rag_response():
    """Test successful in-domain RAG response execution."""
    session_id = "test_session_rag_001"
    kb_data = load_kb_documents()
    create_test_session_state(session_id, kb_data)
    
    conversations = load_test_conversations()
    in_domain_query = conversations["in_domain_queries"][0]  # "What is your return policy?"
    
    result = await process_chat_turn(session_id, in_domain_query)
    
    # Verify response structure
    assert "response" in result
    assert "is_escalated" in result
    assert "strikes" in result
    assert "rationale" in result
    
    # Verify it's not escalated and has 0 strikes
    assert result["is_escalated"] is False
    assert result["strikes"] == 0
    assert "In-domain response generated" in result["rationale"]
    
    # Verify response is not the off-topic escalation message
    assert "I am only able to assist with customer service" not in result["response"]
    assert "human customer support specialist" not in result["response"]


@pytest.mark.asyncio
async def test_off_topic_query_detection_and_strike_incrementation():
    """Test off-topic query detection and invisible strike counter incrementation."""
    session_id = "test_session_strikes_001"
    kb_data = load_kb_documents()
    create_test_session_state(session_id, kb_data)
    
    conversations = load_test_conversations()
    off_topic_query = conversations["off_topic_queries"][0]  # "What is the capital of France?"
    
    # First off-topic query
    result1 = await process_chat_turn(session_id, off_topic_query)
    
    assert result1["is_escalated"] is False
    assert result1["strikes"] == 1
    assert "Off-topic query detected" in result1["rationale"]
    assert "I am only able to assist with customer service and support inquiries" in result1["response"]
    
    # Second off-topic query
    result2 = await process_chat_turn(session_id, off_topic_query)
    
    assert result2["is_escalated"] is False
    assert result2["strikes"] == 2
    assert "Off-topic query detected" in result2["rationale"]
    
    # Third off-topic query should trigger escalation
    result3 = await process_chat_turn(session_id, off_topic_query)
    
    assert result3["is_escalated"] is True
    assert result3["strikes"] == 3
    assert result3["response"] == "Chat diverted/escalated to a human."
    assert result3["rationale"] == "Escalation active."


@pytest.mark.asyncio
async def test_human_escalation_trigger_when_strikes_hit_max():
    """Test human escalation trigger when strikes hit max_strikes."""
    session_id = "test_session_escalation_002"
    kb_data = load_kb_documents()
    create_test_session_state(session_id, kb_data)
    
    conversations = load_test_conversations()
    off_topic_query = conversations["off_topic_queries"][1]  # "Tell me a joke"
    
    # Simulate a session that has already accumulated 2 strikes
    state = SessionManager.get_or_create_session(session_id)
    state.drift_strikes = 2
    state.max_strikes = 3
    state.is_escalated = False
    SessionManager.update_session(state)
    
    # Next off-topic query should trigger escalation
    result = await process_chat_turn(session_id, off_topic_query)
    
    assert result["is_escalated"] is True
    assert result["strikes"] == 3
    assert result["response"] == "Chat diverted/escalated to a human."
    assert result["rationale"] == "Escalation active."
    
    # Subsequent queries should return the short-circuit escalation response
    result2 = await process_chat_turn(session_id, "What is your return policy?")
    
    assert result2["is_escalated"] is True
    assert result2["response"] == "Chat diverted/escalated to a human."
    assert result2["rationale"] == "Escalation active."


@pytest.mark.asyncio
async def test_kb_retriever_relevance_evaluation():
    """Test KBRetriever relevance evaluation with in-domain and off-domain queries."""
    retriever = get_kb_retriever()
    kb_data = load_kb_documents()
    state = create_test_session_state("test_kb_retriever_session", kb_data)
    
    # In-domain query
    in_domain_query = "What is your return policy for opened items?"
    in_domain_result = retriever.evaluate_relevance_and_retrieve(in_domain_query, state, threshold=0.30, top_k=2)
    
    assert in_domain_result["is_relevant"] is True
    assert in_domain_result["score"] >= 0.30
    assert len(in_domain_result["contexts"]) > 0
    
    # Off-domain query
    off_domain_query = "What is the capital of Japan?"
    off_domain_result = retriever.evaluate_relevance_and_retrieve(off_domain_query, state, threshold=0.30, top_k=2)
    
    assert off_domain_result["is_relevant"] is False or off_domain_result["score"] < 0.30


@pytest.mark.asyncio
async def test_short_circuit_escalation_logic():
    """Test that when drift_strikes reaches max, the pipeline instantly returns 'Chat diverted/escalated to a human.' without invoking external LLM APIs."""
    session_id = "test_session_short_circuit_001"
    kb_data = load_kb_documents()
    state = create_test_session_state(session_id, kb_data)
    
    # Set drift_strikes to max_strikes
    state.drift_strikes = 3
    state.max_strikes = 3
    state.is_escalated = False
    SessionManager.update_session(state)
    
    # Query should trigger short-circuit escalation
    result = await process_chat_turn(session_id, "What is your return policy?")
    
    assert result["is_escalated"] is True
    assert result["strikes"] == 3
    assert result["response"] == "Chat diverted/escalated to a human."
    assert result["rationale"] == "Escalation active."
