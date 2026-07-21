import json
import os
import time
import pytest
import asyncio
from src.guardrails.dst import DialogueStateManager, LessonState
from src.pipeline import run_tutor_turn
from src.guardrails.evaluator import validate_topic_adherence
from src.guardrails.input_rail import validate_user_input


def load_initial_state() -> LessonState:
    """Load state parameters from tests/fixtures/initial_state.json."""
    fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")
    state_file = os.path.join(fixtures_dir, "initial_state.json")
    
    with open(state_file, "r", encoding="utf-8") as f:
        state_data = json.load(f)
    
    return LessonState(**state_data)


def load_user_input(input_name: str) -> str:
    """Load test input from tests/fixtures/user_inputs/."""
    fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures", "user_inputs")
    input_file = os.path.join(fixtures_dir, f"{input_name}.txt")
    
    with open(input_file, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_simulated_response(response_name: str) -> str:
    """Load simulated response from tests/fixtures/."""
    fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")
    response_file = os.path.join(fixtures_dir, f"{response_name}.txt")
    
    with open(response_file, "r", encoding="utf-8") as f:
        return f.read().strip()


@pytest.mark.asyncio
async def test_on_topic_turn():
    """Verify that an on-topic prompt passes validation."""
    # Load state parameters
    state = load_initial_state()
    state_manager = DialogueStateManager(state)
    
    # Read on-topic test input
    user_input = load_user_input("on_topic_input")
    
    # Run the tutor turn
    result = await run_tutor_turn(user_input, state_manager)
    
    # Verify the response structure
    assert "response" in result
    assert "is_on_topic" in result
    assert "rationale" in result
    
    # Verify that on-topic prompt passes validation
    assert result["is_on_topic"] is True, f"Expected is_on_topic to be True, but got: {result['is_on_topic']}. Rationale: {result['rationale']}"


@pytest.mark.asyncio
async def test_off_topic_user_input():
    """Verify that off-topic user input is caught by the Input Rail and short-circuits execution."""
    # Load state parameters
    state = load_initial_state()
    state_manager = DialogueStateManager(state)
    
    # Read off-topic test input
    user_input = load_user_input("off_topic_input")
    
    # Run the tutor turn
    result = await run_tutor_turn(user_input, state_manager)
    
    # Verify the response structure
    assert "response" in result
    assert "is_on_topic" in result
    assert "rationale" in result
    
    # Verify that off-topic user input is blocked by Input Rail
    assert result["is_on_topic"] is False, f"Expected is_on_topic to be False, but got: {result['is_on_topic']}. Rationale: {result['rationale']}"
    assert result["response"] == "Let's stay focused on our lesson goal.", f"Expected short-circuit response, but got: {result['response']}"
    assert "Off-topic user input blocked by Input Rail" in result["rationale"]


@pytest.mark.asyncio
async def test_output_rail_drift_detection():
    """Verify that the Output Rail flags a simulated off-topic LLM response as is_on_topic = False."""
    # Load state parameters
    state = load_initial_state()
    
    # Load simulated off-topic LLM response
    simulated_response = load_simulated_response("simulated_off_topic_response")
    
    # Pass the simulated response directly to validate_topic_adherence (Output Rail)
    evaluation_result = await validate_topic_adherence(simulated_response, state.current_sub_goal)
    
    # Verify the evaluation result structure
    assert "is_on_topic" in evaluation_result
    assert "rationale" in evaluation_result
    
    # Verify that the Output Rail flags the simulated off-topic response
    assert evaluation_result["is_on_topic"] is False, f"Expected is_on_topic to be False, but got: {evaluation_result['is_on_topic']}. Rationale: {evaluation_result['rationale']}"


def test_input_rail_local_execution():
    """Verify that validate_user_input executes locally without invoking the remote LLM client."""
    # Load state parameters
    state = load_initial_state()
    
    # Read test inputs
    on_topic_input = load_user_input("on_topic_input")
    off_topic_input = load_user_input("off_topic_input")
    
    # Test on-topic input
    start_time = time.perf_counter()
    on_topic_result = validate_user_input(on_topic_input, state.current_sub_goal)
    on_topic_time = time.perf_counter() - start_time
    
    # Verify on-topic result structure
    assert "is_relevant" in on_topic_result
    assert "score" in on_topic_result
    assert "rationale" in on_topic_result
    assert on_topic_result["is_relevant"] is True
    
    # Test off-topic input
    start_time = time.perf_counter()
    off_topic_result = validate_user_input(off_topic_input, state.current_sub_goal)
    off_topic_time = time.perf_counter() - start_time
    
    # Verify off-topic result structure
    assert "is_relevant" in off_topic_result
    assert "score" in off_topic_result
    assert "rationale" in off_topic_result
    assert off_topic_result["is_relevant"] is False
    
    # Verify that local execution is fast (should be under 2 seconds for embedding computation)
    # First call may be slower due to model loading, but subsequent calls should be fast
    assert off_topic_time < 2.0, f"Local embedding execution took {off_topic_time:.3f}s, expected < 2.0s"


@pytest.mark.asyncio
async def test_cross_lingual_input_rail():
    """Verify that foreign language input matches the English current_sub_goal with is_relevant = True."""
    # Load state parameters
    state = load_initial_state()
    
    # Read cross-lingual test input
    cross_lingual_input = load_user_input("cross_lingual_on_topic_input")
    
    # Test cross-lingual input with input rail
    start_time = time.perf_counter()
    cross_lingual_result = validate_user_input(cross_lingual_input, state.current_sub_goal)
    cross_lingual_time = time.perf_counter() - start_time
    
    # Verify cross-lingual result structure
    assert "is_relevant" in cross_lingual_result
    assert "score" in cross_lingual_result
    assert "rationale" in cross_lingual_result
    
    # Verify that cross-lingual input is recognized as relevant
    assert cross_lingual_result["is_relevant"] is True, f"Expected is_relevant to be True for cross-lingual input, but got: {cross_lingual_result['is_relevant']}. Rationale: {cross_lingual_result['rationale']}. Score: {cross_lingual_result['score']}"
    
    # Verify that local execution is fast
    assert cross_lingual_time < 2.0, f"Cross-lingual embedding execution took {cross_lingual_time:.3f}s, expected < 2.0s"