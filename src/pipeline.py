import os
import asyncio
from typing import Dict, Any
from openai import AsyncOpenAI
from src.llm_client import get_llm_client
from src.guardrails.dst import DialogueStateManager, LessonState
from src.guardrails.input_rail import validate_user_input
from src.guardrails.evaluator import validate_topic_adherence


def _get_async_client():
    """Initialize and return an AsyncOpenAI client."""
    api_key = os.getenv("TRACE_LLM_API_KEY_TEST")
    base_url = os.getenv("TRACE_LLM_BASE_URL")
    model_id = os.getenv("TEST_MODEL_ID", "qwen3-235b-a22b-instruct-2507")

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url
    )
    return client, model_id


async def run_tutor_turn(user_input: str, state_manager: DialogueStateManager) -> Dict[str, Any]:
    """
    Run a single tutor turn in the pipeline with asynchronous output rail validation.
    
    Args:
        user_input: The user's input string
        state_manager: DialogueStateManager instance
        
    Returns:
        Dict with 'response', 'is_on_topic', and 'rationale'
    """
    # a. Run local Input Rail check (validate_user_input)
    input_evaluation = validate_user_input(user_input, state_manager.state.current_sub_goal)
    
    # b. If is_relevant is False, return immediately with short-circuit response
    if not input_evaluation["is_relevant"]:
        state_manager.increment_turn()
        return {
            "response": "Let's stay focused on our lesson goal.",
            "is_on_topic": False,
            "rationale": "Off-topic user input blocked by Input Rail."
        }
    
    # c. Fetch system context from state_manager
    system_context = state_manager.build_system_context()
    
    # d. Call LLM client using TEST_MODEL_ID with system context and user_input
    client, model_id = _get_async_client()
    
    response = await client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_context},
            {"role": "user", "content": user_input}
        ]
    )
    
    response_text = response.choices[0].message.content
    
    # e. Schedule validate_topic_adherence as a non-blocking concurrent task
    evaluation_task = asyncio.create_task(
        validate_topic_adherence(response_text, state_manager.state.current_sub_goal)
    )
    
    # f. Increment turn count in state_manager
    state_manager.increment_turn()
    
    # g. Wait for evaluation task to complete
    evaluation_result = await evaluation_task
    
    # h. Return dict with response, is_on_topic, and rationale
    return {
        "response": response_text,
        "is_on_topic": evaluation_result["is_on_topic"],
        "rationale": evaluation_result["rationale"]
    }
