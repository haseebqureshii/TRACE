import os
from typing import Dict, Any
from src.llm_client import get_llm_client
from src.guardrails.dst import DialogueStateManager, LessonState
from src.guardrails.input_rail import validate_user_input
from src.guardrails.evaluator import validate_topic_adherence


def run_tutor_turn(user_input: str, state_manager: DialogueStateManager) -> Dict[str, Any]:
    """
    Run a single tutor turn in the pipeline.
    
    Args:
        user_input: The user's input string
        state_manager: DialogueStateManager instance
        
    Returns:
        Dict with 'response', 'is_on_topic', and 'rationale'
    """
    # a. Call validate_user_input (Input Rail)
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
    client, model_id = get_llm_client()
    
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_context},
            {"role": "user", "content": user_input}
        ]
    )
    
    response_text = response.choices[0].message.content
    
    # e. Pass response through validate_topic_adherence (Output Rail)
    evaluation_result = validate_topic_adherence(response_text, state_manager.state.current_sub_goal)
    
    # f. Increment turn count in state_manager
    state_manager.increment_turn()
    
    # g. Return dict with response, is_on_topic, and rationale
    return {
        "response": response_text,
        "is_on_topic": evaluation_result["is_on_topic"],
        "rationale": evaluation_result["rationale"]
    }
