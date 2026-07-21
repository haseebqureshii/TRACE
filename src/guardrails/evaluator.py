import os
from typing import Dict, Any
from pydantic import BaseModel, Field
from src.llm_client import get_llm_client


class TopicAdherenceResult(BaseModel):
    """Result of topic adherence validation."""
    is_on_topic: bool = Field(..., description="Whether the response adheres to the current sub-goal")
    rationale: str = Field(..., description="Brief explanation of why the response is or isn't on topic")


def validate_topic_adherence(response_text: str, current_sub_goal: str) -> Dict[str, Any]:
    """
    Evaluate whether response_text adheres to current_sub_goal using the LLM client.
    
    Args:
        response_text: The response to evaluate
        current_sub_goal: The current sub-goal to check adherence against
        
    Returns:
        Dict with 'is_on_topic' (bool) and 'rationale' (str)
    """
    client, model_id = get_llm_client()
    
    prompt = f"""Evaluate whether the following response is a proper language tutor response that stays on the current sub-goal, or if it is a refusal/pivot indicating the user's input was off-topic.

Current sub-goal: {current_sub_goal}

Response: {response_text}

Does the response adhere to the current sub-goal as a proper language tutor response? Return a JSON object with 'is_on_topic' (boolean) and 'rationale' (string) fields. If the response begins with an apology, refusal, or indicates the user's input was off-topic or outside the tutor's scope, set is_on_topic to False, even if it later pivots to relevant content."""

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "You are an evaluator that checks if responses adhere to a given sub-goal. Return only valid JSON with 'is_on_topic' and 'rationale' fields."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_schema", "json_schema": {
                "name": "topic_adherence_result",
                "schema": {
                    "type": "object",
                    "properties": {
                        "is_on_topic": {"type": "boolean"},
                        "rationale": {"type": "string"}
                    },
                    "required": ["is_on_topic", "rationale"],
                    "additionalProperties": False
                }
            }}
        )
        
        result_text = response.choices[0].message.content
        result = TopicAdherenceResult.model_validate_json(result_text)
        
        return {
            "is_on_topic": result.is_on_topic,
            "rationale": result.rationale
        }
        
    except Exception as e:
        # Fallback: try to parse as JSON manually if response_format fails
        try:
            import json
            result_dict = json.loads(result_text)
            return {
                "is_on_topic": bool(result_dict.get("is_on_topic", False)),
                "rationale": str(result_dict.get("rationale", "Failed to evaluate topic adherence"))
            }
        except Exception:
            return {
                "is_on_topic": False,
                "rationale": f"Error evaluating topic adherence: {str(e)}"
            }