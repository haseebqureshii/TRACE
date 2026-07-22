import os
from typing import List, Dict, Any
from openai import AsyncOpenAI


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


async def validate_output_adherence(response_text: str, retrieved_contexts: List[str]) -> Dict[str, Any]:
    """
    Evaluates whether the LLM's generated response strictly answers using the 
    retrieved KB context without inventing ungrounded policies.
    
    Returns {"is_grounded": bool, "rationale": str}.
    """
    client, model_id = _get_async_client()
    
    contexts_str = "\n".join([f"- {ctx}" for ctx in retrieved_contexts])
    
    prompt = f"""Evaluate whether the following response is strictly grounded in the provided knowledge base contexts, 
without inventing ungrounded policies or information.

Knowledge Base Contexts:
{contexts_str}

Response to evaluate: {response_text}

Is the response strictly grounded in the provided contexts? Return a JSON object with 'is_grounded' (boolean) and 'rationale' (string) fields."""

    try:
        response = await client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "You are an evaluator that checks if responses are strictly grounded in provided knowledge base contexts. Return only valid JSON with 'is_grounded' and 'rationale' fields."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        result_text = response.choices[0].message.content
        import json
        result = json.loads(result_text)
        
        return {
            "is_grounded": bool(result.get("is_grounded", False)),
            "rationale": str(result.get("rationale", "Failed to evaluate output adherence"))
        }
        
    except Exception as e:
        # Fallback: try to parse as JSON manually if response_format fails
        try:
            import json
            if 'result_text' in locals():
                result_dict = json.loads(result_text)
                return {
                    "is_grounded": bool(result_dict.get("is_grounded", False)),
                    "rationale": str(result_dict.get("rationale", "Failed to evaluate output adherence"))
                }
        except Exception:
            pass
        
        return {
            "is_grounded": False,
            "rationale": f"Error evaluating output adherence: {str(e)}"
        }