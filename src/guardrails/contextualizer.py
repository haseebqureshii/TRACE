import os
from typing import List, Dict
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variables from .env file, overriding system environment variables
load_dotenv(override=True)


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


async def contextualize_query(user_input: str, chat_history: List[Dict[str, str]]) -> str:
    """
    If chat_history is empty, return user_input directly.
    If chat_history exists, make a fast call to the LLM client asking it to 
    rewrite user_input into a single standalone, fully qualified search query 
    that resolves pronouns.
    """
    if not chat_history:
        return user_input

    client, model_id = _get_async_client()

    # Build history context for pronoun resolution
    history_context = "\n".join([
        f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
        for msg in chat_history[-5:]  # Last 5 messages for context
    ])

    prompt = f"""You are a query contextualizer for a customer service agent. 
Your task is to rewrite the user's latest input into a single standalone, fully qualified search query 
that resolves any pronouns based on the conversation history.

Conversation history:
{history_context}

Latest user input: {user_input}

Provide only the rewritten standalone query, without any additional explanation or text."""

    try:
        response = await client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "You are a query contextualizer. Return only the rewritten standalone query."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100
        )
        
        contextualized = response.choices[0].message.content.strip()
        return contextualized if contextualized else user_input
        
    except Exception:
        # Fallback to user_input if LLM call fails
        return user_input