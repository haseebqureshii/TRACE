import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file, overriding system environment variables
load_dotenv(override=True)


def get_llm_client():
    """Initialize and return an OpenAI-compatible LLM client."""
    api_key = os.getenv("TRACE_LLM_API_KEY_TEST")
    base_url = os.getenv("TRACE_LLM_BASE_URL")
    model_id = os.getenv("TEST_MODEL_ID", "qwen3-235b-a22b-instruct-2507")

    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    return client, model_id