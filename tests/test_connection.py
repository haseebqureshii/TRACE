import os
from dotenv import load_dotenv
from src.llm_client import get_llm_client


def test_llm_connection():
    """Test LLM connectivity with a simple prompt."""
    # Load environment variables from .env file
    load_dotenv()

    # Get client and model
    client, model_id = get_llm_client()

    # Verify required environment variables are set
    api_key = os.getenv("TRACE_LLM_API_KEY_TEST")
    base_url = os.getenv("TRACE_LLM_BASE_URL")

    if not api_key or api_key == "your_key_here":
        raise ValueError("TRACE_LLM_API_KEY_TEST is not set or is using the default placeholder value.")

    if not base_url or base_url == "http://your-university-llm-endpoint/v1":
        raise ValueError("TRACE_LLM_BASE_URL is not set or is using the default placeholder value.")

    # Test the client with a simple prompt
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "user", "content": "Hello, this is a connection test. Please respond with 'Connection successful.'"}
            ],
            max_tokens=50
        )
        print(f"Connection test successful!")
        print(f"Model: {model_id}")
        print(f"Response: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"Connection test failed: {e}")
        return False


if __name__ == "__main__":
    test_llm_connection()