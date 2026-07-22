import os
import asyncio
from typing import Dict, Any
from dotenv import load_dotenv
from openai import AsyncOpenAI
from src.state import SessionState, SessionManager
from src.guardrails.contextualizer import contextualize_query
from src.guardrails.input_rail import get_kb_retriever
from src.guardrails.evaluator import validate_output_adherence

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


async def process_chat_turn(session_id: str, user_input: str) -> Dict[str, Any]:
    """
    Process a single chat turn in the customer service pipeline.
    
    1. Fetch SessionState via SessionManager.
    2. If state.is_escalated is True OR state.drift_strikes >= state.max_strikes: short-circuit and return escalation response.
    3. Run contextualize_query(user_input, state.chat_history).
    4. Run evaluate_relevance_and_retrieve(contextualized_query).
    5. If is_relevant is False: increment drift_strikes, check max_strikes, escalate or return off-topic response.
    6. If is_relevant is True: generate LLM response, validate output adherence, save state, return response.
    """
    # 1. Fetch SessionState via SessionManager
    state = SessionManager.get_or_create_session(session_id)
    
    # 2. Short-circuit escalation logic: if state.is_escalated is True OR state.drift_strikes >= state.max_strikes
    if state.is_escalated or state.drift_strikes >= state.max_strikes:
        state.is_escalated = True
        SessionManager.update_session(state)
        return {
            "response": "Chat diverted/escalated to a human.",
            "is_escalated": True,
            "strikes": state.drift_strikes,
            "rationale": "Escalation active."
        }
    
    # 3. Run contextualize_query
    contextualized_query = await contextualize_query(user_input, state.chat_history)
    
    # 4. Run evaluate_relevance_and_retrieve
    retriever = get_kb_retriever()
    retrieval_result = retriever.evaluate_relevance_and_retrieve(contextualized_query, state)
    
    # 5. If is_relevant is False
    if not retrieval_result["is_relevant"]:
        state.drift_strikes += 1
        
        if state.drift_strikes >= state.max_strikes:
            state.is_escalated = True
            SessionManager.update_session(state)
            return {
                "response": "Chat diverted/escalated to a human.",
                "is_escalated": True,
                "strikes": state.drift_strikes,
                "rationale": "Escalation active."
            }
        else:
            SessionManager.update_session(state)
            return {
                "response": "I am only able to assist with customer service and support inquiries. Could you please clarify your request?",
                "is_escalated": False,
                "strikes": state.drift_strikes,
                "rationale": "Off-topic query detected."
            }
    
    # 6. If is_relevant is True
    retrieved_contexts = retrieval_result["contexts"]
    
    # Generate LLM response using retrieved KB contexts in system prompt with concise directive
    client, model_id = _get_async_client()
    
    contexts_str = "\n".join([f"- {ctx}" for ctx in retrieved_contexts])
    
    system_prompt = f"""You are a customer service assistant for {state.business_name}. Keep all responses as short, direct, and concise as possible (1–3 sentences maximum). Do not use conversational filler. Strictly answer the query using only the provided context.

Knowledge Base Contexts:
{contexts_str}

If the user's question cannot be answered using the provided contexts, politely indicate that you don't have that information and suggest they speak with a human customer support specialist."""

    try:
        response = await client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
        )
        
        llm_response = response.choices[0].message.content
        
        # Schedule validate_output_adherence asynchronously
        evaluation_task = asyncio.create_task(
            validate_output_adherence(llm_response, retrieved_contexts)
        )
        
        # Append user message and assistant response to state.chat_history
        state.chat_history.append({"role": "user", "content": user_input})
        state.chat_history.append({"role": "assistant", "content": llm_response})
        
        # Keep chat history manageable (last 20 messages)
        if len(state.chat_history) > 20:
            state.chat_history = state.chat_history[-20:]
        
        # Save state
        SessionManager.update_session(state)
        
        # Wait for evaluation task to complete (but don't fail if it does)
        try:
            eval_result = await asyncio.wait_for(evaluation_task, timeout=5.0)
            if not eval_result.get("is_grounded", True):
                # If not grounded, we could escalate or adjust response, but for now just note it
                pass
        except asyncio.TimeoutError:
            pass
        
        return {
            "response": llm_response,
            "is_escalated": False,
            "strikes": state.drift_strikes,
            "rationale": "In-domain response generated."
        }
        
    except Exception as e:
        return {
            "response": "I'm sorry, I encountered an error processing your request. Please try again or speak with a human customer support specialist.",
            "is_escalated": False,
            "strikes": state.drift_strikes,
            "rationale": f"Error generating response: {str(e)}"
        }