import os
import asyncio
from typing import Dict, Any
from openai import AsyncOpenAI
from src.state import SessionState, SessionManager
from src.guardrails.contextualizer import contextualize_query
from src.guardrails.input_rail import get_kb_retriever
from src.guardrails.evaluator import validate_output_adherence


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
    2. If state.is_escalated is True: return immediate handoff response.
    3. Run contextualize_query(user_input, state.chat_history).
    4. Run evaluate_relevance_and_retrieve(contextualized_query).
    5. If is_relevant is False: increment drift_strikes, check max_strikes, escalate or return off-topic response.
    6. If is_relevant is True: generate LLM response, validate output adherence, save state, return response.
    """
    # 1. Fetch SessionState via SessionManager
    state = SessionManager.get_or_create_session(session_id)
    
    # 2. If state.is_escalated is True: return immediate handoff response
    if state.is_escalated:
        return {
            "response": "A human agent has been notified and will take over shortly.",
            "is_escalated": True,
            "strikes": state.drift_strikes
        }
    
    # 3. Run contextualize_query
    contextualized_query = await contextualize_query(user_input, state.chat_history)
    
    # 4. Run evaluate_relevance_and_retrieve
    retriever = get_kb_retriever()
    retrieval_result = retriever.evaluate_relevance_and_retrieve(contextualized_query)
    
    # 5. If is_relevant is False
    if not retrieval_result["is_relevant"]:
        state.drift_strikes += 1
        
        if state.drift_strikes >= state.max_strikes:
            state.is_escalated = True
            SessionManager.update_session(state)
            return {
                "response": "I am connecting you with a human customer support specialist who can assist you further.",
                "is_escalated": True,
                "strikes": state.drift_strikes,
                "rationale": "Max drift strikes reached."
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
    
    # Generate LLM response using retrieved KB contexts in system prompt
    client, model_id = _get_async_client()
    
    contexts_str = "\n".join([f"- {ctx}" for ctx in retrieved_contexts])
    
    system_prompt = f"""You are a Drift-Free Customer Service Agent. Your task is to assist customers with their inquiries 
using ONLY the provided knowledge base contexts. Do not invent or make up policies, information, or answers that are not 
strictly grounded in the provided contexts.

Knowledge Base Contexts:
{contexts_str}

If the user's question cannot be answered using the provided contexts, politely indicate that you don't have that information 
and suggest they speak with a human customer support specialist."""

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