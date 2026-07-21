import os
from typing import Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer, util


# Initialize the SentenceTransformer model on module load
_model = None


def get_model():
    """Get or initialize the SentenceTransformer model."""
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return util.cos_sim(vec1, vec2).item()


def generate_embedding(text: str):
    """Generate embedding for a given text."""
    model = get_model()
    return model.encode(text, convert_to_tensor=True)


def validate_user_input(user_input: str, current_sub_goal: str, threshold: float = 0.35) -> Dict[str, Any]:
    """
    Evaluate whether user_input is a reasonable attempt to engage with the lesson state
    or an explicit off-topic distraction using local embedding-based classification.
    
    Args:
        user_input: The user's input to evaluate
        current_sub_goal: The current sub-goal of the lesson
        threshold: Cosine similarity threshold for relevance (default: 0.35)
        
    Returns:
        Dict with 'is_relevant' (bool), 'score' (float), and 'rationale' (str)
    """
    # Generate embeddings
    user_input_embedding = generate_embedding(user_input)
    sub_goal_embedding = generate_embedding(current_sub_goal)
    
    # Compute cosine similarity
    similarity_score = cosine_similarity(user_input_embedding, sub_goal_embedding)
    
    # Determine relevance based on threshold
    is_relevant = similarity_score >= threshold
    
    # Generate rationale
    if is_relevant:
        rationale = f"User input is relevant to the lesson sub-goal (similarity score: {similarity_score:.3f} >= {threshold})."
    else:
        rationale = f"User input is off-topic or a distraction from the lesson sub-goal (similarity score: {similarity_score:.3f} < {threshold})."
    
    return {
        "is_relevant": is_relevant,
        "score": float(similarity_score),
        "rationale": rationale
    }
