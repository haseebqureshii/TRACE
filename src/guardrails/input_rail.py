import os
import json
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer, util
from src.state import KBDocument, SessionState


# Initialize the SentenceTransformer model on module load
_model = None


def get_model():
    """Get or initialize the SentenceTransformer multilingual model."""
    global _model
    if _model is None:
        _model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    return _model


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return util.cos_sim(vec1, vec2).item()


def generate_embedding(text: str):
    """Generate embedding for a given text."""
    model = get_model()
    return model.encode(text, convert_to_tensor=True)


class KBRetriever:
    def embed_kb_documents(self, documents: List[KBDocument]) -> np.ndarray:
        """Embed KB documents using paraphrase-multilingual-MiniLM-L12-v2."""
        embeddings = []
        for doc in documents:
            embedding = generate_embedding(doc.content)
            embeddings.append(embedding)
        return np.array(embeddings)

    def evaluate_relevance_and_retrieve(self, contextualized_query: str, session_state: SessionState, threshold: float = 0.30, top_k: int = 2) -> Dict[str, Any]:
        """
        a. Embed contextualized_query.
        b. Compute cosine similarities against session_state.kb_embeddings.
        c. If max similarity < threshold: return {"is_relevant": False, "score": float, "contexts": []}.
        d. If max similarity >= threshold: return {"is_relevant": True, "score": float, "contexts": [top_k_matching_text_snippets]}.
        """
        if session_state.kb_embeddings is None or len(session_state.kb_embeddings) == 0:
            return {
                "is_relevant": False,
                "score": 0.0,
                "contexts": []
            }
        
        query_embedding = generate_embedding(contextualized_query)
        
        similarities = []
        for i, emb in enumerate(session_state.kb_embeddings):
            sim = cosine_similarity(query_embedding, emb)
            similarities.append((i, sim, session_state.kb_documents[i].content))
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        if not similarities:
            return {
                "is_relevant": False,
                "score": 0.0,
                "contexts": []
            }
        
        max_score = similarities[0][1]
        
        if max_score < threshold:
            return {
                "is_relevant": False,
                "score": float(max_score),
                "contexts": []
            }
        
        # Get top_k matching contexts
        contexts = []
        for i in range(min(top_k, len(similarities))):
            _, _, content = similarities[i]
            contexts.append(content)
        
        return {
            "is_relevant": True,
            "score": float(max_score),
            "contexts": contexts
        }


# Global retriever instance
_kb_retriever = None


def get_kb_retriever() -> KBRetriever:
    """Get or create the global KBRetriever instance."""
    global _kb_retriever
    if _kb_retriever is None:
        _kb_retriever = KBRetriever()
    return _kb_retriever
