import os
import json
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer, util


# Initialize the SentenceTransformer model on module load
_model = None
_kb_documents = None
_kb_embeddings = None


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


def load_kb_documents() -> List[Dict[str, Any]]:
    """Load KB documents from tests/fixtures/kb_documents.json."""
    global _kb_documents
    if _kb_documents is not None:
        return _kb_documents
    
    # Try to find the kb_documents.json file
    kb_paths = [
        os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'fixtures', 'kb_documents.json'),
        os.path.join(os.path.dirname(__file__), '..', 'tests', 'fixtures', 'kb_documents.json'),
        os.path.join('tests', 'fixtures', 'kb_documents.json'),
        os.path.join(os.getcwd(), 'tests', 'fixtures', 'kb_documents.json')
    ]
    
    kb_data = None
    for path in kb_paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                kb_data = json.load(f)
                break
    
    if kb_data is None or 'documents' not in kb_data:
        raise FileNotFoundError("Could not find kb_documents.json with 'documents' key")
    
    _kb_documents = kb_data['documents']
    return _kb_documents


def compute_kb_embeddings():
    """Compute vector embeddings for all KB documents on load."""
    global _kb_embeddings, _kb_documents
    if _kb_embeddings is not None:
        return _kb_embeddings
    
    documents = load_kb_documents()
    embeddings = []
    for doc in documents:
        embedding = generate_embedding(doc['content'])
        embeddings.append(embedding)
    
    _kb_embeddings = embeddings
    return _kb_embeddings


class KBRetriever:
    def __init__(self):
        self.documents = load_kb_documents()
        self.embeddings = compute_kb_embeddings()

    def evaluate_relevance_and_retrieve(self, contextualized_query: str, threshold: float = 0.30, top_k: int = 2) -> Dict[str, Any]:
        """
        a. Embed contextualized_query.
        b. Compute cosine similarities against KB document embeddings.
        c. If max similarity < threshold: return {"is_relevant": False, "score": float, "contexts": []}.
        d. If max similarity >= threshold: return {"is_relevant": True, "score": float, "contexts": [top_k_matching_text_snippets]}.
        """
        query_embedding = generate_embedding(contextualized_query)
        
        similarities = []
        for i, emb in enumerate(self.embeddings):
            sim = cosine_similarity(query_embedding, emb)
            similarities.append((i, sim, self.documents[i]['content']))
        
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