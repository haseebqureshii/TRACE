import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from numpy.linalg import norm

class TraceEngine:
    """
    Project TRACE v4.1: The Production Stack (WordPiece Tokenizer)
    Gate 1: DistilBERT Injection Classifier (Zero-Dependency Intent)
    Gate 2: MiniLM Bi-Encoder EWMA (Topic Drift)
    """
    def __init__(
        self, 
        anchor_text: str, 
        tau_roll: float = 0.40,
        default_alpha: float = 0.35,
        bi_encoder_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        classifier_name: str = "fmops/distilbert-prompt-injection",
        cache_dir: str = "/scratch/aqures16/TRACE/model_cache"
    ):
        self.tau_roll = tau_roll
        self.default_alpha = default_alpha
        self.cache_dir = cache_dir
        self.anchor_text = anchor_text
        
        os.makedirs(self.cache_dir, exist_ok=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[TRACE] Initializing v4.1 Architecture on {self.device.upper()}...")
        
        # 1. Load Bi-Encoder (Gate 2: Fast Rolling Math)
        self.bi_encoder = SentenceTransformer(
            bi_encoder_name, 
            cache_folder=self.cache_dir,
            device=self.device
        )
        
        # 2. Load Dedicated Injection Classifier (Gate 1: Intent)
        self.intent_classifier = pipeline(
            "text-classification",
            model=classifier_name,
            model_kwargs={"cache_dir": self.cache_dir},
            device=0 if self.device == 'cuda' else -1
        )
        
        self.anchor_vector = self._embed(anchor_text)
        self.context_vector = np.copy(self.anchor_vector)
        self.turn_count = 0

    def _embed(self, text: str) -> np.ndarray:
        vector = self.bi_encoder.encode(text, convert_to_numpy=True)
        if norm(vector) == 0: return vector
        return vector / norm(vector)

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return float(np.dot(v1, v2) / (norm(v1) * norm(v2)))

    def process_turn(self, user_text: str, dynamic_alpha: float = None) -> dict:
        self.turn_count += 1
        alpha = dynamic_alpha if dynamic_alpha is not None else self.default_alpha
        
        # --- GATE 1: DistilBERT Injection Classification ---
        intent_result = self.intent_classifier(user_text, truncation=True, max_length=512)[0]
        label = str(intent_result['label']).upper()
        
        # Standard classifier models output 'INJECTION' or 'LABEL_1' for malicious payloads
        is_adversarial = ('INJECTION' in label or 'LABEL_1' in label) and intent_result['score'] > 0.5
        
        # --- GATE 2: Bi-Encoder Rolling Drift Check ---
        v_t = self._embed(user_text)
        self.context_vector = (alpha * v_t) + ((1 - alpha) * self.context_vector)
        roll_sim = self._cosine_similarity(self.context_vector, self.anchor_vector)
        
        is_breached = is_adversarial or (roll_sim < self.tau_roll)
        
        return {
            "turn": self.turn_count,
            "text": user_text,
            "is_injection": is_adversarial,
            "roll_sim": round(roll_sim, 4),
            "is_breached": is_breached,
            "applied_alpha": alpha
        }

    def reset_context(self):
        self.context_vector = np.copy(self.anchor_vector)
        self.turn_count = 0