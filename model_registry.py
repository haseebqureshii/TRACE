import os
import requests

class ModelRegistry:
    """
    Dynamically fetches and validates available LLMs from the ASU Voyager API.
    """
    def __init__(self, base_url: str = "https://openai.rc.asu.edu/v1"):
        self.base_url = base_url
        self.api_key = os.environ.get("VOYAGER_API_KEY")
        
        if not self.api_key:
            raise ValueError("[!] VOYAGER_API_KEY environment variable is not set.")
            
        self.available_models = self._fetch_available_models()

    def _fetch_available_models(self) -> list:
        """Hits the /models endpoint and extracts the model IDs."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(f"{self.base_url}/models", headers=headers, timeout=10)
            response.raise_for_status()
            
            # The OpenAI spec returns data in a 'data' array, where each item has an 'id'
            data = response.json()
            models = [model["id"] for model in data.get("data", [])]
            return models
        except requests.exceptions.RequestException as e:
            print(f"[!] Failed to fetch models from ASU endpoint: {e}")
            return []

    def get_models(self) -> list:
        """Returns the list of currently available models."""
        return self.available_models

    def validate_model(self, model_name: str, fallback: str = "llama4-scout-17b") -> str:
        """
        Ensures the requested model is authorized by the key.
        If not, alerts the user and defaults to a known safe fallback.
        """
        if model_name in self.available_models:
            return model_name
            
        print(f"[WARNING] Model '{model_name}' is not authorized or available.")
        print(f"[WARNING] Defaulting to fallback model: '{fallback}'")
        return fallback

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    registry = ModelRegistry()
    
    print("--- Available ASU Models ---")
    for m in registry.get_models():
        print(f" - {m}")
        
    print("\n--- Validation Test ---")
    # Simulating a user trying to select an invalid model
    selected = registry.validate_model("gpt-99-turbo") 
    print(f"Final Selected Model: {selected}")