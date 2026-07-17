import os
import json
from openai import OpenAI

# Initialize the client pointing to the ASU Research Computing endpoint
# Ensure you have run: export VOYAGER_API_KEY="your_key" in your terminal
client = OpenAI(
    base_url="https://openai.rc.asu.edu/v1",
    api_key=os.environ.get("VOYAGER_API_KEY", "MISSING_KEY")
)

# You can check Voyager for available models, using Llama-3 as a strong default
MODEL_NAME = "llama4-scout-17b" 

def generate_transcripts(archetype: str, instructions: str, count: int = 5) -> list:
    print(f"Generating {count} '{archetype}' transcripts...")
    
    system_prompt = f"""
    You are an AI generating synthetic test data for a vector-drift algorithm.
    The lesson anchor is: "Discussing travel plans and booking a hotel."
    Generate exactly {count} distinct conversations. Each conversation must be a JSON array of 5 strings (user turns).
    
    ARCHETYPE INSTRUCTION:
    {instructions}
    
    Output strictly as a raw JSON list of lists of strings. No markdown formatting, no explanations.
    Example format: [["turn1", "turn2", "turn3", "turn4", "turn5"], ["turn1", ...]]
    """

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": system_prompt}],
        temperature=0.8
    )
    
    try:
        # Parse the raw text into a Python list
        raw_text = response.choices[0].message.content.strip()
        # Clean up any potential markdown code blocks the LLM might stubbornly include
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:-3]
        return json.loads(raw_text)
    except Exception as e:
        print(f"Failed to parse JSON for {archetype}: {e}")
        return []

if __name__ == "__main__":
    if os.environ.get("VOYAGER_API_KEY") is None:
        print("ERROR: Please set your VOYAGER_API_KEY environment variable.")
        exit(1)

    dataset = {
        "strictly_aligned": generate_transcripts(
            "strictly_aligned", 
            "The user must stay entirely on topic for all 5 turns. Ask about rooms, prices, amenities, dates."
        ),
        "sharp_injection": generate_transcripts(
            "sharp_injection", 
            "Turns 1 and 2 are on topic. Turn 3 immediately pivots to a completely unrelated and jarring topic (e.g., politics, crypto, sports, cooking). Turns 4 and 5 continue the unrelated topic."
        ),
        "gradual_drift": generate_transcripts(
            "gradual_drift", 
            "Turn 1 is booking a hotel. Turn 2 mentions the city the hotel is in. Turn 3 talks about the weather there. Turn 4 talks about a movie that took place in that weather. Turn 5 is purely about the actors in that movie."
        )
    }

    # Save to disk in your scratch directory
    output_path = "/scratch/aqures16/TRACE/evaluation_dataset.json"
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=4)
        
    print(f"\nDataset successfully generated and saved to {output_path}")