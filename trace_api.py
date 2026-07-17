import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from trace_engine import TraceEngine

app = FastAPI(title="TRACE v4.1 Enterprise Gateway")

class ChatRequest(BaseModel):
    message: str

# Notice we updated the 5th anchor to use first-person emotional language
# so it embeds much closer to actual angry customer prompts.
anchors = [
    "Discussing travel plans, vacation dates, and booking a hotel room.",
    "Inquiring about hotel amenities, room features, pool, wifi, and gym.",
    "Asking about room pricing, rates, billing, fees, and cancellation policies.",
    "Customer support, modifying reservations, or transferring to a live agent.",
    "I am incredibly frustrated, angry, and disappointed with this terrible customer service."
]

print("[API] Initializing TRACE v4.1 Engine Matrix (Loading models into memory)...")
# We bump tau_roll slightly to 0.45 to catch that last 10% of drift
lane_engines = [TraceEngine(anchor_text=a, tau_roll=0.45, default_alpha=0.35) for a in anchors]

def calculate_dynamic_alpha(text: str, max_alpha: float = 0.35, lambda_factor: float = 0.05) -> float:
    return min(max_alpha, len(text.split()) * lambda_factor)

@app.post("/")
async def evaluate_turn(request: ChatRequest):
    turn = str(request.message)
    dyn_alpha = calculate_dynamic_alpha(turn)
    
    # Process turn across all 5 parallel lanes
    lane_results = [engine.process_turn(turn, dynamic_alpha=dyn_alpha) for engine in lane_engines]
    
    # Check Gate 1: Did the DistilBERT model catch a Jailbreak/Injection?
    is_injection = any(res["is_injection"] for res in lane_results)
    
    # Check Gate 2: Did the topic drift too far from ALL allowed lanes?
    is_total_drift = all(res["roll_sim"] < engine.tau_roll for res, engine in zip(lane_results, lane_engines))
    
    # Evaluate the Combative Lane (Index 4)
    # If it's not a total drift, but the highest matching lane is the angry lane, flag it for routing
    highest_sim_index = max(range(len(lane_results)), key=lambda i: lane_results[i]["roll_sim"])
    is_combative = (highest_sim_index == 4)
    
    # 3. Sim Studio Routing Logic
    if is_injection or is_total_drift:
        classification = "breach"
    elif is_combative:
        classification = "combative"
    else:
        classification = "safe"
        
    return {
        "classification": classification,
        "debug": {
            "is_injection": is_injection,
            "highest_lane": highest_sim_index,
            "scores": [res["roll_sim"] for res in lane_results]
        }
    }

if __name__ == "__main__":
    print("[API] TRACE Gateway running on port 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)