import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from trace_engine import TraceEngine

app = FastAPI(title="TRACE Enterprise Gateway")

# Data model matching what Sim Studio POSTs
class ChatRequest(BaseModel):
    message: str

# 1. Define the 5-Lane Matrix
anchors = [
    "Discussing travel plans, vacation dates, and booking a hotel room.",
    "Inquiring about hotel amenities, room features, pool, wifi, and gym.",
    "Asking about room pricing, rates, billing, fees, and cancellation policies.",
    "Customer support, modifying reservations, or transferring to a live agent.",
    "Expressing customer frustration, anger, complaints, or dissatisfaction with the service."
]

print("[API] Initializing TRACE Engine Matrix (Loading models into memory)...")
# Initializes exactly once at startup using your current min-max calibration settings
lane_engines = [TraceEngine(anchor_text=a, tau=0.60, default_alpha=0.35, v_min=0.95) for a in anchors]

def calculate_dynamic_alpha(text: str, max_alpha: float = 0.35, lambda_factor: float = 0.05) -> float:
    return min(max_alpha, len(text.split()) * lambda_factor)

@app.post("/")
async def evaluate_turn(request: ChatRequest):
    turn = str(request.message)
    dyn_alpha = calculate_dynamic_alpha(turn)
    
    # Process turn across all 5 parallel lanes
    lane_results = [engine.process_turn(turn, dynamic_alpha=dyn_alpha) for engine in lane_engines]
    
    # 2. Routing Logic
    # Lane 4 (index 4) is the Combative/Venting lane
    is_combative = not lane_results[4]["is_breached"]
    
    # If it BREACHED ALL 5 lanes, it has left the entire allowed state space
    is_total_breach = all(res["is_breached"] for res in lane_results)
    
    if is_total_breach:
        classification = "breach"
    elif is_combative:
        classification = "combative"
    else:
        classification = "safe"
        
    return {
        "classification": classification,
        "debug_scores": [res["similarity"] for res in lane_results]
    }

if __name__ == "__main__":
    print("[API] TRACE Gateway running on port 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)