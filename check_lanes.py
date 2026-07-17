import json
from trace_engine import TraceEngine

anchors = [
    "Discussing travel plans, vacation dates, and booking a hotel room.",
    "Inquiring about hotel amenities, room features, pool, wifi, and gym.",
    "Asking about room pricing, rates, billing, fees, and cancellation policies.",
    "Customer support, modifying reservations, or transferring to a live agent."
]

engines = [TraceEngine(anchor_text=a, tau=0.60, default_alpha=0.35) for a in anchors]

# Test turn that should heavily favor the "Amenities" lane over "Billing"
sample_turn = "Does the hotel room have high-speed wifi and is the swimming pool open late?"

print("--- Testing Lane Independence ---")
for i, engine in enumerate(engines):
    # Process turn and check internal state if your engine exposes it
    res = engine.process_turn(sample_turn, dynamic_alpha=0.2)
    # Adjust the dictionary key below if your engine returns the score under a different name
    score = res.get("similarity", "N/A") 
    print(f"Lane {i} ({anchors[i][:20]}...): Similarity Score = {score} | Breached = {res['is_breached']}")