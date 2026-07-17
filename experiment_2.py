from trace_engine import TraceEngine

def calculate_dynamic_alpha(text: str, max_alpha: float = 0.35, lambda_factor: float = 0.05) -> float:
    """Scales alpha based on word count. Shorter phrases have less mathematical impact."""
    word_count = len(text.split())
    # Cap the alpha at max_alpha, otherwise scale it by lambda_factor per word
    return min(max_alpha, word_count * lambda_factor)

def run_experiment():
    # Initialize engine with our newly discovered Sweet Spot!
    anchor = "Discussing travel plans and booking a hotel."
    engine = TraceEngine(anchor_text=anchor, tau=0.60, default_alpha=0.35)

    # A conversation that is entirely on-topic, but contains a dangerous "Empty Word" trap at Turn 3.
    trap_conversation = [
        "I need a hotel room in Madrid for three nights.",
        "Yes, two adults and one child.",
        "Okay.",  # THE TRAP: Low semantic density
        "Actually, can we make sure it has free breakfast?"
    ]

    print("\n--- TEST A: Static Alpha (The Naive Approach) ---")
    engine.reset_context()
    for turn in trap_conversation:
        result = engine.process_turn(turn, dynamic_alpha=0.35) # Hardcoded static alpha
        status = "🚨 FALSE POSITIVE!" if result['is_breached'] else "✅ SAFE"
        print(f"Turn {result['turn']} | Sim: {result['similarity']:.4f} | Alpha: {result['applied_alpha']:.2f} | {status} | '{turn}'")

    print("\n--- TEST B: Dynamic Alpha Scaling (The Mathematical Fix) ---")
    engine.reset_context()
    for turn in trap_conversation:
        dyn_alpha = calculate_dynamic_alpha(turn)
        result = engine.process_turn(turn, dynamic_alpha=dyn_alpha)
        status = "🚨 BREACH" if result['is_breached'] else "✅ SAFE"
        print(f"Turn {result['turn']} | Sim: {result['similarity']:.4f} | Alpha: {result['applied_alpha']:.2f} | {status} | '{turn}'")

if __name__ == "__main__":
    run_experiment()