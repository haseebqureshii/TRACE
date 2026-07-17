import json
from trace_engine import TraceEngine

def run_experiment():
    dataset_path = "/scratch/aqures16/TRACE/evaluation_dataset.json"
    
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    # Initialize engine with our standard anchor
    anchor = "Discussing travel plans and booking a hotel."
    engine = TraceEngine(anchor_text=anchor)

    # Dictionary to store the minimum similarity score reached in each transcript
    results = {
        "strictly_aligned": [],
        "sharp_injection": [],
        "gradual_drift": []
    }

    print("\n[1/2] Processing Vector Embeddings (This may take a moment)...")
    for archetype, transcripts in dataset.items():
        for i, transcript in enumerate(transcripts):
            engine.reset_context()
            min_sim = 1.0
            
            for turn in transcript:
                # We only need the similarity score from the engine
                step_result = engine.process_turn(turn)
                if step_result["similarity"] < min_sim:
                    min_sim = step_result["similarity"]
                    
            results[archetype].append(min_sim)
            print(f"  Processed {archetype} [{i+1}/{len(transcripts)}] -> Lowest Sim: {min_sim:.4f}")

    print("\n[2/2] Sweeping Thresholds (Tau)...")
    print("-" * 65)
    print(f"{'Tau':<6} | {'False Positives (Aligned)':<25} | {'True Positives (Drift/Inject)':<25}")
    print("-" * 65)

    # Test a range of thresholds from 0.40 to 0.85
    thresholds = [x / 100.0 for x in range(40, 90, 5)]
    
    for tau in thresholds:
        # False Positives: Aligned conversations that triggered a penalty
        fp_count = sum(1 for sim in results["strictly_aligned"] if sim < tau)
        fp_rate = (fp_count / len(results["strictly_aligned"])) * 100

        # True Positives: Drifts or Injections that successfully triggered a penalty
        drift_count = sum(1 for sim in results["gradual_drift"] if sim < tau)
        inject_count = sum(1 for sim in results["sharp_injection"] if sim < tau)
        total_tp = drift_count + inject_count
        tp_rate = (total_tp / (len(results["gradual_drift"]) + len(results["sharp_injection"]))) * 100

        # Highlight the "Sweet Spot" (High TP, Low FP)
        marker = "⭐" if (tp_rate == 100.0 and fp_rate == 0.0) else "  "
        
        print(f"{tau:.2f} {marker} | {fp_rate:>5.1f}% ({fp_count}/{len(results['strictly_aligned'])})            | {tp_rate:>5.1f}% ({total_tp}/10)")

if __name__ == "__main__":
    run_experiment()