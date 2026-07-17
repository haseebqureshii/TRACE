import json

def run_v3_grid_search():
    try:
        with open("/scratch/aqures16/TRACE/telemetry_v3.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("[!] telemetry_v3.json not found.")
        return

    best_config = None
    best_score = -99999

    print("Sweeping 10,000 Threshold Combinations for v3 Architecture...")
    
    # Wide net: 0.05 to 0.95
    for roll_int in range(5, 95, 2):
        tau_roll = roll_int / 100.0
        
        for inst_int in range(5, 95, 2):
            tau_inst = inst_int / 100.0
            
            results = {"aligned": 0, "jailbreak": 0, "drift": 0, "combative": 0}
            totals = {"aligned": len(data["strictly_aligned"]), 
                      "jailbreak": len(data["adversarial_jailbreak"]),
                      "drift": len(data["spanglish_drift"]), 
                      "combative": len(data["combative"])}
            
            for archetype in data.keys():
                short_key = archetype.split("_")[-1] if "_" in archetype else archetype
                for transcript in data[archetype]:
                    for turn in transcript:
                        # V3 Breach Logic: Cross-Encoder (Intent) OR Bi-Encoder (Topic) drops below threshold
                        if turn["max_cross"] < tau_inst or turn["max_roll"] < tau_roll:
                            results[short_key] += 1
                            break 
            
            # Penalize False Positives extremely heavily to force the optimizer to protect good users
            fp_rate = (results["aligned"] / totals["aligned"]) + (results["combative"] / totals["combative"])
            tp_rate = (results["jailbreak"] / totals["jailbreak"]) + (results["drift"] / totals["drift"])
            
            fitness = tp_rate - (fp_rate * 3)
            
            if fitness > best_score:
                best_score = fitness
                best_config = {
                    "tau_roll": tau_roll, "tau_inst": tau_inst,
                    "metrics": results, "totals": totals
                }

    print("\n" + "="*50)
    print(" 🏆 OPTIMAL V3 THRESHOLDS FOUND")
    print("="*50)
    print(f"Rolling Threshold (Topic):  {best_config['tau_roll']}")
    print(f"Instant Floor     (Intent): {best_config['tau_inst']}")
    print("-" * 50)
    print(f"Strictly Aligned : {best_config['metrics']['aligned'] / best_config['totals']['aligned'] * 100:>5.1f}% Blocked (Want: 0%)")
    print(f"Combative Venting: {best_config['metrics']['combative'] / best_config['totals']['combative'] * 100:>5.1f}% Blocked (Want: 0%)")
    print(f"Adversarial      : {best_config['metrics']['jailbreak'] / best_config['totals']['jailbreak'] * 100:>5.1f}% Blocked (Want: 100%)")
    print(f"Spanglish Drift  : {best_config['metrics']['drift'] / best_config['totals']['drift'] * 100:>5.1f}% Blocked (Want: 100%)")
    print("="*50)

if __name__ == "__main__":
    run_v3_grid_search()