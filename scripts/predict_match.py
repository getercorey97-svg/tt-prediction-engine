import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from unified_engine import LearningCore, CombinatorialEngine, DatabaseManager, MODEL_PATH

def predict_custom_match(pA, pB):
    print("==========================================================")
    print(f"🎯 ON-DEMAND MATCH PREDICTION: {pA} vs {pB}")
    print("==========================================================")
    
    DatabaseManager.initialize()
    if not os.path.exists(MODEL_PATH):
        print("[Error] Calibrated model artifact missing. Run burn-in or training first.")
        return

    model = joblib.load(MODEL_PATH)
    learning_core = LearningCore()
    combinatorial = CombinatorialEngine()
    
    pA_id = learning_core.register_player(pA)
    pB_id = learning_core.register_player(pB)
    dA = learning_core.state["players"][pA_id]
    dB = learning_core.state["players"][pB_id]

    # Combinatorial evaluation for exact set distributions & TDI momentum
    set_dist = combinatorial.evaluate_set_probabilities(dA["spw"], dB["spw"], 0, 0, 0.0)
    p_win_a_math = set_dist["3-0"] + set_dist["3-1"] + set_dist["3-2"]
    
    # 10-feature array alignment
    features = pd.DataFrame([{
        'glicko_rating_diff': float(dA["rating"] - dB["rating"]),
        'melo_vector_distance': float(np.linalg.norm(np.array(dA["melo_vector"]) - np.array(dB["melo_vector"]))),
        'markov_match_win_prob_diff': p_win_a_math - (1.0 - p_win_a_math),
        'age_diff': 0.0, 'height_diff': 0.0,
        'wttr_pos_diff': 0.0, 'wttr_points_diff': 0.0,
        'recent_win_ratio_diff': 0.0, 
        'spw_diff': dA["spw"] - dB["spw"], 
        'rpw_diff': dA["rpw"] - dB["rpw"]
    }])

    prob_a = float(model.predict_proba(features)[0, 1])
    winner = pA if prob_a >= 0.50 else pB
    confidence = prob_a if prob_a >= 0.50 else (1.0 - prob_a)
    
    dist_str = f"3-0: {set_dist['3-0']*100:.0f}% | 3-1: {set_dist['3-1']*100:.0f}% | 3-2: {set_dist['3-2']*100:.0f}%"
    
    print(f"Projected Winner : {winner}")
    print(f"Model Confidence : {confidence*100:.2f}%")
    print(f"Exact Score Dist : {dist_str}")
    print("==========================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--player-a", required=True, help="Player A Name")
    parser.add_argument("--player-b", required=True, help="Player B Name")
    args = parser.parse_args()
    
    predict_custom_match(args.player_a, args.player_b)
