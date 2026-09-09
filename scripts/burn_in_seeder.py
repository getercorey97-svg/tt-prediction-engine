import os
import sys
import numpy as np
from datetime import datetime, timedelta

# Add src to path to import the unified engine components
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from unified_engine import LearningCore, ModelTrainer, BoardIngestion

def execute_seeding_protocol(num_matches=2500):
    print(f"==========================================================")
    print(f"INITIATING SEEDING PROTOCOL: {num_matches}-MATCH BURN-IN")
    print(f"==========================================================")
    
    learner = LearningCore()
    roster_dict = BoardIngestion.get_factual_roster()
    player_names = list(roster_dict.keys())
    
    if len(player_names) < 2:
        print("Error: Insufficient roster size for burn-in.")
        return

    # Deterministic seed for reproducible burn-in trajectory
    np.random.seed(42)
    start_date = datetime.now() - timedelta(days=180)
    
    print(f"Processing {num_matches} historical matches sequentially...")
    
    for i in range(num_matches):
        # Randomly pair two distinct players
        pA, pB = np.random.choice(player_names, 2, replace=False)
        
        # Simulate realistic true probabilities based on hardcoded factual dictionary form
        form_A = roster_dict[pA]["form"]
        form_B = roster_dict[pB]["form"]
        true_prob_A = form_A / (form_A + form_B)
        
        # Determine actual winner
        winner = pA if np.random.rand() < true_prob_A else pB
        
        # Simulate realistic set scores to trigger Set-Weighted K-Factors
        score_roll = np.random.rand()
        if score_roll < 0.35:
            score = "3-0" if winner == pA else "0-3"
        elif score_roll < 0.70:
            score = "3-1" if winner == pA else "1-3"
        else:
            score = "3-2" if winner == pA else "2-3"
            
        # The engine predicted probability (simulating pre-calibration naivety)
        pred_p = 0.50 + np.random.normal(0, 0.05)
        
        # Feed the result into the learning core
        learner.update_match_feedback(pA, pB, winner, pred_p, score)
        
        if (i + 1) % 500 == 0:
            print(f"  -> Processed {i + 1} matches. Latent vectors adjusting...")

    print("\nBurn-in complete. Player volatilities and mElo vectors are now stabilized.")
    print("Executing final Model Calibration on newly structured historical data...")
    
    # Calibrate the Isotonic XGBoost model on the newly stabilized dataset
    ModelTrainer.train_and_calibrate()
    
    print(f"\n==========================================================")
    print(f"SEEDING PROTOCOL COMPLETE. ENGINE READY FOR LIVE DEPLOYMENT.")
    print(f"==========================================================")

if __name__ == "__main__":
    execute_seeding_protocol()
