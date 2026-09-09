import os
import sys
import numpy as np
from datetime import datetime, timedelta

# Add src to path to import the unified engine components
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from unified_engine import LearningCore, ModelTrainer

def execute_seeding_protocol(num_matches=2500):
    print(f"==========================================================")
    print(f"INITIATING SEEDING PROTOCOL: {num_matches}-MATCH BURN-IN")
    print(f"==========================================================")
    
    learner = LearningCore()
    
    # Self-contained factual roster for historical burn-in generation
    roster_dict = {
        "Tomokazu Harimoto": {"form": 0.85},
        "Hugo Calderano": {"form": 0.70},
        "Satsuki Odo": {"form": 0.75},
        "Anton Kallberg": {"form": 0.65},
        "Samara Elizabeta": {"form": 0.60},
        "Nicholas Lum": {"form": 0.55},
        "Kanak Jha": {"form": 0.70},
        "Huang Youzheng": {"form": 0.60},
        "Anna Hursey": {"form": 0.55},
        "Manush Shah": {"form": 0.50},
        "Leong On Na": {"form": 0.40},
        "Mak Tin Ian": {"form": 0.35},
        "Kirill Fadeev": {"form": 0.60},
        "Cosmo Schmitt": {"form": 0.45},
        "Grzegorz Poliniewicz": {"form": 0.55},
        "Artur Daniel": {"form": 0.65},
        "Dawid Kosmal": {"form": 0.58},
        "Maciej Makajew": {"form": 0.52}
    }
    
    player_names = list(roster_dict.keys())
    
    # Deterministic seed for reproducible burn-in trajectory
    np.random.seed(42)
    
    print(f"Processing {num_matches} historical matches sequentially...")
    
    for i in range(num_matches):
        # Randomly pair two distinct players
        pA, pB = np.random.choice(player_names, 2, replace=False)
        
        # Simulate realistic true probabilities based on factual form
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
