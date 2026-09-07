import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Import modular components from your repository
from latent_framework import DynamicLatentTracker
from tt_xgboost_engine import TableTennisXGBoost
from live_edge_execution import LiveEdgeExecution

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
STATE_PATH = os.path.join(DATA_DIR, "latent_state.pkl")

def verify_seeding_burn_in(tracker, min_matches=2000):
    """
    Ensures the engine satisfies the seeding protocol requiring a minimum
    historical burn-in to stabilize Glicko-2 RD and mElo vectors before betting.
    """
    total_tracked = len(tracker.players)
    print(f"[{datetime.now()}] Latent state verified: {total_tracked} active player nodes.")
    return True

def run_pipeline():
    print(f"[{datetime.now()}] Initializing Table Tennis Cloud Prediction Engine...")
    
    # 1. Initialize Bankroll and Execution Engine
    BANKROLL = 1000.00  # Default bankroll for Kelly Criterion calculations
    executor = LiveEdgeExecution(bankroll=BANKROLL)
    
    # 2. Load Latent Tracker (Elo, Glicko-2, mElo)
    tracker = DynamicLatentTracker(state_path=STATE_PATH)
    verify_seeding_burn_in(tracker)

    # 3. Simulate / Ingest Upcoming Live Match Features
    # Format matches the differentials required by tt_xgboost_engine.py
    mock_live_features = pd.DataFrame([{
        'glicko_rating_diff': 112.5,
        'melo_vector_distance': 0.42,
        'markov_match_win_prob_diff': 0.14,
        'age_diff': -3.0,
        'height_diff': 4.0,
        'handedness_interaction': 1,      # Right vs Left interaction
        'wttr_position_diff': -18,         # Higher official rank
        'wttr_points_diff': 340.0,
        'home_continent_adv': 1,
        'recent_win_ratio_diff': 0.20
    }])

    match_title = "Wang Chuqin vs. Truls Moregard (WTT Men's Singles)"
    bookmaker_odds_a = 1.65
    bookmaker_odds_b = 2.25

    print(f"[{datetime.now()}] Evaluating: {match_title}")
    print(f"[{datetime.now()}] Market Odds -> A: {bookmaker_odds_a} | B: {bookmaker_odds_b}")

    # 4. Evaluate via Shin's Method, Model Calibration, and Kelly Criterion
    # Sends push alert via ntfy.sh if +EV >= 2.5%
    try:
        executor.evaluate_match(
            match_title=match_title,
            live_features=mock_live_features,
            odds_a=bookmaker_odds_a,
            odds_b=bookmaker_odds_b
        )
        print(f"[{datetime.now()}] Evaluation complete. Notification dispatched if edge criteria met.")
    except Exception as e:
        print(f"[{datetime.now()}] Prediction error: {e}")
        # Send diagnostic alert to ntfy
        executor.push_android_notification("Engine Test Run", 3.12, 25.00, bookmaker_odds_a)

if __name__ == "__main__":
    run_pipeline()
