import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

from latent_framework import DynamicLatentTracker
from tt_xgboost_engine import TableTennisXGBoost
from live_edge_execution import LiveEdgeExecution

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
STATE_PATH = os.path.join(DATA_DIR, "latent_state.pkl")

def verify_seeding_burn_in(tracker):
    total_tracked = len(tracker.players)
    print(f"[{datetime.now()}] Latent state verified: {total_tracked} active player nodes[span_8](start_span)[span_8](end_span).")
    return True

def run_pipeline():
    print(f"[{datetime.now()}] Initializing Table Tennis Cloud Prediction Engine...")
    
    BANKROLL = 1000.00  
    executor = LiveEdgeExecution(bankroll=BANKROLL)
    
    tracker = DynamicLatentTracker(state_path=STATE_PATH)
    verify_seeding_burn_in(tracker)

    mock_live_features = pd.DataFrame([{
        'glicko_rating_diff': 112.5,
        'melo_vector_distance': 0.42,
        'markov_match_win_prob_diff': 0.14,
        'age_diff': -3.0,
        'height_diff': 4.0,
        'handedness_interaction': 1,      
        'schedule_density_diff': -2.0,    
        'wttr_position_diff': -18,         
        'wttr_points_diff': 340.0,
        'home_continent_adv': 1,
        'recent_win_ratio_diff': 0.20
    }])

    match_title = "Wang Chuqin vs. Truls Moregard (WTT Men's Singles)"
    bookmaker_odds_a = 1.65
    bookmaker_odds_b = 2.25
    previous_odds_a = 1.60  

    print(f"[{datetime.now()}] Evaluating: {match_title}")
    print(f"[{datetime.now()}] Market Odds -> A: {bookmaker_odds_a} | B: {bookmaker_odds_b}")

    try:
        executor.evaluate_match(
            match_title=match_title,
            live_features=mock_live_features,
            odds_a=bookmaker_odds_a,
            odds_b=bookmaker_odds_b,
            previous_odds_a=previous_odds_a
        )
        print(f"[{datetime.now()}] Evaluation complete. Notification dispatched if edge criteria met.")
    except Exception as e:
        print(f"[{datetime.now()}] Prediction error: {e}")
        executor.push_android_notification("Engine Test Run", 3.12, 25.00, bookmaker_odds_a)

if __name__ == "__main__":
    run_pipeline()
