import os
import sys
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime

from latent_framework import DynamicLatentTracker, DTMC_Engine
from live_edge_execution import PureMathExecution

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
DB_PATH = os.path.join(DATA_DIR, "table_tennis_global.db")
STATE_PATH = os.path.join(DATA_DIR, "latent_state.pkl")

def run_pipeline():
    print(f"[{datetime.now()}] Initializing Pure Mathematical Prediction Engine...")
    
    executor = PureMathExecution()
    tracker = DynamicLatentTracker(state_path=STATE_PATH)
    
    total_tracked = len(tracker.players)
    print(f"[{datetime.now()}] Latent state verified: {total_tracked} active player nodes.")

    # Connect to the data lake to fetch current active matches on the board
    if not os.path.exists(DB_PATH):
        print(f"[{datetime.now()}] Database not found. Run the sync and seeding protocol first.")
        return

    conn = sqlite3.connect(DB_PATH)
    
    # Target all newly scraped matches that have not been processed into the historical burn-in yet
    query = "SELECT * FROM matches WHERE processed = 0"
    try:
        live_board = pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"[{datetime.now()}] Database query failed: {e}")
        conn.close()
        return
    
    if live_board.empty:
        print(f"[{datetime.now()}] No new active matches found on the board. Awaiting data sync.")
        # Diagnostic ping to ensure connection remains alive
        executor.push_android_notification("Diagnostic Sync Check", "Engine Online", 1.00)
        conn.close()
        return
        
    print(f"[{datetime.now()}] Found {len(live_board)} matches. Evaluating mathematical win probabilities...")
    
    for _, row in live_board.iterrows():
        pA = row['player_a_id']
        pB = row['player_b_id']
        match_title = f"{pA} vs. {pB}"
        
        # Ensure players are initialized in the tracker
        tracker._initialize_player(pA)
        tracker._initialize_player(pB)
        
        # Dynamically reconstruct the intrinsic math features[span_2](start_span)[span_2](end_span)
        g_diff = tracker.players[pA]['glicko_rating'] - tracker.players[pB]['glicko_rating']
        m_dist = float(np.linalg.norm(tracker.players[pA]['melo_vector'] - tracker.players[pB]['melo_vector']))
        
        # Reconstruct Markov state trajectories based on Glicko skill advantage[span_3](start_span)[span_3](end_span)
        p_serve_a = 0.55 if g_diff > 0 else 0.48
        p_receive_a = 0.50 if g_diff > 0 else 0.45
        markov_pA = DTMC_Engine.calculate_absorption_probability(p_serve_a, p_receive_a)
        markov_diff = markov_pA - (1.0 - markov_pA)
        
        mock_live_features = pd.DataFrame([{
            'glicko_rating_diff': g_diff,
            'melo_vector_distance': m_dist,
            'markov_match_win_prob_diff': markov_diff,
            'age_diff': row.get('age_diff', 0.0),
            'height_diff': row.get('height_diff', 0.0),
            'handedness_interaction': row.get('handedness_interaction', 0),      
            'schedule_density_diff': row.get('schedule_density_diff', 0.0),    
            'wttr_pos_diff': row.get('wttr_pos_diff', 0.0),         
            'wttr_points_diff': row.get('wttr_points_diff', 0.0),
            'home_continent_adv': row.get('home_continent_adv', 0),
            'recent_win_ratio_diff': row.get('recent_win_ratio_diff', 0.0)
        }])
        
        print(f"[{datetime.now()}] Evaluating: {match_title}")
        executor.evaluate_match_math(match_title, pA, pB, mock_live_features)
        
    conn.close()
    print(f"[{datetime.now()}] Pure math evaluation cycle complete.")

if __name__ == "__main__":
    run_pipeline()
