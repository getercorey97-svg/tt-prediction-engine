import os
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
    
    if not os.path.exists(DB_PATH):
        return

    conn = sqlite3.connect(DB_PATH)
    live_board = pd.read_sql_query("SELECT * FROM matches WHERE processed = 0", conn)
    
    if live_board.empty:
        conn.close()
        return
        
    for _, row in live_board.iterrows():
        pA = row['player_a_id']
        pB = row['player_b_id']
        match_title = f"{pA} vs. {pB}"
        
        tracker._initialize_player(pA)
        tracker._initialize_player(pB)
        
        # Cold Start Detection: Flags unseeded players resting at default
        is_cold_start = (tracker.players[pA]['glicko_rating'] == 1500.0 or tracker.players[pB]['glicko_rating'] == 1500.0)
        
        g_diff = tracker.players[pA]['glicko_rating'] - tracker.players[pB]['glicko_rating']
        
        # Calculates mElo distance to assess stylistic vectors[span_3](start_span)[span_3](end_span)
        m_dist = float(np.linalg.norm(tracker.players[pA]['melo_vector'] - tracker.players[pB]['melo_vector']))
        points_diff = row.get('wttr_points_diff', 0.0)
        
        # Override baseline metrics if players are uncalibrated
        if is_cold_start:
            p_serve_a = 0.55 if points_diff > 1000 else (0.52 if points_diff > 200 else (0.48 if points_diff < -200 else 0.50))
            p_receive_a = 0.50 if points_diff > 1000 else (0.48 if points_diff > 200 else (0.45 if points_diff < -200 else 0.47))
            proxy_glicko_diff = points_diff / 10.0
        else:
            p_serve_a = 0.55 if g_diff > 0 else 0.48
            p_receive_a = 0.50 if g_diff > 0 else 0.45
            proxy_glicko_diff = g_diff
            
        # Computes exact probability of reaching an absorbing set-win state[span_4](start_span)[span_4](end_span)
        markov_pA = DTMC_Engine.calculate_absorption_probability(p_serve_a, p_receive_a)
        markov_diff = markov_pA - (1.0 - markov_pA)
        
        mock_live_features = pd.DataFrame([{
            'glicko_rating_diff': proxy_glicko_diff,
            'melo_vector_distance': m_dist,
            'markov_match_win_prob_diff': markov_diff,
            'age_diff': row.get('age_diff', 0.0),
            'height_diff': row.get('height_diff', 0.0),
            'handedness_interaction': row.get('handedness_interaction', 0),      
            'schedule_density_diff': row.get('schedule_density_diff', 0.0),    
            'wttr_pos_diff': row.get('wttr_pos_diff', 0.0),         
            'wttr_points_diff': points_diff,
            'home_continent_adv': row.get('home_continent_adv', 0),
            'recent_win_ratio_diff': row.get('recent_win_ratio_diff', 0.0)
        }])
        
        executor.evaluate_match_math(match_title, pA, pB, mock_live_features)
        
        # Mark as processed to prevent redundant alerts
        cursor = conn.cursor()
        cursor.execute("UPDATE matches SET processed = 1 WHERE match_id = ?", (row['match_id'],))
        
    conn.commit()
    conn.close()

if __name__ == "__main__":
    run_pipeline()
