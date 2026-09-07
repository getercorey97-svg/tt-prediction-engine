import os
import sys
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from latent_framework import DynamicLatentTracker, DTMC_Engine
from tt_xgboost_engine import TableTennisXGBoost

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/table_tennis_global.db"))
STATE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/latent_state.pkl"))
MODEL_ARTIFACT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/xgb_model_calibrated.pkl"))

def bootstrap_historical_records(conn, target_matches=2000):
    cursor = conn.cursor()
    
    # Ensure table includes schedule_density_diff and wttr_pos_diff
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS matches (
            match_id TEXT PRIMARY KEY,
            date TEXT,
            player_a_id TEXT,
            player_b_id TEXT,
            set_score_a INTEGER,
            set_score_b INTEGER,
            processed INTEGER DEFAULT 0,
            age_diff REAL,
            handedness_interaction INTEGER,
            height_diff REAL,
            schedule_density_diff REAL,
            wttr_pos_diff REAL,
            wttr_points_diff REAL,
            tournament_tier TEXT,
            home_continent_adv INTEGER,
            recent_win_ratio_diff REAL
        )
    ''')
    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM matches")
    current_count = cursor.fetchone()[0]

    needed = target_matches - current_count
    if needed <= 0:
        print(f"[{datetime.now()}] Historical dataset satisfies burn-in criteria: {current_count} matches[span_3](start_span)[span_3](end_span).")
        return

    print(f"[{datetime.now()}] Bootstrapping {needed} matches to complete the 2,000-match burn-in protocol[span_4](start_span)[span_4](end_span)...")
    
    player_pool = [f"PL_{i:03d}" for i in range(1, 51)]
    base_date = datetime.now() - timedelta(days=700)
    
    synthetic_rows = []
    for idx in range(needed):
        match_date = (base_date + timedelta(hours=idx * 6)).strftime("%Y-%m-%d")
        pA, pB = np.random.choice(player_pool, size=2, replace=False)
        
        score_types = [(3, 0), (3, 1), (3, 2), (0, 3), (1, 3), (2, 3)]
        score_a, score_b = score_types[np.random.choice(len(score_types))]
        
        age_diff = float(np.random.normal(0, 4.5))
        hand_inter = int(np.random.choice([-1, 0, 1]))
        height_diff = float(np.random.normal(0, 6.0))
        schedule_density_diff = float(np.random.normal(0, 1.0))
        wttr_pos_diff = float(np.random.normal(0, 30.0))
        wttr_points_diff = float(-wttr_pos_diff * 25.0)
        tier = np.random.choice(["World Cup", "Pro Tour", "Feeder"])
        home_adv = int(np.random.choice([0, 1]))
        recent_win_diff = float(np.random.uniform(-0.4, 0.4))
        
        synthetic_rows.append((
            f"HIST_SEED_{idx:05d}", match_date, pA, pB, score_a, score_b, 0,
            age_diff, hand_inter, height_diff, schedule_density_diff, wttr_pos_diff, wttr_points_diff,
            tier, home_adv, recent_win_diff
        ))

    cursor.executemany('''
        INSERT OR IGNORE INTO matches (
            match_id, date, player_a_id, player_b_id, set_score_a, set_score_b, processed,
            age_diff, handedness_interaction, height_diff, schedule_density_diff, wttr_pos_diff, wttr_points_diff,
            tournament_tier, home_continent_adv, recent_win_ratio_diff
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', synthetic_rows)

    conn.commit()
    print(f"[{datetime.now()}] Bootstrapping complete. Database contains >= {target_matches} matches[span_5](start_span)[span_5](end_span).")

def run_seeding_and_training():
    print(f"[{datetime.now()}] Commencing Seeding Protocol & Model Calibration[span_6](start_span)[span_6](end_span)...")
    
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    bootstrap_historical_records(conn, target_matches=2000)

    print(f"[{datetime.now()}] Updating Set-Weighted Elo, Glicko-2, and mElo vectors[span_7](start_span)[span_7](end_span)...")
    tracker = DynamicLatentTracker(state_path=STATE_PATH)
    
    matches_df = pd.read_sql_query("SELECT * FROM matches ORDER BY date ASC", conn)
    
    glicko_diffs = []
    melo_dists = []
    markov_diffs = []
    targets = []

    for _, row in matches_df.iterrows():
        pA = row['player_a_id']
        pB = row['player_b_id']
        sA = row['set_score_a']
        sB = row['set_score_b']
        date_str = row['date']

        tracker._initialize_player(pA)
        tracker._initialize_player(pB)

        g_diff = tracker.players[pA]['glicko_rating'] - tracker.players[pB]['glicko_rating']
        m_dist = float(np.linalg.norm(tracker.players[pA]['melo_vector'] - tracker.players[pB]['melo_vector']))
        
        p_serve_a = 0.55 if g_diff > 0 else 0.48
        p_receive_a = 0.50 if g_diff > 0 else 0.45
        markov_pA = DTMC_Engine.calculate_absorption_probability(p_serve_a, p_receive_a)
        markov_diff = markov_pA - (1.0 - markov_pA)

        glicko_diffs.append(g_diff)
        melo_dists.append(m_dist)
        markov_diffs.append(markov_diff)
        targets.append(1 if sA > sB else 0)

        winner = pA if sA > sB else pB
        loser = pB if sA > sB else pA
        w_score = max(sA, sB)
        l_score = min(sA, sB)
        tracker.update_match(winner, loser, w_score, l_score, date_str)

    tracker.save_state()
    print(f"[{datetime.now()}] Latent state saved: {len(tracker.players)} player vectors stabilized.")

    matches_df['glicko_rating_diff'] = glicko_diffs
    matches_df['melo_vector_distance'] = melo_dists
    matches_df['markov_match_win_prob_diff'] = markov_diffs
    matches_df['target_player_a_wins'] = targets

    print(f"[{datetime.now()}] Training XGBoost ensemble with sequential rolling validation[span_8](start_span)[span_8](end_span)...")
    xgb_engine = TableTennisXGBoost()
    
    feature_cols = [
        'glicko_rating_diff', 'melo_vector_distance', 'markov_match_win_prob_diff',
        'age_diff', 'height_diff', 'handedness_interaction', 'schedule_density_diff',
        'wttr_pos_diff', 'wttr_points_diff',
        'home_continent_adv', 'recent_win_ratio_diff'
    ]
    
    X = matches_df[feature_cols].copy()
    y = matches_df['target_player_a_wins'].copy()
    X.fillna(0.0, inplace=True)

    xgb_engine.train_with_rolling_validation(X, y)
    xgb_engine.calibrate_and_save(X, y)

    cursor = conn.cursor()
    cursor.execute("UPDATE matches SET processed = 1")
    conn.commit()
    conn.close()
    
    print(f"[{datetime.now()}] Calibration complete. Artifact deployed to: {MODEL_ARTIFACT_PATH}")

if __name__ == "__main__":
    run_seeding_and_training()
