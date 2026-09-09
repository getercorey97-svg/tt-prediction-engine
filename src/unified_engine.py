import os
import sqlite3
import pickle
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests
import joblib
import warnings

import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from scipy.stats import nbinom

warnings.filterwarnings('ignore')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "table_tennis_global.db")
MODEL_PATH = os.path.join(DATA_DIR, "xgb_model_calibrated.pkl")
STATE_PATH = os.path.join(DATA_DIR, "latent_state.pkl")
os.makedirs(DATA_DIR, exist_ok=True)

# =====================================================================
# 1. DATABASE SCHEMA & AUTO-MIGRATION
# =====================================================================
class DatabaseManager:
    @staticmethod
    def get_connection():
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def initialize():
        with DatabaseManager.get_connection() as conn:
            c = conn.cursor()
            c.execute('''
                CREATE TABLE IF NOT EXISTS matches (
                    match_id TEXT PRIMARY KEY, date TEXT, player_a_id TEXT, player_b_id TEXT,
                    set_score_a INTEGER DEFAULT 0, set_score_b INTEGER DEFAULT 0,
                    is_fanduel INTEGER DEFAULT 0, is_live INTEGER DEFAULT 0, processed INTEGER DEFAULT 0,
                    predicted_prob_a REAL, actual_winner TEXT, final_set_score TEXT,
                    closing_odds_a REAL, closing_odds_b REAL, brier_error REAL,
                    age_diff REAL, height_diff REAL, wttr_pos_diff REAL, wttr_points_diff REAL,
                    recent_win_ratio_diff REAL, spw_diff REAL, rpw_diff REAL
                )
            ''')
            conn.commit()

# =====================================================================
# 2. COMBINATORIAL EVALUATION & TDI MOMENTUM
# =====================================================================
class CombinatorialEngine:
    def __init__(self):
        self.Omega = np.array([[0.0, 1.0], [-1.0, 0.0]])

    def calculate_stylistic_advantage(self, vector_a, vector_b):
        va, vb = np.array(vector_a[:2]), np.array(vector_b[:2])
        return float(va.T @ self.Omega @ vb)
        
    def evaluate_set_probabilities(self, p_serve_a, p_serve_b, sets_a_won=0, sets_b_won=0, tdi_momentum=0.0):
        """
        Evaluates the final score mathematically without recursive stepping[span_6](start_span)[span_6](end_span).
        Uses Negative Binomial approximations for set absorption and applies TDI momentum[span_7](start_span)[span_7](end_span).
        """
        # Apply Trend Direction Index (TDI) to service parameters
        pA = np.clip(p_serve_a + (tdi_momentum * 0.05), 0.1, 0.9)
        pB = np.clip(p_serve_b - (tdi_momentum * 0.05), 0.1, 0.9)
        
        # Combinatorial approximation of winning a single 11-point set
        prob_set_a = pA * (1 - pB) / (pA * (1 - pB) + pB * (1 - pA) + 0.01)
        prob_set_a = np.clip(prob_set_a, 0.05, 0.95)
        prob_set_b = 1.0 - prob_set_a

        # Calculate remaining sets needed (Best of 5)
        req_a = 3 - sets_a_won
        req_b = 3 - sets_b_won

        dist = {"3-0": 0.0, "3-1": 0.0, "3-2": 0.0, "0-3": 0.0, "1-3": 0.0, "2-3": 0.0}
        
        if req_a <= 0: return {k: 1.0 if "3-" in k else 0.0 for k in dist}
        if req_b <= 0: return {k: 1.0 if "-3" in k else 0.0 for k in dist}

        # Exact combinatorial set distributions
        for sets_lost in range(0, req_b):
            prob = nbinom.pmf(sets_lost, req_a, prob_set_a)
            final_a = sets_a_won + req_a
            final_b = sets_b_won + sets_lost
            dist[f"{final_a}-{final_b}"] = prob
            
        for sets_lost in range(0, req_a):
            prob = nbinom.pmf(sets_lost, req_b, prob_set_b)
            final_b = sets_b_won + req_b
            final_a = sets_a_won + sets_lost
            dist[f"{final_a}-{final_b}"] = prob
            
        total = sum(dist.values())
        return {k: v/total for k, v in dist.items()}

# =====================================================================
# 3. SET-WEIGHTED LEARNING CORE & ROOKIE ENCODING
# =====================================================================
class LearningCore:
    def __init__(self):
        self.engine = CombinatorialEngine()
        self.state = self.load_state()

    def load_state(self):
        default = {"players": {}, "global_brier_history": []}
        if os.path.exists(STATE_PATH):
            try:
                with open(STATE_PATH, "rb") as f:
                    s = pickle.load(f)
                    if isinstance(s, dict) and "players" in s: return s
            except: pass
        return default

    def save_state(self):
        with open(STATE_PATH, "wb") as f:
            pickle.dump(self.state, f)

    def register_player(self, player_id):
        """Encodes unknown newcomers as 999999 to prevent systemic errors[span_8](start_span)[span_8](end_span)."""
        is_rookie = False if len(str(player_id)) > 2 else True
        pid = "999999" if is_rookie else player_id
        
        if pid not in self.state["players"]:
            self.state["players"][pid] = {
                "rating": 1500.0, "rd": 350.0, "volatility": 0.06,
                "melo_vector": np.random.normal(0, 0.1, 2).tolist(),
                "spw": 0.50, "rpw": 0.50, "matches_played": 0
            }
        return pid

    def update_match_feedback(self, pA, pB, winner, pred_p, final_score):
        pA_id = self.register_player(pA)
        pB_id = self.register_player(pB)
        
        y_actual = 1.0 if winner == pA else 0.0
        brier = (pred_p - y_actual) ** 2
        self.state["global_brier_history"].append(brier)

        # Set-Weighted K-Factor Adjustment[span_9](start_span)[span_9](end_span)
        if final_score in ["3-0", "0-3"]:
            k_base = 40.0
        elif final_score in ["3-1", "1-3"]:
            k_base = 30.0
        else:
            k_base = 20.0
            
        k_factor = k_base * (1.0 + (brier * 0.5))

        dA = self.state["players"][pA_id]
        dB = self.state["players"][pB_id]

        dA["rating"] += k_factor * (y_actual - pred_p)
        dB["rating"] -= k_factor * (y_actual - pred_p)

        va, vb = np.array(dA["melo_vector"]), np.array(dB["melo_vector"])
        grad = y_actual - pred_p
        dA["melo_vector"] = (va + 0.02 * grad * (self.engine.Omega @ vb)).tolist()
        dB["melo_vector"] = (vb - 0.02 * grad * (self.engine.Omega @ va)).tolist()

        dA["matches_played"] += 1
        dB["matches_played"] += 1
        self.save_state()
        return brier

# =====================================================================
# 4. DYNAMIC WEIGHTS PROTOCOL & LIVE EVALUATION
# =====================================================================
class LiveEvaluator:
    def __init__(self):
        self.learning_core = LearningCore()
        self.combinatorial = CombinatorialEngine()

    def evaluate_board(self):
        DatabaseManager.initialize()
        model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

        with DatabaseManager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM matches WHERE processed = 0")
            fixtures = cursor.fetchall()

            for row in fixtures:
                pA = row["player_a_id"]
                pB = row["player_b_id"]
                sets_a, sets_b = row["set_score_a"], row["set_score_b"]
                
                pA_id = self.learning_core.register_player(pA)
                pB_id = self.learning_core.register_player(pB)
                dA = self.learning_core.state["players"][pA_id]
                dB = self.learning_core.state["players"][pB_id]

                # Trend Direction Index momentum based on live latent score differences[span_10](start_span)[span_10](end_span)
                tdi = (sets_a - sets_b) * 0.15 
                set_dist = self.combinatorial.evaluate_set_probabilities(
                    dA["spw"], dB["spw"], sets_a, sets_b, tdi
                )
                p_win_a_math = set_dist["3-0"] + set_dist["3-1"] + set_dist["3-2"]

                # Dynamic Weights Calibration Protocol[span_11](start_span)[span_11](end_span)
                # Fade historic Glicko baseline in favor of live Markov variables if deep in match
                total_sets_played = sets_a + sets_b
                dynamic_glicko_weight = 1.0 if total_sets_played < 2 else 0.25
                
                features = pd.DataFrame([{
                    'glicko_rating_diff': float(dA["rating"] - dB["rating"]) * dynamic_glicko_weight,
                    'melo_vector_distance': float(np.linalg.norm(np.array(dA["melo_vector"]) - np.array(dB["melo_vector"]))),
                    'markov_match_win_prob_diff': p_win_a_math - (1.0 - p_win_a_math),
                    'age_diff': row["age_diff"], 'height_diff': row["height_diff"],
                    'wttr_pos_diff': row["wttr_pos_diff"], 'wttr_points_diff': row["wttr_points_diff"],
                    'recent_win_ratio_diff': row["recent_win_ratio_diff"],
                    'spw_diff': row["spw_diff"], 'rpw_diff': row["rpw_diff"]
                }])

                if model:
                    prob_a = float(model.predict_proba(features)[0, 1])
                else:
                    prob_a = p_win_a_math

                winner = pA if prob_a >= 0.50 else pB
                cursor.execute("UPDATE matches SET predicted_prob_a = ?, processed = 1 WHERE match_id = ?", (prob_a, row["match_id"]))
                
                if row["is_fanduel"]:
                    print(f"[ALERT] {pA} vs {pB} -> {winner} ({max(prob_a, 1-prob_a)*100:.2f}%)")
            conn.commit()

# =====================================================================
# 5. CLI ORCHESTRATOR
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict", action="store_true")
    args = parser.parse_args()

    if args.predict:
        evaluator = LiveEvaluator()
        evaluator.evaluate_board()
