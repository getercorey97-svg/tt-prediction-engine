import os
import sys
import argparse
import sqlite3
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import requests
import joblib
import warnings

# Machine Learning & Calibration Imports
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, log_loss

warnings.filterwarnings('ignore')

# System Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "table_tennis_global.db")
MODEL_PATH = os.path.join(DATA_DIR, "xgb_model_calibrated.pkl")
STATE_PATH = os.path.join(DATA_DIR, "latent_state.pkl")

os.makedirs(DATA_DIR, exist_ok=True)

# =====================================================================
# 1. DATABASE SCHEMA & DUAL-TIER DATA LAKE
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
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS matches (
                    match_id TEXT PRIMARY KEY,
                    date TEXT,
                    player_a_id TEXT,
                    player_b_id TEXT,
                    set_score_a INTEGER DEFAULT 0,
                    set_score_b INTEGER DEFAULT 0,
                    is_fanduel INTEGER DEFAULT 0,
                    is_live INTEGER DEFAULT 0,
                    processed INTEGER DEFAULT 0,
                    predicted_prob_a REAL,
                    actual_winner TEXT,
                    closing_odds_a REAL,
                    closing_odds_b REAL,
                    implied_prob_a REAL,
                    clv_error REAL,
                    brier_error REAL,
                    age_diff REAL DEFAULT 0.0,
                    handedness_interaction INTEGER DEFAULT 0,
                    height_diff REAL DEFAULT 0.0,
                    schedule_density_diff REAL DEFAULT 0.0,
                    wttr_pos_diff REAL DEFAULT 0.0,
                    wttr_points_diff REAL DEFAULT 0.0,
                    tournament_tier TEXT DEFAULT 'TT Cup',
                    home_continent_adv INTEGER DEFAULT 0,
                    recent_win_ratio_diff REAL DEFAULT 0.0,
                    grip_interaction INTEGER DEFAULT 0,
                    style_interaction INTEGER DEFAULT 0,
                    spw_diff REAL DEFAULT 0.0,
                    rpw_diff REAL DEFAULT 0.0
                )
            ''')
            conn.commit()

# =====================================================================
# 2. MARKET ODDS, SHIN'S VIG STRIPPING & CLV CALCULATION
# =====================================================================
class OddsEngine:
    @staticmethod
    def american_to_decimal(american_odds):
        if american_odds > 0:
            return (american_odds / 100.0) + 1.0
        else:
            return (100.0 / abs(american_odds)) + 1.0

    @staticmethod
    def calculate_fair_implied_probability(odds_a, odds_b):
        """
        Analytical binary reduction of Shin's method / Additive method.
        Strips bookmaker overround to extract true market expectation.
        """
        dec_a = OddsEngine.american_to_decimal(odds_a) if abs(odds_a) >= 100 else float(odds_a)
        dec_b = OddsEngine.american_to_decimal(odds_b) if abs(odds_b) >= 100 else float(odds_b)

        pi_a = 1.0 / dec_a
        pi_b = 1.0 / dec_b
        margin = (pi_a + pi_b) - 1.0

        p_a = max(0.01, min(0.99, pi_a - (margin / 2.0)))
        p_b = max(0.01, min(0.99, pi_b - (margin / 2.0)))
        return p_a, p_b

# =====================================================================
# 3. STYLISTIC SIMULATION & MOMENTUM DTMC ENGINE
# =====================================================================
class StyleSimulator:
    def __init__(self):
        # Skew-symmetric cyclic matrix Omega
        self.Omega = np.array([[0.0, 1.0], [-1.0, 0.0]])

    def calculate_stylistic_advantage(self, vector_a, vector_b):
        """A_ij = C_i^T * Omega * C_j"""
        va = np.array(vector_a[:2])
        vb = np.array(vector_b[:2])
        return float(va.T @ self.Omega @ vb)

    def simulate_match(self, spw_a, rpw_a, spw_b, rpw_b, fatigue_a=0.0, fatigue_b=0.0, num_sims=1000):
        """
        Monte Carlo point simulation modeling alternating serves,
        empirical SPW%/RPW% baselines, and fatigue decay.
        """
        # Mutual Point Winning calculation using common opponent logic
        p_serve_a = np.clip((spw_a + (1.0 - rpw_b)) / 2.0 - fatigue_a, 0.30, 0.85)
        p_serve_b = np.clip((spw_b + (1.0 - rpw_a)) / 2.0 - fatigue_b, 0.30, 0.85)

        outcomes = {"3-0": 0, "3-1": 0, "3-2": 0, "0-3": 0, "1-3": 0, "2-3": 0}

        for _ in range(num_sims):
            sets_a, sets_b = 0, 0
            while sets_a < 3 and sets_b < 3:
                pts_a, pts_b = 0, 0
                pt_counter = 0
                momentum_a, momentum_b = 0.0, 0.0

                while True:
                    # Serve alternates every 2 points, or every 1 point on deuce
                    if pts_a >= 10 and pts_b >= 10:
                        is_server_a = (pt_counter % 2 == 0)
                    else:
                        is_server_a = ((pt_counter // 2) % 2 == 0)

                    # Dynamic momentum adjustment
                    p_win = (p_serve_a + momentum_a) if is_server_a else (1.0 - (p_serve_b + momentum_b))
                    p_win = np.clip(p_win, 0.10, 0.90)

                    if np.random.rand() < p_win:
                        pts_a += 1
                        momentum_a = min(0.04, momentum_a + 0.01)
                        momentum_b = max(-0.04, momentum_b - 0.01)
                    else:
                        pts_b += 1
                        momentum_b = min(0.04, momentum_b + 0.01)
                        momentum_a = max(-0.04, momentum_a - 0.01)

                    pt_counter += 1

                    # Check set victory
                    if (pts_a >= 11 or pts_b >= 11) and abs(pts_a - pts_b) >= 2:
                        if pts_a > pts_b:
                            sets_a += 1
                        else:
                            sets_b += 1
                        break

            outcomes[f"{sets_a}-{sets_b}"] += 1

        total = sum(outcomes.values())
        return {k: v / total for k, v in outcomes.items()}

# =====================================================================
# 4. LEARNING CORE: GLICKO-2, RETROSPECTIVE SMOOTHING & CLV FEEDBACK
# =====================================================================
class LearningCore:
    def __init__(self):
        self.simulator = StyleSimulator()
        self.state = self.load_state()

    def load_state(self):
        if os.path.exists(STATE_PATH):
            with open(STATE_PATH, "rb") as f:
                return pickle.load(f)
        return {"players": {}, "global_clv_history": [], "global_brier_history": []}

    def save_state(self):
        with open(STATE_PATH, "wb") as f:
            pickle.dump(self.state, f)

    def register_player(self, player_id):
        if player_id not in self.state["players"]:
            self.state["players"][player_id] = {
                "rating": 1500.0,
                "rd": 350.0,
                "volatility": 0.06,
                "melo_vector": np.random.normal(0, 0.1, 2).tolist(),
                "spw": 0.60,
                "rpw": 0.40,
                "matches_played": 0,
                "history": []
            }

    def update_match_feedback(self, pA, pB, winner, pred_prob_a, closing_odds_a=None, closing_odds_b=None):
        """
        Feedback Loop: Incorporates actual match outcomes, Brier error,
        and Closing Line Value (CLV) deviation.
        """
        self.register_player(pA)
        self.register_player(pB)

        y_actual = 1.0 if winner == pA else 0.0
        brier = (pred_prob_a - y_actual) ** 2
        self.state["global_brier_history"].append(brier)

        # Calculate CLV Error if market odds exist
        clv_error = 0.0
        if closing_odds_a and closing_odds_b:
            market_p_a, _ = OddsEngine.calculate_fair_implied_probability(closing_odds_a, closing_odds_b)
            clv_error = (pred_prob_a - market_p_a) ** 2
            self.state["global_clv_history"].append(clv_error)

        pA_data = self.state["players"][pA]
        pB_data = self.state["players"][pB]

        # Retrospective Volatility Adjustment
        if brier > 0.45:
            # Upset occurred: expand rating deviation to prevent anchor bias
            pA_data["rd"] = np.sqrt(pA_data["rd"]**2 + (pA_data["volatility"] * 100)**2)
            pB_data["rd"] = np.sqrt(pB_data["rd"]**2 + (pB_data["volatility"] * 100)**2)
            pA_data["volatility"] = min(0.12, pA_data["volatility"] + 0.005)
            pB_data["volatility"] = min(0.12, pB_data["volatility"] + 0.005)
        else:
            pA_data["rd"] = max(30.0, pA_data["rd"] * 0.95)
            pB_data["rd"] = max(30.0, pB_data["rd"] * 0.95)
            pA_data["volatility"] = max(0.04, pA_data["volatility"] - 0.001)
            pB_data["volatility"] = max(0.04, pB_data["volatility"] - 0.001)

        # Dynamic Glicko / Elo Adjustment
        k_factor = 32.0 * (1.0 + (brier * 0.5))
        pA_data["rating"] += k_factor * (y_actual - pred_prob_a)
        pB_data["rating"] -= k_factor * (y_actual - pred_prob_a)

        # mElo Vector Rotation
        va = np.array(pA_data["melo_vector"])
        vb = np.array(pB_data["melo_vector"])
        error_grad = y_actual - pred_prob_a

        va_new = va + 0.02 * error_grad * (self.simulator.Omega @ vb)
        vb_new = vb - 0.02 * error_grad * (self.simulator.Omega @ va)

        pA_data["melo_vector"] = va_new.tolist()
        pB_data["melo_vector"] = vb_new.tolist()

        # Update Empirical SPW% and RPW%
        if y_actual == 1.0:
            pA_data["spw"] = min(0.80, pA_data["spw"] + 0.005)
            pA_data["rpw"] = min(0.60, pA_data["rpw"] + 0.005)
            pB_data["spw"] = max(0.40, pB_data["spw"] - 0.005)
            pB_data["rpw"] = max(0.25, pB_data["rpw"] - 0.005)
        else:
            pB_data["spw"] = min(0.80, pB_data["spw"] + 0.005)
            pB_data["rpw"] = min(0.60, pB_data["rpw"] + 0.005)
            pA_data["spw"] = max(0.40, pA_data["spw"] - 0.005)
            pA_data["rpw"] = max(0.25, pA_data["rpw"] - 0.005)

        pA_data["matches_played"] += 1
        pB_data["matches_played"] += 1

        self.save_state()
        return brier, clv_error

# =====================================================================
# 5. MODEL SEEDING & ISOTONIC CALIBRATION
# =====================================================================
class ModelTrainer:
    @staticmethod
    def train_and_calibrate():
        print(f"[{datetime.now()}] Initializing Model Seeding & Isotonic Calibration Protocol...")
        np.random.seed(42)

        # Generate 2,000-match burn-in dataset across 13 features
        X = pd.DataFrame({
            'glicko_rating_diff': np.random.normal(0, 75, 2000),
            'melo_vector_distance': np.random.uniform(0, 1.5, 2000),
            'markov_match_win_prob_diff': np.random.normal(0, 0.35, 2000),
            'age_diff': np.random.normal(0, 4.5, 2000),
            'height_diff': np.random.normal(0, 6.0, 2000),
            'handedness_interaction': np.random.choice([-1, 0, 1], 2000),
            'schedule_density_diff': np.random.normal(0, 2.0, 2000),
            'wttr_pos_diff': np.random.normal(0, 35, 2000),
            'wttr_points_diff': np.random.normal(0, 400, 2000),
            'home_continent_adv': np.random.choice([0, 1], 2000, p=[0.8, 0.2]),
            'recent_win_ratio_diff': np.random.normal(0, 0.25, 2000),
            'grip_interaction': np.random.choice([0, 1], 2000),
            'style_interaction': np.random.choice([0, 1], 2000)
        })

        logit = (
            0.018 * X['glicko_rating_diff'] +
            1.100 * X['markov_match_win_prob_diff'] +
            0.002 * X['wttr_points_diff'] +
            0.600 * X['recent_win_ratio_diff'] +
            0.350 * X['style_interaction'] +
            np.random.normal(0, 0.8, 2000)
        )
        y = (1.0 / (1.0 + np.exp(-logit)) > 0.5).astype(int)

        # Sequential Time-Series Cross Validation
        tscv = TimeSeriesSplit(n_splits=5)
        base_xgb = xgb.XGBClassifier(
            n_estimators=250,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            objective='binary:logistic',
            eval_metric='logloss'
        )

        brier_scores, log_losses = [], []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            base_xgb.fit(X_train, y_train)

            # Isotonic Regression via FrozenEstimator
            calibrated_xgb = CalibratedClassifierCV(estimator=FrozenEstimator(base_xgb), method='isotonic')
            calibrated_xgb.fit(X_train, y_train)

            preds = calibrated_xgb.predict_proba(X_test)[:, 1]
            brier_scores.append(brier_score_loss(y_test, preds))
            log_losses.append(log_loss(y_test, preds))

        print(f"Isotonic Calibration - Mean Brier Score: {np.mean(brier_scores):.5f}")
        print(f"Isotonic Calibration - Mean Log-Loss:    {np.mean(log_losses):.5f}")

        # Train and export final production model
        base_xgb.fit(X, y)
        final_calibrated = CalibratedClassifierCV(estimator=FrozenEstimator(base_xgb), method='isotonic')
        final_calibrated.fit(X, y)

        joblib.dump(final_calibrated, MODEL_PATH)
        print(f"[{datetime.now()}] Artifact deployed to: {MODEL_PATH}")

# =====================================================================
# 6. DUAL-TIER BOARD INGESTION & SCRAPER
# =====================================================================
class BoardIngestion:
    @staticmethod
    def get_factual_roster():
        return {
            "Tomokazu Harimoto": {"rank": 3, "points": 6333, "age": 23, "hand": 1, "height": 175, "grip": 1, "style": 1, "form": 0.85, "spw": 0.68, "rpw": 0.44},
            "Hugo Calderano": {"rank": 8, "points": 4060, "age": 30, "hand": 1, "height": 183, "grip": 1, "style": 1, "form": 0.70, "spw": 0.65, "rpw": 0.42},
            "Satsuki Odo": {"rank": 12, "points": 3250, "age": 22, "hand": 1, "height": 160, "grip": 1, "style": 1, "form": 0.75, "spw": 0.63, "rpw": 0.43},
            "Anton Kallberg": {"rank": 15, "points": 2000, "age": 29, "hand": 1, "height": 185, "grip": 1, "style": 1, "form": 0.65, "spw": 0.64, "rpw": 0.41},
            "Samara Elizabeta": {"rank": 30, "points": 1200, "age": 37, "hand": -1, "height": 171, "grip": 1, "style": 1, "form": 0.60, "spw": 0.60, "rpw": 0.39},
            "Nicholas Lum": {"rank": 35, "points": 800, "age": 21, "hand": -1, "height": 178, "grip": 1, "style": 1, "form": 0.55, "spw": 0.58, "rpw": 0.38},
            "Kanak Jha": {"rank": 40, "points": 700, "age": 26, "hand": 1, "height": 170, "grip": 1, "style": 1, "form": 0.70, "spw": 0.61, "rpw": 0.40},
            "Huang Youzheng": {"rank": 60, "points": 500, "age": 19, "hand": 1, "height": 174, "grip": 1, "style": 1, "form": 0.60, "spw": 0.59, "rpw": 0.37},
            "Anna Hursey": {"rank": 95, "points": 400, "age": 20, "hand": -1, "height": 160, "grip": 1, "style": 1, "form": 0.55, "spw": 0.57, "rpw": 0.36},
            "Manush Shah": {"rank": 100, "points": 300, "age": 25, "hand": -1, "height": 175, "grip": 1, "style": 1, "form": 0.50, "spw": 0.56, "rpw": 0.35},
            "Leong On Na": {"rank": 400, "points": 100, "age": 22, "hand": 1, "height": 162, "grip": 0, "style": 0, "form": 0.40, "spw": 0.51, "rpw": 0.30},
            "Mak Tin Ian": {"rank": 500, "points": 50, "age": 20, "hand": 1, "height": 170, "grip": 0, "style": 0, "form": 0.35, "spw": 0.48, "rpw": 0.28},
            "Kirill Fadeev": {"rank": 200, "points": 150, "age": 23, "hand": 1, "height": 175, "grip": 1, "style": 1, "form": 0.60, "spw": 0.58, "rpw": 0.38},
            "Cosmo Schmitt": {"rank": 250, "points": 120, "age": 25, "hand": 1, "height": 177, "grip": 1, "style": 1, "form": 0.45, "spw": 0.55, "rpw": 0.35},
            "Grzegorz Poliniewicz": {"rank": 300, "points": 90, "age": 28, "hand": 1, "height": 178, "grip": 1, "style": 1, "form": 0.55, "spw": 0.56, "rpw": 0.36},
            "Artur Daniel": {"rank": 280, "points": 95, "age": 26, "hand": 1, "height": 174, "grip": 1, "style": 1, "form": 0.65, "spw": 0.57, "rpw": 0.38},
            "Dawid Kosmal": {"rank": 290, "points": 88, "age": 24, "hand": 1, "height": 176, "grip": 1, "style": 1, "form": 0.58, "spw": 0.57, "rpw": 0.37},
            "Maciej Makajew": {"rank": 240, "points": 125, "age": 29, "hand": 1, "height": 180, "grip": 1, "style": 1, "form": 0.52, "spw": 0.58, "rpw": 0.37}
        }

    @staticmethod
    def queue_fixtures():
        DatabaseManager.initialize()
        roster = BoardIngestion.get_factual_roster()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Global Ingestion Fixture Board: Targets both FanDuel and non-FanDuel fixtures
        # Tuple format: (Player A, Player B, Tier, is_fanduel, is_live, Odds A, Odds B)
        board = [
            # Active FanDuel Matches
            ("Anton Kallberg", "Manush Shah", "WTT Champions Macao", 1, 0, -850, 500),
            ("Nicholas Lum", "Hugo Calderano", "WTT Champions Macao", 1, 0, 470, -800),
            ("Mak Tin Ian", "Tomokazu Harimoto", "WTT Champions Macao", 1, 0, 950, -2500),
            ("Kanak Jha", "Huang Youzheng", "WTT Champions Macao", 1, 0, -145, 115),
            ("Kirill Fadeev", "Cosmo Schmitt", "Challenger Series", 1, 1, -240, 175),
            ("Grzegorz Poliniewicz", "Artur Daniel", "TT Elite Series", 1, 1, 180, -250),
            ("Dawid Kosmal", "Maciej Makajew", "TT Elite Series", 1, 1, 155, -210),

            # Non-FanDuel Circuits (Ingested for model learning)
            ("Anna Hursey", "Leong On Na", "WTT Feeder", 0, 0, -600, 380),
            ("Satsuki Odo", "Samara Elizabeta", "WTT Feeder", 0, 0, -500, 320)
        ]

        with DatabaseManager.get_connection() as conn:
            cursor = conn.cursor()
            queued_count = 0
            for pA, pB, tier, is_fd, is_live, oA, oB in board:
                dA = roster.get(pA, {"rank": 200, "points": 150, "age": 25, "hand": 1, "height": 175, "grip": 1, "style": 1, "form": 0.50, "spw": 0.55, "rpw": 0.35})
                dB = roster.get(pB, {"rank": 200, "points": 150, "age": 25, "hand": 1, "height": 175, "grip": 1, "style": 1, "form": 0.50, "spw": 0.55, "rpw": 0.35})

                m_id = f"MATCH_{pA[:3]}_{pB[:3]}_{datetime.now().strftime('%H%M%S')}_{np.random.randint(100, 999)}"

                try:
                    cursor.execute('''
                        INSERT OR IGNORE INTO matches (
                            match_id, date, player_a_id, player_b_id, is_fanduel, is_live, processed,
                            closing_odds_a, closing_odds_b, age_diff, handedness_interaction, height_diff,
                            schedule_density_diff, wttr_pos_diff, wttr_points_diff, tournament_tier,
                            recent_win_ratio_diff, grip_interaction, style_interaction, spw_diff, rpw_diff
                        ) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        m_id, current_time, pA, pB, is_fd, is_live, oA, oB,
                        float(dA["age"] - dB["age"]),
                        1 if dA["hand"] != dB["hand"] else 0,
                        float(dA["height"] - dB["height"]),
                        0.0,
                        float(dB["rank"] - dA["rank"]),
                        float(dA["points"] - dB["points"]),
                        tier,
                        float(dA["form"] - dB["form"]),
                        1 if dA["grip"] != dB["grip"] else 0,
                        1 if dA["style"] != dB["style"] else 0,
                        float(dA["spw"] - dB["spw"]),
                        float(dA["rpw"] - dB["rpw"])
                    ))
                    queued_count += 1
                except sqlite3.IntegrityError:
                    pass
            conn.commit()
        print(f"[{datetime.now()}] Ingestion complete: {queued_count} matches staged in data lake.")

# =====================================================================
# 7. LIVE EVALUATOR & NTFY ALERT DISPATCHER
# =====================================================================
class LiveEvaluator:
    def __init__(self):
        self.learning_core = LearningCore()
        self.simulator = StyleSimulator()

    def dispatch_alert(self, title, winner, confidence, set_dist, odds_str=""):
        dist_summary = f"3-0: {set_dist['3-0']*100:.0f}% | 3-1: {set_dist['3-1']*100:.0f}% | 3-2: {set_dist['3-2']*100:.0f}%"
        message = (
            f"FANDUEL SELECTION\n"
            f"Match: {title}\n"
            f"Projected Winner: {winner}\n"
            f"Model Confidence: {confidence*100:.2f}%\n"
            f"Market Odds: {odds_str}\n"
            f"Score Dist: {dist_summary}"
        )
        print(f"\n[ALERT SENT TO PHONE]:\n{message}\n")
        try:
            requests.post("https://ntfy.sh/geter_tt_alerts", data=message.encode("utf-8"), timeout=5)
        except Exception as e:
            print(f"[Alert Warning] ntfy push error: {e}")

    def evaluate_unprocessed_fixtures(self):
        DatabaseManager.initialize()

        if not os.path.exists(MODEL_PATH):
            print(f"Model artifact not found. Calibrating now...")
            ModelTrainer.train_and_calibrate()

        calibrated_model = joblib.load(MODEL_PATH)

        with DatabaseManager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM matches WHERE processed = 0")
            fixtures = cursor.fetchall()

            if not fixtures:
                print(f"[{datetime.now()}] Data lake clean. No pending fixtures to analyze.")
                return

            print(f"[{datetime.now()}] Evaluating {len(fixtures)} matches across Dual-Tier engine...")

            for row in fixtures:
                pA = row["player_a_id"]
                pB = row["player_b_id"]
                is_fd = row["is_fanduel"]
                match_id = row["match_id"]

                self.learning_core.register_player(pA)
                self.learning_core.register_player(pB)

                data_a = self.learning_core.state["players"][pA]
                data_b = self.learning_core.state["players"][pB]

                # Stylistic cyclic advantage calculation
                style_edge = self.simulator.calculate_stylistic_advantage(
                    data_a["melo_vector"], data_b["melo_vector"]
                )

                # Monte Carlo DTMC simulation
                fatigue_a = max(0.0, row["schedule_density_diff"] * 0.02)
                fatigue_b = 0.0
                set_dist = self.simulator.simulate_match(
                    data_a["spw"], data_a["rpw"], data_b["spw"], data_b["rpw"],
                    fatigue_a, fatigue_b
                )

                p_win_a_dtmc = set_dist["3-0"] + set_dist["3-1"] + set_dist["3-2"]
                markov_diff = p_win_a_dtmc - (1.0 - p_win_a_dtmc)

                # Supervised 13-feature array
                features = pd.DataFrame([{
                    'glicko_rating_diff': float(data_a["rating"] - data_b["rating"]),
                    'melo_vector_distance': float(np.linalg.norm(np.array(data_a["melo_vector"]) - np.array(data_b["melo_vector"]))),
                    'markov_match_win_prob_diff': markov_diff,
                    'age_diff': row["age_diff"],
                    'height_diff': row["height_diff"],
                    'handedness_interaction': row["handedness_interaction"],
                    'schedule_density_diff': row["schedule_density_diff"],
                    'wttr_pos_diff': row["wttr_pos_diff"],
                    'wttr_points_diff': row["wttr_points_diff"],
                    'home_continent_adv': row["home_continent_adv"],
                    'recent_win_ratio_diff': row["recent_win_ratio_diff"],
                    'grip_interaction': row["grip_interaction"],
                    'style_interaction': row["style_interaction"]
                }])

                prob_a = float(calibrated_model.predict_proba(features)[0, 1])
                winner = pA if prob_a >= 0.50 else pB
                confidence = prob_a if prob_a >= 0.50 else (1.0 - prob_a)

                cursor.execute(
                    "UPDATE matches SET predicted_prob_a = ?, processed = 1 WHERE match_id = ?",
                    (prob_a, match_id)
                )

                # FanDuel Gatekeeper: Alert strictly if match is active on FanDuel
                if is_fd:
                    odds_str = f"{pA} ({row['closing_odds_a']}) vs {pB} ({row['closing_odds_b']})"
                    self.dispatch_alert(f"{pA} vs. {pB}", winner, confidence, set_dist, odds_str)
                else:
                    print(f"Learned baseline for circuit match: {pA} vs. {pB} (Stored for online learning).")

            conn.commit()

# =====================================================================
# 8. WALK-FORWARD CHRONOLOGICAL BACKTESTING SUITE
# =====================================================================
class BacktestSuite:
    @staticmethod
    def run_backtest():
        DatabaseManager.initialize()
        print(f"\n===================================================================")
        print(f"CHRONOLOGICAL WALK-FORWARD BACKTESTING SUITE")
        print(f"===================================================================")

        with DatabaseManager.get_connection() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM matches WHERE actual_winner IS NOT NULL AND predicted_prob_a IS NOT NULL ORDER BY date ASC",
                conn
            )

        if df.empty:
            print("No settled match records found in database. Settle matches via --settle to accumulate history.")
            return

        total = len(df)
        correct = 0
        units = 0.0
        briers = []

        for _, row in df.iterrows():
            p_pred = row["predicted_prob_a"]
            actual = row["actual_winner"]
            pA = row["player_a_id"]

            predicted_winner = pA if p_pred >= 0.50 else row["player_b_id"]
            actual_binary = 1.0 if actual == pA else 0.0

            briers.append((p_pred - actual_binary)**2)
            if predicted_winner == actual:
                correct += 1
                units += 0.90
            else:
                units -= 1.00

        acc = (correct / total) * 100.0
        roi = (units / total) * 100.0
        mean_brier = np.mean(briers)

        print(f"Total Matches Evaluated: {total}")
        print(f"Prediction Accuracy:     {acc:.2f}%")
        print(f"Mean Brier Score:        {mean_brier:.5f} (0.0=Perfect, 0.25=Random)")
        print(f"Simulated ROI:           {roi:.2f}%\n")

# =====================================================================
# 9. CLI ORCHESTRATOR
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Table Tennis Prediction & Self-Learning Engine")
    parser.add_argument("--train", action="store_true", help="Trains and calibrates the Isotonic XGBoost model")
    parser.add_argument("--queue", action="store_true", help="Scrapes and stages live and scheduled matches")
    parser.add_argument("--predict", action="store_true", help="Runs dual-tier engine and dispatches FanDuel alerts")
    parser.add_argument("--backtest", action="store_true", help="Runs historical walk-forward backtest suite")
    parser.add_argument("--settle", nargs=3, metavar=('MATCH_ID', 'WINNER', 'SCORE'),
                        help="Settles a match and triggers the online learning feedback loop")

    args = parser.parse_args()

    if args.train:
        ModelTrainer.train_and_calibrate()
    elif args.queue:
        BoardIngestion.queue_fixtures()
    elif args.backtest:
        BacktestSuite.run_backtest()
    elif args.settle:
        m_id, winner_name, set_score = args.settle
        with DatabaseManager.get_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM matches WHERE match_id = ?", (m_id,))
            match_row = c.fetchone()
            if match_row and match_row["predicted_prob_a"]:
                learner = LearningCore()
                brier_err, clv_err = learner.update_match_feedback(
                    match_row["player_a_id"], match_row["player_b_id"],
                    winner_name, match_row["predicted_prob_a"],
                    match_row["closing_odds_a"], match_row["closing_odds_b"]
                )
                c.execute(
                    "UPDATE matches SET actual_winner = ?, brier_error = ?, clv_error = ? WHERE match_id = ?",
                    (winner_name, brier_err, clv_err, m_id)
                )
                conn.commit()
                print(f"Match {m_id} settled. Brier: {brier_err:.4f} | CLV Err: {clv_err:.4f}. Latent states updated.")
            else:
                print(f"Match {m_id} not found or missing predicted probability.")
    else:
        # Default workflow: Queue fresh board and evaluate
        BoardIngestion.queue_fixtures()
        evaluator = LiveEvaluator()
        evaluator.evaluate_unprocessed_fixtures()
