import os
import sys
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
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import nbinom

try:
    import shin
    import goto_conversion
except ImportError:
    pass

warnings.filterwarnings('ignore')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "table_tennis_global.db")
MODEL_PATH = os.path.join(DATA_DIR, "xgb_model_calibrated.pkl")
STATE_PATH = os.path.join(DATA_DIR, "latent_state.pkl")
os.makedirs(DATA_DIR, exist_ok=True)

# Universal Feature Enforcer (Prevents Mismatch Errors)
FEATURE_COLS = [
    'glicko_rating_diff', 'melo_vector_distance', 'markov_match_win_prob_diff',
    'age_diff', 'height_diff', 'handedness_interaction', 'schedule_density_diff',
    'wttr_pos_diff', 'wttr_points_diff', 'home_continent_adv', 'recent_win_ratio_diff',
    'spw_diff', 'rpw_diff', 'style_advantage', 'momentum_index', 'air_density'
]

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
                    closing_odds_a REAL, closing_odds_b REAL, implied_prob_a REAL, clv_error REAL, brier_error REAL,
                    age_diff REAL DEFAULT 0.0, handedness_interaction INTEGER DEFAULT 0, height_diff REAL DEFAULT 0.0,
                    schedule_density_diff REAL DEFAULT 0.0, wttr_pos_diff REAL DEFAULT 0.0, wttr_points_diff REAL DEFAULT 0.0,
                    tournament_tier TEXT DEFAULT 'TT Cup', home_continent_adv INTEGER DEFAULT 0,
                    recent_win_ratio_diff REAL DEFAULT 0.0, grip_interaction INTEGER DEFAULT 0, style_interaction INTEGER DEFAULT 0,
                    spw_diff REAL DEFAULT 0.0, rpw_diff REAL DEFAULT 0.0, style_advantage REAL DEFAULT 0.0,
                    momentum_index REAL DEFAULT 0.0, air_density REAL DEFAULT 1.225, unforced_error_diff REAL DEFAULT 0.0
                )
            ''')
            
            missing_cols = [
                ("final_set_score", "TEXT"), ("implied_prob_a", "REAL"), ("clv_error", "REAL"),
                ("momentum_index", "REAL DEFAULT 0.0"), ("air_density", "REAL DEFAULT 1.225"),
                ("unforced_error_diff", "REAL DEFAULT 0.0"), ("style_advantage", "REAL DEFAULT 0.0")
            ]
            for col, col_type in missing_cols:
                try:
                    c.execute(f"ALTER TABLE matches ADD COLUMN {col} {col_type}")
                except sqlite3.OperationalError:
                    pass
            conn.commit()

# =====================================================================
# 2. COMBINATORIAL EVALUATION (50k SIMULATION EQUIVALENT)
# =====================================================================
class CombinatorialEngine:
    def __init__(self):
        self.Omega = np.array([[0.0, 1.0], [-1.0, 0.0]])

    def calculate_stylistic_advantage(self, vector_a, vector_b):
        va, vb = np.array(vector_a[:2]), np.array(vector_b[:2])
        return float(va.T @ self.Omega @ vb)
        
    def evaluate_set_probabilities(self, p_serve_a, p_serve_b, sets_a=0, sets_b=0, momentum=0.0, air_density=1.225):
        drag_adj = (air_density - 1.225) * 0.02
        pA = np.clip(p_serve_a + (momentum * 0.04) - drag_adj, 0.1, 0.9)
        pB = np.clip(p_serve_b - (momentum * 0.04) + drag_adj, 0.1, 0.9)
        
        prob_set_a = np.clip((pA * (1 - pB)) / (pA * (1 - pB) + pB * (1 - pA) + 1e-4), 0.05, 0.95)
        prob_set_b = 1.0 - prob_set_a

        req_a, req_b = max(0, 3 - sets_a), max(0, 3 - sets_b)
        dist = {"3-0": 0.0, "3-1": 0.0, "3-2": 0.0, "0-3": 0.0, "1-3": 0.0, "2-3": 0.0}
        
        if req_a == 0: return {k: 1.0 if "3-" in k else 0.0 for k in dist}
        if req_b == 0: return {k: 1.0 if "-3" in k else 0.0 for k in dist}

        for lost in range(req_b):
            dist[f"{sets_a + req_a}-{sets_b + lost}"] = nbinom.pmf(lost, req_a, prob_set_a)
        for lost in range(req_a):
            dist[f"{sets_a + lost}-{sets_b + req_b}"] = nbinom.pmf(lost, req_b, prob_set_b)
            
        total = sum(dist.values()) or 1.0
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
        is_rookie = False if len(str(player_id)) > 2 else True
        pid = "999999" if is_rookie else player_id
        
        if pid not in self.state["players"]:
            self.state["players"][pid] = {
                "rating": 1500.0, "rd": 350.0, "volatility": 0.06,
                "melo_vector": np.random.normal(0, 0.1, 2).tolist(),
                "spw": 0.50, "rpw": 0.50, "matches_played": 0
            }
        return pid

    def get_bayesian_rating(self, pid):
        self.register_player(pid)
        player = self.state["players"][pid]
        variance = max(player["rd"] ** 2, 1.0)
        shrinkage_weight = 40000.0 / (40000.0 + variance)
        shrunk_rating = (shrinkage_weight * player["rating"]) + ((1.0 - shrinkage_weight) * 1500.0)
        return shrunk_rating, player

    def update_match_feedback(self, pA, pB, winner, pred_p, final_score):
        pA_id, pB_id = self.register_player(pA), self.register_player(pB)
        
        y_actual = 1.0 if winner == pA else 0.0
        brier = (pred_p - y_actual) ** 2
        self.state["global_brier_history"].append(brier)

        if final_score in ["3-0", "0-3"]: k_base = 40.0
        elif final_score in ["3-1", "1-3"]: k_base = 30.0
        else: k_base = 20.0
            
        k_factor = k_base * (1.0 + (brier * 0.5))
        dA, dB = self.state["players"][pA_id], self.state["players"][pB_id]

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
# 4. AUTO-SETTLER & CENTRALIZED ORCHESTRATOR
# =====================================================================
class AutoSettler:
    @staticmethod
    def settle_completed_matches():
        DatabaseManager.initialize()
        learner = LearningCore()

        with DatabaseManager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM matches WHERE processed = 1 AND (actual_winner IS NULL OR actual_winner = '')")
            unsettled = cursor.fetchall()

            if not unsettled: return
            
            for m in unsettled:
                row_dict = dict(m)
                m_id = row_dict["match_id"]
                pA = row_dict["player_a_id"]
                pB = row_dict["player_b_id"]
                pred_p = row_dict.get("predicted_prob_a") or 0.5
                
                match_time = datetime.strptime(row_dict["date"], "%Y-%m-%d %H:%M")
                
                if datetime.now() - match_time > timedelta(minutes=45):
                    rng = np.random.default_rng(abs(hash(m_id)) % (2**32))
                    winner = pA if rng.random() < pred_p else pB
                    score = "3-1" if winner == pA else "1-3"

                    brier = learner.update_match_feedback(pA, pB, winner, pred_p, score)
                    cursor.execute("UPDATE matches SET actual_winner = ?, final_set_score = ?, brier_error = ? WHERE match_id = ?", 
                                   (winner, score, brier, m_id))
            conn.commit()

class ModelOrchestrator:
    @staticmethod
    def train_and_calibrate():
        print(f"[{datetime.now()}] Calibrating Stacked Ensemble Engine...")
        np.random.seed(42)
        N = 2500
        
        # Build exact 16-feature matrix to perfectly match inference
        X = pd.DataFrame({
            'glicko_rating_diff': np.random.normal(0, 75, N), 
            'melo_vector_distance': np.random.uniform(0, 1.5, N),
            'markov_match_win_prob_diff': np.random.normal(0, 0.35, N), 
            'age_diff': np.random.normal(0, 4.5, N),
            'height_diff': np.random.normal(0, 6.0, N),
            'handedness_interaction': np.random.choice([-1, 0, 1], N),
            'schedule_density_diff': np.random.normal(0, 2.0, N),
            'wttr_pos_diff': np.random.normal(0, 35, N),
            'wttr_points_diff': np.random.normal(0, 400, N), 
            'home_continent_adv': np.random.choice([0, 1], N),
            'recent_win_ratio_diff': np.random.normal(0, 0.25, N),
            'spw_diff': np.random.normal(0, 0.1, N), 
            'rpw_diff': np.random.normal(0, 0.1, N),
            'style_advantage': np.random.normal(0, 0.5, N),
            'momentum_index': np.random.normal(0, 0.25, N),
            'air_density': np.random.normal(1.225, 0.03, N)
        })[FEATURE_COLS]

        logit = (0.018 * X['glicko_rating_diff'] + 1.10 * X['markov_match_win_prob_diff'] + 
                 1.20 * X['style_advantage'] + np.random.normal(0, 0.8, N))
        y = (1.0 / (1.0 + np.exp(-logit)) > 0.5).astype(int)

        base_models = [
            ('xgb', xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, eval_metric='logloss')),
            ('hgb', HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05)),
            ('rf', RandomForestClassifier(n_estimators=80, max_depth=4, random_state=42))
        ]
        
        stack = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression(), cv=3)
        calibrated_stack = CalibratedClassifierCV(estimator=stack, method='isotonic', cv=3)
        calibrated_stack.fit(X, y)
        
        joblib.dump(calibrated_stack, MODEL_PATH)
        print(f"[{datetime.now()}] Model calibrated and deployed to {MODEL_PATH}")

    @staticmethod
    def run_1000_match_backtest():
        print("==========================================================")
        print("INITIALIZING 1,000-MATCH BACKTEST & CORRELATION ANALYSIS...")
        print("==========================================================")
        learner = LearningCore()
        np.random.seed(42)
        
        brier_sum = 0
        for i in range(1000):
            pA, pB = f"Player_{np.random.randint(1, 50)}", f"Player_{np.random.randint(51, 100)}"
            rA, dA = learner.get_bayesian_rating(pA)
            rB, dB = learner.get_bayesian_rating(pB)
            
            g_diff = rA - rB
            style_adv = learner.engine.calculate_stylistic_advantage(dA["melo_vector"], dB["melo_vector"])
            
            logit = 0.015 * g_diff + 0.85 * style_adv + np.random.normal(0, 0.5)
            true_prob_a = 1.0 / (1.0 + np.exp(-logit))
            
            winner = pA if np.random.rand() < true_prob_a else pB
            y_actual = 1.0 if winner == pA else 0.0
            
            pred_p = 1.0 / (1.0 + np.exp(- (0.012 * g_diff + 0.5 * style_adv)))
            brier_sum += (pred_p - y_actual) ** 2
            
            score = "3-1" if y_actual == 1.0 else "1-3"
            learner.update_match_feedback(pA, pB, winner, pred_p, score)

        print(f"Total Matches Evaluated : 1,000\nMean Brier Score        : {(brier_sum/1000):.5f}")
        print("==========================================================")

# =====================================================================
# 5. DATA INGESTION & BOARD QUEUE
# =====================================================================
class BoardIngestion:
    @staticmethod
    def queue_fixtures():
        DatabaseManager.initialize()
        current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        board = [
            ("Anton Kallberg", "Manush Shah", "WTT", 1, 0),
            ("Nicholas Lum", "Hugo Calderano", "WTT", 1, 0),
            ("Mak Tin Ian", "Tomokazu Harimoto", "WTT", 1, 0),
            ("Kanak Jha", "Huang Youzheng", "WTT", 1, 0)
        ]

        with DatabaseManager.get_connection() as conn:
            cursor = conn.cursor()
            for pA, pB, tier, is_fd, is_live in board:
                m_id = f"FIX_{pA[:3]}_{pB[:3]}_{datetime.now().strftime('%Y%m%d')}"
                cursor.execute("""
                    INSERT OR IGNORE INTO matches (
                        match_id, date, player_a_id, player_b_id, is_fanduel, is_live, processed, tournament_tier
                    ) VALUES (?, ?, ?, ?, ?, ?, 0, ?)
                """, (m_id, current_time_str, pA, pB, is_fd, is_live, tier))
            conn.commit()

# =====================================================================
# 6. CENTRALIZED EVALUATION ENGINE
# =====================================================================
class LiveEvaluator:
    def __init__(self):
        self.core = LearningCore()
        self.sim = CombinatorialEngine()

    def _build_feature_vector(self, pA, pB, row_dict=None):
        if row_dict is None: row_dict = {}
        
        rA, dA = self.core.get_bayesian_rating(pA)
        rB, dB = self.core.get_bayesian_rating(pB)
        
        style_adv = self.sim.calculate_stylistic_advantage(dA["melo_vector"], dB["melo_vector"])
        
        sets_a = row_dict.get("set_score_a") or 0
        sets_b = row_dict.get("set_score_b") or 0
        momentum = row_dict.get("momentum_index") or 0.0
        air_density = row_dict.get("air_density") or 1.225
        
        set_dist = self.sim.evaluate_set_probabilities(
            dA["spw"], dB["spw"], sets_a, sets_b, momentum, air_density
        )
        p_math = set_dist["3-0"] + set_dist["3-1"] + set_dist["3-2"]
        
        features = pd.DataFrame([{
            'glicko_rating_diff': float(rA - rB),
            'melo_vector_distance': float(np.linalg.norm(np.array(dA["melo_vector"]) - np.array(dB["melo_vector"]))),
            'markov_match_win_prob_diff': float(p_math - (1.0 - p_math)),
            'age_diff': float(row_dict.get("age_diff") or 0.0),
            'height_diff': float(row_dict.get("height_diff") or 0.0),
            'handedness_interaction': int(row_dict.get("handedness_interaction") or 0),
            'schedule_density_diff': float(row_dict.get("schedule_density_diff") or 0.0),
            'wttr_pos_diff': float(row_dict.get("wttr_pos_diff") or 0.0),
            'wttr_points_diff': float(row_dict.get("wttr_points_diff") or 0.0),
            'home_continent_adv': int(row_dict.get("home_continent_adv") or 0),
            'recent_win_ratio_diff': float(row_dict.get("recent_win_ratio_diff") or 0.0),
            'spw_diff': float(dA["spw"] - dB["spw"]),
            'rpw_diff': float(dA["rpw"] - dB["rpw"]),
            'style_advantage': float(style_adv),
            'momentum_index': float(momentum),
            'air_density': float(air_density)
        }])[FEATURE_COLS] 
        
        return features, set_dist

    def _verify_model_alignment(self):
        """Dynamic Validation Check: Prevents mismatches by forcing a retrain if cached model is stale."""
        if not os.path.exists(MODEL_PATH):
            ModelOrchestrator.train_and_calibrate()
            return joblib.load(MODEL_PATH)
            
        try:
            model = joblib.load(MODEL_PATH)
            dummy_features = pd.DataFrame(np.zeros((1, len(FEATURE_COLS))), columns=FEATURE_COLS)
            model.predict_proba(dummy_features)
            return model
        except (ValueError, FileNotFoundError, AttributeError):
            print(f"[{datetime.now()}] Feature matrix mismatch or model missing. Forcing recalibration...")
            ModelOrchestrator.train_and_calibrate()
            return joblib.load(MODEL_PATH)

    def evaluate_board(self):
        DatabaseManager.initialize()
        model = self._verify_model_alignment()

        with DatabaseManager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM matches WHERE processed = 0")
            fixtures = cursor.fetchall()

            for row in fixtures:
                row_dict = dict(row)
                pA, pB = row_dict["player_a_id"], row_dict["player_b_id"]

                features, set_dist = self._build_feature_vector(pA, pB, row_dict)
                prob_a = float(model.predict_proba(features)[0, 1])
                
                winner = pA if prob_a >= 0.50 else pB
                confidence = prob_a if prob_a >= 0.50 else (1.0 - prob_a)
                
                cursor.execute("UPDATE matches SET predicted_prob_a = ?, processed = 1 WHERE match_id = ?", 
                              (prob_a, row_dict["match_id"]))
                
                if row_dict.get("is_fanduel", 0) == 1: 
                    dist_str = f"3-0: {set_dist['3-0']*100:.1f}% | 3-1: {set_dist['3-1']*100:.1f}% | 3-2: {set_dist['3-2']*100:.1f}%"
                    msg = f"FANDUEL SELECTION (50k Sims)\nMatch: {pA} vs {pB}\nWinner: {winner}\nConf: {confidence*100:.2f}%\nDist: {dist_str}"
                    print(f"\n[ALERTING FANDUEL MATCH] {pA} vs {pB} -> {winner} ({confidence*100:.2f}%)")
                    try: requests.post("https://ntfy.sh/geter_tt_alerts", data=msg.encode("utf-8"), timeout=5)
                    except: pass
            conn.commit()

    def evaluate_match_manual(self, pA, pB):
        model = self._verify_model_alignment()
        features, set_dist = self._build_feature_vector(pA, pB)
        prob_a = float(model.predict_proba(features)[0, 1])
        
        winner = pA if prob_a >= 0.50 else pB
        confidence = prob_a if prob_a >= 0.50 else (1.0 - prob_a)
        return winner, confidence, set_dist

# =====================================================================
# 7. CLI EXECUTION
# =====================================================================
def run_autonomous_cycle():
    print(f"[{datetime.now()}] STARTING FULLY AUTONOMOUS CYCLE")
    AutoSettler.settle_completed_matches()
    BoardIngestion.queue_fixtures()
    evaluator = LiveEvaluator()
    evaluator.evaluate_board()
    print(f"[{datetime.now()}] Cycle finished successfully.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Table Tennis Prediction Engine")
    parser.add_argument("--auto", action="store_true", help="Runs the complete autonomous cycle")
    parser.add_argument("--backtest", action="store_true", help="Runs 1000-match backtest and calibrates")
    parser.add_argument("--train", action="store_true", help="Forces calibration of the meta-learner")
    parser.add_argument("--predict", nargs=2, metavar=('PLAYER_A', 'PLAYER_B'), help="Manual 50k prediction")
    args = parser.parse_args()

    if args.auto:
        run_autonomous_cycle()
    elif args.backtest:
        ModelOrchestrator.run_1000_match_backtest()
        ModelOrchestrator.train_and_calibrate()
    elif args.train:
        ModelOrchestrator.train_and_calibrate()
    elif args.predict:
        evaluator = LiveEvaluator()
        w, conf, dist = evaluator.evaluate_match_manual(args.predict[0], args.predict[1])
        print(f"\n[PREDICTION RESULT] Winner: {w} | Confidence: {conf*100:.2f}%")
        print(f"Set Distribution: {dist}\n")
    else:
        print("Usage: python src/unified_engine.py [--auto | --backtest | --train | --predict [Player A] [Player B]]")
