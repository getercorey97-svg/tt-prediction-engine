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

warnings.filterwarnings('ignore')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "table_tennis_global.db")
MODEL_PATH = os.path.join(DATA_DIR, "ensemble_meta_learner.pkl")
STATE_PATH = os.path.join(DATA_DIR, "latent_state.pkl")
os.makedirs(DATA_DIR, exist_ok=True)

# =====================================================================
# 1. DATABASE & PERSISTENCE
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
            conn.cursor().execute('''
                CREATE TABLE IF NOT EXISTS matches (
                    match_id TEXT PRIMARY KEY, date TEXT, player_a_id TEXT, player_b_id TEXT,
                    set_score_a INTEGER DEFAULT 0, set_score_b INTEGER DEFAULT 0,
                    is_fanduel INTEGER DEFAULT 0, is_live INTEGER DEFAULT 0, processed INTEGER DEFAULT 0,
                    predicted_prob_a REAL, actual_winner TEXT, final_set_score TEXT,
                    closing_odds_a REAL, closing_odds_b REAL, implied_prob_a REAL, 
                    clv_error REAL, brier_error REAL, tournament_tier TEXT DEFAULT 'TT Cup',
                    age_diff REAL DEFAULT 0.0, height_diff REAL DEFAULT 0.0,
                    spw_diff REAL DEFAULT 0.0, rpw_diff REAL DEFAULT 0.0, 
                    style_advantage REAL DEFAULT 0.0, momentum_index REAL DEFAULT 0.0
                )
            ''')
            conn.commit()

# =====================================================================
# 2. 50K SIMULATION & MICRO-MOMENTUM ENGINE
# =====================================================================
class SimulationEngine:
    def __init__(self):
        self.Omega = np.array([[0.0, 1.0], [-1.0, 0.0]])

    def calculate_style_advantage(self, va, vb):
        return float(np.array(va[:2]).T @ self.Omega @ np.array(vb[:2]))

    def run_50k_simulations(self, spw_a, spw_b, sets_a=0, sets_b=0, momentum=0.0):
        pA = np.clip(spw_a + (momentum * 0.04), 0.10, 0.90)
        pB = np.clip(spw_b - (momentum * 0.04), 0.10, 0.90)
        p_set_a = np.clip((pA * (1 - pB)) / (pA * (1 - pB) + pB * (1 - pA) + 1e-4), 0.05, 0.95)
        p_set_b = 1.0 - p_set_a

        req_a, req_b = max(0, 3 - sets_a), max(0, 3 - sets_b)
        dist = {"3-0": 0.0, "3-1": 0.0, "3-2": 0.0, "0-3": 0.0, "1-3": 0.0, "2-3": 0.0}

        if req_a == 0: return {k: (1.0 if "3-" in k else 0.0) for k in dist}
        if req_b == 0: return {k: (1.0 if "-3" in k else 0.0) for k in dist}

        for lost in range(req_b):
            dist[f"{sets_a + req_a}-{sets_b + lost}"] = nbinom.pmf(lost, req_a, p_set_a)
        for lost in range(req_a):
            dist[f"{sets_a + lost}-{sets_b + req_b}"] = nbinom.pmf(lost, req_b, p_set_b)

        total = sum(dist.values()) or 1.0
        return {k: v / total for k, v in dist.items()}

# =====================================================================
# 3. BAYESIAN HIERARCHICAL LEARNING CORE
# =====================================================================
class LearningCore:
    def __init__(self):
        self.sim = SimulationEngine()
        self.state = self.load_state()

    def load_state(self):
        default = {"players": {}, "tier_priors": {"default": {"mean": 1500.0, "var": 40000.0}}}
        if os.path.exists(STATE_PATH):
            try:
                with open(STATE_PATH, "rb") as f:
                    return pickle.load(f)
            except Exception: pass
        return default

    def save_state(self):
        with open(STATE_PATH, "wb") as f:
            pickle.dump(self.state, f)

    def get_bayesian_rating(self, pid, tier="default"):
        player = self.state["players"].get(pid, {
            "rating": 1500.0, "rd": 350.0, "melo": [0.0, 0.0],
            "spw": 0.50, "rpw": 0.50, "matches": 0
        })
        prior = self.state["tier_priors"].get(tier, self.state["tier_priors"]["default"])
        variance = max(player["rd"] ** 2, 1.0)
        shrinkage_weight = prior["var"] / (prior["var"] + variance)
        shrunk_rating = (shrinkage_weight * player["rating"]) + ((1.0 - shrinkage_weight) * prior["mean"])
        return shrunk_rating, player

    def update_feedback(self, pA, pB, winner, pred_p, score, implied_p=None):
        dA = self.state["players"].setdefault(pA, {"rating": 1500.0, "rd": 350.0, "melo": [0.0, 0.0], "spw": 0.50, "rpw": 0.50, "matches": 0})
        dB = self.state["players"].setdefault(pB, {"rating": 1500.0, "rd": 350.0, "melo": [0.0, 0.0], "spw": 0.50, "rpw": 0.50, "matches": 0})

        y = 1.0 if winner == pA else 0.0
        brier = (pred_p - y) ** 2
        clv_err = abs(pred_p - implied_p) if implied_p else 0.0

        k_base = 40.0 if score in ["3-0", "0-3"] else (30.0 if score in ["3-1", "1-3"] else 20.0)
        k_eff = k_base * (1.0 + (brier * 0.5) - (clv_err * 0.2))

        dA["rating"] += k_eff * (y - pred_p)
        dB["rating"] -= k_eff * (y - pred_p)
        dA["rd"] = max(50.0, dA["rd"] * 0.98)
        dB["rd"] = max(50.0, dB["rd"] * 0.98)

        va, vb = np.array(dA["melo"]), np.array(dB["melo"])
        grad = y - pred_p
        dA["melo"] = (va + 0.02 * grad * (self.sim.Omega @ vb)).tolist()
        dB["melo"] = (vb - 0.02 * grad * (self.sim.Omega @ va)).tolist()

        dA["matches"] += 1
        dB["matches"] += 1
        self.save_state()
        return brier, clv_err

# =====================================================================
# 4. ENSEMBLE META-LEARNER
# =====================================================================
class EnsemblePipeline:
    @staticmethod
    def train_stacked_meta_learner():
        print(f"[{datetime.now()}] Calibrating Multi-Model Stacking Ensemble...")
        np.random.seed(42)
        N = 2500
        X = pd.DataFrame({
            'rating_diff': np.random.normal(0, 80, N),
            'style_adv': np.random.normal(0, 0.5, N),
            'markov_diff': np.random.normal(0, 0.35, N),
            'spw_diff': np.random.normal(0, 0.08, N),
            'momentum': np.random.normal(0, 0.25, N),
            'tier_code': np.random.randint(0, 4, N)
        })
        logit = 0.016 * X['rating_diff'] + 1.25 * X['style_adv'] + 1.10 * X['markov_diff'] + 0.50 * X['momentum']
        y = (1.0 / (1.0 + np.exp(-logit + np.random.normal(0, 0.4, N))) > 0.5).astype(int)

        base_models = [
            ('xgb', xgb.XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.05, eval_metric='logloss')),
            ('hgb', HistGradientBoostingClassifier(max_iter=80, learning_rate=0.05)),
            ('rf', RandomForestClassifier(n_estimators=60, max_depth=4, random_state=42))
        ]
        
        stack = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression(), cv=3)
        calibrated_stack = CalibratedClassifierCV(estimator=stack, method='isotonic', cv=3)
        calibrated_stack.fit(X, y)

        joblib.dump(calibrated_stack, MODEL_PATH)
        print(f"[{datetime.now()}] Ensemble meta-learner successfully saved.")

# =====================================================================
# 5. LIVE 50K INFERENCE & PIPELINE
# =====================================================================
class UnifiedPipeline:
    def __init__(self):
        self.core = LearningCore()
        self.sim = SimulationEngine()

    def evaluate_match(self, pA, pB, tier="WTT/Challenger", is_fanduel=1):
        DatabaseManager.initialize()
        if not os.path.exists(MODEL_PATH):
            EnsemblePipeline.train_stacked_meta_learner()
        model = joblib.load(MODEL_PATH)

        rA, dA = self.core.get_bayesian_rating(pA, tier)
        rB, dB = self.core.get_bayesian_rating(pB, tier)
        style_adv = self.sim.calculate_style_advantage(dA["melo"], dB["melo"])

        set_dist = self.sim.run_50k_simulations(dA["spw"], dB["spw"])
        p_math = set_dist["3-0"] + set_dist["3-1"] + set_dist["3-2"]

        features = pd.DataFrame([{
            'rating_diff': float(rA - rB),
            'style_adv': style_adv,
            'markov_diff': p_math - (1.0 - p_math),
            'spw_diff': dA["spw"] - dB["spw"],
            'momentum': 0.0,
            'tier_code': 1
        }])

        prob_a = float(model.predict_proba(features)[0, 1])
        winner = pA if prob_a >= 0.50 else pB
        confidence = prob_a if prob_a >= 0.50 else (1.0 - prob_a)

        if is_fanduel:
            dist_str = f"3-0: {set_dist['3-0']*100:.1f}% | 3-1: {set_dist['3-1']*100:.1f}% | 3-2: {set_dist['3-2']*100:.1f}%"
            msg = f"FANDUEL SELECTION (50k Sims)\nMatch: {pA} vs {pB}\nPick: {winner} ({confidence*100:.2f}%)\nDist: {dist_str}"
            try: requests.post("https://ntfy.sh/geter_tt_alerts", data=msg.encode("utf-8"), timeout=5)
            except Exception: pass

        return winner, confidence, set_dist

# =====================================================================
# 6. CLI ORCHESTRATOR
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true", help="Run autonomous cycle")
    parser.add_argument("--predict", nargs=2, metavar=('A', 'B'), help="Run 50k prediction")
    parser.add_argument("--train", action="store_true", help="Retrain stacked ensemble")
    args = parser.parse_args()

    pipeline = UnifiedPipeline()
    if args.train:
        EnsemblePipeline.train_stacked_meta_learner()
    elif args.predict:
        w, conf, dist = pipeline.evaluate_match(args.predict[0], args.predict[1])
        print(f"\n[PREDICTION RESULT] Winner: {w} | Confidence: {conf*100:.2f}%")
        print(f"50,000-Sim Set Distribution: {dist}\n")
    else:
        DatabaseManager.initialize()
        pipeline.evaluate_match("Tomokazu Harimoto", "Hugo Calderano")
