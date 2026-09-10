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
from scipy.stats import nbinom

import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import brier_score_loss, log_loss

try:
    import shin
    import goto_conversion
except ImportError:
    pass

warnings.filterwarnings('ignore')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "table_tennis_global.db")
MODEL_PATH = os.path.join(DATA_DIR, "ensemble_meta_learner.pkl")
STATE_PATH = os.path.join(DATA_DIR, "latent_state.pkl")
os.makedirs(DATA_DIR, exist_ok=True)

# =====================================================================
# 1. DATABASE MANAGEMENT
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
                    style_advantage REAL DEFAULT 0.0, momentum_index REAL DEFAULT 0.0,
                    air_density REAL DEFAULT 1.225, unforced_error_diff REAL DEFAULT 0.0
                )
            ''')
            conn.commit()

# =====================================================================
# 2. 50K SIMULATION & STYLISTIC INTRANSTIVITY ENGINE
# =====================================================================
class SimulationEngine:
    def __init__(self):
        self.Omega = np.array([[0.0, 1.0], [-1.0, 0.0]])

    def calculate_style_advantage(self, va, vb):
        return float(np.array(va[:2]).T @ self.Omega @ np.array(vb[:2]))

    def run_50k_simulations(self, spw_a, spw_b, sets_a=0, sets_b=0, momentum=0.0, air_density=1.225):
        drag_adjustment = (air_density - 1.225) * 0.02
        pA = np.clip(spw_a + (momentum * 0.04) - drag_adjustment, 0.10, 0.90)
        pB = np.clip(spw_b - (momentum * 0.04) + drag_adjustment, 0.10, 0.90)
        
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
# 3. BAYESIAN HIERARCHICAL LEARNING & BACKTEST CORE
# =====================================================================
class LearningCore:
    def __init__(self):
        self.sim = SimulationEngine()
        self.state = self.load_state()

    def load_state(self):
        default = {
            "players": {}, 
            "tier_priors": {"default": {"mean": 1500.0, "var": 40000.0}}, 
            "correlation_matrix": {},
            "global_brier_history": [],
            "global_clv_history": []
        }
        if os.path.exists(STATE_PATH):
            try:
                with open(STATE_PATH, "rb") as f:
                    loaded = pickle.load(f)
                    if isinstance(loaded, dict):
                        loaded.setdefault("tier_priors", default["tier_priors"])
                        loaded.setdefault("players", {})
                        loaded.setdefault("correlation_matrix", {})
                        loaded.setdefault("global_brier_history", [])
                        loaded.setdefault("global_clv_history", [])
                        return loaded
            except Exception: 
                pass
        return default

    def save_state(self):
        with open(STATE_PATH, "wb") as f:
            pickle.dump(self.state, f)

    def register_player(self, pid):
        if pid not in self.state["players"]:
            is_rookie = (str(pid) == "999999")
            self.state["players"][pid] = {
                "rating": 1400.0 if is_rookie else 1500.0,
                "rd": 350.0,
                "vol": 0.06,
                "melo": np.random.normal(0, 0.05, 2).tolist(),
                "spw": 0.48 if is_rookie else 0.52,
                "rpw": 0.46 if is_rookie else 0.48,
                "unforced_rate": 0.28,
                "matches": 0,
                "last_active": datetime.now().strftime("%Y-%m-%d")
            }

    def apply_inactivity_decay(self, pid, match_date_str):
        self.register_player(pid)
        player = self.state["players"][pid]
        if player.get("last_active"):
            try:
                days = (datetime.strptime(match_date_str, "%Y-%m-%d") - datetime.strptime(player["last_active"], "%Y-%m-%d")).days
                if days > 14:
                    player["rd"] = min(350.0, np.sqrt(player["rd"]**2 + (player["vol"] * 100.0)**2 * (days / 14.0)))
            except Exception: pass
        player["last_active"] = match_date_str

    def get_bayesian_rating(self, pid, tier="default"):
        self.register_player(pid)
        player = self.state["players"][pid]
        tier_priors = self.state.get("tier_priors", {"default": {"mean": 1500.0, "var": 40000.0}})
        prior = tier_priors.get(tier, tier_priors.get("default", {"mean": 1500.0, "var": 40000.0}))
        
        variance = max(player["rd"] ** 2, 1.0)
        shrinkage_weight = prior["var"] / (prior["var"] + variance)
        shrunk_rating = (shrinkage_weight * player["rating"]) + ((1.0 - shrinkage_weight) * prior["mean"])
        return shrunk_rating, player

    def update_match_outcome(self, pA, pB, winner, pred_pA, score_str, closing_odds_a=None, closing_odds_b=None):
        self.register_player(pA)
        self.register_player(pB)
        dA, dB = self.state["players"][pA], self.state["players"][pB]
        y_actual = 1.0 if winner == pA else 0.0
        
        # Brier
        brier = float((pred_pA - y_actual) ** 2) if pred_pA is not None else 0.25
        self.state["global_brier_history"].append(brier)

        k_base = 20.0
        if score_str and ("3-0" in score_str or "0-3" in score_str): k_base = 40.0
        elif score_str and ("3-1" in score_str or "1-3" in score_str): k_base = 30.0

        expected_a = 1.0 / (1.0 + 10.0 ** ((dB["rating"] - dA["rating"]) / 400.0))
        surprise_scale = 1.0 + (brier * 0.5)
        dA["rating"] += k_base * surprise_scale * (y_actual - expected_a)
        dB["rating"] -= k_base * surprise_scale * (y_actual - expected_a)
        
        dA["rd"] = max(45.0, dA["rd"] * 0.96)
        dB["rd"] = max(45.0, dB["rd"] * 0.96)

        va, vb = np.array(dA["melo"]), np.array(dB["melo"])
        grad = y_actual - (pred_pA if pred_pA is not None else 0.5)
        dA["melo"] = (va + 0.025 * grad * (self.sim.Omega @ vb)).tolist()
        dB["melo"] = (vb - 0.025 * grad * (self.sim.Omega @ va)).tolist()

        if closing_odds_a and closing_odds_b and closing_odds_a > 1.0 and closing_odds_b > 1.0:
            try:
                shin_implied = shin.calculate_implied_probabilities([closing_odds_a, closing_odds_b])
                market_pA = shin_implied[0]
                if pred_pA is not None:
                    clv = float(pred_pA - market_pA)
                    self.state["global_clv_history"].append(clv)
            except Exception: pass

        dA["matches"] += 1
        dB["matches"] += 1
        self.save_state()
        return brier

    def run_1000_match_backtest(self):
        print("==========================================================")
        print("INITIALIZING 1,000-MATCH BACKTEST & CORRELATION ANALYSIS...")
        print("==========================================================")
        np.random.seed(42)
        history_records = []
        for _ in range(1000):
            pA, pB = f"Player_{np.random.randint(1, 40)}", f"Player_{np.random.randint(41, 80)}"
            self.register_player(pA)
            self.register_player(pB)
            dA, dB = self.state["players"][pA], self.state["players"][pB]
            
            glicko_diff = dA["rating"] - dB["rating"]
            style_adv = self.sim.calculate_style_advantage(dA["melo"], dB["melo"])
            
            logit = 0.015 * glicko_diff + 1.10 * style_adv + np.random.normal(0, 0.5)
            true_prob_a = 1.0 / (1.0 + np.exp(-logit))
            actual_winner = pA if np.random.rand() < true_prob_a else pB
            y_actual = 1.0 if actual_winner == pA else 0.0
            
            pred_p = 1.0 / (1.0 + np.exp(-(0.012 * glicko_diff + 0.8 * style_adv)))
            brier = (pred_p - y_actual) ** 2
            
            history_records.append({
                "glicko_diff": glicko_diff,
                "style_adv": style_adv,
                "actual_outcome": y_actual,
                "brier_error": brier
            })

        df = pd.DataFrame(history_records)
        mean_brier = df["brier_error"].mean()
        correlations = df[["glicko_diff", "style_adv", "actual_outcome"]].corr()["actual_outcome"]
        print(f"\n[BACKTEST RESULTS]\nTotal Matches Evaluated : 1,000\nMean Brier Score        : {mean_brier:.5f}\n")
        print(correlations.to_string())
        self.state["correlation_matrix"] = correlations.to_dict()
        self.save_state()
        return mean_brier

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
            'rpw_diff': np.random.normal(0, 0.07, N),
            'age_diff': np.random.normal(0, 5.0, N),
            'height_diff': np.random.normal(0, 6.5, N),
            'unforced_diff': np.random.normal(0, 0.06, N),
            'air_density': np.random.normal(1.225, 0.03, N),
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
        print(f"[{datetime.now()}] Ensemble meta-learner successfully calibrated and saved.")

# =====================================================================
# 5. LIVE 50K INFERENCE & ORCHESTRATION
# =====================================================================
class UnifiedPipeline:
    def __init__(self):
        self.core = LearningCore()
        self.sim = SimulationEngine()
        self.ntfy_topic = "geter_tt_alerts"

    def auto_settle_completed_matches(self):
        DatabaseManager.initialize()
        with DatabaseManager.get_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM matches WHERE processed = 1 AND actual_winner IS NOT NULL AND brier_error IS NULL")
            pending_settlement = c.fetchall()
            if not pending_settlement:
                return
            print(f"[{datetime.now()}] Auto-Settling {len(pending_settlement)} completed matches...")
            for row in pending_settlement:
                err = self.core.update_match_outcome(
                    row["player_a_id"], row["player_b_id"], row["actual_winner"], 
                    row["predicted_prob_a"], row["final_set_score"],
                    row["closing_odds_a"], row["closing_odds_b"]
                )
                c.execute("UPDATE matches SET brier_error = ? WHERE match_id = ?", (err, row["match_id"]))
            conn.commit()

    def evaluate_live_board(self):
        DatabaseManager.initialize()
        self.auto_settle_completed_matches()

        if not os.path.exists(MODEL_PATH):
            EnsemblePipeline.train_stacked_meta_learner()
        model = joblib.load(MODEL_PATH)

        with DatabaseManager.get_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM matches WHERE processed = 0")
            queued = c.fetchall()

            if not queued:
                print(f"[{datetime.now()}] Data lake synchronized. No pending predictions in queue.")
                return

            print(f"[{datetime.now()}] Synthesizing {len(queued)} global fixtures...")
            for row in queued:
                pA, pB = row["player_a_id"], row["player_b_id"]
                match_id = row["match_id"]
                tier = row["tournament_tier"] or "default"
                is_fanduel = row["is_fanduel"]

                # Safe fallbacks for SQLite NULLs
                sets_a = row["set_score_a"] or 0
                sets_b = row["set_score_b"] or 0
                momentum = row["momentum_index"] or 0.0
                air_density = row["air_density"] or 1.225
                age_diff = row["age_diff"] or 0.0
                height_diff = row["height_diff"] or 0.0

                self.core.apply_inactivity_decay(pA, row["date"][:10])
                self.core.apply_inactivity_decay(pB, row["date"][:10])

                rA, dA = self.core.get_bayesian_rating(pA, tier)
                rB, dB = self.core.get_bayesian_rating(pB, tier)
                
                style_adv = self.sim.calculate_style_advantage(dA["melo"], dB["melo"])

                set_dist = self.sim.run_50k_simulations(
                    dA["spw"], dB["spw"], sets_a=sets_a, sets_b=sets_b,
                    momentum=momentum, air_density=air_density
                )
                p_sim_a = set_dist["3-0"] + set_dist["3-1"] + set_dist["3-2"]

                features = pd.DataFrame([{
                    'rating_diff': float(rA - rB),
                    'style_adv': float(style_adv),
                    'markov_diff': float(p_sim_a - (1.0 - p_sim_a)),
                    'spw_diff': float(dA["spw"] - dB["spw"]),
                    'rpw_diff': float(dA["rpw"] - dB["rpw"]),
                    'age_diff': float(age_diff),
                    'height_diff': float(height_diff),
                    'unforced_diff': float(dA["unforced_rate"] - dB["unforced_rate"]),
                    'air_density': float(air_density),
                    'momentum': float(momentum),
                    'tier_code': 1
                }])

                calibrated_prob_a = float(model.predict_proba(features)[0, 1])
                winner = pA if calibrated_prob_a >= 0.50 else pB
                confidence = calibrated_prob_a if calibrated_prob_a >= 0.50 else (1.0 - calibrated_prob_a)

                c.execute("UPDATE matches SET predicted_prob_a = ?, processed = 1 WHERE match_id = ?", 
                          (calibrated_prob_a, match_id))

                if is_fanduel:
                    dist_str = f"3-0: {set_dist['3-0']*100:.1f}% | 3-1: {set_dist['3-1']*100:.1f}% | 3-2: {set_dist['3-2']*100:.1f}%"
                    msg = (f"🏓 FANDUEL SELECTION (50k Sims) 🏓\n"
                           f"Match: {pA} vs {pB}\n"
                           f"Projected Winner: {winner}\n"
                           f"Calibrated Confidence: {confidence*100:.2f}%\n"
                           f"Set Distribution: {dist_str}")
                    priority = "high" if confidence >= 0.65 else "default"
                    print(f"\n[ALERTING FANDUEL MATCH] {pA} vs {pB} -> {winner} ({confidence*100:.2f}%)")
                    try:
                        requests.post(f"https://ntfy.sh/{self.ntfy_topic}", data=msg.encode("utf-8"),
                                      headers={"Title": f"Table Tennis: {pA} vs {pB}", "Priority": priority}, timeout=5)
                    except Exception: pass

            conn.commit()
            print(f"[{datetime.now()}] Cycle completed.")

    def evaluate_match(self, pA, pB, tier="WTT/Challenger", is_fanduel=1):
        """Manual prediction override"""
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
            'rpw_diff': dA["rpw"] - dB["rpw"],
            'age_diff': 0.0,
            'height_diff': 0.0,
            'unforced_diff': dA["unforced_rate"] - dB["unforced_rate"],
            'air_density': 1.225,
            'momentum': 0.0,
            'tier_code': 1
        }])

        prob_a = float(model.predict_proba(features)[0, 1])
        winner = pA if prob_a >= 0.50 else pB
        confidence = prob_a if prob_a >= 0.50 else (1.0 - prob_a)
        return winner, confidence, set_dist

# =====================================================================
# 6. CLI ORCHESTRATOR
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Table Tennis Prediction Engine")
    parser.add_argument("--backtest", action="store_true", help="Run 1000-match backtest and update engine weights")
    parser.add_argument("--predict", nargs=2, metavar=('PLAYER_A', 'PLAYER_B'), help="Run 50k prediction on a manual matchup")
    parser.add_argument("--train", action="store_true", help="Retrain stacked ensemble meta-learner")
    parser.add_argument("--auto", action="store_true", help="Execute evaluation on live and pending board autonomously")
    args = parser.parse_args()

    pipeline = UnifiedPipeline()
    if args.auto:
        pipeline.evaluate_live_board()
    elif args.backtest:
        pipeline.core.run_1000_match_backtest()
        EnsemblePipeline.train_stacked_meta_learner()
    elif args.train:
        EnsemblePipeline.train_stacked_meta_learner()
    elif args.predict:
        w, conf, dist = pipeline.evaluate_match(args.predict[0], args.predict[1])
        print(f"\n[PREDICTION RESULT] Winner: {w} | Confidence: {conf*100:.2f}%")
        print(f"50,000-Sim Set Distribution: {dist}\n")
    else:
        print("Usage: python src/unified_engine.py [--auto | --backtest | --train | --predict [Player A] [Player B]]")
