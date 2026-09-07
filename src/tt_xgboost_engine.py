import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
import joblib

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
MODEL_ARTIFACT_PATH = os.path.join(DATA_DIR, "xgb_model_calibrated.pkl")

class TableTennisXGBoost:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            learning_rate=0.05,
            max_depth=5,
            n_estimators=300,
            subsample=0.8,
            colsample_bytree=0.8
        )
        self.calibrated_model = None

    def engineer_differentials(self, df):
        """
        Safely transforms and aligns relative strength differentials,
        incorporating Schedule Density (Fatigue) and stylistic metrics without key errors.
        """
        df = df.copy()
        
        expected_features = [
            'glicko_rating_diff', 'melo_vector_distance', 'markov_match_win_prob_diff',
            'age_diff', 'height_diff', 'handedness_interaction', 'schedule_density_diff',
            'wttr_pos_diff', 'wttr_points_diff',
            'home_continent_adv', 'recent_win_ratio_diff'
        ]
        
        # Ensure all features exist safely in the dataframe
        for feat in expected_features:
            if feat not in df.columns:
                if feat == 'glicko_rating_diff' and 'player_a_glicko' in df.columns and 'player_b_glicko' in df.columns:
                    df[feat] = df['player_a_glicko'] - df['player_b_glicko']
                elif feat == 'schedule_density_diff' and 'player_a_sets_last_48h' in df.columns and 'player_b_sets_last_48h' in df.columns:
                    df[feat] = df['player_a_sets_last_48h'] - df['player_b_sets_last_48h']
                elif feat == 'wttr_pos_diff' and 'player_a_wttr_pos' in df.columns and 'player_b_wttr_pos' in df.columns:
                    df[feat] = df['player_a_wttr_pos'] - df['player_b_wttr_pos']
                else:
                    df[feat] = 0.0
                    
        target = df['target_player_a_wins'] if 'target_player_a_wins' in df.columns else pd.Series([1] * len(df))
        return df[expected_features], target

    def train_with_rolling_validation(self, X, y):
        tscv = TimeSeriesSplit(n_splits=5)
        brier_scores = []
        log_losses = []

        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            self.model.fit(X_train, y_train)
            preds = self.model.predict_proba(X_test)[:, 1]
            
            brier_scores.append(brier_score_loss(y_test, preds))
            log_losses.append(log_loss(y_test, preds))

        print(f"Average Brier Score: {np.mean(brier_scores):.5f}")
        print(f"Average Log-Loss: {np.mean(log_losses):.5f}")

    def calibrate_and_save(self, X, y):
        os.makedirs(DATA_DIR, exist_ok=True)
        self.calibrated_model = CalibratedClassifierCV(self.model, method='isotonic', cv=3)
        self.calibrated_model.fit(X, y)
        
        joblib.dump(self.calibrated_model, MODEL_ARTIFACT_PATH)
        print(f"Calibrated model artifact saved to: {MODEL_ARTIFACT_PATH}")

    def load_and_predict(self, X_live):
        if not self.calibrated_model:
            if os.path.exists(MODEL_ARTIFACT_PATH):
                self.calibrated_model = joblib.load(MODEL_ARTIFACT_PATH)
            else:
                raise FileNotFoundError(f"Model artifact not found at {MODEL_ARTIFACT_PATH}. Run seeding protocol first.")
        
        # Ensure live input goes through feature alignment
        X_processed, _ = self.engineer_differentials(X_live)
        return self.calibrated_model.predict_proba(X_processed)[:, 1]
