import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
import joblib

MODEL_ARTIFACT_PATH = "../data/xgb_model_calibrated.pkl"

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
        Transforms absolute player identities into relative strength differentials,
        incorporating Schedule Density (Fatigue) and micro-stylistic metrics[span_1](start_span)[span_1](end_span).
        """
        df['glicko_rating_diff'] = df['player_a_glicko'] - df['player_b_glicko']
        df['melo_vector_distance'] = np.linalg.norm(df['player_a_melo'] - df['player_b_melo'], axis=1)
        df['markov_match_win_prob_diff'] = df['player_a_markov'] - df['player_b_markov']
        
        df['age_diff'] = df['player_a_age'] - df['player_b_age']
        df['height_diff'] = df['player_a_height'] - df['player_b_height']
        df['handedness_interaction'] = df['player_a_hand'] - df['player_b_hand']
        
        # Schedule Density Differential (Sets played in past 48 hours)
        df['schedule_density_diff'] = df['player_a_sets_last_48h'] - df['player_b_sets_last_48h']
        
        df['wttr_position_diff'] = df['player_a_wttr_pos'] - df['player_b_wttr_pos']
        df['wttr_points_diff'] = df['player_a_wttr_points'] - df['player_b_wttr_points']
        
        df['home_continent_adv'] = df['player_a_home_continent'] - df['player_b_home_continent']
        df['recent_win_ratio_diff'] = df['player_a_recent_win_ratio'] - df['player_b_recent_win_ratio']
        
        features = [
            'glicko_rating_diff', 'melo_vector_distance', 'markov_match_win_prob_diff',
            'age_diff', 'height_diff', 'handedness_interaction', 'schedule_density_diff',
            'wttr_position_diff', 'wttr_points_diff',
            'home_continent_adv', 'recent_win_ratio_diff'
        ]
        return df[features], df['target_player_a_wins']

    def train_with_rolling_validation(self, X, y):
        """
        Implements sequential rolling-window cross-validation to prevent look-ahead bias[span_2](start_span)[span_2](end_span).
        """
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
        """
        Applies isotonic regression calibration for Brier Score and Log-Loss minimization[span_3](start_span)[span_3](end_span).
        """
        self.calibrated_model = CalibratedClassifierCV(self.model, method='isotonic', cv='prefit')
        self.model.fit(X, y)
        self.calibrated_model.fit(X, y)
        
        joblib.dump(self.calibrated_model, MODEL_ARTIFACT_PATH)
        print("Calibrated model artifact saved for edge execution.")

    def load_and_predict(self, X_live):
        if not self.calibrated_model:
            self.calibrated_model = joblib.load(MODEL_ARTIFACT_PATH)
        return self.calibrated_model.predict_proba(X_live)[:, 1]
