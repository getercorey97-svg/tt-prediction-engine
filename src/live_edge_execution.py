import pandas as pd
import requests
from tt_xgboost_engine import TableTennisXGBoost

class PureMathExecution:
    def __init__(self):
        self.ntfy_topic = "geter_tt_alerts" 
        self.engine = TableTennisXGBoost()

    def push_android_notification(self, match_title, projected_winner, win_prob):
        """
        Dynamically formats the notification based purely on algorithmic player mathematics.
        """
        confidence = win_prob * 100.0
        
        message = (f"🏓 PURE MATH PROJECTION 🏓\n"
                   f"Projected Winner: {projected_winner}\n"
                   f"Mathematical Confidence: {confidence:.2f}%")
        
        # Flag high-confidence algorithmic mismatches
        priority = "high" if confidence >= 65.0 else "default"
        tags = "crystal_ball,ping_pong"

        requests.post(
            f"https://ntfy.sh/{self.ntfy_topic}",
            data=message.encode('utf-8'),
            headers={
                "Title": f"Match: {match_title}",
                "Priority": priority,
                "Tags": tags
            }
        )
        print(f"Mathematical projection pushed for {match_title} | Winner: {projected_winner} ({confidence:.2f}%)")

    def evaluate_match_math(self, match_title, player_a_name, player_b_name, live_features):
        """
        Executes raw algorithmic predictions directly from XGBoost without market influence.
        """
        # Engine probability is strictly the likelihood of Player A winning
        engine_prob_a = self.engine.load_and_predict(live_features)[0]
        
        if engine_prob_a >= 0.50:
            projected_winner = player_a_name
            win_prob = engine_prob_a
        else:
            projected_winner = player_b_name
            win_prob = 1.0 - engine_prob_a
            
        self.push_android_notification(match_title, projected_winner, win_prob)
