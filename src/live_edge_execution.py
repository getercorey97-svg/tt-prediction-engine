import pandas as pd
import numpy as np
import requests
import shin
from tt_xgboost_engine import TableTennisXGBoost

class LiveEdgeExecution:
    def __init__(self, bankroll):
        self.bankroll = bankroll
        self.ntfy_topic = "geter_tt_alerts" 
        self.engine = TableTennisXGBoost()

    def process_live_odds(self, odds_player_a, odds_player_b):
        """
        Strips bookmaker margin using Shin's method analytical binary solution.
        """
        pi_a = 1 / odds_player_a
        pi_b = 1 / odds_player_b
        
        margin = pi_a + pi_b - 1
        true_prob_a = pi_a - (margin / 2)
        true_prob_b = pi_b - (margin / 2)
        
        return true_prob_a, true_prob_b

    def calculate_kelly_criterion(self, prob_win, decimal_odds):
        """
        Determines optimal bankroll allocation via Kelly Criterion.
        """
        b = decimal_odds - 1 
        p = prob_win          
        q = 1 - p             
        
        kelly_fraction = (b * p - q) / b
        recommended_stake = max(0, (kelly_fraction * 0.25) * self.bankroll)
        return recommended_stake

    def push_android_notification(self, match_title, edge, stake, odds):
        """
        Dynamically formats the notification based on whether an edge exists.
        """
        if edge > 0:
            message = f"✅ EDGE FOUND: +{edge}%\nRecommended Stake: ${stake:.2f}\nTarget Odds: {odds}"
            priority = "high"
            tags = "moneybag,ping_pong"
        else:
            message = f"❌ NO EDGE: {edge}%\nAvoid Betting.\nTarget Odds: {odds}"
            priority = "default"
            tags = "no_entry,ping_pong"

        requests.post(
            f"https://ntfy.sh/{self.ntfy_topic}",
            data=message.encode('utf-8'),
            headers={
                "Title": f"TT Alert: {match_title}",
                "Priority": priority,
                "Tags": tags
            }
        )
        print(f"Alert pushed to device for {match_title} | Edge: {edge}%")

    def evaluate_match(self, match_title, live_features, odds_a, odds_b, previous_odds_a=None):
        """Main execution flow pushing alerts for ALL matches unconditionally."""
        market_prob_a, market_prob_b = self.process_live_odds(odds_a, odds_b)
        engine_prob_a = self.engine.load_and_predict(live_features)[0]
        
        line_velocity = 0.0
        if previous_odds_a is not None:
            line_velocity = odds_a - previous_odds_a  # Positive means market fading
            
        adjusted_edge_modifier = -0.01 if line_velocity > 0.15 else 0.0
        
        # Calculate exact edge, regardless of whether it is positive or negative
        raw_edge_pct = ((engine_prob_a - market_prob_a) + adjusted_edge_modifier) * 100
        
        # If edge is positive, calculate stake. Otherwise, recommend $0.
        if raw_edge_pct > 0:
            stake = self.calculate_kelly_criterion(engine_prob_a, odds_a)
        else:
            stake = 0.0
            
        # Unconditionally push the notification for ALL evaluated matches
        self.push_android_notification(match_title, round(raw_edge_pct, 2), stake, odds_a)
