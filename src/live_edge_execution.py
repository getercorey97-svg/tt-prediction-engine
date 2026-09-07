import pandas as pd
import numpy as np
import requests
import shin

# Updated import to match the new isolated file name
from tt_xgboost_engine import TableTennisXGBoost

class LiveEdgeExecution:
    def __init__(self, bankroll):
        self.bankroll = bankroll
        self.ntfy_topic = "geter_tt_alerts" 
        self.engine = TableTennisXGBoost()

    def process_live_odds(self, odds_player_a, odds_player_b):
        """
        Strips the bookmaker overround (margin) using the analytical binary solution 
        of Shin's Method to extract the true market implied probabilities.
        """
        pi_a = 1 / odds_player_a
        pi_b = 1 / odds_player_b
        
        margin = pi_a + pi_b - 1
        true_prob_a = pi_a - (margin / 2)
        true_prob_b = pi_b - (margin / 2)
        
        return true_prob_a, true_prob_b

    def calculate_kelly_criterion(self, prob_win, decimal_odds):
        """
        Determines the optimal percentage of the bankroll to wager based on 
        the mathematical edge over the market.
        """
        b = decimal_odds - 1 
        p = prob_win          
        q = 1 - p             
        
        kelly_fraction = (b * p - q) / b
        
        recommended_stake = max(0, (kelly_fraction * 0.25) * self.bankroll)
        return recommended_stake

    def push_android_notification(self, match_title, edge, stake, odds):
        """
        Pushes the actionable betting alert directly to the Galaxy S26 Ultra 
        via the ntfy.sh REST API.
        """
        message = (
            f"Edge Identified: +{edge}%\n"
            f"Recommended Stake: ${stake:.2f}\n"
            f"Target Odds: {odds}"
        )
        
        requests.post(
            f"https://ntfy.sh/{self.ntfy_topic}",
            data=message.encode('utf-8'),
            headers={
                "Title": f"TT Engine Alert: {match_title}",
                "Priority": "urgent",
                "Tags": "moneybag,ping_pong"
            }
        )
        print(f"Alert pushed to device for {match_title}")

    def evaluate_match(self, match_title, live_features, odds_a, odds_b):
        """Main execution flow for a live match."""
        market_prob_a, market_prob_b = self.process_live_odds(odds_a, odds_b)
        engine_prob_a = self.engine.load_and_predict(live_features)[0]
        
        if engine_prob_a > market_prob_a:
            edge_pct = (engine_prob_a - market_prob_a) * 100
            if edge_pct >= 2.5:  
                stake = self.calculate_kelly_criterion(engine_prob_a, odds_a)
                if stake > 0:
                    self.push_android_notification(match_title, round(edge_pct, 2), stake, odds_a)

