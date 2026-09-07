import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta

class SportradarTTEngine:
    def __init__(self):
        # API Key must be stored securely in GitHub Secrets, not hardcoded
        self.api_key = os.environ.get("SPORTRADAR_API_KEY")
        if not self.api_key:
            raise ValueError("CRITICAL: SPORTRADAR_API_KEY environment variable is missing.")
        
        # Base URL for Sportradar Table Tennis v2 API
        self.base_url = "https://api.sportradar.com/table-tennis/trial/v2/en"
        self.headers = {"accept": "application/json"}
        self.rate_limit_delay = 1.2 

    def _make_request(self, endpoint):
        """Handles rate-limited GET requests to Sportradar."""
        url = f"{self.base_url}/{endpoint}?api_key={self.api_key}"
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            time.sleep(self.rate_limit_delay)
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"[{datetime.now()}] Sportradar API Error: {e}")
            return None

    def fetch_daily_summaries(self, target_date):
        """Polls the Daily Summaries endpoint for finalized matches on a specific date."""
        date_str = target_date.strftime("%Y-%m-%d")
        endpoint = f"schedules/{date_str}/summaries.json"
        
        print(f"[{datetime.now()}] Polling Sportradar for matches on {date_str}...")
        payload = self._make_request(endpoint)
        
        if not payload or 'summaries' not in payload:
            return []
            
        return payload['summaries']

    def parse_match_data(self, summary):
        """Extracts structural facts from the API payload."""
        sport_event = summary.get('sport_event', {})
        sport_event_status = summary.get('sport_event_status', {})
        
        if sport_event_status.get('status') != 'closed':
            return None

        competitors = sport_event.get('competitors', [])
        if len(competitors) != 2:
            return None

        player_a, player_b = competitors[0], competitors[1]
        score_a = sport_event_status.get('home_score', 0)
        score_b = sport_event_status.get('away_score', 0)
        tournament = sport_event.get('tournament', {})
        tier = tournament.get('category', {}).get('name', 'Unknown')

        return {
            "match_id": sport_event.get('id'),
            "date": sport_event.get('start_time', '').split('T')[0],
            "player_a_id": player_a.get('id'),
            "player_b_id": player_b.get('id'),
            "set_score_a": score_a,
            "set_score_b": score_b,
            "tournament_tier": tier,
        }

    def execute_factual_burn_in(self, target_match_count=2000):
        """Iterates backward chronologically to extract 2,000 empirical matches."""
        historical_matches = []
        current_date = datetime.now()

        print(f"[{datetime.now()}] Commencing factual historical extraction from Sportradar...")
        
        while len(historical_matches) < target_match_count:
            summaries = self.fetch_daily_summaries(current_date)
            
            for summary in summaries:
                parsed_match = self.parse_match_data(summary)
                if parsed_match:
                    historical_matches.append(parsed_match)
                    if len(historical_matches) >= target_match_count:
                        break
            
            current_date -= timedelta(days=1)
            
        print(f"[{datetime.now()}] Successfully extracted {len(historical_matches)} empirical matches.")
        return pd.DataFrame(historical_matches)

if __name__ == "__main__":
    engine = SportradarTTEngine()
