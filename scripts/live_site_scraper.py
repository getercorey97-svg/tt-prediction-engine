import os
import sqlite3
import requests
from datetime import datetime
import numpy as np

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/table_tennis_global.db"))

class UniversalMatchScraper:
    def __init__(self):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        self.conn = sqlite3.connect(DB_PATH)
        
        # Replace these with your actual odds/schedule API endpoints
        self.rankings_api_url = "https://api.example.com/v1/ittf/rankings" 
        self.upcoming_api_url = "https://api.example.com/v1/fixtures/upcoming"
        
        self.api_headers = {"Authorization": "Bearer YOUR_API_KEY_HERE", "Content-Type": "application/json"}
        self.rankings_cache = {}
        
        self.setup_database()
        self.preload_rankings()

    def setup_database(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS matches (
                match_id TEXT PRIMARY KEY, date TEXT, player_a_id TEXT, player_b_id TEXT,
                set_score_a INTEGER, set_score_b INTEGER, processed INTEGER DEFAULT 0,
                age_diff REAL, handedness_interaction INTEGER, height_diff REAL,
                schedule_density_diff REAL, wttr_pos_diff REAL, wttr_points_diff REAL,
                tournament_tier TEXT, home_continent_adv INTEGER, recent_win_ratio_diff REAL, 
                grip_interaction INTEGER, style_interaction INTEGER, is_live INTEGER DEFAULT 0
            )
        ''')
        self.conn.commit()

    def preload_rankings(self):
        print(f"[{datetime.now()}] Pulling live ITTF rankings via JSON API...")
        try:
            response = requests.get(self.rankings_api_url, headers=self.api_headers, timeout=5)
            if response.status_code == 200:
                for player in response.json().get('data', []):
                    self.rankings_cache[player.get('player_name')] = {
                        "rank": player.get('rank', 200),
                        "points": player.get('points', 150),
                        "age": player.get('age', 25), 
                        "hand": 1 if player.get('handedness', 'Right') == 'Right' else -1,
                        "height": player.get('height', 175),
                        "grip": player.get('grip', 1),
                        "style": player.get('style', 1),
                        "recent_form": player.get('recent_win_ratio', 0.50)
                    }
        except requests.exceptions.RequestException:
            pass

    def get_factual_player_data(self, player_name):
        if player_name in self.rankings_cache:
            return self.rankings_cache[player_name]
        
        # Baseline fallback mapped to prevent cold-start anomalies
        return {"rank": 200, "points": 150, "age": 25, "hand": 1, "height": 175, "grip": 1, "style": 1, "recent_form": 0.50}

    def fetch_global_board(self):
        print(f"[{datetime.now()}] Scanning global networks for Upcoming scheduled fixtures...")
        matches = []
        current_date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        try:
            # Polling the upcoming schedule API
            response = requests.get(self.upcoming_api_url, headers=self.api_headers, timeout=10)
            
            if response.status_code == 200:
                # Assuming the API returns a JSON list of scheduled events
                upcoming_data = response.json().get('events', [])
                
                for event in upcoming_data:
                    # Filter for matches that have strictly not started
                    if event.get('status') not in ['scheduled', 'not_started']:
                        continue
                        
                    pA = event.get('player_a')
                    pB = event.get('player_b')
                    tier = event.get('tournament_tier', 'Unknown')
                    
                    data_A = self.get_factual_player_data(pA)
                    data_B = self.get_factual_player_data(pB)
                    
                    matches.append({
                        "match_id": f"UPCOMING_{pA[:3]}_{pB[:3]}_{event.get('start_time', '0000')}_{np.random.randint(1000,9999)}",
                        "date": current_date_str,
                        "player_a_id": pA,
                        "player_b_id": pB,
                        "set_score_a": 0, # Neutral 0-0 state for pre-match baseline
                        "set_score_b": 0,
                        "tournament_tier": tier,
                        "age_diff": float(data_A["age"] - data_B["age"]),
                        "handedness_interaction": 1 if data_A["hand"] != data_B["hand"] else 0,
                        "height_diff": float(data_A["height"] - data_B["height"]),
                        "schedule_density_diff": 0.0,
                        "wttr_pos_diff": float(data_B["rank"] - data_A["rank"]), 
                        "wttr_points_diff": float(data_A["points"] - data_B["points"]),
                        "home_continent_adv": 0,
                        "recent_win_ratio_diff": float(data_A["recent_form"] - data_B["recent_form"]),
                        "grip_interaction": 1 if data_A["grip"] != data_B["grip"] else 0,
                        "style_interaction": 1 if data_A["style"] != data_B["style"] else 0,
                        "is_live": 0 # Explicitly flags the match as not started
                    })
                print(f"[{datetime.now()}] API pulled {len(matches)} scheduled fixtures.")
            else:
                print(f"[{datetime.now()}] API Error {response.status_code}. Please verify endpoint status.")
                
        except requests.exceptions.RequestException as e:
            print(f"[{datetime.now()}] Connection error: {e}.")

        return matches

    def save_to_database(self, matches):
        cursor = self.conn.cursor()
        added_count = 0
        for m in matches:
            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO matches (
                        match_id, date, player_a_id, player_b_id, set_score_a, set_score_b,
                        processed, age_diff, handedness_interaction, height_diff,
                        schedule_density_diff, wttr_pos_diff, wttr_points_diff, tournament_tier, 
                        home_continent_adv, recent_win_ratio_diff, grip_interaction, style_interaction, is_live
                    ) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    m["match_id"], m["date"], m["player_a_id"], m["player_b_id"],
                    m["set_score_a"], m["set_score_b"], m["age_diff"], m["handedness_interaction"],
                    m["height_diff"], m["schedule_density_diff"], m["wttr_pos_diff"], m["wttr_points_diff"],
                    m["tournament_tier"], m["home_continent_adv"], m["recent_win_ratio_diff"], 
                    m["grip_interaction"], m["style_interaction"], m["is_live"]
                ))
                added_count += 1
            except sqlite3.IntegrityError:
                pass
        self.conn.commit()
        self.conn.close()
        print(f"[{datetime.now()}] Successfully stored {added_count} upcoming API matches.")

if __name__ == "__main__":
    scraper = UniversalMatchScraper()
    scraper.save_to_database(scraper.fetch_global_board())
