import os
import sqlite3
import requests
import numpy as np
from datetime import datetime

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/table_tennis_global.db"))

class GlobalBoardScraper:
    def __init__(self):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        self.conn = sqlite3.connect(DB_PATH)
        self.setup_database()

    def setup_database(self):
        cursor = self.conn.cursor()
        cursor.execute('''
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
        self.conn.commit()

    def fetch_environmental_density(self):
        """Fetches live temperature and pressure to calculate barometric air density."""
        try:
            url = "https://api.open-meteo.com/v1/forecast?latitude=22.1987&longitude=113.5439&current=temperature_2m,relative_humidity_2m,surface_pressure"
            r = requests.get(url, timeout=3).json()
            curr = r.get("current", {})
            temp = curr.get("temperature_2m", 22.0)
            rh = curr.get("relative_humidity_2m", 50.0)
            press = curr.get("surface_pressure", 1013.25)
            
            # Tetens equation for vapor pressure
            p_total = press * 100.0
            e_sat = 6.1078 * (10.0 ** ((7.5 * temp) / (237.3 + temp))) * 100.0
            pv = (rh / 100.0) * e_sat
            pd = p_total - pv
            return float((pd / (287.058 * (temp + 273.15))) + (pv / (461.495 * (temp + 273.15))))
        except Exception:
            return 1.225

    def get_player_metadata(self, name):
        roster = {
            "Tomokazu Harimoto": {"age": 23, "height": 175, "spw": 0.62, "rpw": 0.54, "tier": "WTT"},
            "Hugo Calderano": {"age": 30, "height": 183, "spw": 0.60, "rpw": 0.51, "tier": "WTT"},
            "Satsuki Odo": {"age": 22, "height": 160, "spw": 0.59, "rpw": 0.53, "tier": "WTT"},
            "Anton Kallberg": {"age": 29, "height": 185, "spw": 0.58, "rpw": 0.49, "tier": "WTT"},
            "Samara Elizabeta": {"age": 37, "height": 171, "spw": 0.55, "rpw": 0.48, "tier": "WTT"},
            "Nicholas Lum": {"age": 21, "height": 178, "spw": 0.54, "rpw": 0.47, "tier": "WTT"},
            "Kanak Jha": {"age": 26, "height": 170, "spw": 0.57, "rpw": 0.50, "tier": "WTT"},
            "Huang Youzheng": {"age": 19, "height": 174, "spw": 0.56, "rpw": 0.48, "tier": "WTT"},
            "Anna Hursey": {"age": 20, "height": 160, "spw": 0.53, "rpw": 0.47, "tier": "WTT"},
            "Manush Shah": {"age": 25, "height": 175, "spw": 0.52, "rpw": 0.46, "tier": "WTT"},
            "Leong On Na": {"age": 22, "height": 162, "spw": 0.49, "rpw": 0.43, "tier": "WTT"},
            "Mak Tin Ian": {"age": 20, "height": 170, "spw": 0.48, "rpw": 0.42, "tier": "WTT"},
            "Kirill Fadeev": {"age": 23, "height": 175, "spw": 0.53, "rpw": 0.48, "tier": "Challenger"},
            "Cosmo Schmitt": {"age": 25, "height": 177, "spw": 0.51, "rpw": 0.46, "tier": "Challenger"},
            "Grzegorz Poliniewicz": {"age": 28, "height": 178, "spw": 0.52, "rpw": 0.47, "tier": "TT Elite"},
            "Artur Daniel": {"age": 26, "height": 174, "spw": 0.54, "rpw": 0.49, "tier": "TT Elite"},
            "Dawid Kosmal": {"age": 24, "height": 176, "spw": 0.53, "rpw": 0.48, "tier": "TT Elite"},
            "Maciej Makajew": {"age": 29, "height": 180, "spw": 0.52, "rpw": 0.47, "tier": "TT Elite"}
        }
        return roster.get(name, {"age": 25, "height": 175, "spw": 0.50, "rpw": 0.46, "tier": "TT Cup"})

    def sync_board(self):
        air_density = self.fetch_environmental_density()
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Global match matrix: Enforces dual-tier separation (is_fanduel = 1 vs 0)
        fixtures = [
            # WTT Champions (Hosted on FanDuel)
            ("Anton Kallberg", "Manush Shah", "WTT", 1, 0),
            ("Mak Tin Ian", "Tomokazu Harimoto", "WTT", 1, 0),
            ("Nicholas Lum", "Hugo Calderano", "WTT", 1, 0),
            ("Kanak Jha", "Huang Youzheng", "WTT", 1, 0),
            ("Anna Hursey", "Leong On Na", "WTT", 1, 0),
            ("Satsuki Odo", "Samara Elizabeta", "WTT", 1, 0),
            # Regional Circuits (Background Model Learning Lake)
            ("Kirill Fadeev", "Cosmo Schmitt", "Challenger", 0, 0),
            ("Grzegorz Poliniewicz", "Artur Daniel", "TT Elite", 0, 0),
            ("Dawid Kosmal", "Maciej Makajew", "TT Elite", 0, 0)
        ]

        cursor = self.conn.cursor()
        added = 0
        for pA, pB, tier, is_fd, is_live in fixtures:
            mA = self.get_player_metadata(pA)
            mB = self.get_player_metadata(pB)
            match_id = f"FIX_{pA[:3]}_{pB[:3]}_{datetime.now().strftime('%Y%m%d_%H%M')}"

            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO matches (
                        match_id, date, player_a_id, player_b_id, is_fanduel, is_live,
                        tournament_tier, age_diff, height_diff, spw_diff, rpw_diff, air_density
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    match_id, date_str, pA, pB, is_fd, is_live, tier,
                    float(mA["age"] - mB["age"]), float(mA["height"] - mB["height"]),
                    float(mA["spw"] - mB["spw"]), float(mA["rpw"] - mB["rpw"]), air_density
                ))
                added += 1
            except sqlite3.IntegrityError:
                pass

        self.conn.commit()
        self.conn.close()
        print(f"[{datetime.now()}] Data lake synchronized: {added} fixtures queued. Air density: {air_density:.4f} kg/m^3.")

if __name__ == "__main__":
    scraper = GlobalBoardScraper()
    scraper.sync_board()
