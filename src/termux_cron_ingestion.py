import os
import sqlite3
import pandas as pd
from datetime import datetime
import subprocess

from latent_framework import DynamicLatentTracker

DB_PATH = "../data/table_tennis_global.db"
STATE_PATH = "../data/latent_state.pkl"

def pull_github_updates():
    try:
        subprocess.run(["git", "pull", "origin", "main"], check=True, capture_output=True)
        print(f"[{datetime.now()}] GitHub synchronization successful.")
    except subprocess.CalledProcessError as e:
        print(f"[{datetime.now()}] Git pull failed: {e.stderr}")

def ingest_and_update_ratings():
    conn = sqlite3.connect(DB_PATH)
    
    query = """
        SELECT match_id, date, winner_id, loser_id, set_score_winner, set_score_loser 
        FROM matches 
        WHERE processed = 0 
        ORDER BY date ASC
    """
    new_matches = pd.read_sql_query(query, conn)
    
    if new_matches.empty:
        print(f"[{datetime.now()}] No new matches to process.")
        return

    tracker = DynamicLatentTracker(state_path=STATE_PATH)
    
    for _, row in new_matches.iterrows():
        tracker.update_match(
            winner=row['winner_id'],
            loser=row['loser_id'],
            score_w=row['set_score_winner'],
            score_l=row['set_score_loser'],
            date=row['date']
        )
        
        cursor = conn.cursor()
        cursor.execute("UPDATE matches SET processed = 1 WHERE match_id = ?", (row['match_id'],))
    
    conn.commit()
    conn.close()
    
    tracker.save_state()
    print(f"[{datetime.now()}] Processed {len(new_matches)} matches. Latent state updated.")

if __name__ == "__main__":
    pull_github_updates()
    ingest_and_update_ratings()
