import os
import sqlite3
import pandas as pd
from datetime import datetime
import subprocess

# Local imports from our mathematical framework
from latent_framework import DynamicLatentTracker

DB_PATH = "../data/table_tennis_global.db"
STATE_PATH = "../data/latent_state.pkl"

def pull_github_updates():
    """Executes a git pull to sync the local Termux environment with the cloud data lake."""
    try:
        subprocess.run(["git", "pull", "origin", "main"], check=True, capture_output=True)
        print(f"[{datetime.now()}] GitHub synchronization successful.")
    except subprocess.CalledProcessError as e:
        print(f"[{datetime.now()}] Git pull failed: {e.stderr}")

def ingest_and_update_ratings():
    """Ingests newly synced matches and updates latent Glicko-2 and mElo vectors."""
    conn = sqlite3.connect(DB_PATH)
    
    # Fetch unprocessed matches (assuming a 'processed' flag exists in the schema)
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

    # Initialize or load existing latent state tracking
    tracker = DynamicLatentTracker(state_path=STATE_PATH)
    
    for _, row in new_matches.iterrows():
        # Update Latent Variables
        tracker.update_match(
            winner=row['winner_id'],
            loser=row['loser_id'],
            score_w=row['set_score_winner'],
            score_l=row['set_score_loser'],
            date=row['date']
        )
        
        # Mark as processed
        cursor = conn.cursor()
        cursor.execute("UPDATE matches SET processed = 1 WHERE match_id = ?", (row['match_id'],))
    
    conn.commit()
    conn.close()
    
    # Persist the updated latent rating vectors
    tracker.save_state()
    print(f"[{datetime.now()}] Processed {len(new_matches)} matches. Latent state updated.")

if __name__ == "__main__":
    pull_github_updates()
    ingest_and_update_ratings()
