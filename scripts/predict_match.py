import os
import sys
import argparse

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from unified_engine import UnifiedPipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player-a", required=True, help="Player A Name")
    parser.add_argument("--player-b", required=True, help="Player B Name")
    args = parser.parse_args()

    pipeline = UnifiedPipeline()
    winner, confidence, set_dist = pipeline.evaluate_match(args.player_a, args.player_b, is_fanduel=1)

    dist_str = f"3-0: {set_dist['3-0']*100:.1f}% | 3-1: {set_dist['3-1']*100:.1f}% | 3-2: {set_dist['3-2']*100:.1f}%"
    print("==========================================================")
    print(f"🎯 50,000-SIM PREDICTION: {args.player_a} vs {args.player_b}")
    print("==========================================================")
    print(f"Projected Winner : {winner}")
    print(f"Model Confidence : {confidence*100:.2f}%")
    print(f"Set Distribution : {dist_str}")
    print("==========================================================")

if __name__ == "__main__":
    main()
