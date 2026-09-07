import os
import pickle
import numpy as np
from datetime import datetime

class DynamicLatentTracker:
    def __init__(self, state_path=None):
        self.state_path = state_path or os.path.join(
            os.path.dirname(__file__), "../data/latent_state.pkl"
        )
        self.players = {}  # Unified mapping for Elo, Glicko-2, and mElo vectors
        self.load_state()

    def _initialize_player(self, player_id):
        """Initializes unranked players with baseline priors."""
        if player_id not in self.players:
            self.players[player_id] = {
                'elo': 1500.0,
                'glicko_rating': 1500.0,
                'rd': 350.0,         # Default uncalibrated rating deviation
                'vol': 0.06,         # Rating volatility (sigma)
                'melo_vector': np.random.normal(0.0, 0.1, 4),  # 4D latent style vector
                'last_active': None
            }

    def _calculate_set_weighted_k(self, score_w, score_l):
        """
        Dynamically scales the Elo K-factor based on margin of victory:
        - 3-0 Sweep: K = 40
        - 3-1 Moderate: K = 30
        - 3-2 Narrow: K = 20
        """
        margin = score_w - score_l
        if margin >= 3:
            return 40.0
        elif margin == 2:
            return 30.0
        else:
            return 20.0

    def _apply_inactivity_decay(self, player_id, current_date):
        """
        Applies exponential decay on Rating Deviation (RD) during inactivity:
        phi_new = sqrt(phi_old^2 + sigma^2)
        """
        player = self.players[player_id]
        if player['last_active'] is not None:
            days_inactive = (current_date - player['last_active']).days
            if days_inactive > 30:
                # Expand uncertainty interval
                new_rd = np.sqrt(player['rd']**2 + (player['vol'] * 100)**2)
                player['rd'] = min(float(new_rd), 350.0)

    def update_match(self, winner, loser, score_w, score_l, date_str):
        """Updates player skill baselines and vectors following a match."""
        self._initialize_player(winner)
        self._initialize_player(loser)

        match_date = datetime.strptime(date_str, "%Y-%m-%d")

        # 1. Apply inactivity decay prior to rating update
        self._apply_inactivity_decay(winner, match_date)
        self._apply_inactivity_decay(loser, match_date)

        # 2. Set-Weighted Elo Calculation
        k_factor = self._calculate_set_weighted_k(score_w, score_l)
        w_elo = self.players[winner]['elo']
        l_elo = self.players[loser]['elo']

        expected_w = 1.0 / (1.0 + 10.0 ** ((l_elo - w_elo) / 400.0))
        self.players[winner]['elo'] += k_factor * (1.0 - expected_w)
        self.players[loser]['elo'] += k_factor * (0.0 - (1.0 - expected_w))

        # 3. Basic Glicko-2 Style Rating Drift
        # Approximate variance update for winner and loser
        q = np.log(10) / 400.0
        g_rd = 1.0 / np.sqrt(1.0 + 3.0 * (q**2) * (self.players[loser]['rd']**2) / (np.pi**2))
        d2 = 1.0 / ((q**2) * (g_rd**2) * expected_w * (1.0 - expected_w))
        
        self.players[winner]['glicko_rating'] += (q / ((1.0 / self.players[winner]['rd']**2) + (1.0 / d2))) * g_rd * (1.0 - expected_w)
        self.players[loser]['glicko_rating'] -= (q / ((1.0 / self.players[loser]['rd']**2) + (1.0 / d2))) * g_rd * expected_w

        # 4. Mark timestamps
        self.players[winner]['last_active'] = match_date
        self.players[loser]['last_active'] = match_date

    def save_state(self):
        """Persists player latent nodes to disk."""
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        with open(self.state_path, 'wb') as f:
            pickle.dump(self.players, f)

    def load_state(self):
        """Loads player latent nodes from disk if present."""
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, 'rb') as f:
                    self.players = pickle.load(f)
            except Exception:
                self.players = {}
        else:
            self.players = {}


class DTMC_Engine:
    @staticmethod
    def calculate_absorption_probability(p_serve, p_receive):
        """
        Combinatorial Discrete-Time Markov Chain evaluation.
        Computes the fundamental matrix N = (I - Q)^-1 to bypass recursive overhead.
        """
        # Formulate transient states Q for an 11-point table tennis game
        # Simplified abstraction representing regular non-deuce state absorption
        dim = 20
        Q_matrix = np.zeros((dim, dim))
        
        # Populate upper transient transitions based on serve alternating rule
        for i in range(dim - 1):
            Q_matrix[i, i + 1] = p_serve if (i // 2) % 2 == 0 else (1.0 - p_receive)

        I_matrix = np.eye(dim)
        try:
            # Fundamental matrix computation
            N_matrix = np.linalg.inv(I_matrix - Q_matrix)
            win_prob = np.clip(N_matrix[0, -1], 0.01, 0.99)
            return float(win_prob)
        except np.linalg.LinAlgError:
            # Fallback heuristic if singular
            return float(p_serve / (p_serve + p_receive))
