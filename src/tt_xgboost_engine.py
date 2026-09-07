    def engineer_differentials(self, df):
        """
        Transforms absolute player identities into relative strength differentials,
        incorporating Schedule Density (Fatigue) and micro-stylistic metrics.
        """
        # Algorithmic Baselines
        df['glicko_rating_diff'] = df['player_a_glicko'] - df['player_b_glicko']
        df['melo_vector_distance'] = np.linalg.norm(df['player_a_melo'] - df['player_b_melo'], axis=1)
        df['markov_match_win_prob_diff'] = df['player_a_markov'] - df['player_b_markov']
        
        # Physiological & Fatigue Variables
        df['age_diff'] = df['player_a_age'] - df['player_b_age']
        df['height_diff'] = df['player_a_height'] - df['player_b_height']
        df['handedness_interaction'] = df['player_a_hand'] - df['player_b_hand']
        
        # NEW: Schedule Density Differential (Sets played in past 48 hours)
        df['schedule_density_diff'] = df['player_a_sets_last_48h'] - df['player_b_sets_last_48h']
        
        # Official Rankings
        df['wttr_position_diff'] = df['player_a_wttr_pos'] - df['player_b_wttr_pos']
        df['wttr_points_diff'] = df['player_a_wttr_points'] - df['player_b_wttr_points']
        
        # Match Context & Momentum
        df['home_continent_adv'] = df['player_a_home_continent'] - df['player_b_home_continent']
        df['recent_win_ratio_diff'] = df['player_a_recent_win_ratio'] - df['player_b_recent_win_ratio']
        
        features = [
            'glicko_rating_diff', 'melo_vector_distance', 'markov_match_win_prob_diff',
            'age_diff', 'height_diff', 'handedness_interaction', 'schedule_density_diff',
            'wttr_position_diff', 'wttr_points_diff',
            'home_continent_adv', 'recent_win_ratio_diff'
        ]
        return df[features], df['target_player_a_wins']
