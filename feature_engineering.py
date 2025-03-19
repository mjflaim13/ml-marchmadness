# feature_engineering.py
import pandas as pd
import numpy as np
import os

def load_matchup_data():
    """Load processed matchup data"""
    matchup_path = 'data/processed/matchups.csv'
    
    if not os.path.exists(matchup_path):
        raise FileNotFoundError("Matchup data not found. Run data_preprocessing.py first.")
    
    matchups_df = pd.read_csv(matchup_path)
    return matchups_df

def create_relative_features(df):
    """
    Create relative features comparing both teams
    """
    print("Creating relative features...")
    
    # Seed difference already exists as 'SeedDiff'
    
    # Create relative metrics for all KenPom and other metrics, checking if they exist first
    feature_pairs = [
        # KenPom metrics
        ('AdjEM', 'RelativeAdjEM'),
        ('AdjO', 'RelativeAdjO'),
        ('AdjD', 'RelativeAdjD'),
        ('AdjT', 'RelativeAdjT'),
        ('Luck', 'RelativeLuck'),
        ('SOS', 'RelativeSOS'),
        ('NCSOS', 'RelativeNCSOS'),
        ('KenPomRank', 'RelativeKenPomRank'),
        
        # Other metrics
        ('ELO', 'RelativeELO'),
        ('OffenseRating', 'RelativeOffenseRating'),
        ('DefenseRating', 'RelativeDefenseRating'),
        ('TempoRating', 'RelativeTempoRating'),
        ('Q1Wins', 'RelativeQ1Wins')
        
        # Removed Heat Check features
    ]
    
    for base_feature, relative_feature in feature_pairs:
        team1_feature = f'Team1_{base_feature}'
        team2_feature = f'Team2_{base_feature}'
        
        if team1_feature in df.columns and team2_feature in df.columns:
            print(f"Creating {relative_feature}")
            # For rank features (lower is better), subtract Team1 from Team2
            if base_feature == 'KenPomRank':
                df[relative_feature] = df[team2_feature] - df[team1_feature]
            else:
                # For other metrics (higher is better), subtract Team2 from Team1
                df[relative_feature] = df[team1_feature] - df[team2_feature]
        else:
            print(f"Skipping {relative_feature} - required columns not found")
    
    return df

def create_momentum_features(df, tournament_data):
    """
    Create momentum features based on team's recent performance
    Uses real historical tournament data
    """
    print("Creating momentum features...")
    
    # Get all team names
    all_teams = set(df['Team1'].unique()) | set(df['Team2'].unique())
    
    # Create a lookup for momentum 
    momentum_lookup = {}
    
    # Process each team
    for team in all_teams:
        # Get most recent tournament performance
        team_wins = tournament_data[tournament_data['WinningTeam'] == team]
        team_losses = tournament_data[tournament_data['LosingTeam'] == team]
        
        if len(team_wins) > 0 or len(team_losses) > 0:
            # Calculate win percentage
            total_games = len(team_wins) + len(team_losses)
            win_percentage = len(team_wins) / total_games if total_games > 0 else 0
            
            # Calculate how far the team advanced in their most recent tournament
            rounds_reached = []
            for _, game in team_wins.iterrows():
                year = game['Year']
                round_name = game['Round']
                round_value = {'R64': 1, 'R32': 2, 'S16': 3, 'E8': 4, 'F4': 5, 'NCG': 6}.get(round_name, 0)
                rounds_reached.append((year, round_value))
            
            for _, game in team_losses.iterrows():
                year = game['Year']
                round_name = game['Round']
                round_value = {'R64': 1, 'R32': 2, 'S16': 3, 'E8': 4, 'F4': 5, 'NCG': 6}.get(round_name, 0)
                rounds_reached.append((year, round_value))
            
            # Sort by year (descending) and get most recent tournament performance
            if rounds_reached:
                most_recent = sorted(rounds_reached, key=lambda x: x[0], reverse=True)[0]
                most_recent_round = most_recent[1]
            else:
                most_recent_round = 0
            
            # Combine factors to calculate momentum
            momentum = (win_percentage * 2) + (most_recent_round / 6)  # Scale from 0 to 3
            momentum_lookup[team] = momentum
        else:
            # No tournament history
            momentum_lookup[team] = 0
    
    # Add momentum features to the DataFrame
    df['Team1_Momentum'] = df['Team1'].map(momentum_lookup)
    df['Team2_Momentum'] = df['Team2'].map(momentum_lookup)
    df['RelativeMomentum'] = df['Team1_Momentum'] - df['Team2_Momentum']
    
    # Fill missing values
    df['Team1_Momentum'].fillna(0, inplace=True)
    df['Team2_Momentum'].fillna(0, inplace=True)
    df['RelativeMomentum'].fillna(0, inplace=True)
    
    return df

def create_volatility_features(df, tournament_data):
    """
    Create volatility features (consistency of team performance)
    Uses real historical tournament data
    """
    print("Creating volatility features...")
    
    # Get all team names
    all_teams = set(df['Team1'].unique()) | set(df['Team2'].unique())
    
    # Create a lookup for volatility
    volatility_lookup = {}
    
    # Use tournament data to calculate volatility
    for team in all_teams:
        # Get games where team won
        won_games = tournament_data[tournament_data['WinningTeam'] == team]
        
        # Get games where team lost
        lost_games = tournament_data[tournament_data['LosingTeam'] == team]
        
        # Calculate score differences
        win_margins = won_games['ScoreDiff'].values if len(won_games) > 0 else []
        loss_margins = lost_games['ScoreDiff'].values if len(lost_games) > 0 else []
        
        # Combine all margins
        all_margins = list(win_margins) + list(loss_margins)
        
        if len(all_margins) >= 2:
            # Calculate volatility as standard deviation of score differences
            volatility = np.std(all_margins)
        elif len(all_margins) == 1:
            # With just one game, use a moderate volatility
            volatility = 10
        else:
            # Default volatility if no tournament history
            volatility = 10  # Average volatility
        
        # Store volatility
        volatility_lookup[team] = volatility
    
    # Add volatility features to the DataFrame
    df['Team1_Volatility'] = df['Team1'].map(volatility_lookup)
    df['Team2_Volatility'] = df['Team2'].map(volatility_lookup)
    df['RelativeVolatility'] = df['Team1_Volatility'] - df['Team2_Volatility']
    
    # Fill missing values
    df['Team1_Volatility'].fillna(10, inplace=True)
    df['Team2_Volatility'].fillna(10, inplace=True)
    df['RelativeVolatility'].fillna(0, inplace=True)
    
    return df

def create_efficiency_features(df):
    """
    Create additional efficiency features from existing KenPom metrics
    """
    print("Creating efficiency features...")
    
    # Offensive to defensive efficiency ratio
    df['Team1_OffDefRatio'] = df['Team1_AdjO'] / df['Team1_AdjD']
    df['Team2_OffDefRatio'] = df['Team2_AdjO'] / df['Team2_AdjD']
    df['RelativeOffDefRatio'] = df['Team1_OffDefRatio'] - df['Team2_OffDefRatio']
    
    # Efficiency * Tempo (approximating points)
    df['Team1_ExpectedOutput'] = df['Team1_AdjO'] * (df['Team1_AdjT'] / 100)
    df['Team2_ExpectedOutput'] = df['Team2_AdjO'] * (df['Team2_AdjT'] / 100)
    df['RelativeExpectedOutput'] = df['Team1_ExpectedOutput'] - df['Team2_ExpectedOutput']
    
    return df

def create_elo_features(df, tournament_data):
    """
    Create ELO-style ratings based on historical tournament performances
    Uses real tournament results
    """
    print("Creating ELO-style features...")
    
    # Get all team names
    all_teams = set(df['Team1'].unique()) | set(df['Team2'].unique())
    
    # Initialize ELO ratings (default 1500)
    elo_lookup = {team: 1500 for team in all_teams}
    
    # Check if we already have ELO ratings in the data
    if 'Team1_ELO' in df.columns and 'Team2_ELO' in df.columns:
        print("Using existing ELO ratings from the dataset")
        return df
    
    # Sort tournament data by year and round to simulate chronological order
    round_order = {'R64': 0, 'R32': 1, 'S16': 2, 'E8': 3, 'F4': 4, 'NCG': 5}
    sorted_games = tournament_data.copy()
    sorted_games['RoundOrder'] = sorted_games['Round'].map(round_order)
    sorted_games = sorted_games.sort_values(['Year', 'RoundOrder']).reset_index(drop=True)
    
    # K-factor for ELO calculations
    K = 32
    
    # Update ELO ratings based on tournament results
    for _, game in sorted_games.iterrows():
        winner = game['WinningTeam']
        loser = game['LosingTeam']
        
        if winner in elo_lookup and loser in elo_lookup:
            # Get current ratings
            winner_elo = elo_lookup[winner]
            loser_elo = elo_lookup[loser]
            
            # Calculate expected outcome
            expected_winner = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
            expected_loser = 1 - expected_winner
            
            # Update ratings
            winner_new_elo = winner_elo + K * (1 - expected_winner)
            loser_new_elo = loser_elo + K * (0 - expected_loser)
            
            # Store updated ratings
            elo_lookup[winner] = winner_new_elo
            elo_lookup[loser] = loser_new_elo
    
    # Add ELO features to the DataFrame if they don't already exist
    df['Team1_ELO'] = df['Team1'].map(elo_lookup)
    df['Team2_ELO'] = df['Team2'].map(elo_lookup)
    df['RelativeELO'] = df['Team1_ELO'] - df['Team2_ELO']
    
    return df

def prepare_features(df):
    """
    Prepare final feature set for modeling
    """
    print("Preparing final feature set...")
    
    # Define core features that should be present
    core_features = [
        'SeedDiff',
        'RelativeAdjEM', 'RelativeAdjO', 'RelativeAdjD', 'RelativeAdjT',
        'RelativeLuck', 'RelativeSOS', 'RelativeNCSOS', 'RelativeKenPomRank',
        'RelativeMomentum', 'RelativeVolatility',
        'RelativeOffDefRatio', 'RelativeExpectedOutput',
        'RelativeELO'
    ]
    
    # Get all relative features from the dataframe
    all_relative_features = [col for col in df.columns if col.startswith('Relative')]
    
    # Combine core features with any additional relative features
    features = list(set(core_features) | set(all_relative_features))
    
    # Ensure all features exist in the dataframe
    for feature in list(features):
        if feature not in df.columns:
            print(f"Warning: Feature '{feature}' not found in dataframe, it will be excluded")
            features.remove(feature)
    
    # Create X (features) and y (target)
    X = df[features]
    y = df['Team1_Won']
    
    return X, y, features

def main():
    """Main feature engineering function"""
    # Load data
    try:
        matchups_df = load_matchup_data()
        print("Matchup data loaded successfully.")
    except FileNotFoundError as e:
        print(e)
        return
    
    # Load tournament data for momentum and volatility features
    tournament_path = 'data/raw/tournament_results.csv'
    
    if not os.path.exists(tournament_path):
        print("Tournament results file not found. Run data_collection.py first.")
        return
    
    tournament_data = pd.read_csv(tournament_path)
    
    # Create features
    matchups_df = create_relative_features(matchups_df)
    matchups_df = create_momentum_features(matchups_df, tournament_data)
    matchups_df = create_volatility_features(matchups_df, tournament_data)
    matchups_df = create_efficiency_features(matchups_df)
    matchups_df = create_elo_features(matchups_df, tournament_data)
    
    # Prepare final feature set
    X, y, features = prepare_features(matchups_df)
    
    # Save engineered features
    feature_data = pd.concat([X, y], axis=1)
    feature_data.to_csv('data/processed/engineered_features.csv', index=False)
    
    # Save feature list for later use
    pd.DataFrame({'feature': features}).to_csv('data/processed/feature_list.csv', index=False)
    
    print("Feature engineering complete!")
    print(f"Engineered features saved to data/processed/engineered_features.csv")

if __name__ == "__main__":
    main()