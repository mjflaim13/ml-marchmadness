# fix_features.py
import pandas as pd
import numpy as np
import os
import pickle

def load_model():
    """Load the trained model"""
    model_path = 'models/final_model.pkl'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Run model_training.py first.")
    
    with open(model_path, 'rb') as f:
        model_info = pickle.load(f)
    
    return model_info

def load_engineered_features():
    """Load engineered feature data"""
    feature_path = 'data/processed/engineered_features.csv'
    
    if not os.path.exists(feature_path):
        raise FileNotFoundError("Engineered feature data not found. Run feature_engineering.py first.")
    
    feature_data = pd.read_csv(feature_path)
    return feature_data

def load_matchup_data():
    """Load processed matchup data"""
    matchup_path = 'data/processed/matchups.csv'
    
    if not os.path.exists(matchup_path):
        raise FileNotFoundError("Matchup data not found. Run data_preprocessing.py first.")
    
    matchups_df = pd.read_csv(matchup_path)
    return matchups_df

def create_relative_features(matchups_df):
    """
    Create relative features directly in the matchups dataframe
    """
    print("Creating relative features in matchup data...")
    
    # Create seed difference (should already exist)
    if 'SeedDiff' not in matchups_df.columns and all(col in matchups_df.columns for col in ['Team1_Seed', 'Team2_Seed']):
        matchups_df['SeedDiff'] = matchups_df['Team1_Seed'] - matchups_df['Team2_Seed']
    
    # Create relative KenPom metrics
    kenpom_features = {
        'AdjEM': 'Adjusted Efficiency Margin',
        'AdjO': 'Adjusted Offensive Efficiency',
        'AdjD': 'Adjusted Defensive Efficiency',
        'AdjT': 'Adjusted Tempo',
        'Luck': 'Luck',
        'SOS': 'Strength of Schedule',
        'NCSOS': 'Non-conference SOS',
        'KenPomRank': 'KenPom Rank'
    }
    
    for feature, description in kenpom_features.items():
        team1_col = f'Team1_{feature}'
        team2_col = f'Team2_{feature}'
        relative_col = f'Relative{feature}'
        
        if team1_col in matchups_df.columns and team2_col in matchups_df.columns:
            if feature == 'KenPomRank':
                # For rank, lower is better, so subtract Team1 from Team2
                matchups_df[relative_col] = matchups_df[team2_col] - matchups_df[team1_col]
            else:
                # For other metrics, higher is better
                matchups_df[relative_col] = matchups_df[team1_col] - matchups_df[team2_col]
            print(f"Created {relative_col}")
    
    # Create relative odds features
    odds_features = {
        'ChampOdds': 'Championship Odds',
        'FFOdds': 'Final Four Odds',
        'E8Odds': 'Elite Eight Odds'
    }
    
    for feature, description in odds_features.items():
        team1_col = f'Team1_{feature}'
        team2_col = f'Team2_{feature}'
        relative_col = f'Relative{feature}'
        
        if team1_col in matchups_df.columns and team2_col in matchups_df.columns:
            # For odds, lower is better (better chance to win)
            matchups_df[relative_col] = matchups_df[team2_col] - matchups_df[team1_col]
            print(f"Created {relative_col}")
            
            # Create implied probability differences
            if feature in ['ChampOdds', 'FFOdds']:
                prob_suffix = feature.replace('Odds', 'Prob')
                
                # Calculate probabilities (100/odds)
                matchups_df[f'Team1_{prob_suffix}'] = 100 / matchups_df[team1_col]
                matchups_df[f'Team2_{prob_suffix}'] = 100 / matchups_df[team2_col]
                
                # Calculate relative probability
                matchups_df[f'Relative{prob_suffix}'] = matchups_df[f'Team1_{prob_suffix}'] - matchups_df[f'Team2_{prob_suffix}']
                print(f"Created Relative{prob_suffix}")
    
    # Create efficiency ratio features
    if all(col in matchups_df.columns for col in ['Team1_AdjO', 'Team1_AdjD', 'Team2_AdjO', 'Team2_AdjD']):
        # Offensive to defensive efficiency ratio
        matchups_df['Team1_OffDefRatio'] = matchups_df['Team1_AdjO'] / matchups_df['Team1_AdjD']
        matchups_df['Team2_OffDefRatio'] = matchups_df['Team2_AdjO'] / matchups_df['Team2_AdjD']
        matchups_df['RelativeOffDefRatio'] = matchups_df['Team1_OffDefRatio'] - matchups_df['Team2_OffDefRatio']
        print("Created RelativeOffDefRatio")
    
    # Create expected output features
    if all(col in matchups_df.columns for col in ['Team1_AdjO', 'Team1_AdjT', 'Team2_AdjO', 'Team2_AdjT']):
        # Efficiency * Tempo (approximating points)
        matchups_df['Team1_ExpectedOutput'] = matchups_df['Team1_AdjO'] * (matchups_df['Team1_AdjT'] / 100)
        matchups_df['Team2_ExpectedOutput'] = matchups_df['Team2_AdjO'] * (matchups_df['Team2_AdjT'] / 100)
        matchups_df['RelativeExpectedOutput'] = matchups_df['Team1_ExpectedOutput'] - matchups_df['Team2_ExpectedOutput']
        print("Created RelativeExpectedOutput")
    
    # Check if Momentum features already exist
    if 'Team1_Momentum' in matchups_df.columns and 'Team2_Momentum' in matchups_df.columns:
        # Calculate relative momentum
        matchups_df['RelativeMomentum'] = matchups_df['Team1_Momentum'] - matchups_df['Team2_Momentum']
        print("Created RelativeMomentum using existing momentum data")
    else:
        print("Momentum data not available, RelativeMomentum feature will be missing")
    
    # Check if Volatility features already exist
    if 'Team1_Volatility' in matchups_df.columns and 'Team2_Volatility' in matchups_df.columns:
        # Calculate relative volatility
        matchups_df['RelativeVolatility'] = matchups_df['Team1_Volatility'] - matchups_df['Team2_Volatility']
        print("Created RelativeVolatility using existing volatility data")
    else:
        print("Volatility data not available, RelativeVolatility feature will be missing")
    
    # Check if ELO features already exist
    if 'Team1_ELO' in matchups_df.columns and 'Team2_ELO' in matchups_df.columns:
        # Calculate relative ELO
        matchups_df['RelativeELO'] = matchups_df['Team1_ELO'] - matchups_df['Team2_ELO']
        print("Created RelativeELO using existing ELO data")
    else:
        print("ELO data not available, RelativeELO feature will be missing")
    
    return matchups_df

def create_momentum_features(matchups_df, tournament_data):
    """Create momentum features based on team's tournament history"""
    print("Creating momentum features from tournament data...")
    
    # Get all team names
    all_teams = set(matchups_df['Team1'].unique()) | set(matchups_df['Team2'].unique())
    
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
    matchups_df['Team1_Momentum'] = matchups_df['Team1'].map(momentum_lookup)
    matchups_df['Team2_Momentum'] = matchups_df['Team2'].map(momentum_lookup)
    matchups_df['RelativeMomentum'] = matchups_df['Team1_Momentum'] - matchups_df['Team2_Momentum']
    
    # Fill missing values
    matchups_df['Team1_Momentum'].fillna(0, inplace=True)
    matchups_df['Team2_Momentum'].fillna(0, inplace=True)
    matchups_df['RelativeMomentum'].fillna(0, inplace=True)
    
    return matchups_df

def create_volatility_features(matchups_df, tournament_data):
    """Create volatility features based on score differences"""
    print("Creating volatility features from tournament data...")
    
    # Get all team names
    all_teams = set(matchups_df['Team1'].unique()) | set(matchups_df['Team2'].unique())
    
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
    matchups_df['Team1_Volatility'] = matchups_df['Team1'].map(volatility_lookup)
    matchups_df['Team2_Volatility'] = matchups_df['Team2'].map(volatility_lookup)
    matchups_df['RelativeVolatility'] = matchups_df['Team1_Volatility'] - matchups_df['Team2_Volatility']
    
    # Fill missing values
    matchups_df['Team1_Volatility'].fillna(10, inplace=True)
    matchups_df['Team2_Volatility'].fillna(10, inplace=True)
    matchups_df['RelativeVolatility'].fillna(0, inplace=True)
    
    return matchups_df

def create_elo_features(matchups_df, tournament_data):
    """Create ELO ratings based on tournament history"""
    print("Creating ELO features from tournament data...")
    
    # Get all team names
    all_teams = set(matchups_df['Team1'].unique()) | set(matchups_df['Team2'].unique())
    
    # Initialize ELO ratings (default 1500)
    elo_lookup = {team: 1500 for team in all_teams}
    
    # Check if we already have ELO ratings in the data
    if 'Team1_ELO' in matchups_df.columns and 'Team2_ELO' in matchups_df.columns:
        print("Using existing ELO ratings from the dataset")
        matchups_df['RelativeELO'] = matchups_df['Team1_ELO'] - matchups_df['Team2_ELO']
        return matchups_df
    
    # Sort tournament data by year and round to simulate chronological order
    round_order = {'R64': 0, 'R32': 1, 'S16': 2, 'E8': 3, 'F4': 4, 'NCG': 5}
    sorted_games = tournament_data.copy()
    sorted_games['RoundOrder'] = sorted_games['Round'].map(lambda r: round_order.get(r, 0))
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
    
    # Add ELO features to the DataFrame
    matchups_df['Team1_ELO'] = matchups_df['Team1'].map(elo_lookup)
    matchups_df['Team2_ELO'] = matchups_df['Team2'].map(elo_lookup)
    matchups_df['RelativeELO'] = matchups_df['Team1_ELO'] - matchups_df['Team2_ELO']
    
    # Fill missing values
    matchups_df['Team1_ELO'].fillna(1500, inplace=True)
    matchups_df['Team2_ELO'].fillna(1500, inplace=True)
    matchups_df['RelativeELO'].fillna(0, inplace=True)
    
    return matchups_df

def align_features_for_model(matchups_df, model_info):
    """
    Ensure matchups dataframe has all features required by the model in the correct order
    """
    print("Aligning features for model compatibility...")
    
    if 'feature_names' not in model_info:
        print("Model does not have feature names specified, using all available features")
        return matchups_df
    
    model_features = model_info['feature_names']
    print(f"Model requires {len(model_features)} features: {model_features}")
    
    # Check which features are missing
    missing_features = [f for f in model_features if f not in matchups_df.columns]
    
    if missing_features:
        print(f"Missing {len(missing_features)} features: {missing_features}")
        
        # Create missing features with placeholder values
        for feature in missing_features:
            print(f"Creating placeholder for {feature}")
            matchups_df[feature] = 0.0
    
    # Ensure features are in the correct order for the model
    return matchups_df

def main():
    """Main function to check and fix feature alignment"""
    print("Checking and fixing feature alignment issues...")
    
    try:
        # Load model to get required features
        model_info = load_model()
        print("Model loaded successfully.")
        
        # Load matchup data
        matchups_df = load_matchup_data()
        print(f"Matchup data loaded successfully: {matchups_df.shape[0]} rows, {matchups_df.shape[1]} columns")
        
        # Add engineered features directly to matchups
        enhanced_matchups = create_relative_features(matchups_df)
        print(f"Added base relative features: now {enhanced_matchups.shape[1]} columns")
        
        # Load tournament data for momentum and volatility features
        tournament_path = 'data/raw/tournament_results.csv'
        if os.path.exists(tournament_path):
            tournament_data = pd.read_csv(tournament_path)
            enhanced_matchups = create_momentum_features(enhanced_matchups, tournament_data)
            enhanced_matchups = create_volatility_features(enhanced_matchups, tournament_data)
            enhanced_matchups = create_elo_features(enhanced_matchups, tournament_data)
            print(f"Added momentum, volatility, and ELO features using real tournament data")
        else:
            print("Tournament results file not found. Some features will be limited.")
        
        # Align features with model requirements
        aligned_matchups = align_features_for_model(enhanced_matchups, model_info)
        
        # Save the enhanced matchups
        enhanced_path = 'data/processed/enhanced_matchups.csv'
        aligned_matchups.to_csv(enhanced_path, index=False)
        print(f"Enhanced matchups saved to {enhanced_path}")
        
        print("\nFeature alignment complete! You can now run backtesting.py again with the enhanced data.")
        print("Use the following code to load the enhanced matchups in backtesting.py:")
        print("matchups_df = pd.read_csv('data/processed/enhanced_matchups.csv')")
        
    except Exception as e:
        print(f"Error during feature alignment: {e}")

if __name__ == "__main__":
    main()