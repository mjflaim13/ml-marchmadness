# predict_bracket.py
from enhanced_prediction import predict_with_enhanced_model
import pandas as pd
import numpy as np
import os
import pickle
import json
from sklearn.preprocessing import StandardScaler
def predict_with_enhanced_model(model_info, X):
    """Make predictions using enhanced model"""
    # Preprocess data
    X_imputed = model_info['imputer'].transform(X)
    X_scaled = model_info['scaler'].transform(X_imputed)
    
    # Check if we have a voting ensemble or single model
    if 'voting_ensemble' in model_info:
        # Get predictions from voting ensemble
        predictions = model_info['voting_ensemble'].predict(X_scaled)
        probabilities = model_info['voting_ensemble'].predict_proba(X_scaled)[:, 1]
    else:
        # Use standard model
        predictions = model_info['model'].predict(X_scaled)
        probabilities = model_info['model'].predict_proba(X_scaled)[:, 1] if hasattr(model_info['model'], 'predict_proba') else predictions
    
    return predictions, probabilities
def make_json_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {make_json_serializable(k): make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(i) for i in obj]
    else:
        return obj

def load_model():
    """Load the trained model, preferring enhanced model if available"""
    enhanced_model_path = 'models/enhanced_model.pkl'
    standard_model_path = 'models/final_model.pkl'
    
    if os.path.exists(enhanced_model_path):
        print("Using enhanced ensemble model for predictions")
        with open(enhanced_model_path, 'rb') as f:
            model_info = pickle.load(f)
        # Flag as enhanced model
        model_info['is_enhanced'] = True
    elif os.path.exists(standard_model_path):
        print("Using standard model for predictions")
        with open(standard_model_path, 'rb') as f:
            model_info = pickle.load(f)
        # Flag as standard model
        model_info['is_enhanced'] = False
    else:
        raise FileNotFoundError("No model found. Run model_training.py or run_enhanced_ml.py first.")
    
    return model_info

def load_current_data():
    """Load current year data"""
    bracket_path = 'data/raw/bracket_2025.csv'
    kenpom_path = 'data/raw/kenpom_ratings.csv'
    
    # Define paths to additional data sources
    resumes_path = 'data/raw/Resumes.csv'
    evan_miya_path = 'data/raw/EvanMiya.csv'
    team_rankings_path = 'data/raw/TeamRankings.csv'
    
    # Check if essential files exist
    if not all(os.path.exists(path) for path in [bracket_path, kenpom_path]):
        raise FileNotFoundError("Current year data not found. Run data_collection.py first.")
    
    # Load bracket and KenPom data
    bracket_df = pd.read_csv(bracket_path)
    kenpom_df = pd.read_csv(kenpom_path)
    
    # Filter KenPom data for current year
    current_year = 2025
    kenpom_df = kenpom_df[kenpom_df['Year'] == current_year]
    
    # Load additional data sources
    additional_data = {}
    for key, path in [
        ('resumes', resumes_path),
        ('evan_miya', evan_miya_path),
        ('team_rankings', team_rankings_path)
    ]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Filter for current year if possible
            if 'YEAR' in df.columns:
                df = df[df['YEAR'] == current_year]
            additional_data[key] = df
            print(f"Loaded {key} data: {len(df)} teams")
        else:
            print(f"Note: {path} not found, some features will be limited")
            additional_data[key] = pd.DataFrame()
    
    return bracket_df, kenpom_df, additional_data

def create_matchup_features(team1, team2, kenpom_df, additional_data):
    """
    Create features for a single matchup using available data sources
    """
    # Get team data from KenPom
    team1_kenpom = kenpom_df[kenpom_df['Team'] == team1]
    team2_kenpom = kenpom_df[kenpom_df['Team'] == team2]
    
    # If teams not found in KenPom, use reasonable defaults
    if team1_kenpom.empty:
        print(f"WARNING: Team {team1} not found in KenPom data")
        team1_kenpom = pd.DataFrame({
            'AdjEM': [0], 'AdjO': [100], 'AdjD': [100], 'AdjT': [70],
            'Luck': [0], 'SOS': [0], 'NCSOS': [0], 'KenPomRank': [200]
        })
    else:
        team1_kenpom = team1_kenpom.iloc[0:1]
    
    if team2_kenpom.empty:
        print(f"WARNING: Team {team2} not found in KenPom data")
        team2_kenpom = pd.DataFrame({
            'AdjEM': [0], 'AdjO': [100], 'AdjD': [100], 'AdjT': [70],
            'Luck': [0], 'SOS': [0], 'NCSOS': [0], 'KenPomRank': [200]
        })
    else:
        team2_kenpom = team2_kenpom.iloc[0:1]
    
    # Initialize additional metrics with default values
    team1_elo = 1500
    team2_elo = 1500
    team1_offense_rating = 100
    team2_offense_rating = 100
    team1_defense_rating = 100
    team2_defense_rating = 100
    team1_tempo_rating = 70
    team2_tempo_rating = 70
    team1_q1_wins = 0
    team2_q1_wins = 0
    
    # Get data from Resumes if available
    resumes_df = additional_data.get('resumes', pd.DataFrame())
    if not resumes_df.empty:
        team1_resumes = resumes_df[resumes_df['TEAM'] == team1]
        if not team1_resumes.empty:
            if 'ELO' in team1_resumes.columns:
                team1_elo = team1_resumes['ELO'].iloc[0]
            if 'Q1 W' in team1_resumes.columns:
                team1_q1_wins = team1_resumes['Q1 W'].iloc[0]
        
        team2_resumes = resumes_df[resumes_df['TEAM'] == team2]
        if not team2_resumes.empty:
            if 'ELO' in team2_resumes.columns:
                team2_elo = team2_resumes['ELO'].iloc[0]
            if 'Q1 W' in team2_resumes.columns:
                team2_q1_wins = team2_resumes['Q1 W'].iloc[0]
    
    # Get data from EvanMiya if available
    evan_miya_df = additional_data.get('evan_miya', pd.DataFrame())
    if not evan_miya_df.empty:
        team1_evan_miya = evan_miya_df[evan_miya_df['TEAM'] == team1]
        if not team1_evan_miya.empty:
            if 'O RATE' in team1_evan_miya.columns:
                team1_offense_rating = team1_evan_miya['O RATE'].iloc[0]
            if 'D RATE' in team1_evan_miya.columns:
                team1_defense_rating = team1_evan_miya['D RATE'].iloc[0]
            if 'TRUE TEMPO' in team1_evan_miya.columns:
                team1_tempo_rating = team1_evan_miya['TRUE TEMPO'].iloc[0]
        
        team2_evan_miya = evan_miya_df[evan_miya_df['TEAM'] == team2]
        if not team2_evan_miya.empty:
            if 'O RATE' in team2_evan_miya.columns:
                team2_offense_rating = team2_evan_miya['O RATE'].iloc[0]
            if 'D RATE' in team2_evan_miya.columns:
                team2_defense_rating = team2_evan_miya['D RATE'].iloc[0]
            if 'TRUE TEMPO' in team2_evan_miya.columns:
                team2_tempo_rating = team2_evan_miya['TRUE TEMPO'].iloc[0]
    
    # Create relative features
    features = {
        'SeedDiff': 0,  # Will be updated later with actual seeds
        'RelativeAdjEM': team1_kenpom['AdjEM'].iloc[0] - team2_kenpom['AdjEM'].iloc[0],
        'RelativeAdjO': team1_kenpom['AdjO'].iloc[0] - team2_kenpom['AdjO'].iloc[0],
        'RelativeAdjD': team1_kenpom['AdjD'].iloc[0] - team2_kenpom['AdjD'].iloc[0],
        'RelativeAdjT': team1_kenpom['AdjT'].iloc[0] - team2_kenpom['AdjT'].iloc[0],
        'RelativeLuck': team1_kenpom['Luck'].iloc[0] - team2_kenpom['Luck'].iloc[0],
        'RelativeSOS': team1_kenpom['SOS'].iloc[0] - team2_kenpom['SOS'].iloc[0],
        'RelativeNCSOS': team1_kenpom['NCSOS'].iloc[0] - team2_kenpom['NCSOS'].iloc[0],
        'RelativeKenPomRank': team2_kenpom['KenPomRank'].iloc[0] - team1_kenpom['KenPomRank'].iloc[0],
        'RelativeELO': team1_elo - team2_elo,
        'RelativeOffenseRating': team1_offense_rating - team2_offense_rating,
        'RelativeDefenseRating': team1_defense_rating - team2_defense_rating,
        'RelativeQ1Wins': team1_q1_wins - team2_q1_wins
    }
    
    # Add offensive to defensive efficiency ratio
    team1_offdef = team1_kenpom['AdjO'].iloc[0] / team1_kenpom['AdjD'].iloc[0]
    team2_offdef = team2_kenpom['AdjO'].iloc[0] / team2_kenpom['AdjD'].iloc[0]
    features['RelativeOffDefRatio'] = team1_offdef - team2_offdef
    
    # Add expected output (scoring potential)
    team1_expected = team1_kenpom['AdjO'].iloc[0] * (team1_kenpom['AdjT'].iloc[0] / 100)
    team2_expected = team2_kenpom['AdjO'].iloc[0] * (team2_kenpom['AdjT'].iloc[0] / 100)
    features['RelativeExpectedOutput'] = team1_expected - team2_expected
    
    # Include momentum and volatility features (simplified)
    features['RelativeMomentum'] = 0
    features['RelativeVolatility'] = 0
    
    return features

def predict_matchup(model_info, features):
    """
    Predict the winner of a matchup
    """
    # Convert features to DataFrame
    features_df = pd.DataFrame([features])
    
    # Ensure correct order of features
    if 'feature_names' in model_info:
        # Only use features that the model was trained on
        model_features = model_info['feature_names']
        missing_features = set(model_features) - set(features_df.columns)
        
        if missing_features:
            # Add missing features with zero values
            for feature in missing_features:
                features_df[feature] = 0.0
                
        # Ensure correct order
        features_df = features_df[model_features]
    
    # Check if we're using the enhanced model
    if model_info.get('is_enhanced', False):
        # Make prediction using enhanced model
        prediction, probability = predict_with_enhanced_model(model_info, features_df)
        return prediction[0], probability[0]
    else:
        # Original model prediction logic
        model = model_info['model']
        scaler = model_info['scaler']
        
        # Scale features
        features_scaled = scaler.transform(features_df)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        # Get probability if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_scaled)[0]
            probability = probabilities[1] if prediction == 1 else probabilities[0]
        else:
            probability = 0.5
        
        return prediction, probability
def simulate_tournament(model_info, bracket_df, kenpom_df, additional_data):
    """
    Simulate the entire tournament bracket with proper NCAA advancement structure
    """
    print("Simulating 2025 March Madness tournament...")
    
    # Structure for brackets
    regions = bracket_df['Region'].unique()
    bracket_structure = {}
    
    # Standard seeding order for first round matchups
    first_round_matchups = [
        (1, 16), (8, 9),    # These winners play each other in second round
        (5, 12), (4, 13),   # These winners play each other in second round
        (6, 11), (3, 14),   # These winners play each other in second round
        (7, 10), (2, 15)    # These winners play each other in second round
    ]
    
    # Build initial bracket
    for region in regions:
        region_teams = bracket_df[bracket_df['Region'] == region]
        bracket_structure[region] = {}
        
        # Create initial matchups based on standard NCAA format
        for seed1, seed2 in first_round_matchups:
            try:
                team1 = region_teams[region_teams['Seed'] == seed1]['Team'].values[0]
                team2 = region_teams[region_teams['Seed'] == seed2]['Team'].values[0]
                
                bracket_structure[region][f"R64_{seed1}_vs_{seed2}"] = {
                    "teams": [team1, team2],
                    "seeds": [seed1, seed2]
                }
            except (IndexError, KeyError) as e:
                print(f"Warning: Could not create matchup for {region} {seed1} vs {seed2}: {e}")
                # Use placeholder team names if needed
                team1 = f"Team_{region}_{seed1}"
                team2 = f"Team_{region}_{seed2}"
                
                bracket_structure[region][f"R64_{seed1}_vs_{seed2}"] = {
                    "teams": [team1, team2],
                    "seeds": [seed1, seed2]
                }
    
    # Rounds to simulate
    rounds = [
        {"name": "R64", "games_per_region": 8},
        {"name": "R32", "games_per_region": 4},
        {"name": "S16", "games_per_region": 2},
        {"name": "E8", "games_per_region": 1},
        {"name": "F4", "games_per_region": 0},  # Final Four is cross-region
        {"name": "NCG", "games_per_region": 0}  # Championship is cross-region
    ]
    
    # Track advancing teams
    advancing = {}
    for region in regions:
        advancing[region] = {}
    
    # Simulate rounds
    for r_idx, round_info in enumerate(rounds):
        round_name = round_info["name"]
        games_per_region = round_info["games_per_region"]
        
        print(f"Simulating {round_name}...")
        
        if round_name == "R64":
            # First round uses initial bracket
            for region in regions:
                advancing[region][round_name] = []
                
                # Process matchups in the order they appear in the bracket
                for seed1, seed2 in first_round_matchups:
                    matchup_key = f"R64_{seed1}_vs_{seed2}"
                    
                    if matchup_key in bracket_structure[region]:
                        matchup = bracket_structure[region][matchup_key]
                        team1, team2 = matchup["teams"]
                        seed1, seed2 = matchup["seeds"]
                        
                        # Create features
                        features = create_matchup_features(team1, team2, kenpom_df, additional_data)
                        features['SeedDiff'] = seed1 - seed2  # Update seed difference
                        
                        # Predict winner
                        prediction, probability = predict_matchup(model_info, features)
                        
                        # Determine winner
                        winner = team1 if prediction == 1 else team2
                        winner_seed = seed1 if prediction == 1 else seed2
                        
                        # Store result - IMPORTANT: Order is preserved for proper bracket progression
                        advancing[region][round_name].append({
                            "team": winner,
                            "seed": winner_seed,
                            "probability": probability,
                            "matchup": f"{seed1}_vs_{seed2}"  # Store the matchup for reference
                        })
                        
                        print(f"  {region} - {seed1} {team1} vs {seed2} {team2}: {winner} wins ({probability:.2f})")
        
        elif round_name == "R32":
            # Second round - winners play according to bracket structure
            # In R32: (1/16 vs 8/9) and (5/12 vs 4/13) and (6/11 vs 3/14) and (7/10 vs 2/15)
            prev_round = rounds[r_idx - 1]["name"]
            
            for region in regions:
                advancing[region][round_name] = []
                prev_winners = advancing[region][prev_round]
                
                # Match winners according to bracket structure
                # For R32, create the standard pairings
                r32_matchups = [
                    (0, 1),  # 1/16 winner vs 8/9 winner
                    (2, 3),  # 5/12 winner vs 4/13 winner
                    (4, 5),  # 6/11 winner vs 3/14 winner
                    (6, 7)   # 7/10 winner vs 2/15 winner
                ]
                
                for idx1, idx2 in r32_matchups:
                    if idx1 < len(prev_winners) and idx2 < len(prev_winners):
                        team1 = prev_winners[idx1]["team"]
                        team2 = prev_winners[idx2]["team"]
                        seed1 = prev_winners[idx1]["seed"]
                        seed2 = prev_winners[idx2]["seed"]
                        
                        # Create features
                        features = create_matchup_features(team1, team2, kenpom_df, additional_data)
                        features['SeedDiff'] = seed1 - seed2  # Update seed difference
                        
                        # Predict winner
                        prediction, probability = predict_matchup(model_info, features)
                        
                        # Determine winner
                        winner = team1 if prediction == 1 else team2
                        winner_seed = seed1 if prediction == 1 else seed2
                        
                        # Store result
                        advancing[region][round_name].append({
                            "team": winner,
                            "seed": winner_seed,
                            "probability": probability
                        })
                        
                        print(f"  {region} - {seed1} {team1} vs {seed2} {team2}: {winner} wins ({probability:.2f})")
        
        elif round_name == "S16":
            # Sweet 16 - second round winners play according to bracket structure
            # In S16: (1/16/8/9 vs 5/12/4/13) and (6/11/3/14 vs 7/10/2/15)
            prev_round = rounds[r_idx - 1]["name"]
            
            for region in regions:
                advancing[region][round_name] = []
                prev_winners = advancing[region][prev_round]
                
                # Match winners according to bracket structure
                # For S16, match indices 0&1, 2&3
                s16_matchups = [
                    (0, 1),  # Top quarter winner vs second quarter winner
                    (2, 3)   # Third quarter winner vs bottom quarter winner
                ]
                
                for idx1, idx2 in s16_matchups:
                    if idx1 < len(prev_winners) and idx2 < len(prev_winners):
                        team1 = prev_winners[idx1]["team"]
                        team2 = prev_winners[idx2]["team"]
                        seed1 = prev_winners[idx1]["seed"]
                        seed2 = prev_winners[idx2]["seed"]
                        
                        # Create features
                        features = create_matchup_features(team1, team2, kenpom_df, additional_data)
                        features['SeedDiff'] = seed1 - seed2  # Update seed difference
                        
                        # Predict winner
                        prediction, probability = predict_matchup(model_info, features)
                        
                        # Determine winner
                        winner = team1 if prediction == 1 else team2
                        winner_seed = seed1 if prediction == 1 else seed2
                        
                        # Store result
                        advancing[region][round_name].append({
                            "team": winner,
                            "seed": winner_seed,
                            "probability": probability
                        })
                        
                        print(f"  {region} - {seed1} {team1} vs {seed2} {team2}: {winner} wins ({probability:.2f})")
        
        elif round_name == "E8":
            # Elite 8 - sweet 16 winners play for region championship
            # In E8: (1/16/8/9/5/12/4/13 vs 6/11/3/14/7/10/2/15)
            prev_round = rounds[r_idx - 1]["name"]
            
            for region in regions:
                advancing[region][round_name] = []
                prev_winners = advancing[region][prev_round]
                
                # Match winners according to bracket structure
                # For E8, there should be exactly 2 teams left in each region
                if len(prev_winners) == 2:
                    team1 = prev_winners[0]["team"]
                    team2 = prev_winners[1]["team"]
                    seed1 = prev_winners[0]["seed"]
                    seed2 = prev_winners[1]["seed"]
                    
                    # Create features
                    features = create_matchup_features(team1, team2, kenpom_df, additional_data)
                    features['SeedDiff'] = seed1 - seed2  # Update seed difference
                    
                    # Predict winner
                    prediction, probability = predict_matchup(model_info, features)
                    
                    # Determine winner
                    winner = team1 if prediction == 1 else team2
                    winner_seed = seed1 if prediction == 1 else seed2
                    
                    # Store result
                    advancing[region][round_name].append({
                        "team": winner,
                        "seed": winner_seed,
                        "probability": probability
                    })
                    
                    print(f"  {region} - {seed1} {team1} vs {seed2} {team2}: {winner} wins ({probability:.2f})")
                else:
                    print(f"Error: Expected 2 teams in {region} for {round_name}, but found {len(prev_winners)}")
        
        elif round_name == "F4":
            # Final Four - region winners play according to traditional matchups
            # Traditionally: (East vs West, South vs Midwest)
            advancing["Final Four"] = {}
            advancing["Final Four"][round_name] = []
            
            # Get region winners
            region_winners = []
            for region in regions:
                if "E8" in advancing[region] and advancing[region]["E8"]:
                    winner = advancing[region]["E8"][0]
                    winner["region"] = region
                    region_winners.append(winner)
            
            # Ensure we have 4 regions
            if len(region_winners) == 4:
                # Traditional Final Four matchups:
                # 1. East vs West
                # 2. South vs Midwest
                region_matchups = []
                
                # Find East and West regions
                east_idx = next((i for i, w in enumerate(region_winners) if w['region'] == 'East'), -1)
                west_idx = next((i for i, w in enumerate(region_winners) if w['region'] == 'West'), -1)
                
                # Find South and Midwest regions
                south_idx = next((i for i, w in enumerate(region_winners) if w['region'] == 'South'), -1)
                midwest_idx = next((i for i, w in enumerate(region_winners) if w['region'] == 'Midwest'), -1)
                
                # If we have all traditional regions, use them
                if east_idx >= 0 and west_idx >= 0 and south_idx >= 0 and midwest_idx >= 0:
                    region_matchups = [(east_idx, west_idx), (south_idx, midwest_idx)]
                else:
                    # Fallback: just match regions in pairs as they appear
                    region_matchups = [(0, 1), (2, 3)]
                
                # Simulate Final Four matchups
                for idx1, idx2 in region_matchups:
                    team1 = region_winners[idx1]["team"]
                    team2 = region_winners[idx2]["team"]
                    seed1 = region_winners[idx1]["seed"]
                    seed2 = region_winners[idx2]["seed"]
                    region1 = region_winners[idx1]["region"]
                    region2 = region_winners[idx2]["region"]
                    
                    # Create features
                    features = create_matchup_features(team1, team2, kenpom_df, additional_data)
                    features['SeedDiff'] = seed1 - seed2  # Update seed difference
                    
                    # Predict winner
                    prediction, probability = predict_matchup(model_info, features)
                    
                    # Determine winner
                    winner = team1 if prediction == 1 else team2
                    winner_seed = seed1 if prediction == 1 else seed2
                    winner_region = region1 if prediction == 1 else region2
                    
                    # Store result
                    advancing["Final Four"][round_name].append({
                        "team": winner,
                        "seed": winner_seed,
                        "region": winner_region,
                        "probability": probability
                    })
                    
                    print(f"  Final Four - {seed1} {team1} ({region1}) vs {seed2} {team2} ({region2}): {winner} wins ({probability:.2f})")
            else:
                print(f"Error: Expected 4 region winners for Final Four, but found {len(region_winners)}")
        
        elif round_name == "NCG":
            # National Championship Game
            advancing["Final Four"][round_name] = []
            
            # Get Final Four winners
            ff_winners = advancing["Final Four"]["F4"]
            
            if len(ff_winners) == 2:
                team1 = ff_winners[0]["team"]
                team2 = ff_winners[1]["team"]
                seed1 = ff_winners[0]["seed"]
                seed2 = ff_winners[1]["seed"]
                region1 = ff_winners[0]["region"]
                region2 = ff_winners[1]["region"]
                
                # Create features
                features = create_matchup_features(team1, team2, kenpom_df, additional_data)
                features['SeedDiff'] = seed1 - seed2  # Update seed difference
                
                # Predict winner
                prediction, probability = predict_matchup(model_info, features)
                
                # Determine winner
                winner = team1 if prediction == 1 else team2
                winner_seed = seed1 if prediction == 1 else seed2
                winner_region = region1 if prediction == 1 else region2
                
                # Store result
                advancing["Final Four"][round_name].append({
                    "team": winner,
                    "seed": winner_seed,
                    "region": winner_region,
                    "probability": probability
                })
                
                print(f"\nNATIONAL CHAMPION: {winner_seed} {winner} from {winner_region} ({probability:.2f})")
            else:
                print(f"Error: Expected 2 Final Four winners for championship, but found {len(ff_winners)}")
    
    # Save results
    tournament_results = {
        "bracket": bracket_structure,
        "results": advancing,
        "champion": advancing["Final Four"]["NCG"][0]["team"] if "Final Four" in advancing and "NCG" in advancing["Final Four"] and advancing["Final Four"]["NCG"] else "Unknown"
    }
    
    os.makedirs('output', exist_ok=True)
    with open('output/tournament_prediction.json', 'w') as f:
        serializable_results = make_json_serializable(tournament_results)
        json.dump(serializable_results, f, indent=2)
    
    print("Tournament predictions saved to output/tournament_prediction.json")
    
    return tournament_results

def main():
    """Main prediction function"""
    # Load model
    try:
        model_info = load_model()
        print("Model loaded successfully.")
    except FileNotFoundError as e:
        print(e)
        return
    
    # Load current data
    try:
        bracket_df, kenpom_df, additional_data = load_current_data()
        print("Current data loaded successfully.")
        print(f"Bracket: {len(bracket_df)} teams")
        print(f"KenPom data: {len(kenpom_df)} teams")
    except FileNotFoundError as e:
        print(e)
        return
    
    # Simulate tournament
    tournament_results = simulate_tournament(model_info, bracket_df, kenpom_df, additional_data)
    
    print("Bracket prediction complete!")

if __name__ == "__main__":
    main()