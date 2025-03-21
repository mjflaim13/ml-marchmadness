# backtesting.py
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.metrics import accuracy_score

def load_model():
    """Load the trained model"""
    model_path = 'models/final_model.pkl'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Run model_training.py first.")
    
    with open(model_path, 'rb') as f:
        model_info = pickle.load(f)
    
    return model_info

def load_matchup_data():
    """Load enhanced matchup data if available, otherwise regular matchups"""
    enhanced_path = 'data/processed/enhanced_matchups.csv'
    regular_path = 'data/processed/matchups.csv'
    
    if os.path.exists(enhanced_path):
        print("Loading enhanced matchups with engineered features")
        matchups_df = pd.read_csv(enhanced_path)
    elif os.path.exists(regular_path):
        print("Loading regular matchups")
        matchups_df = pd.read_csv(regular_path)
    else:
        raise FileNotFoundError("Matchup data not found. Run data_preprocessing.py first.")
    
    return matchups_df

def load_feature_list():
    """Load list of features from feature_list.csv"""
    feature_list_path = 'data/processed/feature_list.csv'
    
    if not os.path.exists(feature_list_path):
        print("Feature list not found, will use available features in data")
        return None
    
    feature_df = pd.read_csv(feature_list_path)
    if 'feature' in feature_df.columns:
        return feature_df['feature'].tolist()
    else:
        print("Feature list file does not have expected 'feature' column")
        return None

def ensure_model_features(matchups_df, model_info):
    """Ensure the dataframe has all features required by the model"""
    if 'feature_names' not in model_info:
        print("Model does not specify feature names")
        return matchups_df, matchups_df.columns.tolist()
    
    model_features = model_info['feature_names']
    
    # Check which features are missing
    missing_features = [f for f in model_features if f not in matchups_df.columns]
    
    if missing_features:
        print(f"WARNING: Missing {len(missing_features)} features required by model")
        print(f"Missing features: {missing_features}")
        print("Creating placeholders for missing features (filled with zeros)")
        
        # Create missing features with zero values
        for feature in missing_features:
            matchups_df[feature] = 0.0
            
    return matchups_df, model_features

def backtest_by_year(model_info, matchups_df):
    """
    Backtest the model on historical tournament data, year by year
    """
    print("Backtesting model on historical tournament data...")
    
    # Get model components
    model = model_info['model']
    scaler = model_info['scaler']
    
    # Ensure dataframe has all required features
    matchups_df, features = ensure_model_features(matchups_df, model_info)
    
    # Get unique years
    years = matchups_df['Year'].unique()
    
    results = {}
    
    for year in years:
        print(f"Backtesting for {year}...")
        
        # Filter data for the current year
        year_data = matchups_df[matchups_df['Year'] == year]
        
        # We need to ensure we don't double-count games due to our symmetrical matchup creation
        # Get unique games based on Team1, Team2, and Round
        unique_games = year_data.drop_duplicates(subset=['Team1', 'Team2', 'Round'])
        
        # Extract features
        X = unique_games[features]
        y_true = unique_games['Team1_Won']
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make predictions
        y_pred = model.predict(X_scaled)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        # Store results
        results[year] = {
            'accuracy': accuracy,
            'games': len(unique_games),
            'correct': (y_true == y_pred).sum()
        }
        
        print(f"Year {year} - Accuracy: {accuracy:.4f} ({results[year]['correct']}/{results[year]['games']} games)")
    
    # Calculate overall accuracy
    total_games = sum(results[year]['games'] for year in years)
    total_correct = sum(results[year]['correct'] for year in years)
    overall_accuracy = total_correct / total_games
    
    print(f"\nOverall Accuracy: {overall_accuracy:.4f} ({total_correct}/{total_games} games)")
    
    return results

def backtest_by_round(model_info, matchups_df):
    """
    Backtest the model on historical tournament data, round by round
    """
    print("\nBacktesting model by tournament round...")
    
    # Get model components
    model = model_info['model']
    scaler = model_info['scaler']
    
    # Ensure dataframe has all required features
    matchups_df, features = ensure_model_features(matchups_df, model_info)
    
    # Define round order
    rounds = ['R64', 'R32', 'S16', 'E8', 'F4', 'NCG']
    
    results = {}
    
    for round_name in rounds:
        print(f"Backtesting for {round_name}...")
        
        # Filter data for the current round
        round_data = matchups_df[matchups_df['Round'] == round_name]
        
        if len(round_data) == 0:
            print(f"No data found for round {round_name}")
            continue
        
        # Get unique games
        unique_games = round_data.drop_duplicates(subset=['Team1', 'Team2', 'Year'])
        
        # Extract features
        X = unique_games[features]
        y_true = unique_games['Team1_Won']
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make predictions
        y_pred = model.predict(X_scaled)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        # Store results
        results[round_name] = {
            'accuracy': accuracy,
            'games': len(unique_games),
            'correct': (y_true == y_pred).sum()
        }
        
        print(f"Round {round_name} - Accuracy: {accuracy:.4f} ({results[round_name]['correct']}/{results[round_name]['games']} games)")
    
    return results

def backtest_by_seed_diff(model_info, matchups_df):
    """
    Backtest the model based on seed differences
    """
    print("\nBacktesting model by seed difference...")
    
    # Get model components
    model = model_info['model']
    scaler = model_info['scaler']
    
    # Ensure dataframe has all required features
    matchups_df, features = ensure_model_features(matchups_df, model_info)
    
    # Check if we have seed data
    has_seed_data = all(col in matchups_df.columns for col in ['Team1_Seed', 'Team2_Seed'])
    
    if not has_seed_data:
        print("WARNING: Seed data missing, skipping seed difference backtesting")
        return {}
    
    # Define seed difference groups
    seed_diff_groups = [
        ('Favorites (Seeds 1-4 vs 13-16)', lambda x: (x['Team1_Seed'] <= 4 and x['Team2_Seed'] >= 13) or (x['Team2_Seed'] <= 4 and x['Team1_Seed'] >= 13)),
        ('Mid-range matchups (Seeds 5-8 vs 9-12)', lambda x: (x['Team1_Seed'] >= 5 and x['Team1_Seed'] <= 8 and x['Team2_Seed'] >= 9 and x['Team2_Seed'] <= 12) or 
                                                              (x['Team2_Seed'] >= 5 and x['Team2_Seed'] <= 8 and x['Team1_Seed'] >= 9 and x['Team1_Seed'] <= 12)),
        ('Similar seeds (Diff ≤ 2)', lambda x: abs(x['Team1_Seed'] - x['Team2_Seed']) <= 2),
        ('Upsets (Lower seed won)', lambda x: (x['Team1_Seed'] > x['Team2_Seed'] and x['Team1_Won'] == 1) or (x['Team2_Seed'] > x['Team1_Seed'] and x['Team1_Won'] == 0))
    ]
    
    results = {}
    
    for group_name, filter_func in seed_diff_groups:
        print(f"Backtesting for {group_name}...")
        
        try:
            # Filter data for the current group
            group_data = matchups_df[matchups_df.apply(filter_func, axis=1)]
            
            # Get unique games
            unique_games = group_data.drop_duplicates(subset=['Team1', 'Team2', 'Year', 'Round'])
            
            if len(unique_games) == 0:
                print(f"No games found for {group_name}")
                continue
            
            # Extract features
            X = unique_games[features]
            y_true = unique_games['Team1_Won']
            
            # Scale features
            X_scaled = scaler.transform(X)
            
            # Make predictions
            y_pred = model.predict(X_scaled)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_true, y_pred)
            
            # Store results
            results[group_name] = {
                'accuracy': accuracy,
                'games': len(unique_games),
                'correct': (y_true == y_pred).sum()
            }
            
            print(f"{group_name} - Accuracy: {accuracy:.4f} ({results[group_name]['correct']}/{results[group_name]['games']} games)")
        except Exception as e:
            print(f"Error processing {group_name}: {e}")
    
    return results

def simulate_brackets(model_info, matchups_df, years=None):
    """
    Simulate full tournament brackets for given years and evaluate performance
    """
    print("\nSimulating full tournament brackets...")
    
    # Get model components
    model = model_info['model']
    scaler = model_info['scaler']
    
    # Ensure dataframe has all required features
    matchups_df, features = ensure_model_features(matchups_df, model_info)
    
    if years is None:
        years = matchups_df['Year'].unique()
    
    # Define round progression
    rounds = ['R64', 'R32', 'S16', 'E8', 'F4', 'NCG']
    
    bracket_results = {}
    
    for year in years:
        print(f"Simulating {year} bracket...")
        
        # Filter data for the current year
        year_data = matchups_df[matchups_df['Year'] == year]
        
        if len(year_data) == 0:
            print(f"No data found for year {year}")
            continue
        
        # Create a dict to track advancing teams
        advancing_teams = {}
        
        # Start with first round (R64)
        round_games = year_data[year_data['Round'] == 'R64'].drop_duplicates(subset=['Team1', 'Team2'])
        
        if len(round_games) == 0:
            print(f"No first round games found for {year}")
            continue
        
        for r_idx, round_name in enumerate(rounds):
            print(f"  Round: {round_name}")
            
            # For first round, use actual matchups
            if r_idx == 0:
                matchups = round_games
            else:
                # For later rounds, create matchups from advancing teams
                prev_round = rounds[r_idx - 1]
                if prev_round not in advancing_teams or not advancing_teams[prev_round]:
                    print(f"No advancing teams from {prev_round}")
                    break
                
                # Get advancing teams from previous round
                teams = advancing_teams[prev_round]
                
                # Create matchups based on tournament bracket structure
                # This is a simplified version that may not match the actual bracket exactly
                matchups = []
                for i in range(0, len(teams), 2):
                    if i + 1 < len(teams):
                        team1 = teams[i]
                        team2 = teams[i + 1]
                        
                        # Find actual matchup data for these teams if it exists
                        actual_matchup = year_data[
                            (((year_data['Team1'] == team1) & (year_data['Team2'] == team2)) |
                             ((year_data['Team1'] == team2) & (year_data['Team2'] == team1))) &
                            (year_data['Round'] == round_name)
                        ]
                        
                        if len(actual_matchup) > 0:
                            matchups.append(actual_matchup.iloc[0])
                
                if not matchups:
                    print(f"Could not create matchups for {round_name}")
                    break
                
                matchups = pd.DataFrame(matchups)
            
            if len(matchups) == 0:
                print(f"No matchups for round {round_name}")
                break
                
            # Extract features
            X = matchups[features]
            
            # Scale features
            X_scaled = scaler.transform(X)
            
            # Make predictions
            y_pred = model.predict(X_scaled)
            
            # Get winning teams
            winning_teams = []
            for i, pred in enumerate(y_pred):
                if pred == 1:
                    winning_teams.append(matchups.iloc[i]['Team1'])
                else:
                    winning_teams.append(matchups.iloc[i]['Team2'])
            
            # Store advancing teams
            advancing_teams[round_name] = winning_teams
        
        # Store results for this year
        bracket_results[year] = advancing_teams
    
    return bracket_results

def main():
    """Main backtesting function"""
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Load model
    try:
        model_info = load_model()
        print("Model loaded successfully.")
    except FileNotFoundError as e:
        print(e)
        return
    
    # Load matchup data
    try:
        matchups_df = load_matchup_data()
        print(f"Matchup data loaded successfully: {matchups_df.shape[0]} rows, {matchups_df.shape[1]} columns")
    except FileNotFoundError as e:
        print(e)
        return
    
    # Backtest by year
    year_results = backtest_by_year(model_info, matchups_df)
    
    # Backtest by round
    round_results = backtest_by_round(model_info, matchups_df)
    
    # Backtest by seed difference
    seed_diff_results = backtest_by_seed_diff(model_info, matchups_df)
    
    # Simulate brackets
    bracket_results = simulate_brackets(model_info, matchups_df)
    
    # Save backtesting results
    backtesting_results = {
        'year_results': year_results,
        'round_results': round_results,
        'seed_diff_results': seed_diff_results,
        'bracket_results': bracket_results
    }
    
    with open('output/backtesting_results.pkl', 'wb') as f:
        pickle.dump(backtesting_results, f)
    
    print("Backtesting results saved to output/backtesting_results.pkl")
    
    # Create summary report
    summary = {
        'overall_accuracy': sum(year_results[y]['correct'] for y in year_results) / 
                           sum(year_results[y]['games'] for y in year_results) if year_results else 0,
        'year_accuracies': {y: year_results[y]['accuracy'] for y in year_results} if year_results else {},
        'round_accuracies': {r: round_results[r]['accuracy'] for r in round_results} if round_results else {},
        'seed_diff_accuracies': {s: seed_diff_results[s]['accuracy'] for s in seed_diff_results} if seed_diff_results else {}
    }
    
    # Save summary as CSV
    summary_df = pd.DataFrame({
        'Category': ['Overall'] + 
                    [f'Year: {y}' for y in summary['year_accuracies']] + 
                    [f'Round: {r}' for r in summary['round_accuracies']] +
                    [f'Matchup: {s}' for s in summary['seed_diff_accuracies']],
        'Accuracy': [summary['overall_accuracy']] + 
                    list(summary['year_accuracies'].values()) + 
                    list(summary['round_accuracies'].values()) +
                    list(summary['seed_diff_accuracies'].values())
    })
    
    summary_df.to_csv('output/backtesting_summary.csv', index=False)
    print("Backtesting summary saved to output/backtesting_summary.csv")
    
    print("Backtesting complete!")

if __name__ == "__main__":
    main()