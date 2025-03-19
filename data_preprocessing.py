# data_preprocessing.py
import pandas as pd
import numpy as np
import os

def load_raw_data():
    """Load all raw data files"""
    # Define paths to the data files
    kenpom_path = 'data/raw/kenpom_ratings.csv'
    tournament_path = 'data/raw/tournament_results.csv'  # This file doesn't exist
    tournament_matchups_path = 'data/raw/Tournament Matchups.csv'  # This is the file you actually have
    
    # Check if essential files exist - now checking for Tournament Matchups instead
    if not os.path.exists(kenpom_path) or not os.path.exists(tournament_matchups_path):
        print("ERROR: Missing essential data files:")
        if not os.path.exists(kenpom_path):
            print(f"  - {kenpom_path}")
        if not os.path.exists(tournament_matchups_path):
            print(f"  - {tournament_matchups_path}")
        
        # Print files that exist in the data/raw directory
        raw_dir = 'data/raw'
        if os.path.exists(raw_dir):
            print("\nFiles found in data/raw directory:")
            for file in os.listdir(raw_dir):
                print(f"  - {os.path.join(raw_dir, file)}")
            
        raise FileNotFoundError("Raw data files not found. Run data_collection.py first.")
    
    # Load essential data
    kenpom_data = pd.read_csv(kenpom_path)
    
    # Load and process Tournament Matchups instead of tournament_results
    tournament_matchups = pd.read_csv(tournament_matchups_path)
    print(f"Loaded Tournament Matchups data: {len(tournament_matchups)} rows")
    
    # Process tournament matchups into the format we need
    tournament_data = process_tournament_matchups(tournament_matchups)
    print(f"Processed Tournament data: {len(tournament_data)} games")
    
    # Load additional data sources
    additional_data = {}
    
    # Define additional data files to check
    additional_files = {
        'resumes': 'data/raw/Resumes.csv',
        'team_rankings': 'data/raw/TeamRankings.csv',
        'evan_miya': 'data/raw/EvanMiya.csv'
        # Removed Heat Check Ratings
    }
    
    # Load each additional file if it exists
    for key, file_path in additional_files.items():
        if os.path.exists(file_path):
            try:
                additional_data[key] = pd.read_csv(file_path)
                print(f"Loaded {key} data: {additional_data[key].shape[0]} rows")
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
        else:
            print(f"Note: {file_path} not found, some features will be limited")
    
    return kenpom_data, tournament_data, additional_data

def process_tournament_matchups(matchups_df):
    """
    Process Tournament Matchups data into the format needed for matchup preparation
    Convert from individual team rows to complete game results
    """
    print("Processing Tournament Matchups data...")
    
    # Create a list to store game results
    results = []
    
    # Make a copy to avoid modifying the original
    df = matchups_df.copy()
    
    # Clean team names for consistency
    if 'TEAM' in df.columns:
        df['TEAM'] = df['TEAM'].apply(lambda x: clean_team_name(x) if isinstance(x, str) else x)
    
    # Determine which column identifies the game
    game_id_col = None
    if 'BY GAME NO' in df.columns:
        game_id_col = 'BY GAME NO'
    elif 'BY YEAR NO' in df.columns:
        game_id_col = 'BY YEAR NO'
    
    # Check if we have the right columns for processing
    required_cols = ['YEAR', 'TEAM', 'SEED', 'ROUND', 'SCORE']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        print(f"WARNING: Tournament Matchups is missing columns: {missing}")
        print("Available columns:", df.columns.tolist())
        # Create a minimal dataset with default values
        return pd.DataFrame(columns=['Year', 'Round', 'GameID', 'WinningTeam', 'WinningSeed', 
                                    'WinningScore', 'LosingTeam', 'LosingSeed', 'LosingScore', 'ScoreDiff'])
    
    if game_id_col:
        print(f"Using '{game_id_col}' to identify game pairs")
        
        # Group by year and game number to find teams that played against each other
        df.sort_values(['YEAR', game_id_col], inplace=True)
        
        # Process games by grouping teams with the same YEAR and game number
        game_groups = df.groupby(['YEAR', game_id_col])
        
        for (year, game_no), game_group in game_groups:
            # Skip if we don't have exactly 2 teams in a game
            if len(game_group) != 2:
                print(f"Warning: Game {game_no} in year {year} has {len(game_group)} teams, skipping")
                continue
                
            teams = game_group.reset_index(drop=True)
            team1 = teams.iloc[0]
            team2 = teams.iloc[1]
            
            # When both teams have a SCORE, determine winner and loser
            if pd.notna(team1['SCORE']) and pd.notna(team2['SCORE']):
                if team1['SCORE'] > team2['SCORE']:
                    winner, loser = team1, team2
                else:
                    winner, loser = team2, team1
                
                # Convert round number to name
                round_name = convert_round_to_name(int(winner['ROUND']))
                
                # Create game result entry
                game_result = {
                    'Year': int(year),
                    'Round': round_name,
                    'GameID': int(game_no),  # Use the game number as GameID
                    'WinningTeam': winner['TEAM'],
                    'WinningSeed': int(winner['SEED']),
                    'WinningScore': float(winner['SCORE']),
                    'LosingTeam': loser['TEAM'],
                    'LosingSeed': int(loser['SEED']),
                    'LosingScore': float(loser['SCORE']),
                    'ScoreDiff': float(winner['SCORE'] - loser['SCORE'])
                }
                
                results.append(game_result)
            else:
                # If scores aren't available, this may be a future matchup
                print(f"Note: Scores not available for game {game_no} in year {year}, skipping")
    else:
        # Fall back to the old consecutive row pairing method
        print("WARNING: No game identifier column found. Using sequential pairing which may be inaccurate.")
        print("Available columns:", df.columns.tolist())
        print("Consider adding 'BY GAME NO' column to identify which teams played each other.")
        
        # Add a sequential game ID for tracking
        df['GameID'] = range(len(df))
        
        # Process rows in pairs (each game has 2 teams)
        for i in range(0, len(df) - 1, 2):
            # Get the two teams in this game
            try:
                team1 = df.iloc[i]
                team2 = df.iloc[i + 1]
                
                # Verify teams are from the same year (basic validation)
                if team1['YEAR'] != team2['YEAR']:
                    print(f"Warning: Teams at index {i} and {i+1} are from different years ({team1['YEAR']} vs {team2['YEAR']}), skipping")
                    continue
                
                # When both teams have a SCORE, determine winner and loser
                if pd.notna(team1['SCORE']) and pd.notna(team2['SCORE']):
                    if team1['SCORE'] > team2['SCORE']:
                        winner, loser = team1, team2
                    else:
                        winner, loser = team2, team1
                    
                    # Convert round number to name
                    round_name = convert_round_to_name(int(winner['ROUND']))
                    
                    # Create game result entry
                    game_result = {
                        'Year': int(winner['YEAR']),
                        'Round': round_name,
                        'GameID': i // 2,  # Generate sequential game ID
                        'WinningTeam': winner['TEAM'],
                        'WinningSeed': int(winner['SEED']),
                        'WinningScore': float(winner['SCORE']),
                        'LosingTeam': loser['TEAM'],
                        'LosingSeed': int(loser['SEED']),
                        'LosingScore': float(loser['SCORE']),
                        'ScoreDiff': float(winner['SCORE'] - loser['SCORE'])
                    }
                    
                    results.append(game_result)
                else:
                    # If scores aren't available, this may be a future matchup
                    print(f"Note: Scores not available for game at index {i} and {i+1}, skipping")
            except Exception as e:
                print(f"Error processing game at index {i}: {e}")
    
    # Create DataFrame from results
    tournament_df = pd.DataFrame(results)
    
    if tournament_df.empty:
        print("WARNING: No tournament results could be extracted from the matchups data")
        return pd.DataFrame(columns=['Year', 'Round', 'GameID', 'WinningTeam', 'WinningSeed', 
                                    'WinningScore', 'LosingTeam', 'LosingSeed', 'LosingScore', 'ScoreDiff'])
    
    # Save processed tournament data for reuse
    tournament_df.to_csv('data/raw/tournament_results.csv', index=False)
    print(f"Saved processed tournament results to data/raw/tournament_results.csv ({len(tournament_df)} games)")
    
    return tournament_df

def convert_round_to_name(round_num):
    """Convert numeric round to standard round name"""
    round_mapping = {
        1: 'R64',  # First Round (Round of 64)
        2: 'R32',  # Second Round (Round of 32)
        3: 'S16',  # Sweet Sixteen
        4: 'E8',   # Elite Eight
        5: 'F4',   # Final Four
        6: 'NCG'   # National Championship Game
    }
    
    return round_mapping.get(round_num, f'Round{round_num}')

def clean_team_name(name):
    """Clean and standardize team names for consistency across data sources"""
    if not isinstance(name, str):
        return str(name)
        
    # Remove common suffixes
    name = name.strip()
    
    # Handle specific name variations
    mappings = {
        'UNC': 'North Carolina',
        'NC State': 'North Carolina State',
        'USC': 'Southern California',
        'UCF': 'Central Florida',
        'UCONN': 'Connecticut',
        'UConn': 'Connecticut',
        'Pitt': 'Pittsburgh',
        'Saint': 'St.',
        'St ': 'St. ',
    }
    
    for key, value in mappings.items():
        if key == name:
            return value
    
    return name

def prepare_matchup_data(tournament_data, kenpom_data, additional_data):
    """
    Prepare matchup data for model training using real data sources
    """
    print("Preparing matchup data...")
    
    # Get tournament years
    years = tournament_data['Year'].unique()
    print(f"Processing tournament years: {years}")
    
    # Create empty list for matchups
    matchups = []
    matchup_count = 0
    skipped_count = 0
    
    for year in years:
        year_tournament = tournament_data[tournament_data['Year'] == year]
        year_kenpom = kenpom_data[kenpom_data['Year'] == year]
        
        # Get additional data for this year
        year_resumes = additional_data.get('resumes', pd.DataFrame())
        if not year_resumes.empty:
            year_resumes = year_resumes[year_resumes['YEAR'] == year]
        
        year_team_rankings = additional_data.get('team_rankings', pd.DataFrame())
        if not year_team_rankings.empty:
            year_team_rankings = year_team_rankings[year_team_rankings['YEAR'] == year]
        
        year_evan_miya = additional_data.get('evan_miya', pd.DataFrame())
        if not year_evan_miya.empty:
            year_evan_miya = year_evan_miya[year_evan_miya['YEAR'] == year]
        
        # Removed Heat Check ratings
            
        print(f"Processing {year} tournament ({len(year_tournament)} games)")
        print(f"KenPom teams for {year}: {len(year_kenpom)}")
        print(f"Additional data sources available:")
        print(f"  - Resumes: {len(year_resumes) if not year_resumes.empty else 0} teams")
        print(f"  - TeamRankings: {len(year_team_rankings) if not year_team_rankings.empty else 0} teams")
        print(f"  - EvanMiya: {len(year_evan_miya) if not year_evan_miya.empty else 0} teams")
        
        # Process each game
        for _, game in year_tournament.iterrows():
            team1 = game['WinningTeam']
            team2 = game['LosingTeam']
            team1_seed = game['WinningSeed']
            team2_seed = game['LosingSeed']
            round_name = game['Round']
            
            # Ensure team names exist in KenPom data
            team1_kenpom = year_kenpom[year_kenpom['Team'] == team1]
            team2_kenpom = year_kenpom[year_kenpom['Team'] == team2]
            
            if team1_kenpom.empty:
                print(f"WARNING: Team {team1} not found in {year} KenPom data")
                # Look for similar team names as a fallback
                similar_teams = year_kenpom[year_kenpom['Team'].str.contains(team1.split('_')[1], case=False) if '_' in team1 else year_kenpom['Team'].str.contains(team1, case=False)]
                if not similar_teams.empty:
                    team1_kenpom = similar_teams.iloc[0:1]
                    print(f"  Using {team1_kenpom['Team'].iloc[0]} as a substitute")
            
            if team2_kenpom.empty:
                print(f"WARNING: Team {team2} not found in {year} KenPom data")
                # Look for similar team names as a fallback
                similar_teams = year_kenpom[year_kenpom['Team'].str.contains(team2.split('_')[1], case=False) if '_' in team2 else year_kenpom['Team'].str.contains(team2, case=False)]
                if not similar_teams.empty:
                    team2_kenpom = similar_teams.iloc[0:1]
                    print(f"  Using {team2_kenpom['Team'].iloc[0]} as a substitute")
            
            # Skip if either team is still not found
            if team1_kenpom.empty or team2_kenpom.empty:
                skipped_count += 1
                if skipped_count <= 5:  # Limit number of skipped warnings
                    print(f"Skipping matchup: {team1} vs {team2} - KenPom data not found")
                elif skipped_count == 6:
                    print("Additional matchups skipped...")
                continue
            
            # Get KenPom data
            team1_kenpom_row = team1_kenpom.iloc[0]
            team2_kenpom_row = team2_kenpom.iloc[0]
            
            # Initialize additional metrics with default values
            team1_elo = 1500
            team2_elo = 1500
            team1_offense_rating = None
            team2_offense_rating = None
            team1_defense_rating = None
            team2_defense_rating = None
            team1_q1_wins = 0
            team2_q1_wins = 0
            
            # Get data from Resumes if available
            if not year_resumes.empty:
                team1_resumes = year_resumes[year_resumes['TEAM'] == team1]
                if not team1_resumes.empty:
                    if 'ELO' in team1_resumes.columns:
                        team1_elo = team1_resumes['ELO'].iloc[0]
                    if 'Q1 W' in team1_resumes.columns:
                        team1_q1_wins = team1_resumes['Q1 W'].iloc[0]
                
                team2_resumes = year_resumes[year_resumes['TEAM'] == team2]
                if not team2_resumes.empty:
                    if 'ELO' in team2_resumes.columns:
                        team2_elo = team2_resumes['ELO'].iloc[0]
                    if 'Q1 W' in team2_resumes.columns:
                        team2_q1_wins = team2_resumes['Q1 W'].iloc[0]
            
            # Get data from EvanMiya if available
            if not year_evan_miya.empty:
                team1_evan_miya = year_evan_miya[year_evan_miya['TEAM'] == team1]
                if not team1_evan_miya.empty:
                    if 'O RATE' in team1_evan_miya.columns:
                        team1_offense_rating = team1_evan_miya['O RATE'].iloc[0]
                    if 'D RATE' in team1_evan_miya.columns:
                        team1_defense_rating = team1_evan_miya['D RATE'].iloc[0]
                
                team2_evan_miya = year_evan_miya[year_evan_miya['TEAM'] == team2]
                if not team2_evan_miya.empty:
                    if 'O RATE' in team2_evan_miya.columns:
                        team2_offense_rating = team2_evan_miya['O RATE'].iloc[0]
                    if 'D RATE' in team2_evan_miya.columns:
                        team2_defense_rating = team2_evan_miya['D RATE'].iloc[0]
            
            # Removed Heat Check features
            
            # Create matchup features
            matchup = {
                'Year': year,
                'Round': round_name,
                'Team1': team1,
                'Team2': team2,
                'Team1_Seed': team1_seed,
                'Team2_Seed': team2_seed,
                'SeedDiff': team1_seed - team2_seed,
                
                # KenPom features
                'Team1_AdjEM': team1_kenpom_row['AdjEM'],
                'Team2_AdjEM': team2_kenpom_row['AdjEM'],
                'Team1_AdjO': team1_kenpom_row['AdjO'],
                'Team2_AdjO': team2_kenpom_row['AdjO'],
                'Team1_AdjD': team1_kenpom_row['AdjD'],
                'Team2_AdjD': team2_kenpom_row['AdjD'],
                'Team1_AdjT': team1_kenpom_row['AdjT'],
                'Team2_AdjT': team2_kenpom_row['AdjT'],
                'Team1_Luck': team1_kenpom_row['Luck'],
                'Team2_Luck': team2_kenpom_row['Luck'],
                'Team1_SOS': team1_kenpom_row['SOS'],
                'Team2_SOS': team2_kenpom_row['SOS'],
                'Team1_NCSOS': team1_kenpom_row['NCSOS'],
                'Team2_NCSOS': team2_kenpom_row['NCSOS'],
                'Team1_KenPomRank': team1_kenpom_row['KenPomRank'],
                'Team2_KenPomRank': team2_kenpom_row['KenPomRank'],
                
                # ELO ratings (from Resumes)
                'Team1_ELO': team1_elo,
                'Team2_ELO': team2_elo,
                
                # Quality win counts (from Resumes)
                'Team1_Q1Wins': team1_q1_wins,
                'Team2_Q1Wins': team2_q1_wins,
                
                # Target variable: Team1 won (always true in this loop)
                'Team1_Won': 1
            }
            
            # Add offense and defense ratings if available
            if team1_offense_rating is not None:
                matchup['Team1_OffenseRating'] = team1_offense_rating
            if team2_offense_rating is not None:
                matchup['Team2_OffenseRating'] = team2_offense_rating
            if team1_defense_rating is not None:
                matchup['Team1_DefenseRating'] = team1_defense_rating
            if team2_defense_rating is not None:
                matchup['Team2_DefenseRating'] = team2_defense_rating
            
            # Removed Heat Check features
            
            # Add to matchups list
            matchups.append(matchup)
            matchup_count += 1
            
            # Add reverse matchup (to avoid bias)
            reverse_matchup = matchup.copy()
            reverse_matchup['Team1'] = team2
            reverse_matchup['Team2'] = team1
            reverse_matchup['Team1_Seed'] = team2_seed
            reverse_matchup['Team2_Seed'] = team1_seed
            reverse_matchup['SeedDiff'] = team2_seed - team1_seed
            
            # Swap KenPom features
            for feature in ['AdjEM', 'AdjO', 'AdjD', 'AdjT', 'Luck', 'SOS', 'NCSOS', 'KenPomRank']:
                reverse_matchup[f'Team1_{feature}'] = matchup[f'Team2_{feature}']
                reverse_matchup[f'Team2_{feature}'] = matchup[f'Team1_{feature}']
                
            # Swap additional metrics
            for feature in ['ELO', 'Q1Wins']:
                reverse_matchup[f'Team1_{feature}'] = matchup[f'Team2_{feature}']
                reverse_matchup[f'Team2_{feature}'] = matchup[f'Team1_{feature}']
            
            # Swap offense and defense ratings if they exist
            for feature in ['OffenseRating', 'DefenseRating']:
                if f'Team1_{feature}' in matchup and f'Team2_{feature}' in matchup:
                    reverse_matchup[f'Team1_{feature}'] = matchup[f'Team2_{feature}']
                    reverse_matchup[f'Team2_{feature}'] = matchup[f'Team1_{feature}']
            
            # Removed Heat Check features swap
            
            # Reverse target (Team1 lost)
            reverse_matchup['Team1_Won'] = 0
            
            # Add reverse matchup
            matchups.append(reverse_matchup)
            matchup_count += 1
    
    print(f"Created {matchup_count} matchups (including reverse matchups)")
    print(f"Skipped {skipped_count} matchups due to missing data")
    
    # If there are no valid matchups, we cannot proceed
    if not matchups:
        raise ValueError("No valid matchups could be created from the real data. Please check your data sources.")
    
    # Create DataFrame
    matchups_df = pd.DataFrame(matchups)
    
    # Save to CSV
    matchups_df.to_csv('data/processed/matchups.csv', index=False)
    print(f"Matchup data saved to data/processed/matchups.csv ({len(matchups_df)} rows)")
    
    # Print sample
    if not matchups_df.empty:
        print("\nSample matchup data:")
        print(matchups_df.head())
    
    return matchups_df

def load_current_bracket(year=2025):
    """Load the current year's bracket"""
    bracket_path = f'data/raw/bracket_{year}.csv'
    
    if not os.path.exists(bracket_path):
        raise FileNotFoundError(f"Current bracket file not found. Run data_collection.py first.")
    
    bracket_df = pd.read_csv(bracket_path)
    return bracket_df

def main():
    """Main preprocessing function"""
    # Create directories if they don't exist
    os.makedirs('data/processed', exist_ok=True)
    
    print("Starting data preprocessing...")
    
    try:
        # Load raw data
        kenpom_data, tournament_data, additional_data = load_raw_data()
        print("Raw data loaded successfully.")
        
        # Prepare matchup data
        matchups_df = prepare_matchup_data(tournament_data, kenpom_data, additional_data)
        print(f"Successfully created matchup data with {len(matchups_df)} rows.")
        
        # Load current bracket
        try:
            current_bracket = load_current_bracket(2025)
            print("Current bracket loaded successfully.")
            print(f"Current bracket has {len(current_bracket)} teams.")
        except FileNotFoundError as e:
            print(e)
        
        print("Data preprocessing complete!")
    except Exception as e:
        print(f"Error in data preprocessing: {e}")
        raise

if __name__ == "__main__":
    main()