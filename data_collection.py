# data_collection.py
import os
import pandas as pd
import numpy as np
import re
import argparse
import sys
import logging
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Tournament years (excluding 2020 due to COVID)
TOURNAMENT_YEARS = [2018, 2019, 2021, 2022, 2023, 2024]
TARGET_YEAR = 2025

def create_data_dir():
    """Create data directories if they don't exist"""
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)

def clean_team_name(name):
    """Clean and standardize team names for consistency across data sources"""
    if not isinstance(name, str):
        return str(name)
        
    # Remove common suffixes
    name = re.sub(r'\s+(University|College|State University|State College)$', '', name, flags=re.IGNORECASE)
    
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
        name = re.sub(r'\b' + re.escape(key) + r'\b', value, name)
    
    # Remove any trailing/leading whitespace and standardize format
    name = name.strip()
    
    return name

def get_kenpom_file_pattern(year):
    """Get the KenPom file pattern based on the year"""
    if year == 2025:
        return f"summary{str(year)[-2:]}.csv"  # summary25.csv
    else:
        return f"summary{str(year)[-2:]}_pt.csv"  # summary18_pt.csv, etc.

def load_manual_kenpom_data(data_dir='data/raw', years=None):
    """
    Load manually downloaded KenPom data files with the specific naming convention
    
    Expected formats: 
    - summary25.csv for 2025
    - summary18_pt.csv, summary19_pt.csv, etc. for other years
    """
    if years is None:
        years = TOURNAMENT_YEARS + [TARGET_YEAR]
    
    logger.info(f"Loading manually downloaded KenPom data for years: {years}")
    
    # Create a DataFrame to store all years' data
    all_data = pd.DataFrame()
    
    for year in years:
        # Get the correct file pattern based on year
        file_pattern = get_kenpom_file_pattern(year)
        kenpom_file = os.path.join(data_dir, file_pattern)
        
        if not os.path.exists(kenpom_file):
            logger.warning(f"KenPom data file not found for {year}: {kenpom_file}")
            continue
        
        logger.info(f"Loading KenPom data for {year} from {kenpom_file}")
        
        try:
            # Load KenPom data from CSV
            year_df = pd.read_csv(kenpom_file)
            
            # Check if Year column exists, add it if not
            if 'Season' in year_df.columns:
                # Rename to Year for consistency
                year_df.rename(columns={'Season': 'Year'}, inplace=True)
            elif 'Year' not in year_df.columns:
                year_df['Year'] = year
            
            # Clean and standardize team names
            if 'TeamName' in year_df.columns:
                year_df['Team'] = year_df['TeamName'].apply(clean_team_name)
            elif 'Team' in year_df.columns:
                year_df['Team'] = year_df['Team'].apply(clean_team_name)
            else:
                logger.error(f"No team name column found in {kenpom_file}")
                continue
                
            # Based on the actual column names seen in the image
            column_mapping = {
                'TeamName': 'Team',
                'RankAdjEM': 'KenPomRank',  # Use the AdjEM rank as the overall rank
                'AdjEM': 'AdjEM',
                'AdjOE': 'AdjO',
                'AdjDE': 'AdjD',
                'AdjTempo': 'AdjT',
                'Luck': 'Luck',  # Assuming this exists
            }
            
            # Apply column mapping
            for old_col, new_col in column_mapping.items():
                if old_col in year_df.columns and new_col not in year_df.columns:
                    year_df.rename(columns={old_col: new_col}, inplace=True)
            
            # Ensure required columns exist
            required_cols = ['Year', 'Team', 'AdjEM', 'AdjO', 'AdjD', 'AdjT']
            missing_cols = [col for col in required_cols if col not in year_df.columns]
            
            if missing_cols:
                logger.warning(f"Missing columns in KenPom data for {year}: {missing_cols}")
                
                # Try to derive missing columns if possible
                if 'AdjO' not in year_df.columns and 'AdjOE' in year_df.columns:
                    year_df['AdjO'] = year_df['AdjOE']
                if 'AdjD' not in year_df.columns and 'AdjDE' in year_df.columns:
                    year_df['AdjD'] = year_df['AdjDE']
                if 'AdjT' not in year_df.columns and 'AdjTempo' in year_df.columns:
                    year_df['AdjT'] = year_df['AdjTempo']
                
                # Check again after derivation
                missing_cols = [col for col in required_cols if col not in year_df.columns]
                if missing_cols:
                    logger.error(f"Still missing required columns after derivation: {missing_cols}")
                    continue
            
            # Add placeholder columns for any missing non-essential columns
            if 'Luck' not in year_df.columns:
                year_df['Luck'] = 0.0
            if 'SOS' not in year_df.columns:
                year_df['SOS'] = 0.0
            if 'NCSOS' not in year_df.columns:
                year_df['NCSOS'] = 0.0
            if 'KenPomRank' not in year_df.columns:
                # Use RankAdjEM as KenPomRank if available
                if 'RankAdjEM' in year_df.columns:
                    year_df['KenPomRank'] = year_df['RankAdjEM']
                else:
                    # Create a rank based on AdjEM
                    year_df['KenPomRank'] = year_df['AdjEM'].rank(ascending=False).astype(int)
            
            logger.info(f"Loaded {len(year_df)} teams for {year}")
            all_data = pd.concat([all_data, year_df], ignore_index=True)
            
        except Exception as e:
            logger.error(f"Error loading KenPom data for {year}: {e}")
    
    # Save combined data to CSV
    if all_data.empty:
        raise ValueError("No KenPom data loaded from any year")
        
    # Ensure we have the necessary columns
    essential_columns = ['Year', 'Team', 'KenPomRank', 'AdjEM', 'AdjO', 'AdjD', 'AdjT', 'Luck', 'SOS', 'NCSOS']
    missing_essentials = [col for col in essential_columns if col not in all_data.columns]
    
    if missing_essentials:
        logger.warning(f"Missing essential columns in combined data: {missing_essentials}")
        # Add missing columns with placeholder values
        for col in missing_essentials:
            all_data[col] = 0.0
    
    # Select only the columns we need
    columns_to_keep = essential_columns
    all_data = all_data[columns_to_keep]
    
    # Save the processed data
    all_data.to_csv('data/raw/kenpom_ratings.csv', index=False)
    logger.info(f"Combined KenPom data saved to data/raw/kenpom_ratings.csv ({len(all_data)} rows)")
    
    return all_data

def process_tournament_matchups(file_path='data/raw/Tournament Matchups.csv'):
    """
    Process tournament matchups data to extract:
    1. Past tournament results (who played whom and who won)
    2. Current year's bracket structure
    """
    logger.info(f"Processing tournament matchups from {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"Tournament matchups file not found: {file_path}")
        raise FileNotFoundError(f"Tournament matchups file not found: {file_path}")
    
    # Load matchups data
    matchups_df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(matchups_df)} rows of tournament matchup data")
    
    # Clean team names for consistency
    matchups_df['TEAM'] = matchups_df['TEAM'].apply(clean_team_name)
    
    # Process historical tournament results
    tournament_results = extract_tournament_results(matchups_df)
    
    # Process current year's bracket
    current_bracket = extract_current_bracket(matchups_df)
    
    return tournament_results, current_bracket

def extract_tournament_results(matchups_df):
    """Extract historical tournament results, pairing consecutive rows as matchups"""
    logger.info("Extracting historical tournament results")
    
    # Filter to only include past years (not TARGET_YEAR)
    hist_matchups = matchups_df[matchups_df['YEAR'] < TARGET_YEAR].copy()
    
    # Determine which column identifies the game
    game_id_col = None
    if 'BY GAME NO' in hist_matchups.columns:
        game_id_col = 'BY GAME NO'
    elif 'BY YEAR NO' in hist_matchups.columns:
        game_id_col = 'BY YEAR NO'
        
    if not game_id_col:
        logger.warning("Neither 'BY GAME NO' nor 'BY YEAR NO' columns found. Using sequential pairing.")
        # Sort by year to ensure basic organization
        hist_matchups.sort_values(['YEAR'], inplace=True)
        
        # Create a list to store game results
        results = []
        
        # Process in pairs (every 2 rows should be a complete game)
        for i in range(0, len(hist_matchups) - 1, 2):
            # Get the two teams in this game
            team1 = hist_matchups.iloc[i]
            team2 = hist_matchups.iloc[i + 1]
            
            # Verify this is a valid pair (same year)
            if team1['YEAR'] != team2['YEAR']:
                logger.warning(f"Mismatched game pair at index {i}, skipping")
                continue
            
            # Determine winner based on score
            if team1['SCORE'] > team2['SCORE']:
                winner, loser = team1, team2
            else:
                winner, loser = team2, team1
            
            # Convert round number to standard name
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
        logger.info(f"Using '{game_id_col}' to identify game pairs")
        
        # Sort by year and game ID to ensure correct pairing
        if 'BY ROUND NO' in hist_matchups.columns:
            hist_matchups.sort_values(['YEAR', game_id_col, 'BY ROUND NO'], inplace=True)
        else:
            hist_matchups.sort_values(['YEAR', game_id_col], inplace=True)
        
        # Create a list to store game results
        results = []
        
        # Group by year and game number to find teams that played against each other
        game_groups = hist_matchups.groupby(['YEAR', game_id_col])
        
        for (year, game_id), game_group in game_groups:
            # Skip if we don't have exactly 2 teams in a game
            if len(game_group) != 2:
                logger.warning(f"Game {game_id} in year {year} has {len(game_group)} teams, skipping")
                continue
                
            teams = game_group.reset_index(drop=True)
            team1 = teams.iloc[0]
            team2 = teams.iloc[1]
            
            # Determine winner based on score
            if team1['SCORE'] > team2['SCORE']:
                winner, loser = team1, team2
            else:
                winner, loser = team2, team1
            
            # Convert round number to standard name
            round_name = convert_round_to_name(int(winner['ROUND']))
            
            # Create game result entry
            game_result = {
                'Year': int(year),
                'Round': round_name,
                'GameID': int(game_id),  # Use the game number as GameID
                'WinningTeam': winner['TEAM'],
                'WinningSeed': int(winner['SEED']),
                'WinningScore': float(winner['SCORE']),
                'LosingTeam': loser['TEAM'],
                'LosingSeed': int(loser['SEED']),
                'LosingScore': float(loser['SCORE']),
                'ScoreDiff': float(winner['SCORE'] - loser['SCORE'])
            }
            
            results.append(game_result)
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    if results_df.empty:
        logger.warning("No historical tournament results found")
        return pd.DataFrame(columns=['Year', 'Round', 'GameID', 'WinningTeam', 'WinningSeed', 
                                    'WinningScore', 'LosingTeam', 'LosingSeed', 'LosingScore', 'ScoreDiff'])
    
    # Save to CSV
    results_df.to_csv('data/raw/tournament_results.csv', index=False)
    logger.info(f"Saved {len(results_df)} historical tournament games to data/raw/tournament_results.csv")
    
    return results_df

def extract_current_bracket(matchups_df):
    """Extract current year's tournament bracket"""
    logger.info(f"Extracting {TARGET_YEAR} tournament bracket")
    
    # Filter to only include TARGET_YEAR
    current_matchups = matchups_df[matchups_df['YEAR'] == TARGET_YEAR].copy()
    
    if current_matchups.empty:
        logger.warning(f"No data found for {TARGET_YEAR}, using most recent year as template")
        most_recent_year = matchups_df['YEAR'].max()
        current_matchups = matchups_df[matchups_df['YEAR'] == most_recent_year].copy()
        current_matchups['YEAR'] = TARGET_YEAR  # Replace year with TARGET_YEAR
    
    logger.info(f"Found {len(current_matchups)} total entries for {TARGET_YEAR}")
    
    # Determine which column identifies the game
    game_id_col = None
    if 'BY GAME NO' in current_matchups.columns:
        game_id_col = 'BY GAME NO'
    elif 'BY YEAR NO' in current_matchups.columns:
        game_id_col = 'BY YEAR NO'
    
    # Sort the matchups to ensure consistent processing
    if game_id_col and 'BY ROUND NO' in current_matchups.columns:
        current_matchups.sort_values([game_id_col, 'BY ROUND NO'], inplace=True)
    elif game_id_col:
        current_matchups.sort_values([game_id_col], inplace=True)
    else:
        # No game identifier column, just sort by whatever is available
        logger.warning("No game identifier column found. Using sequential order.")
        
    current_matchups.reset_index(drop=True, inplace=True)
    
    # Define the standard regions
    regions = ['East', 'West', 'South', 'Midwest']
    teams_per_region = 16
    
    # We need 64 teams total
    expected_teams = 64
    
    # Check if we have enough teams
    if len(current_matchups) < expected_teams:
        logger.warning(f"Not enough teams for {TARGET_YEAR}: found {len(current_matchups)}, expected {expected_teams}")
        logger.warning("Will use available teams and create placeholder entries if needed")
    
    # Extract all teams for the current year, regardless of round
    # We'll take the first 64 teams after sorting
    bracket_entries = []
    
    # Process each team (limit to first 64)
    for i, team in current_matchups.iterrows():
        if i >= expected_teams:
            break
            
        # Calculate region (0-3) based on position
        region_idx = (i // teams_per_region) % len(regions)
        region = regions[region_idx]
        
        bracket_entry = {
            'Year': TARGET_YEAR,
            'Region': region,
            'Seed': int(team['SEED']),
            'Team': team['TEAM'],
            'TeamNo': int(team['TEAM NO'])
        }
        bracket_entries.append(bracket_entry)
    
    # Check if we have enough teams, create placeholders if needed
    if len(bracket_entries) < expected_teams:
        logger.warning(f"Creating {expected_teams - len(bracket_entries)} placeholder entries to reach 64 teams")
        
        # Create placeholder entries to reach 64 teams
        for i in range(len(bracket_entries), expected_teams):
            region_idx = (i // teams_per_region) % len(regions)
            region = regions[region_idx]
            
            # For placeholder teams, use a sequential number as seed
            placeholder_seed = (i % teams_per_region) + 1
            
            bracket_entries.append({
                'Year': TARGET_YEAR,
                'Region': region,
                'Seed': placeholder_seed,
                'Team': f"Team_{region}_{placeholder_seed}",
                'TeamNo': -i  # Use negative numbers for placeholder teams
            })
    
    # Create DataFrame
    bracket_df = pd.DataFrame(bracket_entries)
    
    # Check for duplicate seeds within regions and fix if needed
    for region in regions:
        region_df = bracket_df[bracket_df['Region'] == region]
        seed_counts = region_df['Seed'].value_counts()
        
        # Check for missing seeds (1-16)
        missing_seeds = set(range(1, 17)) - set(region_df['Seed'])
        if missing_seeds:
            logger.warning(f"Region {region} is missing seeds: {missing_seeds}")
        
        # Check for duplicate seeds
        duplicate_seeds = [seed for seed, count in seed_counts.items() if count > 1]
        if duplicate_seeds:
            logger.warning(f"Region {region} has duplicate seeds: {duplicate_seeds}")
            
            # This would require seed reassignment which is beyond the scope
            # of this fix and might interfere with real data
    
    # Save to CSV
    bracket_df.to_csv(f'data/raw/bracket_{TARGET_YEAR}.csv', index=False)
    logger.info(f"Saved {TARGET_YEAR} bracket with {len(bracket_df)} teams to data/raw/bracket_{TARGET_YEAR}.csv")
    
    return bracket_df

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

def load_team_data(file_path, year_col='YEAR', team_col='TEAM', team_id_col='TEAM NO'):
    """Generic function to load team data from CSV files"""
    if not os.path.exists(file_path):
        logger.warning(f"Data file not found: {file_path}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} rows from {file_path}")
        
        # Clean team names
        if team_col in df.columns:
            df[team_col] = df[team_col].apply(clean_team_name)
        
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        return pd.DataFrame()

def combine_team_data(kenpom_data, tournament_data, bracket_data):
    """Combine all data sources using the team identifier as the key"""
    logger.info("Combining team data from all sources")
    
    # Load additional data sources
    evan_miya_df = load_team_data('data/raw/EvanMiya.csv')
    resumes_df = load_team_data('data/raw/Resumes.csv')
    team_rankings_df = load_team_data('data/raw/TeamRankings.csv')
    
    # Get all years across datasets
    years = set(kenpom_data['Year'].unique())
    
    if not evan_miya_df.empty:
        years.update(evan_miya_df['YEAR'].unique())
    
    if not resumes_df.empty:
        years.update(resumes_df['YEAR'].unique())
    
    if not team_rankings_df.empty:
        years.update(team_rankings_df['YEAR'].unique())
    
    # Get tournament teams by year
    tournament_teams = {}
    for year in tournament_data['Year'].unique():
        year_data = tournament_data[tournament_data['Year'] == year]
        teams = set(year_data['WinningTeam'].unique()) | set(year_data['LosingTeam'].unique())
        tournament_teams[year] = teams
    
    # Create a list to hold all combined team data
    combined_data = []
    
    # Process each year
    for year in sorted(years):
        logger.info(f"Processing team data for year {year}")
        
        # Get KenPom data for this year
        year_kenpom = kenpom_data[kenpom_data['Year'] == year]
        
        # Get other data sources for this year
        year_evan_miya = evan_miya_df[evan_miya_df['YEAR'] == year] if not evan_miya_df.empty else pd.DataFrame()
        year_resumes = resumes_df[resumes_df['YEAR'] == year] if not resumes_df.empty else pd.DataFrame()
        year_rankings = team_rankings_df[team_rankings_df['YEAR'] == year] if not team_rankings_df.empty else pd.DataFrame()
        
        # Get all unique teams across datasets
        all_teams = set()
        
        # Add teams from KenPom
        all_teams.update(year_kenpom['Team'].unique())
        
        # Add teams from Evan Miya
        if not year_evan_miya.empty:
            all_teams.update(year_evan_miya['TEAM'].unique())
        
        # Add teams from Resumes
        if not year_resumes.empty:
            all_teams.update(year_resumes['TEAM'].unique())
        
        # Add teams from TeamRankings
        if not year_rankings.empty:
            all_teams.update(year_rankings['TEAM'].unique())
        
        # Process each team
        for team in all_teams:
            # Create base entry
            team_entry = {
                'Year': year,
                'Team': team,
                'InTournament': team in tournament_teams.get(year, set())
            }
            
            # Add KenPom data
            team_kenpom = year_kenpom[year_kenpom['Team'] == team]
            if not team_kenpom.empty:
                for col in ['KenPomRank', 'AdjEM', 'AdjO', 'AdjD', 'AdjT', 'Luck', 'SOS', 'NCSOS']:
                    if col in team_kenpom.columns:
                        team_entry[col] = team_kenpom[col].iloc[0]
            
            # Add Evan Miya data
            team_evan_miya = year_evan_miya[year_evan_miya['TEAM'] == team] if not year_evan_miya.empty else pd.DataFrame()
            if not team_evan_miya.empty:
                # Add team ID if available
                if 'TEAM NO' in team_evan_miya.columns:
                    team_entry['TeamNo'] = team_evan_miya['TEAM NO'].iloc[0]
                
                # Add seed if available
                if 'SEED' in team_evan_miya.columns:
                    team_entry['Seed'] = team_evan_miya['SEED'].iloc[0]
                
                # Add metrics
                metrics = ['O RATE', 'D RATE', 'RELATIVE RATING', 'TRUE TEMPO']
                for metric in metrics:
                    if metric in team_evan_miya.columns:
                        col_name = metric.replace(' ', '_')
                        team_entry[col_name] = team_evan_miya[metric].iloc[0]
            
            # Add Resume data
            team_resumes = year_resumes[year_resumes['TEAM'] == team] if not year_resumes.empty else pd.DataFrame()
            if not team_resumes.empty:
                # Add team ID if not already added
                if 'TeamNo' not in team_entry and 'TEAM NO' in team_resumes.columns:
                    team_entry['TeamNo'] = team_resumes['TEAM NO'].iloc[0]
                
                # Add seed if not already added
                if 'Seed' not in team_entry and 'SEED' in team_resumes.columns:
                    team_entry['Seed'] = team_resumes['SEED'].iloc[0]
                
                # Add metrics
                metrics = ['NET RPI', 'ELO', 'Q1 W', 'Q2 W', 'Q1 PLUS Q2 W']
                for metric in metrics:
                    if metric in team_resumes.columns:
                        col_name = metric.replace(' ', '_')
                        team_entry[col_name] = team_resumes[metric].iloc[0]
            
            # Add TeamRankings data
            team_rankings = year_rankings[year_rankings['TEAM'] == team] if not year_rankings.empty else pd.DataFrame()
            if not team_rankings.empty:
                # Add team ID if not already added
                if 'TeamNo' not in team_entry and 'TEAM NO' in team_rankings.columns:
                    team_entry['TeamNo'] = team_rankings['TEAM NO'].iloc[0]
                
                # Add seed if not already added
                if 'Seed' not in team_entry and 'SEED' in team_rankings.columns:
                    team_entry['Seed'] = team_rankings['SEED'].iloc[0]
                
                # Add metrics
                metrics = ['TR RANK', 'TR RATING', 'SOS RANK', 'LUCK RANK', 'CONSISTENCY RANK']
                for metric in metrics:
                    if metric in team_rankings.columns:
                        col_name = metric.replace(' ', '_')
                        team_entry[col_name] = team_rankings[metric].iloc[0]
            
            # Add to combined data
            combined_data.append(team_entry)
    
    # Create DataFrame from combined data
    combined_df = pd.DataFrame(combined_data)
    
    # Save to CSV
    combined_df.to_csv('data/raw/combined_team_data.csv', index=False)
    logger.info(f"Saved combined team data with {len(combined_df)} rows to data/raw/combined_team_data.csv")
    
    return combined_df

def main():
    """Main function to collect and process all data sources"""
    create_data_dir()
    
    logger.info(f"Starting data collection for March Madness prediction model")
    logger.info(f"Target prediction year: {TARGET_YEAR}")
    logger.info(f"Historical tournament years: {TOURNAMENT_YEARS}")
    
    try:
        # Load KenPom data from CSV files
        kenpom_data = load_manual_kenpom_data()
        
        # Process tournament matchups to get historical results and current bracket
        tournament_data, bracket_data = process_tournament_matchups()
        
        # Combine all team data sources
        combined_data = combine_team_data(kenpom_data, tournament_data, bracket_data)
        
        logger.info("Data collection and processing complete!")
        return True
    except Exception as e:
        logger.error(f"Error during data collection: {e}")
        raise

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Collect data for March Madness prediction model')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        main()
        logger.info("Data collection completed successfully!")
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        sys.exit(1)