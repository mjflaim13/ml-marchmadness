# utils/visualization.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os
import pickle
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path
import matplotlib.colors as mcolors

def visualize_feature_importance(output_dir='output'):
    """
    Visualize feature importance from the trained model
    """
    print("Visualizing feature importance...")
    
    # Check if feature importance file exists
    feature_path = 'data/processed/feature_importance.csv'
    if not os.path.exists(feature_path):
        print("Feature importance data not found. Run model_training.py first.")
        return
    
    # Load feature importance
    feature_importance = pd.read_csv(feature_path)
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('Importance', ascending=True)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Plot horizontal bar chart
    bars = plt.barh(feature_importance['Feature'], feature_importance['Importance'])
    
    # Color bars based on importance (green for positive, red for negative)
    for i, bar in enumerate(bars):
        if feature_importance['Importance'].iloc[i] > 0:
            bar.set_color('#4CAF50')  # Green
        else:
            bar.set_color('#F44336')  # Red
    
    # Add labels and title
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/feature_importance.png', dpi=300)
    
    print(f"Feature importance visualization saved to {output_dir}/feature_importance.png")

def visualize_backtesting_results(output_dir='output'):
    """
    Visualize backtesting results
    """
    print("Visualizing backtesting results...")
    
    # Check if backtesting results file exists
    backtesting_path = 'output/backtesting_results.pkl'
    if not os.path.exists(backtesting_path):
        print("Backtesting results not found. Run backtesting.py first.")
        return
    
    # Load backtesting results
    with open(backtesting_path, 'rb') as f:
        backtesting_results = pickle.load(f)
    
    # Extract results
    year_results = backtesting_results.get('year_results', {})
    round_results = backtesting_results.get('round_results', {})
    seed_diff_results = backtesting_results.get('seed_diff_results', {})
    
    # Create plots
    
    # 1. Year-by-year accuracy
    plt.figure(figsize=(10, 6))
    
    years = sorted(year_results.keys())
    if not years:
        print("No year results found")
    else:
        accuracies = [year_results[year]['accuracy'] for year in years]
        
        plt.bar(years, accuracies, color='#1976D2')
        
        for i, acc in enumerate(accuracies):
            plt.text(years[i], acc + 0.01, f'{acc:.3f}', ha='center')
        
        plt.xlabel('Year')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy by Year')
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f'{output_dir}/accuracy_by_year.png', dpi=300)
    
    # 2. Round-by-round accuracy
    plt.figure(figsize=(10, 6))
    
    # Define round order
    round_order = ['R64', 'R32', 'S16', 'E8', 'F4', 'NCG']
    round_labels = ['First Round', 'Second Round', 'Sweet 16', 'Elite 8', 'Final Four', 'Championship']
    
    rounds = [r for r in round_order if r in round_results]
    if not rounds:
        print("No round results found")
    else:
        accuracies = [round_results[r]['accuracy'] for r in rounds]
        
        plt.bar(range(len(rounds)), accuracies, color='#673AB7')
        
        for i, acc in enumerate(accuracies):
            plt.text(i, acc + 0.01, f'{acc:.3f}', ha='center')
        
        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy by Tournament Round')
        plt.xticks(range(len(rounds)), [round_labels[round_order.index(r)] for r in rounds], rotation=45)
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f'{output_dir}/accuracy_by_round.png', dpi=300)
    
    # 3. Seed difference accuracy
    plt.figure(figsize=(12, 6))
    
    groups = list(seed_diff_results.keys())
    if not groups:
        print("No seed difference results found")
    else:
        accuracies = [seed_diff_results[g]['accuracy'] for g in groups]
        
        plt.bar(range(len(groups)), accuracies, color='#FF9800')
        
        for i, acc in enumerate(accuracies):
            plt.text(i, acc + 0.01, f'{acc:.3f}', ha='center')
        
        plt.xlabel('Matchup Type')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy by Matchup Type')
        plt.xticks(range(len(groups)), groups, rotation=45, ha='right')
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f'{output_dir}/accuracy_by_matchup.png', dpi=300)
    
    print(f"Backtesting visualizations saved to {output_dir}/")

def get_team_name(team_obj):
    """Helper function to extract team name from either string or dict"""
    if isinstance(team_obj, dict):
        return team_obj.get('team', 'Unknown')
    return str(team_obj)

def get_team_seed(team_obj):
    """Helper function to extract team seed from either string or dict"""
    if isinstance(team_obj, dict):
        return team_obj.get('seed', 0)
    return 0

def get_team_region(team_obj):
    """Helper function to extract team region from either string or dict"""
    if isinstance(team_obj, dict):
        return team_obj.get('region', 'Unknown')
    return 'Unknown'

def visualize_bracket(output_dir='output'):
    """
    Visualize the predicted tournament bracket in a traditional NCAA format
    """
    print("Visualizing tournament bracket prediction...")
    
    # Check if tournament prediction file exists
    prediction_path = 'output/tournament_prediction.json'
    if not os.path.exists(prediction_path):
        print("Tournament prediction not found. Run predict_bracket.py first.")
        return
    
    # Load tournament prediction
    with open(prediction_path, 'r') as f:
        prediction = json.load(f)
    
    # Extract results
    bracket = prediction.get('bracket', {})
    results = prediction.get('results', {})
    champion = prediction.get('champion', "Unknown")
    
    # Define regions and their positions in the bracket
    regions = list(bracket.keys())
    if not regions:
        print("No regions found in bracket data")
        return
    
    # Create figure
    plt.figure(figsize=(15, 10), facecolor='white')
    
    # Define bracket parameters
    bracket_width = 14  # Width of the entire bracket
    bracket_height = 9  # Height of the entire bracket
    region_width = bracket_width / 2  # Width of each region
    region_height = bracket_height  # Height of each region
    round_spacing = region_width / 7  # Horizontal space between rounds
    team_height = 0.25  # Height of each team box
    team_spacing = 0.05  # Vertical space between team boxes
    
    # Create a color function based on seed
    def get_seed_color(seed):
        if seed <= 4:
            return '#d0f0d0'  # Light green for top seeds (1-4)
        elif seed <= 8:
            return '#d0d0f0'  # Light blue for middle seeds (5-8)
        elif seed <= 12:
            return '#f0f0d0'  # Light yellow for lower-mid seeds (9-12)
        else:
            return '#f0d0d0'  # Light red for bottom seeds (13-16)
    
    # Function to draw team box
    def draw_team_box(ax, x, y, team, seed, winner=False):
        box_width = round_spacing * 0.9
        
        # Draw box
        rect = Rectangle((x, y), box_width, team_height, 
                       facecolor=get_seed_color(seed),
                       edgecolor='black',
                       linewidth=1.5 if winner else 1.0,
                       alpha=0.8)
        ax.add_patch(rect)
        
        # Draw team name and seed
        ax.text(x + box_width/2, y + team_height/2, 
                f"{seed} {team}", 
                ha='center', va='center',
                fontsize=7,
                fontweight='bold' if winner else 'normal')
    
    # Function to draw a connector line between rounds
    def draw_connector(ax, x1, y1, x2, y2):
        # Horizontal line from first team
        ax.plot([x1, x1 + round_spacing/2], [y1, y1], 'k-', linewidth=1)
        
        # Vertical line to midpoint
        middle_y = (y1 + y2) / 2
        ax.plot([x1 + round_spacing/2, x1 + round_spacing/2], [y1, middle_y], 'k-', linewidth=1)
        
        # Vertical line from midpoint
        ax.plot([x1 + round_spacing/2, x1 + round_spacing/2], [middle_y, y2], 'k-', linewidth=1)
        
        # Horizontal line to second team
        ax.plot([x1 + round_spacing/2, x2], [y2, y2], 'k-', linewidth=1)
    
    # Create axis with the NCAA bracket look
    ax = plt.axes([0.05, 0.05, 0.9, 0.85])
    
    # Draw NCAA March Madness logo at the top center
    ax.text(bracket_width/2, bracket_height + 0.2, "NCAA MARCH MADNESS", 
            ha='center', va='center', fontsize=24, fontweight='bold',
            color='#0066cc')
    
    # Draw year below logo
    ax.text(bracket_width/2, bracket_height - 0.2, "2025 TOURNAMENT", 
            ha='center', va='center', fontsize=18, fontweight='bold')
    
    # Define x-coordinates for each round
    round_x = {
        'R64': [0.5, bracket_width - 0.5],  # First round (left and right)
        'R32': [0.5 + round_spacing, bracket_width - 0.5 - round_spacing],  # Second round
        'S16': [0.5 + 2*round_spacing, bracket_width - 0.5 - 2*round_spacing],  # Sweet 16
        'E8': [0.5 + 3*round_spacing, bracket_width - 0.5 - 3*round_spacing],  # Elite 8
        'F4': [bracket_width/2 - round_spacing, bracket_width/2 + round_spacing],  # Final Four
        'NCG': [bracket_width/2]  # Championship
    }
    
    # Define team starting positions
    team_positions = {}
    
    # Process each region (draw the first 4 rounds)
    for i, region in enumerate(regions[:4]):  # Ensure we only process the 4 main regions
        # Determine if this is a left or right region
        is_left = i < 2
        region_idx = i % 2  # 0 for top regions, 1 for bottom regions
        
        # Set region header position
        region_x = 0.5 if is_left else bracket_width - 2.5
        region_y = bracket_height - 0.5 if region_idx == 0 else 0.5
        
        # Draw region name
        ax.text(region_x + (0 if is_left else 2), region_y, region, 
                ha='center', va='center', fontsize=14, fontweight='bold')
        
        # Draw initial matchups (R64)
        r64_games = [g for g in bracket.get(region, {}).keys() if g.startswith('R64')]
        
        # Sort games to ensure proper bracket ordering
        r64_games.sort()
        
        for j, game_key in enumerate(r64_games):
            game = bracket[region][game_key]
            team1, team2 = game["teams"]
            seed1, seed2 = game["seeds"]
            
            # Calculate y-position for this matchup
            base_y = (bracket_height - (len(r64_games) * (team_height + team_spacing))) / 2
            y1 = base_y + j * (2 * team_height + team_spacing)
            y2 = y1 + team_height
            
            # Adjust x-position based on left/right region
            x = round_x['R64'][0] if is_left else round_x['R64'][1] - round_spacing * 0.9
            
            # Store positions for later rounds
            team_key1 = f"{region}_{seed1}_{team1}"
            team_key2 = f"{region}_{seed2}_{team2}"
            team_positions[team_key1] = (x, y1)
            team_positions[team_key2] = (x, y2)
            
            # Determine winner if available
            winner_team = None
            if 'R64' in results.get(region, {}):
                winners = results[region]['R64']
                if j < len(winners):
                    winner_team = get_team_name(winners[j])
            
            # Draw team boxes
            draw_team_box(ax, x, y1, team1, seed1, winner=winner_team == team1)
            draw_team_box(ax, x, y2, team2, seed2, winner=winner_team == team2)
            
            # Draw connector line
            draw_connector(ax, x + round_spacing * 0.9, y1 + team_height/2, 
                          x + round_spacing * 0.9, y2 + team_height/2)
        
    # Draw subsequent rounds
    for round_name in ['R32', 'S16', 'E8']:
        for i, region in enumerate(regions[:4]):
            is_left = i < 2
            region_idx = i % 2
            
            # Skip if no results for this round
            if round_name not in results.get(region, {}):
                continue
            
            # Get teams that advanced to this round
            teams = results[region][round_name]
            
            for j, team_obj in enumerate(teams):
                team = get_team_name(team_obj)
                
                # Find team in previous positions
                found = False
                for team_key, (prev_x, prev_y) in team_positions.items():
                    region_name, seed_str, team_name = team_key.split('_', 2)
                    
                    if team == team_name and region_name == region:
                        # This is our team
                        seed = int(seed_str)
                        
                        # Calculate new position
                        round_idx = ['R64', 'R32', 'S16', 'E8'].index(round_name)
                        x = round_x[round_name][0 if is_left else 1] - (0 if is_left else round_spacing * 0.9)
                        y_spacing = 2**(round_idx+1) * (team_height + team_spacing)
                        y = (bracket_height - (len(teams) * (team_height + team_spacing))) / 2 + j * y_spacing
                        
                        # Store new position
                        team_positions[team_key] = (x, y)
                        
                        # Determine if this team wins the next round
                        winner = False
                        next_round = ['R32', 'S16', 'E8', 'F4'][round_idx]
                        
                        if next_round in results.get(region, {}):
                            next_winners = [get_team_name(t) for t in results[region][next_round]]
                            if team in next_winners:
                                winner = True
                        
                        # Draw team box
                        draw_team_box(ax, x, y, team, seed, winner=winner)
                        found = True
                        break
                
                if not found and isinstance(team_obj, dict):
                    # Try to use the seed and region from the team_obj
                    seed = team_obj.get('seed', 0)
                    
                    # Calculate new position
                    round_idx = ['R64', 'R32', 'S16', 'E8'].index(round_name)
                    x = round_x[round_name][0 if is_left else 1] - (0 if is_left else round_spacing * 0.9)
                    y_spacing = 2**(round_idx+1) * (team_height + team_spacing)
                    y = (bracket_height - (len(teams) * (team_height + team_spacing))) / 2 + j * y_spacing
                    
                    # Determine if this team wins the next round
                    winner = False
                    next_round = ['R32', 'S16', 'E8', 'F4'][round_idx]
                    
                    if next_round in results.get(region, {}):
                        next_winners = [get_team_name(t) for t in results[region][next_round]]
                        if team in next_winners:
                            winner = True
                    
                    # Draw team box
                    draw_team_box(ax, x, y, team, seed, winner=winner)
                    found = True
    
    # Draw Final Four
    if 'Final Four' in results and 'F4' in results['Final Four']:
        f4_teams = results['Final Four']['F4']
        
        for j, team_obj in enumerate(f4_teams):
            team = get_team_name(team_obj)
            seed = get_team_seed(team_obj)
            region = get_team_region(team_obj)
            
            # Calculate position
            x = round_x['F4'][0] if j == 0 else round_x['F4'][1] - round_spacing * 0.9
            y = bracket_height/2 - team_height - team_spacing if j == 0 else bracket_height/2 + team_spacing
            
            # Determine if this team wins
            is_winner = False
            if 'NCG' in results.get('Final Four', {}):
                champions = [get_team_name(t) for t in results['Final Four']['NCG']]
                if team in champions:
                    is_winner = True
            
            # Draw team box
            draw_team_box(ax, x, y, f"{team} ({region})", seed, winner=is_winner)
    
    # Draw Championship
    if 'Final Four' in results and 'NCG' in results['Final Four'] and results['Final Four']['NCG']:
        champ_obj = results['Final Four']['NCG'][0]
        champ = get_team_name(champ_obj)
        seed = get_team_seed(champ_obj)
        region = get_team_region(champ_obj)
        
        # Draw championship box
        x = round_x['NCG'][0] - round_spacing * 0.45
        y = bracket_height/2 - team_height/2
        
        # Larger box for champion
        champ_width = round_spacing * 0.9
        champ_height = team_height * 1.2
        
        rect = Rectangle((x, y), champ_width, champ_height, 
                       facecolor=get_seed_color(seed),
                       edgecolor='gold',
                       linewidth=2,
                       alpha=0.9)
        ax.add_patch(rect)
        
        # Draw champion text
        ax.text(x + champ_width/2, y + champ_height * 0.7, 
                f"{seed} {champ}", 
                ha='center', va='center',
                fontsize=9,
                fontweight='bold')
        
        ax.text(x + champ_width/2, y + champ_height * 0.3, 
                f"({region})", 
                ha='center', va='center',
                fontsize=8)
        
        # Add crown or trophy icon
        ax.text(x + champ_width/2, y + champ_height + 0.1, 
                "🏆 CHAMPION 🏆", 
                ha='center', va='center',
                fontsize=12,
                fontweight='bold',
                color='#FFD700')  # Gold color
    
    # Add round labels
    round_labels = ["FIRST ROUND", "SECOND ROUND", "SWEET 16", "ELITE 8", "FINAL FOUR", "CHAMPION"]
    round_keys = ['R64', 'R32', 'S16', 'E8', 'F4', 'NCG']
    
    for i, (label, key) in enumerate(zip(round_labels, round_keys)):
        # Left side label
        if i < 4:  # Rounds in the regions
            ax.text(0.5 + i * round_spacing, bracket_height + 0.1, 
                    label, ha='center', va='center', fontsize=8, fontweight='bold')
            
            # Right side label (mirror)
            ax.text(bracket_width - 0.5 - i * round_spacing, bracket_height + 0.1, 
                    label, ha='center', va='center', fontsize=8, fontweight='bold')
        elif i == 4:  # Final Four
            ax.text(bracket_width/2 - round_spacing, bracket_height + 0.1, 
                    label, ha='center', va='center', fontsize=8, fontweight='bold')
            ax.text(bracket_width/2 + round_spacing, bracket_height + 0.1, 
                    label, ha='center', va='center', fontsize=8, fontweight='bold')
        else:  # Championship
            ax.text(bracket_width/2, bracket_height + 0.1, 
                    label, ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Create a legend for seed colors
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor=get_seed_color(1), edgecolor='black', label='Seeds 1-4'),
        Rectangle((0, 0), 1, 1, facecolor=get_seed_color(5), edgecolor='black', label='Seeds 5-8'),
        Rectangle((0, 0), 1, 1, facecolor=get_seed_color(9), edgecolor='black', label='Seeds 9-12'),
        Rectangle((0, 0), 1, 1, facecolor=get_seed_color(13), edgecolor='black', label='Seeds 13-16')
    ]
    
    # Add the legend at the bottom
    ax.legend(handles=legend_elements, loc='lower center', 
              bbox_to_anchor=(0.5, -0.05), ncol=4, frameon=True)
    
    # Turn off the axis
    ax.set_xlim(0, bracket_width)
    ax.set_ylim(0, bracket_height + 0.5)
    ax.axis('off')
    
    # Add algorithm information
    ax.text(0.1, 0.02, "Generated using AlgoMadness: Trading-inspired Tournament Prediction Algorithm", 
            fontsize=8, color='#666666', transform=plt.gcf().transFigure)
    
    # Save visualization
    plt.savefig(f'{output_dir}/bracket_prediction.png', dpi=300, bbox_inches='tight')
    
    print(f"Bracket visualization saved to {output_dir}/bracket_prediction.png")

def visualize_all():
    """Run all visualizations"""
    print("Running all visualizations...")
    
    # Create output directory
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize feature importance
    visualize_feature_importance(output_dir)
    
    # Visualize backtesting results
    visualize_backtesting_results(output_dir)
    
    # Visualize bracket prediction
    visualize_bracket(output_dir)
    
    print("All visualizations complete!")

if __name__ == "__main__":
    visualize_all()