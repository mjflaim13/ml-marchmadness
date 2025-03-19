# main.py
import os
import sys
import time

def print_header(message):
    """Print a header message"""
    print("\n" + "="*80)
    print(f" {message}")
    print("="*80)

def main():
    """Main function to run the entire pipeline"""
    # Create required directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    
    # Step 1: Data Collection
    print_header("Step 1: Data Collection")
    from data_collection import main as collect_data
    collect_data()
    
    # Step 2: Data Preprocessing
    print_header("Step 2: Data Preprocessing")
    from data_preprocessing import main as preprocess_data
    preprocess_data()
    
    # Step 3: Feature Engineering
    print_header("Step 3: Feature Engineering")
    from feature_engineering import main as engineer_features
    engineer_features()
    
    # Step 4: Model Training
    print_header("Step 4: Model Training")
    from model_training import main as train_model
    train_model()
    
    # Step 5: Backtesting
    print_header("Step 5: Backtesting")
    from backtesting import main as backtest_model
    backtest_model()
    
    # Step 6: Bracket Prediction
    print_header("Step 6: Bracket Prediction")
    from predict_bracket import main as predict_bracket
    predict_bracket()
    
    # Step 7: Visualization
    print_header("Step 7: Visualization")
    from utils.visualization import visualize_all
    visualize_all()
    
    print_header("PIPELINE COMPLETE!")
    print("Your March Madness bracket prediction has been generated.")
    print("Check the 'output' directory for the results and visualizations.")

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")