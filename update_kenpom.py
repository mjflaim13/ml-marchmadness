# update_kenpom.py
import os
import shutil
import subprocess

def update_kenpom_data():
    """
    Integrate the new KenPom Barttorvik data into the prediction system
    """
    print("Integrating new KenPom Barttorvik data into the prediction system...")
    
    # Create required directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    
    # Check if new KenPom file exists
    if not os.path.exists('data/raw/KenPom Barttorvik.csv'):
        # Try to copy from current directory
        if os.path.exists('KenPom Barttorvik.csv'):
            print("Copying KenPom Barttorvik.csv to data/raw/")
            shutil.copy('KenPom Barttorvik.csv', 'data/raw/KenPom Barttorvik.csv')
        else:
            print("ERROR: KenPom Barttorvik.csv file not found!")
            return False
    
    # Run the pipeline
    print("\nStep 1: Data Collection")
    subprocess.run(["python", "data_collection.py"])
    
    print("\nStep 2: Data Preprocessing")
    subprocess.run(["python", "data_preprocessing.py"])
    
    print("\nStep 3: Feature Engineering")
    subprocess.run(["python", "feature_engineering.py"])
    
    print("\nStep 4: Model Training")
    subprocess.run(["python", "model_training.py"])
    
    print("\nStep 5: Backtesting")
    subprocess.run(["python", "backtesting.py"])
    
    print("\nStep 6: Bracket Prediction")
    subprocess.run(["python", "predict_bracket.py"])
    
    print("\nStep 7: Visualization")
    subprocess.run(["python", "utils/visualization.py"])
    
    print("\nUpdate complete! Your March Madness bracket prediction has been regenerated.")
    print("Check the 'output' directory for the results and visualizations.")
    
    return True

if __name__ == "__main__":
    update_kenpom_data()