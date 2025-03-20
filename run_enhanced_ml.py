# run_enhanced_ml.py - Fixed NaN handling
import pandas as pd
import numpy as np
import os
import pickle
import json
import time
import subprocess
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

# Start with a basic print to confirm execution
print("Starting enhanced ML pipeline...")

# Check for required packages
try:
    # Base packages
    print("Base packages imported successfully")
    
    # ML packages
    print("Scikit-learn packages imported successfully")
    
    # XGBoost
    try:
        import xgboost as xgb
        print("XGBoost imported successfully")
        has_xgboost = True
    except ImportError:
        print("XGBoost not found - will use other models only")
        has_xgboost = False
        
except ImportError as e:
    print(f"Error importing required packages: {e}")
    print("Please make sure scikit-learn is installed: pip install scikit-learn")
    exit(1)

print("XGBoost imported successfully!")

# Suppress warnings
warnings.filterwarnings('ignore')

def print_header(message):
    """Print a header message"""
    print("\n" + "="*80)
    print(f" {message}")
    print("="*80)

def load_feature_data():
    """Load engineered feature data"""
    feature_path = 'data/processed/engineered_features.csv'
    
    if not os.path.exists(feature_path):
        print(f"Error: File not found - {feature_path}")
        print("Please run data_preprocessing.py and feature_engineering.py first")
        return None, None
    
    print(f"Loading features from {feature_path}")
    feature_data = pd.read_csv(feature_path)
    
    # Separate features and target
    X = feature_data.drop('Team1_Won', axis=1)
    y = feature_data['Team1_Won']
    
    print(f"Loaded {len(X)} samples with {X.shape[1]} features")
    
    # Check for missing values
    nan_count = X.isna().sum().sum()
    if nan_count > 0:
        print(f"WARNING: Data contains {nan_count} missing values")
    
    return X, y

def build_enhanced_model(X, y, test_size=0.2, random_state=42):
    """
    Build an enhanced model architecture with ensembles
    """
    print("Building enhanced model architecture...")
    
    # Handle missing values before splitting
    print("Imputing missing values...")
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Split data for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize individual models
    models = {
        'logistic': LogisticRegression(
            C=0.1, 
            max_iter=2000, 
            class_weight='balanced',
            random_state=random_state
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=200, 
            max_depth=None, 
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            class_weight='balanced',
            bootstrap=True,
            random_state=random_state
        ),
        'gradient_boost': GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=2,
            min_samples_leaf=1,
            subsample=0.8,
            random_state=random_state
        ),
        'neural_net': MLPClassifier(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size='auto',
            learning_rate='adaptive',
            max_iter=1000,
            early_stopping=False,
            random_state=random_state
        )
    }
    
    # Add XGBoost if available
    try:
        models['xgboost'] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            gamma=0,
            random_state=random_state
        )
        print("XGBoost added to model ensemble")
    except Exception as e:
        print(f"Error adding XGBoost to ensemble: {e}")
    
    # Train each model and collect predictions
    print("Training individual models...")
    trained_models = {}
    model_predictions = {}
    model_accuracies = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        try:
            start_time = time.time()
            
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            trained_models[name] = model
            model_predictions[name] = y_pred
            model_accuracies[name] = accuracy
            
            elapsed_time = time.time() - start_time
            print(f"  {name} - Accuracy: {accuracy:.4f} (Time: {elapsed_time:.2f}s)")
        except Exception as e:
            print(f"  Error training {name}: {e}")
    
    # Create a voting ensemble
    print("\nCreating voting ensemble...")
    if len(trained_models) >= 2:
        try:
            voting_clf = VotingClassifier(
                estimators=[(name, model) for name, model in trained_models.items()],
                voting='soft'  # Use probability estimates for voting
            )
            
            voting_clf.fit(X_train_scaled, y_train)
            voting_pred = voting_clf.predict(X_test_scaled)
            voting_accuracy = accuracy_score(y_test, voting_pred)
            
            print(f"Voting Ensemble Accuracy: {voting_accuracy:.4f}")
            print("\nClassification Report for Voting Ensemble:")
            print(classification_report(y_test, voting_pred))
            
            # Train final model on all data
            print("\nTraining final ensemble on all data...")
            
            # Process full dataset
            X_full_scaled = scaler.transform(X_imputed)
            
            # Train each model on full data
            final_models = {}
            for name, model in models.items():
                if name in trained_models:  # Only include models that trained successfully
                    model.fit(X_full_scaled, y)
                    final_models[name] = model
            
            # Train voting ensemble on full data
            final_voting_clf = VotingClassifier(
                estimators=[(name, model) for name, model in final_models.items()],
                voting='soft'
            )
            final_voting_clf.fit(X_full_scaled, y)
            
            # Create enhanced model info
            model_info = {
                'individual_models': final_models,
                'voting_ensemble': final_voting_clf,
                'model_weights': {name: acc/sum(model_accuracies.values()) for name, acc in model_accuracies.items()},
                'imputer': imputer,
                'scaler': scaler,
                'feature_names': X.columns.tolist(),
                'model_accuracies': model_accuracies
            }
            
            return model_info
            
        except Exception as e:
            print(f"Error creating voting ensemble: {e}")
            print("Falling back to best individual model")
    
    # If voting ensemble failed or not enough models, use best individual model
    if not trained_models:
        print("No models trained successfully")
        return None
    
    # Find the best model
    best_model_name = max(model_accuracies, key=model_accuracies.get)
    best_model = trained_models[best_model_name]
    
    print(f"\nUsing best individual model: {best_model_name} (Accuracy: {model_accuracies[best_model_name]:.4f})")
    
    # Train best model on all data
    X_full_scaled = scaler.transform(X_imputed)
    best_model.fit(X_full_scaled, y)
    
    # Standard model interface for compatibility
    model_info = {
        'model': best_model,
        'scaler': scaler,
        'imputer': imputer,
        'model_type': best_model_name,
        'feature_names': X.columns.tolist(),
        'is_enhanced': True  # Flag as enhanced model
    }
    
    return model_info

def extract_feature_importance(model_info, X):
    """Extract feature importance from the model"""
    print("Extracting feature importance...")
    
    # Check if we have a voting ensemble
    if 'individual_models' in model_info:
        print("Using random forest model for feature importance")
        if 'random_forest' in model_info['individual_models']:
            model = model_info['individual_models']['random_forest']
            importance = model.feature_importances_
        elif 'gradient_boost' in model_info['individual_models']:
            model = model_info['individual_models']['gradient_boost']
            importance = model.feature_importances_
        elif 'logistic' in model_info['individual_models']:
            model = model_info['individual_models']['logistic']
            importance = model.coef_[0]
        else:
            print("No model with feature importance found in ensemble")
            return None
    else:
        # Check standard model
        if hasattr(model_info['model'], 'feature_importances_'):
            importance = model_info['model'].feature_importances_
        elif hasattr(model_info['model'], 'coef_'):
            importance = model_info['model'].coef_[0]
        else:
            print("Model doesn't support feature importance")
            return None
    
    # Create DataFrame with feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns.tolist(),
        'Importance': importance
    })
    
    # Sort by absolute importance
    feature_importance['AbsImportance'] = feature_importance['Importance'].abs()
    feature_importance = feature_importance.sort_values('AbsImportance', ascending=False)
    feature_importance = feature_importance.drop('AbsImportance', axis=1)
    
    # Save feature importance
    feature_importance.to_csv('data/processed/feature_importance.csv', index=False)
    print(f"Feature importance saved to data/processed/feature_importance.csv")
    
    # Print top features
    print("\nTop 10 features by importance:")
    print(feature_importance.head(10))
    
    return feature_importance

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

def update_predict_bracket_stub():
    """Create a stub function for using enhanced model in predict_bracket.py"""
    function_code = """
def predict_with_enhanced_model(model_info, X):
    \"\"\"Make predictions using enhanced model\"\"\"
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
"""
    
    with open('enhanced_model_stub.py', 'w') as f:
        f.write(function_code)
    
    print("\nCreated enhanced_model_stub.py")
    print("To use the enhanced model in predict_bracket.py:")
    print("1. Import the function: from enhanced_model_stub import predict_with_enhanced_model")
    print("2. Load the enhanced model: model_info = pickle.load(open('models/enhanced_model.pkl', 'rb'))")
    print("3. Use the function for prediction")

def train_enhanced_model():
    """Main enhanced model training function"""
    print_header("ENHANCED MODEL TRAINING")
    
    # Create output directory
    os.makedirs('models', exist_ok=True)
    
    # Load data
    X, y = load_feature_data()
    if X is None or y is None:
        return None
    
    # Skipping complex feature engineering due to NaN issues
    print("Skipping complex feature engineering to avoid NaN issues")
    print(f"Using {X.shape[1]} base features")
    
    # Build enhanced model
    model_info = build_enhanced_model(X, y)
    if model_info is None:
        return None
    
    # Extract feature importance
    extract_feature_importance(model_info, X)
    
    # Save model info
    model_path = 'models/enhanced_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_info, f)
    
    print(f"Enhanced model saved to {model_path}")
    
    # Create stub for predict_bracket.py
    update_predict_bracket_stub()
    
    return model_info

def main():
    """Main function to run the pipeline"""
    print_header("ENHANCED MARCH MADNESS PREDICTION PIPELINE")
    
    try:
        # Train enhanced model
        model_info = train_enhanced_model()
        if model_info is None:
            print("Enhanced model training failed. Exiting.")
            return
        
        print("Enhanced model training completed successfully!")
        print("\nTo use this model for bracket prediction:")
        print("1. Copy the function from enhanced_model_stub.py")
        print("2. Update your predict_bracket.py to use the enhanced model")
        
    except Exception as e:
        print(f"Error during pipeline execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()