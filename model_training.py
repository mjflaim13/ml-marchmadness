# model_training.py
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_feature_data():
    """Load engineered feature data"""
    feature_path = 'data/processed/engineered_features.csv'
    
    if not os.path.exists(feature_path):
        raise FileNotFoundError("Engineered feature data not found. Run feature_engineering.py first.")
    
    feature_data = pd.read_csv(feature_path)
    
    # Separate features and target
    X = feature_data.drop('Team1_Won', axis=1)
    y = feature_data['Team1_Won']
    
    return X, y

def train_test_evaluation(X, y, test_size=0.2, random_state=42):
    """
    Train and evaluate models using train/test split
    """
    print("Evaluating models using train/test split...")
    
    # Create train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Check if there are any NaN values
    if X_train.isna().any().any() or X_test.isna().any().any():
        print(f"WARNING: Input data contains NaN values. Adding imputation step.")
        print(f"NaN counts per feature:")
        nan_counts = X.isna().sum()
        print(nan_counts[nan_counts > 0])
    
    # Initialize models with pipelines that include imputation
    models = {
        'LogisticRegression': Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(max_iter=1000, random_state=random_state))
        ]),
        'RandomForest': Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('model', RandomForestClassifier(n_estimators=100, random_state=random_state))
        ])
    }
    
    results = {}
    
    # Train and evaluate each model
    for name, pipeline in models.items():
        print(f"Training {name}...")
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Store results
        results[name] = {
            'pipeline': pipeline,
            'accuracy': accuracy,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        print(f"{name} Accuracy: {accuracy:.4f}")
        print(f"Confusion Matrix:\n{results[name]['confusion_matrix']}")
        print(f"Classification Report:\n{results[name]['classification_report']}")
        print()
    
    return results

def cross_validation_evaluation(X, y, cv=5, random_state=42):
    """
    Evaluate models using cross-validation
    """
    print(f"Evaluating models using {cv}-fold cross-validation...")
    
    # Initialize models with pipelines that include imputation
    models = {
        'LogisticRegression': Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(max_iter=1000, random_state=random_state))
        ]),
        'RandomForest': Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('model', RandomForestClassifier(n_estimators=100, random_state=random_state))
        ])
    }
    
    results = {}
    
    # Evaluate each model using cross-validation
    for name, pipeline in models.items():
        print(f"Cross-validating {name}...")
        
        # Perform cross-validation
        cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
        
        # Store results
        results[name] = {
            'cv_scores': cv_scores,
            'mean_cv_score': cv_scores.mean(),
            'std_cv_score': cv_scores.std()
        }
        
        print(f"{name} CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"Individual fold scores: {cv_scores}")
        print()
    
    return results

def train_final_model(X, y, model_type='RandomForest', random_state=42):
    """
    Train the final model on all data
    """
    print(f"Training final {model_type} model on all data...")
    
    # Initialize pipeline with imputation
    if model_type == 'LogisticRegression':
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(max_iter=1000, random_state=random_state))
        ])
    elif model_type == 'RandomForest':
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('model', RandomForestClassifier(n_estimators=100, random_state=random_state))
        ])
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train model on all data
    pipeline.fit(X, y)
    
    # Extract the actual model from the pipeline
    if model_type == 'LogisticRegression':
        model = pipeline.named_steps['model']
        scaler = pipeline.named_steps['scaler']
    else:
        model = pipeline.named_steps['model']
        scaler = None  # RandomForest doesn't need scaling
    
    # Create model info
    model_info = {
        'model': model,
        'pipeline': pipeline,  # Store the full pipeline for prediction
        'scaler': scaler,
        'model_type': model_type,
        'feature_names': X.columns.tolist()
    }
    
    # Calculate feature importance
    if model_type == 'LogisticRegression':
        importance = model.coef_[0]
    else:  # RandomForest
        importance = model.feature_importances_
    
    # Create DataFrame with feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importance
    })
    
    # Sort by importance (absolute value for logistic regression)
    if model_type == 'LogisticRegression':
        feature_importance['AbsImportance'] = feature_importance['Importance'].abs()
        feature_importance = feature_importance.sort_values('AbsImportance', ascending=False)
        feature_importance = feature_importance.drop('AbsImportance', axis=1)
    else:
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    # Save feature importance
    feature_importance.to_csv('data/processed/feature_importance.csv', index=False)
    print(f"Feature importance saved to data/processed/feature_importance.csv")
    
    # Save model info
    os.makedirs('models', exist_ok=True)
    with open(f'models/final_model.pkl', 'wb') as f:
        pickle.dump(model_info, f)
    
    print(f"Final model saved to models/final_model.pkl")
    
    # Print top features
    print("\nTop features by importance:")
    print(feature_importance.head(10))
    
    return model_info

def main():
    """Main model training function"""
    # Create output directory
    os.makedirs('models', exist_ok=True)
    
    # Load data
    try:
        X, y = load_feature_data()
        print("Feature data loaded successfully.")
        print(f"Training data shape: {X.shape}")
    except FileNotFoundError as e:
        print(e)
        return
    
    # Evaluate using train/test split
    train_test_results = train_test_evaluation(X, y)
    
    # Evaluate using cross-validation
    cv_results = cross_validation_evaluation(X, y)
    
    # Select best model based on cross-validation
    logistic_cv_score = cv_results['LogisticRegression']['mean_cv_score']
    rf_cv_score = cv_results['RandomForest']['mean_cv_score']
    
    best_model = 'LogisticRegression' if logistic_cv_score > rf_cv_score else 'RandomForest'
    print(f"\nBest model based on cross-validation: {best_model}")
    
    # Train final model
    final_model_info = train_final_model(X, y, model_type=best_model)
    
    print("Model training complete!")

if __name__ == "__main__":
    main()