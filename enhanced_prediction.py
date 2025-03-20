# enhanced_prediction.py
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