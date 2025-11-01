"""
Inference script for Market Regime Classification Model
"""
import joblib
import pandas as pd


def load_model():
    """Load the trained model."""
    return joblib.load("model.pkl")


# Load model at module level for efficiency
model = load_model()

# Regime mapping
REGIME_NAMES = {
    0: "Bull Market",
    1: "Bear Market", 
    2: "Sideways Market",
    3: "High Volatility"
}


def predict(data):
    """
    Make predictions on input data.
    
    Args:
        data: dict or list of dicts with keys: returns, volatility, volume_ratio
        
    Returns:
        list of dicts with regime, regime_name, and probability
    """
    # Convert to DataFrame
    if isinstance(data, dict):
        data = [data]
    df = pd.DataFrame(data)
    
    # Ensure correct column order
    required_cols = ['returns', 'volatility', 'volume_ratio']
    df = df[required_cols]
    
    # Get predictions and probabilities
    predictions = model.predict(df)
    probabilities = model.predict_proba(df)
    
    # Format results
    results = []
    for pred, proba in zip(predictions, probabilities):
        results.append({
            'regime': int(pred),
            'regime_name': REGIME_NAMES[pred],
            'probability': proba.tolist()
        })
    
    return results
