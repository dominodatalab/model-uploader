"""
Market Regime Classification: Train and Register to MLflow
Combines model training with MLflow registration in a single pipeline.
Now includes saving training data and ground truth for model monitoring.
"""
import argparse
import json
import pickle
import time
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from mlflow.pyfunc import PythonModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split


class PicklePyFunc(PythonModel):
    """MLflow PyFunc wrapper for pickle models."""
    
    def load_context(self, context):
        with open(context.artifacts["model_pkl"], "rb") as f:
            self._model = pickle.load(f)
    
    def predict(self, context, model_input):
        X = model_input if isinstance(model_input, pd.DataFrame) else pd.DataFrame(model_input)
        if hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(X)
        return self._model.predict(X)


def generate_synthetic_data(n_samples=5000, random_state=42):
    """Generate synthetic market data with regime labels."""
    np.random.seed(random_state)
    
    returns = np.random.randn(n_samples) * 0.02
    volatility = np.abs(np.random.randn(n_samples) * 0.15 + 0.20)
    volume_ratio = np.random.gamma(2, 0.5, n_samples)
    
    regimes = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        if returns[i] > 0.01 and volatility[i] < 0.25:
            regimes[i] = 0  # Bull market
        elif returns[i] < -0.01 and volatility[i] < 0.25:
            regimes[i] = 1  # Bear market
        elif volatility[i] > 0.30:
            regimes[i] = 3  # High volatility
        else:
            regimes[i] = 2  # Sideways/neutral
    
    return pd.DataFrame({
        'returns': returns,
        'volatility': volatility,
        'volume_ratio': volume_ratio,
        'regime': regimes
    })


def train_model(df):
    """Train RandomForest classifier and return model with metrics, signature, and splits."""
    X = df[['returns', 'volatility', 'volume_ratio']]
    y = df['regime']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Create signature
    signature = infer_signature(X_test, y_proba)
    
    print("\nModel Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['Bull', 'Bear', 'Sideways', 'High Vol']
    ))
    
    # Return splits for saving
    return model, signature, accuracy, f1, X_train, X_test, y_train, y_test


def save_monitoring_data(X_train, y_train, X_test, y_test, output_dir="monitoring_data"):
    """
    Save training data and ground truth for Domino model monitoring.
    
    Training Data: Features + target used to train the model
    Ground Truth: Test set with actual labels (simulates future ground truth)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save training data (features + target)
    training_data = X_train.copy()
    training_data['regime'] = y_train.values
    training_path = output_dir / "training_data.csv"
    training_data.to_csv(training_path, index=False)
    print(f"✓ Saved training data: {training_path} ({len(training_data)} rows)")
    
    # Save ground truth (test set with actual labels)
    # This simulates what you'd get from actual outcomes
    ground_truth = X_test.copy()
    ground_truth['regime'] = y_test.values
    ground_truth_path = output_dir / "ground_truth.csv"
    ground_truth.to_csv(ground_truth_path, index=False)
    print(f"✓ Saved ground truth: {ground_truth_path} ({len(ground_truth)} rows)")
    
    # Save schema info
    schema = {
        "features": list(X_train.columns),
        "target": "regime",
        "model_type": "classification",
        "classes": {
            "0": "Bull Market",
            "1": "Bear Market", 
            "2": "Sideways Market",
            "3": "High Volatility"
        }
    }
    schema_path = output_dir / "schema.json"
    with open(schema_path, 'w') as f:
        json.dump(schema, f, indent=2)
    print(f"✓ Saved schema: {schema_path}")
    
    return training_path, ground_truth_path, schema_path


def save_artifacts(model, signature, output_dir="artifacts"):
    """Save model, signature, and supporting files to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save model
    pkl_path = output_dir / "model.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save signature
    sig_path = output_dir / "signature.pkl"
    joblib.dump(signature, sig_path)
    
    # Save requirements
    requirements = output_dir / "requirements.txt"
    requirements.write_text(
        "scikit-learn==1.3.2\n"
        "pandas==2.1.0\n"
        "numpy==1.24.3\n"
        "mlflow>=2.9.0\n"
    )
    return pkl_path, sig_path


def register_to_mlflow(pkl_path, sig_path, model_params, registry_name, experiment=None, accuracy=None, f1=None):
    """Register model to MLflow Model Registry using saved signature."""
    if experiment:
        mlflow.set_experiment(experiment)
    
    # Load saved signature
    signature = joblib.load(sig_path)
    
    with mlflow.start_run() as run:
        # Log metrics
        if accuracy is not None:
            mlflow.log_metric("accuracy", accuracy)
        if f1 is not None:
            mlflow.log_metric("f1_score", f1)
        
        # Log parameters
        mlflow.log_params(model_params)
        
        # Log model as PyFunc
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=PicklePyFunc(),
            artifacts={"model_pkl": str(pkl_path)},
            pip_requirements=[
                "mlflow>=2.9.0",
                "pandas>=2.0.0",
                "scikit-learn==1.3.2",
            ],
            signature=signature,
        )
        
        print(f"✓ Logged model to run: {run.info.run_id}")
    
    # Register to Model Registry
    mv = mlflow.register_model(model_uri=model_info.model_uri, name=registry_name)
    
    # Wait for registration to complete
    client = MlflowClient()
    while True:
        mv = client.get_model_version(registry_name, mv.version)
        if mv.status in ("READY", "FAILED_REGISTRATION"):
            break
        time.sleep(1)
    
    print(f"✓ Registered: name={mv.name} version={mv.version} status={mv.status}")
    
    # Create metadata
    metadata = {
        "model_name": registry_name,
        "version": mv.version,
        "description": "Market regime classification model that predicts Bull, Bear, Sideways, or High Volatility conditions",
        "mlflow_run_id": run.info.run_id,
        "training_date": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "metrics": {
            "accuracy": round(accuracy, 4) if accuracy else None,
            "f1_score": round(f1, 4) if f1 else None,
        },
        "regime_labels": {
            "0": "Bull Market",
            "1": "Bear Market",
            "2": "Sideways Market",
            "3": "High Volatility"
        }
    }
    
    with open("metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print("✓ Saved metadata.json")
    
    return mv


def main():
    parser = argparse.ArgumentParser(description="Train and register market regime model")
    parser.add_argument("--name", required=True, help="MLflow Model Registry name")
    parser.add_argument("--experiment", default=None, help="MLflow experiment name")
    parser.add_argument("--samples", type=int, default=5000, help="Number of training samples")
    parser.add_argument("--output-dir", default="artifacts", help="Output directory for artifacts")
    parser.add_argument("--monitoring-dir", default="monitoring_data", help="Output directory for monitoring data")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Market Regime Model: Train and Register")
    print("=" * 60)
    
    # Generate and train
    print("\n1. Generating synthetic data...")
    df = generate_synthetic_data(n_samples=args.samples)
    print(f"   Generated {len(df)} samples")
    
    print("\n2. Training model...")
    model, signature, accuracy, f1, X_train, X_test, y_train, y_test = train_model(df)
    
    # Save monitoring data
    print("\n3. Saving monitoring data...")
    training_path, ground_truth_path, schema_path = save_monitoring_data(
        X_train, y_train, X_test, y_test, args.monitoring_dir
    )
    
    # Save artifacts including signature
    print("\n4. Saving model artifacts...")
    pkl_path, sig_path = save_artifacts(model, signature, args.output_dir)
    
    # Prepare model parameters for logging
    model_params = {
        "n_estimators": model.n_estimators,
        "max_depth": model.max_depth,
        "model_type": "RandomForestClassifier"
    }
    
    # Register to MLflow
    print("\n5. Registering to MLflow...")
    mv = register_to_mlflow(
        pkl_path, sig_path, model_params, args.name,
        experiment=args.experiment,
        accuracy=accuracy,
        f1=f1
    )
    
    print("\n" + "=" * 60)
    print("✓ Complete!")
    print(f"  Model: {mv.name} v{mv.version}")
    print(f"  Status: {mv.status}")
    print(f"\n  Monitoring Data:")
    print(f"    Training: {training_path}")
    print(f"    Ground Truth: {ground_truth_path}")
    print(f"    Schema: {schema_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()