"""
Train and evaluate predictive models for insurance risk.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib


def train_models(input_path, output_dir):
    """
    Train and evaluate multiple models for insurance risk prediction.
    
    Args:
        input_path: Path to the feature-engineered data CSV file
        output_dir: Directory to save model artifacts and results
    """
    print(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare features and target
    print("Preparing features and target variable...")
    
    # Define target variable (high risk = 1, low risk = 0)
    # Using loss ratio threshold to define high/low risk
    risk_threshold = df['LossRatio'].quantile(0.75)  # Top 25% as high risk
    df['HighRisk'] = (df['LossRatio'] > risk_threshold).astype(int)
    
    # Select features
    categorical_features = ['Province', 'Gender', 'VehicleType', 'VehicleMake']
    numerical_features = ['TotalPremium', 'TotalClaims']
    
    # Encode categorical variables
    df_encoded = df.copy()
    label_encoders = {}
    
    for feature in categorical_features:
        if feature in df.columns:
            le = LabelEncoder()
            df_encoded[feature] = le.fit_transform(df[feature])
            label_encoders[feature] = le
    
    # Prepare feature matrix
    feature_columns = [col for col in categorical_features + numerical_features if col in df.columns]
    X = df_encoded[feature_columns]
    y = df_encoded['HighRisk']
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {
        'logistic_regression': LogisticRegression(random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        
        # Train model
        if model_name == 'logistic_regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred)),
            'recall': float(recall_score(y_test, y_pred)),
            'f1': float(f1_score(y_test, y_pred)),
            'roc_auc': float(roc_auc_score(y_test, y_pred_proba))
        }
        
        results[model_name] = metrics
        
        # Save model
        model_path = os.path.join(output_dir, f'{model_name}_model.joblib')
        joblib.dump(model, model_path)
        
        print(f"{model_name} - Accuracy: {metrics['accuracy']:.3f}, ROC-AUC: {metrics['roc_auc']:.3f}")
    
    # Save scaler and encoders
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))
    joblib.dump(label_encoders, os.path.join(output_dir, 'label_encoders.joblib'))
    
    # Save metrics
    with open(os.path.join(output_dir, "model_metrics.json"), "w") as f:
        json.dump(results, f, indent=4)
    
    # Create model summary
    with open(os.path.join(output_dir, "model_summary.md"), "w") as f:
        f.write("# Insurance Risk Predictive Models: Performance Summary\n\n")
        f.write(f"**Dataset:** {len(df)} records\n")
        f.write(f"**Features:** {len(feature_columns)}\n")
        f.write(f"**High Risk Threshold:** Loss Ratio > {risk_threshold:.3f}\n\n")
        
        f.write("## Model Performance\n\n")
        for model_name, metrics in results.items():
            f.write(f"### {model_name.replace('_', ' ').title()}\n")
            for metric, value in metrics.items():
                f.write(f"- **{metric.upper()}:** {value:.3f}\n")
            f.write("\n")
        
        # Determine best model
        best_model = max(results.keys(), key=lambda k: results[k]['roc_auc'])
        f.write(f"## Recommendation\n\n")
        f.write(f"**Best Model:** {best_model.replace('_', ' ').title()} ")
        f.write(f"(ROC-AUC: {results[best_model]['roc_auc']:.3f})\n")
    
    print(f"Model files saved to {output_dir}")
    return results


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python train_model.py <input_path> <output_dir>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_dir = sys.argv[2]
    
    train_models(input_path, output_dir)