"""
Feature engineering for insurance risk modeling.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def engineer_features(input_path, output_path):
    """
    Perform feature engineering on preprocessed insurance data.
    
    Args:
        input_path: Path to the preprocessed data CSV file
        output_path: Path to save the engineered data
    """
    print(f"Loading preprocessed data from {input_path}")
    df = pd.read_csv(input_path)
    
    print(f"Input data shape: {df.shape}")
    
    # Create interaction features
    print("Creating interaction features...")
    
    # Check if necessary columns exist
    if 'VehicleType' in df.columns and 'VehicleAge' in df.columns:
        # Interaction between vehicle type and age
        df['VehicleTypeAge'] = df['VehicleType'] + '_' + df['VehicleAge'].astype(str)
    
    if 'Province' in df.columns and 'VehicleMake' in df.columns:
        # Interaction between province and vehicle make
        df['ProvinceVehicleMake'] = df['Province'] + '_' + df['VehicleMake']
    
    # Create polynomial features for numeric variables
    print("Creating polynomial features...")
    if 'Age' in df.columns:
        df['Age_Squared'] = df['Age'] ** 2
    
    if 'VehicleAge' in df.columns:
        df['VehicleAge_Squared'] = df['VehicleAge'] ** 2
    
    if 'TotalPremium' in df.columns:
        df['Premium_Sqrt'] = np.sqrt(df['TotalPremium'])
    
    # Create categorical encodings
    print("Encoding categorical variables...")
    
    # One-hot encode categorical variables
    categorical_cols = [col for col in ['Province', 'Gender', 'VehicleType', 'VehicleMake'] 
                        if col in df.columns]
    
    for col in categorical_cols:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df = pd.concat([df, dummies], axis=1)
    
    # Scale numeric features if they exist
    print("Scaling numeric features...")
    numeric_features = [col for col in df.columns 
                       if df[col].dtype in ['int64', 'float64'] 
                       and col not in ['PolicyID', 'TotalClaims', 'LossRatio']]
    
    if numeric_features:
        scaler = StandardScaler()
        df[numeric_features] = scaler.fit_transform(df[numeric_features])
    
    print(f"Engineered data shape: {df.shape}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Engineered data saved to {output_path}")
    
    return df


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python feature_engineering.py <input_path> <output_path>")
        sys.exit(1)
        
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    engineer_features(input_path, output_path) 