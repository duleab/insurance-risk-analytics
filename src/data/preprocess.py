"""
Preprocess insurance data for analysis and modeling.
"""

import os
import sys
import pandas as pd


def preprocess_insurance_data(input_path, output_path):
    """
    Preprocess insurance data and save to CSV.
    
    Args:
        input_path: Path to the raw data CSV file
        output_path: Path to save the processed data
    """
    print(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    
    print(f"Raw data shape: {df.shape}")
    
    # Handle missing values
    print("Handling missing values...")
    # For numeric columns, fill with median
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # For categorical columns, fill with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # Feature engineering
    print("Creating derived features...")
    
    # Calculate loss ratio
    df['LossRatio'] = df['TotalClaims'] / df['TotalPremium']
    
    # Replace infinite values with NaN and then with 0
    df['LossRatio'] = df['LossRatio'].replace([float('inf')], float('nan'))
    df['LossRatio'] = df['LossRatio'].fillna(0.0)
    
    # Create claim flag
    df['HasClaim'] = (df['TotalClaims'] > 0).astype(int)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")
    
    return df


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python preprocess.py <input_path> <output_path>")
        sys.exit(1)
        
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    preprocess_insurance_data(input_path, output_path) 