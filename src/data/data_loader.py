"""Data loading and preprocessing module for insurance data."""

import os
import pandas as pd


def load_raw_data(filepath):
    """
    Load raw insurance data from a CSV file.
    
    Args:
        filepath (str): Path to the CSV file.
        
    Returns:
        pandas.DataFrame: Raw insurance data.
    """
    return pd.read_csv(filepath)


def preprocess_data(df):
    """
    Preprocess insurance data for analysis.
    
    Args:
        df (pandas.DataFrame): Raw insurance data.
        
    Returns:
        pandas.DataFrame: Preprocessed insurance data.
    """
    # Make a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Handle missing values
    # For numeric columns, fill with median
    numeric_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    # For categorical columns, fill with mode
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
    
    # Convert date columns if any
    if 'PolicyStartDate' in df_processed.columns:
        df_processed['PolicyStartDate'] = pd.to_datetime(df_processed['PolicyStartDate'])
    
    # Create derived features
    if 'TotalClaims' in df_processed.columns and 'TotalPremium' in df_processed.columns:
        # Calculate loss ratio
        df_processed['LossRatio'] = df_processed['TotalClaims'] / df_processed['TotalPremium']
        # Replace infinite values with NaN and then with a large value
        df_processed['LossRatio'] = df_processed['LossRatio'].replace([float('inf')], float('nan'))
        df_processed['LossRatio'] = df_processed['LossRatio'].fillna(10.0)  # Arbitrary high value
    
    return df_processed


def save_processed_data(df, output_filepath):
    """
    Save processed data to a CSV file.
    
    Args:
        df (pandas.DataFrame): Processed data.
        output_filepath (str): Path to save the CSV file.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    df.to_csv(output_filepath, index=False)


def main(input_filepath, output_filepath):
    """
    Main function to load, process, and save insurance data.
    
    Args:
        input_filepath (str): Path to the raw data.
        output_filepath (str): Path to save the processed data.
    """
    print(f"Loading data from {input_filepath}")
    df = load_raw_data(input_filepath)
    
    print("Preprocessing data")
    df_processed = preprocess_data(df)
    
    print(f"Saving processed data to {output_filepath}")
    save_processed_data(df_processed, output_filepath)
    
    print("Done!")


if __name__ == "__main__":
    # Example usage
    # main("data/raw/insurance_data.csv", "data/processed/insurance_data_processed.csv")
    pass 