"""
Exploratory Data Analysis for Insurance Risk Analytics.

This script performs exploratory data analysis on insurance data
and generates visualizations and insights.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add the parent directory to the path to import custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.data_loader import load_raw_data, preprocess_data
from src.visualization.visualize import (
    plot_loss_ratio_by_category,
    plot_premium_claims_scatter,
    plot_claim_distribution,
    plot_geographical_risk
)


def generate_synthetic_data(n_samples=1000):
    """
    Generate synthetic insurance data for demonstration purposes.
    
    Args:
        n_samples: Number of data points to generate
        
    Returns:
        pandas.DataFrame: Synthetic insurance data
    """
    np.random.seed(42)
    
    # Create dummy data
    df = pd.DataFrame({
        'PolicyID': range(1, n_samples + 1),
        'Province': np.random.choice(['Western Cape', 'Gauteng', 'KwaZulu-Natal', 'Eastern Cape'], n_samples),
        'Gender': np.random.choice(['M', 'F'], n_samples),
        'VehicleType': np.random.choice(['Sedan', 'SUV', 'Hatchback', 'Truck', 'Sports'], n_samples),
        'VehicleMake': np.random.choice(['Toyota', 'Volkswagen', 'Ford', 'BMW', 'Mercedes'], n_samples),
        'CustomValueEstimate': np.random.normal(200000, 50000, n_samples),
        'TotalPremium': np.random.normal(5000, 1000, n_samples),
        'TotalClaims': np.random.exponential(2000, n_samples) * np.random.binomial(1, 0.2, n_samples),
        'PolicyStartDate': pd.date_range(start='2014-02-01', periods=n_samples)
    })
    
    return df


def run_data_overview(df):
    """
    Run data overview analysis and print results.
    
    Args:
        df: DataFrame containing insurance data
    """
    print("=" * 50)
    print("DATA OVERVIEW")
    print("=" * 50)
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumn names: {df.columns.tolist()}")
    
    print("\nData types:")
    print(df.dtypes)
    
    print("\nMissing values:")
    print(df.isnull().sum())
    
    print("\nSample data:")
    print(df.head())
    
    print("\nSummary statistics:")
    print(df.describe(include='all').T)


def run_portfolio_analysis(df):
    """
    Run portfolio performance analysis and print results.
    
    Args:
        df: DataFrame containing insurance data
    """
    print("=" * 50)
    print("PORTFOLIO PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    # Calculate overall loss ratio
    overall_loss_ratio = df['TotalClaims'].sum() / df['TotalPremium'].sum()
    print(f"Overall Loss Ratio: {overall_loss_ratio:.4f}")
    
    # Calculate claim frequency
    claim_frequency = (df['TotalClaims'] > 0).mean()
    print(f"Claim Frequency: {claim_frequency:.4f} ({claim_frequency*100:.2f}%)")
    
    # Calculate average premium and claim
    avg_premium = df['TotalPremium'].mean()
    avg_claim = df[df['TotalClaims'] > 0]['TotalClaims'].mean()
    print(f"Average Premium: {avg_premium:.2f}")
    print(f"Average Claim (for policies with claims): {avg_claim:.2f}")


def run_segment_analysis(df, output_dir='results'):
    """
    Run segment analysis and generate visualizations.
    
    Args:
        df: DataFrame containing insurance data
        output_dir: Directory to save visualization files
    """
    print("=" * 50)
    print("SEGMENT ANALYSIS")
    print("=" * 50)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Loss ratio by province
    print("Analyzing loss ratio by province...")
    ax = plot_loss_ratio_by_category(df, 'Province', title='Loss Ratio by Province')
    plt.savefig(os.path.join(output_dir, 'loss_ratio_by_province.png'))
    plt.close()
    
    # Loss ratio by vehicle type
    print("Analyzing loss ratio by vehicle type...")
    ax = plot_loss_ratio_by_category(df, 'VehicleType', title='Loss Ratio by Vehicle Type')
    plt.savefig(os.path.join(output_dir, 'loss_ratio_by_vehicle_type.png'))
    plt.close()
    
    # Loss ratio by gender
    print("Analyzing loss ratio by gender...")
    ax = plot_loss_ratio_by_category(df, 'Gender', title='Loss Ratio by Gender')
    plt.savefig(os.path.join(output_dir, 'loss_ratio_by_gender.png'))
    plt.close()


def run_premium_claims_analysis(df, output_dir='results'):
    """
    Run premium vs claims analysis and generate visualizations.
    
    Args:
        df: DataFrame containing insurance data
        output_dir: Directory to save visualization files
    """
    print("=" * 50)
    print("PREMIUM VS CLAIMS ANALYSIS")
    print("=" * 50)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Scatter plot of premium vs claims
    print("Generating premium vs claims scatter plot...")
    ax = plot_premium_claims_scatter(df, hue='Province')
    plt.savefig(os.path.join(output_dir, 'premium_vs_claims.png'))
    plt.close()
    
    # Distribution of claim amounts
    print("Analyzing claim distribution...")
    ax = plot_claim_distribution(df)
    plt.savefig(os.path.join(output_dir, 'claim_distribution.png'))
    plt.close()


def run_vehicle_analysis(df, output_dir='results'):
    """
    Run vehicle analysis and generate visualizations.
    
    Args:
        df: DataFrame containing insurance data
        output_dir: Directory to save visualization files
    """
    print("=" * 50)
    print("VEHICLE ANALYSIS")
    print("=" * 50)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze vehicle makes by risk
    print("Analyzing vehicle makes by risk...")
    vehicle_risk = df.groupby('VehicleMake').agg({
        'TotalPremium': 'sum',
        'TotalClaims': 'sum',
        'PolicyID': 'count'
    }).rename(columns={'PolicyID': 'PolicyCount'})
    
    vehicle_risk['LossRatio'] = vehicle_risk['TotalClaims'] / vehicle_risk['TotalPremium']
    vehicle_risk['AvgPremium'] = vehicle_risk['TotalPremium'] / vehicle_risk['PolicyCount']
    vehicle_risk['AvgClaim'] = vehicle_risk['TotalClaims'] / vehicle_risk['PolicyCount']
    
    # Sort by loss ratio
    vehicle_risk_sorted = vehicle_risk.sort_values('LossRatio', ascending=False)
    print(vehicle_risk_sorted)
    
    # Plot vehicle makes by loss ratio
    plt.figure(figsize=(12, 8))
    sns.barplot(x=vehicle_risk_sorted.index, y='LossRatio', data=vehicle_risk_sorted)
    plt.axhline(y=1.0, color='red', linestyle='--', label='Break-even point')
    plt.title('Loss Ratio by Vehicle Make')
    plt.xlabel('Vehicle Make')
    plt.ylabel('Loss Ratio')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_ratio_by_vehicle_make.png'))
    plt.close()


def generate_key_insights(df):
    """
    Generate key insights from the analysis.
    
    Args:
        df: DataFrame containing insurance data
    """
    print("=" * 50)
    print("KEY INSIGHTS AND FINDINGS")
    print("=" * 50)
    
    # Overall portfolio performance
    overall_loss_ratio = df['TotalClaims'].sum() / df['TotalPremium'].sum()
    claim_frequency = (df['TotalClaims'] > 0).mean()
    
    print("1. Overall portfolio performance:")
    print(f"   - Overall loss ratio: {overall_loss_ratio:.4f}")
    print(f"   - Claim frequency: {claim_frequency:.4f} ({claim_frequency*100:.2f}%)")
    
    # Segment insights
    province_lr = df.groupby('Province')['LossRatio'].mean().sort_values()
    vehicle_lr = df.groupby('VehicleType')['LossRatio'].mean().sort_values()
    gender_lr = df.groupby('Gender')['LossRatio'].mean().sort_values()
    
    print("\n2. Segment insights:")
    print(f"   - Provinces with lowest risk: {', '.join(province_lr.index[:2])}")
    print(f"   - Vehicle types with lowest risk: {', '.join(vehicle_lr.index[:2])}")
    print(f"   - Gender-based risk differences: {gender_lr.index[0]} has lower risk than {gender_lr.index[-1]}")
    
    # Vehicle insights
    vehicle_risk = df.groupby('VehicleMake').agg({
        'TotalPremium': 'sum',
        'TotalClaims': 'sum'
    })
    vehicle_risk['LossRatio'] = vehicle_risk['TotalClaims'] / vehicle_risk['TotalPremium']
    vehicle_risk_sorted = vehicle_risk.sort_values('LossRatio')
    
    print("\n3. Vehicle insights:")
    print(f"   - Makes with lowest risk: {', '.join(vehicle_risk_sorted.index[:2])}")
    print(f"   - Makes with highest risk: {', '.join(vehicle_risk_sorted.index[-2:])}")
    
    # Recommendations
    print("\n4. Recommendations for further analysis:")
    print("   - Analyze interaction effects between province and vehicle type")
    print("   - Investigate time-based patterns in claims")
    print("   - Explore additional customer demographics if available")
    print("   - Analyze claim severity separately from claim frequency")


def main():
    """Main function to run the EDA analysis."""
    # Set up output directory
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data or generate synthetic data
    try:
        print("Attempting to load real data...")
        df = load_raw_data('data/raw/insurance_data.csv')
        print(f"Data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print("Data file not found. Creating synthetic data for demonstration...")
        df = generate_synthetic_data(1000)
        print(f"Created synthetic data. Shape: {df.shape}")
    
    # Preprocess data
    print("Preprocessing data...")
    df_processed = preprocess_data(df)
    print(f"Processed data shape: {df_processed.shape}")
    
    # Run analyses
    run_data_overview(df_processed)
    run_portfolio_analysis(df_processed)
    run_segment_analysis(df_processed, output_dir)
    run_premium_claims_analysis(df_processed, output_dir)
    run_vehicle_analysis(df_processed, output_dir)
    generate_key_insights(df_processed)
    
    print("\nAnalysis complete. Visualizations saved to the 'results' directory.")


if __name__ == "__main__":
    main() 