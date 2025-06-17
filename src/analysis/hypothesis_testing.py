"""
Statistical hypothesis testing for insurance risk analysis.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def load_data(input_path):
    """Load the processed data for analysis."""
    print(f"Loading data from {input_path}")
    return pd.read_csv(input_path)


def run_t_test(df, group_col, value_col, group1, group2=None):
    """
    Run a t-test to compare means between groups.
    
    Args:
        df: DataFrame containing the data
        group_col: Column name for grouping
        value_col: Column name for the value to compare
        group1: First group value
        group2: Second group value (if None, compare group1 to all others)
    
    Returns:
        Dictionary with test results
    """
    if group2 is None:
        # Compare one group against all others
        group1_data = df[df[group_col] == group1][value_col].dropna()
        group2_data = df[df[group_col] != group1][value_col].dropna()
        group2_name = "All Others"
    else:
        # Compare two specific groups
        group1_data = df[df[group_col] == group1][value_col].dropna()
        group2_data = df[df[group_col] == group2][value_col].dropna()
        group2_name = group2
    
    # Run t-test
    t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=False)
    
    # Calculate means and difference
    mean1 = group1_data.mean()
    mean2 = group2_data.mean()
    diff = mean1 - mean2
    
    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt(
        ((len(group1_data) - 1) * group1_data.std()**2 + 
         (len(group2_data) - 1) * group2_data.std()**2) / 
        (len(group1_data) + len(group2_data) - 2)
    )
    
    cohens_d = diff / pooled_std if pooled_std != 0 else 0
    
    return {
        'group1': group1,
        'group2': group2_name,
        'mean1': mean1,
        'mean2': mean2,
        'diff': diff,
        'p_value': p_value,
        't_stat': t_stat,
        'cohens_d': cohens_d,
        'significant': p_value < 0.05,
        'better_group': group1 if diff < 0 else group2_name
    }


def run_chi_square_test(df, group_col, binary_col):
    """
    Run a chi-square test for independence between a grouping variable and a binary outcome.
    
    Args:
        df: DataFrame containing the data
        group_col: Column name for grouping
        binary_col: Column name for the binary outcome (0/1)
    
    Returns:
        DataFrame with test results for each group
    """
    # Get unique groups
    groups = df[group_col].unique()
    
    results = []
    
    # Overall rate
    overall_rate = df[binary_col].mean()
    
    for group in groups:
        # Create contingency table
        group_data = df[df[group_col] == group]
        group_rate = group_data[binary_col].mean()
        
        # Count of successes and failures in this group
        success_count = group_data[binary_col].sum()
        failure_count = len(group_data) - success_count
        
        # Expected counts based on overall rate
        expected_success = len(group_data) * overall_rate
        expected_failure = len(group_data) * (1 - overall_rate)
        
        # Chi-square test
        observed = np.array([success_count, failure_count])
        expected = np.array([expected_success, expected_failure])
        
        chi2, p_value = stats.chisquare(observed, expected)
        
        results.append({
            'group': group,
            'observed_rate': group_rate,
            'overall_rate': overall_rate,
            'diff': group_rate - overall_rate,
            'chi2': chi2,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'lower_risk': group_rate < overall_rate
        })
    
    return pd.DataFrame(results)


def analyze_risk_factors(df, output_dir):
    """
    Perform statistical tests to identify significant risk factors.
    
    Args:
        df: DataFrame containing the processed insurance data
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'province': {},
        'gender': {},
        'vehicle_type': {},
        'vehicle_make': {}
    }
    
    # 1. Analyze claim frequency by province
    if 'Province' in df.columns and 'HasClaim' in df.columns:
        print("\nAnalyzing claim frequency by province...")
        province_results = run_chi_square_test(df, 'Province', 'HasClaim')
        province_results = province_results.sort_values('observed_rate')
        results['province']['claim_frequency'] = province_results
        
        # Save results
        province_results.to_csv(f"{output_dir}/province_claim_frequency.csv", index=False)
    
    # 2. Analyze loss ratio by gender
    if 'Gender' in df.columns and 'LossRatio' in df.columns:
        print("\nAnalyzing loss ratio by gender...")
        gender_results = run_t_test(df, 'Gender', 'LossRatio', 'Male', 'Female')
        results['gender']['loss_ratio'] = gender_results
        
        # Save results
        pd.DataFrame([gender_results]).to_csv(f"{output_dir}/gender_loss_ratio.csv", index=False)
    
    # 3. Analyze loss ratio by vehicle type
    if 'VehicleType' in df.columns and 'LossRatio' in df.columns:
        print("\nAnalyzing loss ratio by vehicle type...")
        vehicle_type_results = []
        vehicle_types = df['VehicleType'].unique()
        
        for vtype in vehicle_types:
            test_result = run_t_test(df, 'VehicleType', 'LossRatio', vtype)
            vehicle_type_results.append(test_result)
        
        vehicle_type_df = pd.DataFrame(vehicle_type_results)
        vehicle_type_df = vehicle_type_df.sort_values('mean1')
        results['vehicle_type']['loss_ratio'] = vehicle_type_df
        
        # Save results
        vehicle_type_df.to_csv(f"{output_dir}/vehicle_type_loss_ratio.csv", index=False)
    
    # 4. Analyze claim frequency by vehicle make
    if 'VehicleMake' in df.columns and 'HasClaim' in df.columns:
        print("\nAnalyzing claim frequency by vehicle make...")
        vehicle_make_results = run_chi_square_test(df, 'VehicleMake', 'HasClaim')
        vehicle_make_results = vehicle_make_results.sort_values('observed_rate')
        results['vehicle_make']['claim_frequency'] = vehicle_make_results
        
        # Save results
        vehicle_make_results.to_csv(f"{output_dir}/vehicle_make_claim_frequency.csv", index=False)
    
    # Generate summary report
    generate_summary_report(results, output_dir)
    
    return results


def generate_summary_report(results, output_dir):
    """Generate a summary report of the hypothesis testing results."""
    with open(f"{output_dir}/hypothesis_testing_summary.md", 'w') as f:
        f.write("# Insurance Risk Analytics: Hypothesis Testing Results\n\n")
        
        # Province analysis
        f.write("## Province Analysis\n\n")
        if 'province' in results and 'claim_frequency' in results['province']:
            province_df = results['province']['claim_frequency']
            low_risk_provinces = province_df[
                province_df['lower_risk'] & province_df['significant']
            ]
            
            f.write("### Low Risk Provinces (Significantly Lower Claim Frequency)\n\n")
            f.write("| Province | Observed Rate | Overall Rate | Difference | P-Value | Significant |\n")
            f.write("|----------|---------------|--------------|------------|---------|------------|\n")
            
            for _, row in low_risk_provinces.iterrows():
                f.write(f"| {row['group']} | {row['observed_rate']:.4f} | {row['overall_rate']:.4f} | ")
                f.write(f"{row['diff']:.4f} | {row['p_value']:.4f} | {'Yes' if row['significant'] else 'No'} |\n")
        
        # Gender analysis
        f.write("\n## Gender Analysis\n\n")
        if 'gender' in results and 'loss_ratio' in results['gender']:
            gender_result = results['gender']['loss_ratio']
            
            f.write("### Loss Ratio Comparison\n\n")
            f.write(f"- Male Mean Loss Ratio: {gender_result['mean1']:.4f}\n")
            f.write(f"- Female Mean Loss Ratio: {gender_result['mean2']:.4f}\n")
            f.write(f"- Difference: {gender_result['diff']:.4f}\n")
            f.write(f"- P-Value: {gender_result['p_value']:.4f}\n")
            f.write(f"- Significant: {'Yes' if gender_result['significant'] else 'No'}\n")
            f.write(f"- Lower Risk Group: {gender_result['better_group']}\n")
        
        # Vehicle type analysis
        f.write("\n## Vehicle Type Analysis\n\n")
        if 'vehicle_type' in results and 'loss_ratio' in results['vehicle_type']:
            vehicle_type_df = results['vehicle_type']['loss_ratio']
            low_risk_vehicle_types = vehicle_type_df[
                vehicle_type_df['significant'] & (vehicle_type_df['diff'] < 0)
            ]
            
            f.write("### Low Risk Vehicle Types (Significantly Lower Loss Ratio)\n\n")
            f.write("| Vehicle Type | Mean Loss Ratio | Overall Mean | Difference | P-Value | Significant |\n")
            f.write("|-------------|-----------------|--------------|------------|---------|------------|\n")
            
            for _, row in low_risk_vehicle_types.iterrows():
                f.write(f"| {row['group1']} | {row['mean1']:.4f} | {row['mean2']:.4f} | ")
                f.write(f"{row['diff']:.4f} | {row['p_value']:.4f} | {'Yes' if row['significant'] else 'No'} |\n")
        
        # Vehicle make analysis
        f.write("\n## Vehicle Make Analysis\n\n")
        if 'vehicle_make' in results and 'claim_frequency' in results['vehicle_make']:
            vehicle_make_df = results['vehicle_make']['claim_frequency']
            low_risk_makes = vehicle_make_df[
                vehicle_make_df['lower_risk'] & vehicle_make_df['significant']
            ]
            
            f.write("### Low Risk Vehicle Makes (Significantly Lower Claim Frequency)\n\n")
            f.write("| Vehicle Make | Observed Rate | Overall Rate | Difference | P-Value | Significant |\n")
            f.write("|-------------|---------------|--------------|------------|---------|------------|\n")
            
            for _, row in low_risk_makes.iterrows():
                f.write(f"| {row['group']} | {row['observed_rate']:.4f} | {row['overall_rate']:.4f} | ")
                f.write(f"{row['diff']:.4f} | {row['p_value']:.4f} | {'Yes' if row['significant'] else 'No'} |\n")
        
        # Summary of findings
        f.write("\n## Summary of Low-Risk Customer Segments\n\n")
        f.write("Based on our statistical analysis, the following customer segments demonstrate significantly lower risk:\n\n")
        
        # Provinces
        if 'province' in results and 'claim_frequency' in results['province']:
            province_df = results['province']['claim_frequency']
            low_risk_provinces = province_df[
                province_df['lower_risk'] & province_df['significant']
            ]
            if not low_risk_provinces.empty:
                f.write("1. **Provinces**: ")
                f.write(", ".join(low_risk_provinces['group'].tolist()))
                f.write("\n")
        
        # Gender
        if 'gender' in results and 'loss_ratio' in results['gender']:
            gender_result = results['gender']['loss_ratio']
            if gender_result['significant']:
                f.write(f"2. **Gender**: {gender_result['better_group']}\n")
        
        # Vehicle types
        if 'vehicle_type' in results and 'loss_ratio' in results['vehicle_type']:
            vehicle_type_df = results['vehicle_type']['loss_ratio']
            low_risk_vehicle_types = vehicle_type_df[
                vehicle_type_df['significant'] & (vehicle_type_df['diff'] < 0)
            ]
            if not low_risk_vehicle_types.empty:
                f.write("3. **Vehicle Types**: ")
                f.write(", ".join(low_risk_vehicle_types['group1'].tolist()))
                f.write("\n")
        
        # Vehicle makes
        if 'vehicle_make' in results and 'claim_frequency' in results['vehicle_make']:
            vehicle_make_df = results['vehicle_make']['claim_frequency']
            low_risk_makes = vehicle_make_df[
                vehicle_make_df['lower_risk'] & vehicle_make_df['significant']
            ]
            if not low_risk_makes.empty:
                f.write("4. **Vehicle Makes**: ")
                f.write(", ".join(low_risk_makes['group'].tolist()))
                f.write("\n")
        
        f.write("\nThese segments could be considered for targeted premium reductions to improve customer retention while maintaining profitability.\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python hypothesis_testing.py <input_path> <output_dir>")
        sys.exit(1)
        
    input_path = sys.argv[1]
    output_dir = sys.argv[2]
    
    # Load data
    df = load_data(input_path)
    
    # Run analysis
    analyze_risk_factors(df, output_dir) 