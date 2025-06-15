"""Visualization functions for insurance data analysis."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def set_plotting_style():
    """Set the default plotting style for consistent visualizations."""
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (12, 8)
    plt.rcParams["font.size"] = 12


def plot_loss_ratio_by_category(df, category_col, title=None, figsize=(12, 8)):
    """
    Plot average loss ratio by a categorical variable.
    
    Args:
        df: DataFrame containing the data
        category_col: Column name for categories
        title: Plot title
        figsize: Figure size as tuple (width, height)
    """
    set_plotting_style()
    
    # Group by category and calculate mean loss ratio
    grouped = df.groupby(category_col)['LossRatio'].mean().sort_values()
    
    # Create figure
    plt.figure(figsize=figsize)
    ax = grouped.plot(kind='barh')
    
    # Add title and labels
    if title:
        plt.title(title)
    else:
        plt.title(f'Average Loss Ratio by {category_col}')
    
    plt.xlabel('Loss Ratio (Claims / Premium)')
    plt.ylabel(category_col)
    
    # Add reference line for loss ratio = 1.0
    plt.axvline(x=1.0, color='red', linestyle='--', 
                label='Break-even point')
    
    # Add data labels
    for i, v in enumerate(grouped):
        ax.text(v + 0.02, i, f'{v:.2f}', va='center')
    
    plt.legend()
    plt.tight_layout()
    
    return ax


def plot_premium_claims_scatter(df, x_col='TotalPremium', y_col='TotalClaims', 
                               hue=None, figsize=(12, 8)):
    """
    Create a scatter plot of premium vs claims.
    
    Args:
        df: DataFrame containing the data
        x_col: Column name for x-axis (default: TotalPremium)
        y_col: Column name for y-axis (default: TotalClaims)
        hue: Column name for color grouping
        figsize: Figure size as tuple (width, height)
    """
    set_plotting_style()
    
    plt.figure(figsize=figsize)
    
    # Create scatter plot
    ax = sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue, alpha=0.7)
    
    # Add reference line where premium = claims
    max_val = max(df[x_col].max(), df[y_col].max())
    plt.plot([0, max_val], [0, max_val], 'r--', 
             label='Premium = Claims')
    
    # Add title and labels
    plt.title(f'{y_col} vs {x_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    
    plt.legend()
    plt.tight_layout()
    
    return ax


def plot_claim_distribution(df, figsize=(12, 8)):
    """
    Plot the distribution of claim amounts.
    
    Args:
        df: DataFrame containing the data
        figsize: Figure size as tuple (width, height)
    """
    set_plotting_style()
    
    plt.figure(figsize=figsize)
    
    # Filter to only policies with claims
    claims_df = df[df['TotalClaims'] > 0]
    
    # Create histogram
    ax = sns.histplot(data=claims_df, x='TotalClaims', kde=True)
    
    # Add vertical line for mean
    mean_claim = claims_df['TotalClaims'].mean()
    plt.axvline(mean_claim, color='red', linestyle='--', 
                label=f'Mean: {mean_claim:.2f}')
    
    # Add vertical line for median
    median_claim = claims_df['TotalClaims'].median()
    plt.axvline(median_claim, color='green', linestyle='--', 
                label=f'Median: {median_claim:.2f}')
    
    plt.title('Distribution of Claim Amounts')
    plt.xlabel('Total Claims')
    plt.ylabel('Frequency')
    
    plt.legend()
    plt.tight_layout()
    
    return ax


def plot_geographical_risk(df, geo_col='Province', metric='LossRatio',
                          figsize=(12, 8)):
    """
    Create a bar chart showing risk metrics by geographical region.
    
    Args:
        df: DataFrame containing the data
        geo_col: Column name for geographical regions
        metric: Metric to plot (default: LossRatio)
        figsize: Figure size as tuple (width, height)
    """
    set_plotting_style()
    
    plt.figure(figsize=figsize)
    
    # Group by geography and calculate mean of the metric
    grouped = df.groupby(geo_col)[metric].mean().sort_values(ascending=False)
    
    # Create bar chart
    ax = grouped.plot(kind='bar')
    
    # Add title and labels
    plt.title(f'Average {metric} by {geo_col}')
    plt.xlabel(geo_col)
    plt.ylabel(f'Average {metric}')
    
    # Add reference line if metric is LossRatio
    if metric == 'LossRatio':
        plt.axhline(y=1.0, color='red', linestyle='--', 
                    label='Break-even point')
    
    # Add data labels
    for i, v in enumerate(grouped):
        ax.text(i, v + 0.02, f'{v:.2f}', ha='center')
    
    plt.legend()
    plt.tight_layout()
    
    return ax 