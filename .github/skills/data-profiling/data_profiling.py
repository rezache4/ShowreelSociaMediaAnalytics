"""
Data Profiling Utilities for AFB_Lab EDA

Reusable functions for data quality assessment, missingness analysis,
outlier detection, and cross-platform comparisons.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def generate_quality_report(df: pd.DataFrame, dataset_name: str, platform: Optional[str] = None) -> pd.DataFrame:
    """
    Generate comprehensive data quality scorecard.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
    dataset_name : str
        Name of the dataset (e.g., "Facebook Posts")
    platform : str, optional
        Platform name (e.g., "Facebook", "Instagram", "TikTok", "YouTube")
    
    Returns:
    --------
    pd.DataFrame
        Quality report with metrics per column
    """
    report = []
    
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        dtype = str(df[col].dtype)
        cardinality = df[col].nunique()
        
        # For numeric columns, get outlier count (IQR method)
        outliers = 0
        if pd.api.types.is_numeric_dtype(df[col]):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        
        report.append({
            'Column': col,
            'Type': dtype,
            'Non-Null': len(df) - missing_count,
            'Missing': missing_count,
            'Missing %': f"{missing_pct:.2f}%",
            'Unique': cardinality,
            'Outliers': outliers if pd.api.types.is_numeric_dtype(df[col]) else 'N/A'
        })
    
    report_df = pd.DataFrame(report)
    
    # Print header info
    platform_str = f" ({platform})" if platform else ""
    print(f"\n{'='*80}")
    print(f"DATA QUALITY REPORT: {dataset_name}{platform_str}")
    print(f"{'='*80}")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Duplicates: {df.duplicated().sum()} ({(df.duplicated().sum() / len(df) * 100):.2f}%)")
    print(f"{'='*80}\n")
    
    return report_df


def analyze_missingness(df: pd.DataFrame, plot: bool = True) -> pd.DataFrame:
    """
    Analyze missing data patterns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
    plot : bool, default True
        Generate visualization
    
    Returns:
    --------
    pd.DataFrame
        Missing data summary
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': df.columns,
        'Missing Count': missing.values,
        'Missing %': missing_pct.values
    }).sort_values('Missing %', ascending=False)
    
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar plot
        missing_df[missing_df['Missing Count'] > 0].plot(
            x='Column', y='Missing Count', kind='bar', ax=axes[0], legend=False
        )
        axes[0].set_title('Missing Data Count', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Count')
        axes[0].set_xlabel('')
        
        # Heatmap of missingness
        sns.heatmap(df.isnull(), cbar=True, cmap='RdYlGn_r', ax=axes[1], yticklabels=False)
        axes[1].set_title('Missingness Heatmap (Yellow = Missing)', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    print(f"\nMissingness Summary:")
    print(missing_df[missing_df['Missing Count'] > 0].to_string(index=False))
    
    return missing_df


def detect_outliers_iqr(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                        multiplier: float = 1.5, plot: bool = True) -> Dict[str, List[int]]:
    """
    Detect outliers using Interquartile Range (IQR) method.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
    columns : list, optional
        Columns to check. If None, checks all numeric columns.
    multiplier : float, default 1.5
        IQR multiplier for sensitivity (higher = less sensitive)
    plot : bool, default True
        Generate box plot visualization
    
    Returns:
    --------
    dict
        Outlier indices per column
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outliers = {}
    numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outliers[col] = df[outlier_mask].index.tolist()
        
        if len(outliers[col]) > 0:
            pct = (len(outliers[col]) / len(df)) * 100
            print(f"{col}: {len(outliers[col])} outliers ({pct:.2f}%)")
    
    if plot and numeric_cols:
        fig, axes = plt.subplots(1, len(numeric_cols), figsize=(5*len(numeric_cols), 5))
        if len(numeric_cols) == 1:
            axes = [axes]
        
        for ax, col in zip(axes, numeric_cols):
            df.boxplot(column=col, ax=ax)
            ax.set_title(f'{col} (Outliers: {len(outliers[col])})', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    return outliers


def platform_comparison(dfs_dict: Dict[str, pd.DataFrame], metric_cols: List[str]) -> None:
    """
    Compare metrics across platforms.
    
    Parameters:
    -----------
    dfs_dict : dict
        Dictionary with platform names as keys and DataFrames as values
    metric_cols : list
        Columns to compare across platforms
    """
    comparison = []
    
    for platform, df in dfs_dict.items():
        for col in metric_cols:
            if col in df.columns:
                comparison.append({
                    'Platform': platform,
                    'Metric': col,
                    'Mean': df[col].mean(),
                    'Median': df[col].median(),
                    'Std': df[col].std(),
                    'Min': df[col].min(),
                    'Max': df[col].max()
                })
    
    comparison_df = pd.DataFrame(comparison)
    
    print("\nCross-Platform Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Visualization
    if comparison_df.shape[0] > 0:
        for metric in metric_cols:
            metric_data = comparison_df[comparison_df['Metric'] == metric]
            if len(metric_data) > 0:
                fig, ax = plt.subplots(figsize=(10, 5))
                metric_data.plot(x='Platform', y=['Mean', 'Median'], kind='bar', ax=ax)
                ax.set_title(f'{metric} Across Platforms', fontsize=12, fontweight='bold')
                ax.set_ylabel('Value')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()


def check_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None, report: bool = True) -> Tuple[int, float]:
    """
    Identify duplicate rows.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
    subset : list, optional
        Check duplicates on specific columns only
    report : bool, default True
        Display sample duplicate rows
    
    Returns:
    --------
    tuple
        (duplicate_count, percentage)
    """
    dup_count = df.duplicated(subset=subset).sum()
    dup_pct = (dup_count / len(df)) * 100
    
    print(f"\nDuplicate Analysis:")
    print(f"Total Duplicates: {dup_count} ({dup_pct:.2f}%)")
    
    if report and dup_count > 0:
        dup_mask = df.duplicated(subset=subset, keep=False)
        print(f"\nSample Duplicates (first 5):")
        print(df[dup_mask].head())
    
    return dup_count, dup_pct


def validate_dtypes(df: pd.DataFrame, expected_types: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Validate data types and flag unexpected types.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
    expected_types : dict, optional
        Expected data types {column: dtype_string}
    
    Returns:
    --------
    pd.DataFrame
        Validation report
    """
    validation = []
    
    for col in df.columns:
        current_type = str(df[col].dtype)
        expected_type = expected_types.get(col, 'Not specified') if expected_types else 'Not specified'
        
        is_valid = 'Unknown' if expected_types is None else (
            'Yes' if str(expected_types.get(col, current_type)) == current_type else 'No'
        )
        
        validation.append({
            'Column': col,
            'Current Type': current_type,
            'Expected Type': expected_type,
            'Valid': is_valid
        })
    
    validation_df = pd.DataFrame(validation)
    
    print("\nData Type Validation Report:")
    print(validation_df.to_string(index=False))
    
    if expected_types and 'No' in validation_df['Valid'].values:
        print("\n⚠️  Type mismatches detected. Review above report.")
    
    return validation_df
