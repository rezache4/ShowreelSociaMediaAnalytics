---
name: data-profiling
description: "Use when: you need reusable data profiling functions for quality assessment, missingness analysis, outlier detection, and cross-platform comparisons. Provides ready-made functions to generate quality reports and visualizations."
---

# Data Profiling Utilities

Reusable functions for data quality assessment and exploratory analysis across the Camihawke (platform-specific) and YouTube datasets.

## Available Functions

### `generate_quality_report(df, dataset_name, platform=None)`
Creates a comprehensive data quality scorecard including:
- Missingness percentages
- Data type summary
- Duplicate count
- Cardinality (unique values per column)
- Numeric column statistics (mean, std, min, max, outlier count)

**Usage**:
```python
from data_profiling import generate_quality_report
report = generate_quality_report(df_fb_posts, "Facebook Posts", platform="Facebook")
print(report)
```

### `analyze_missingness(df, plot=True)`
Returns:
- Missing value count and percentage per column
- Visualization if `plot=True`: heatmap and bar chart
- Identifies patterns and high-risk columns

**Usage**:
```python
from data_profiling import analyze_missingness
missing_df = analyze_missingness(df, plot=True)
```

### `detect_outliers_iqr(df, columns=None, multiplier=1.5)`
IQR-based outlier detection for numeric columns:
- Returns outlier count and indices
- Visualizes with box plots
- `multiplier` parameter (default 1.5) for sensitivity tuning

**Usage**:
```python
from data_profiling import detect_outliers_iqr
outliers = detect_outliers_iqr(df, columns=['likes', 'shares'], multiplier=1.5)
```

### `platform_comparison(dfs_dict, metric_cols)`
Compares metrics across platforms:
- `dfs_dict`: Dict with platform names as keys and DataFrames as values
- `metric_cols`: Columns to compare (e.g., ['engagement_rate', 'comment_count'])
- Returns side-by-side statistics and visualization

**Usage**:
```python
from data_profiling import platform_comparison
dfs = {
    'Facebook': df_fb_posts,
    'Instagram': df_ig_posts,
    'TikTok': df_tk_posts
}
platform_comparison(dfs, metric_cols=['likes', 'comments'])
```

### `check_duplicates(df, subset=None, report=True)`
- Identifies duplicate rows
- `subset`: Check duplicates on specific columns only
- Returns count, percentage, and optionally displays sample rows

**Usage**:
```python
from data_profiling import check_duplicates
check_duplicates(df, subset=['post_id'], report=True)
```

### `validate_dtypes(df, expected_types=None)`
- Validates data types match expectations
- Flags unexpected types in numeric or categorical columns
- Returns validation report with recommendations

**Usage**:
```python
from data_profiling import validate_dtypes
validate_dtypes(df, expected_types={'user_id': 'int64', 'timestamp': 'datetime64'})
```

## File Structure

```
.github/skills/data-profiling/
├── SKILL.md                 # This file
└── data_profiling.py        # Reusable functions module
```

## Import in Your Notebook

Add this cell near the top of your notebook:

```python
import sys
sys.path.insert(0, '.github/skills/data-profiling')
from data_profiling import (
    generate_quality_report,
    analyze_missingness,
    detect_outliers_iqr,
    platform_comparison,
    check_duplicates,
    validate_dtypes
)
```

Then use functions throughout your analysis as needed.
