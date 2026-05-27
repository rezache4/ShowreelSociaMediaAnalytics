---
name: data-quality-practices
description: "Auto-apply data cleaning and quality validation best practices across notebooks. Validates missing data handling, dtype consistency, and deduplication standards."
applyTo: ["**/*.ipynb"]
---

# Data Cleaning & Quality Standards

Apply these best practices automatically when working with datasets in this project.

## Missing Data Handling

- **Report missingness** before imputation: Always display `df.isnull().sum()` and percentage
- **Imputation strategy**: Document choice (drop, forward fill, mean, median, interpolation)
- **Preserve original**: Keep an unmodified copy for reference (`df_original = df.copy()`)
- **Post-imputation validation**: Verify no NaN remains and compare distribution before/after

## Data Type Consistency

- **Verify on load**: Use `df.dtypes` immediately after `pd.read_csv()`
- **Fix categorical types**: Convert to `category` dtype for memory efficiency if many repeating values
- **Date parsing**: Always parse datetime columns: `pd.to_datetime(col, errors='coerce')`
- **Numeric validation**: Check for unexpected strings in numeric columns with `pd.numeric(df[col], errors='coerce')`

## Duplicate & Outlier Management

- **Check duplicates early**: Run `df.duplicated().sum()` and `df.drop_duplicates(inplace=True)` if needed
- **Document outlier removal**: Log which rows were removed and threshold used (IQR, Z-score, domain knowledge)
- **Never silently drop**: Always report count and percentage before removing

## Platform-Specific Analysis Workflow

For Camihawke data (Facebook, Instagram, TikTok):

1. **Platform Separation**: Load and analyze each platform independently first
   - fb_posts.csv, fb_comments.csv
   - ig_posts.csv, ig_comments.csv
   - tk_posts.csv, tk_comments.csv

2. **Per-Platform Validation**:
   - Check platform-specific column patterns
   - Validate engagement metrics (likes, shares, comments) ranges
   - Identify platform-specific missing data patterns

3. **Cross-Platform Comparison**:
   - Only after individual validation, merge for comparative analysis
   - Track platform as a categorical feature throughout
   - Compare metrics, user behavior, and content patterns

4. **Data Schema Alignment**:
   - Ensure all platforms have comparable columns before combining
   - Note differences in data availability per platform

## Feature Engineering Notes

- **Engagement rates**: Calculate from raw counts, document formula
- **Time-based features**: Extract hour, day, day_of_week separately for analysis
- **Text features**: Document encoding and cleaning steps (lowercase, removal of special chars, etc.)

## Export & Reproducibility

- **Save processed datasets**: Create `data_processed/` folder for cleaned CSVs with timestamp
- **Log transformations**: Include comment cells documenting all major transformations
- **Version tracking**: Use comments to note which version of raw data was processed
