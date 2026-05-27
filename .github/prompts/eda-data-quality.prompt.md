---
description: "Generate comprehensive EDA with data quality checks: missingness, distributions, correlations, outliers, duplicates, data types, and class imbalance"
name: "EDA & Data Quality Analysis"
argument-hint: "Analyze datasets"
---

## Comprehensive Exploratory Data Analysis

Generate detailed EDA and data quality assessment for the datasets in this project. **IMPORTANT: Camihawke data spans 3 platforms (Facebook, Instagram, TikTok) — analyze each platform separately first, then perform cross-platform comparisons.**

### 1. **Data Loading & Overview**
- Load all available datasets (CSV files from Data/Camihawke and Data/YouTube)
- Display shape, dtypes, and first few rows
- Generate basic info() and describe() summaries
- **For Camihawke**: Load posts and comments for each platform separately (fb_, ig_, tk_ prefixes)

### 2. **Platform-Specific Analysis (Camihawke Data)**

Analyze **each platform independently** before any cross-platform work:

**Facebook Analysis** (fb_posts.csv + fb_comments.csv):
- Posts: engagement metrics, posting frequency, content characteristics
- Comments: user activity levels, comment depth
- Quality issues specific to Facebook data

**Instagram Analysis** (ig_posts.csv + ig_comments.csv):
- Posts: IG-specific metrics, hashtag usage if present
- Comments: engagement style differences vs Facebook
- Data schema differences from Facebook

**TikTok Analysis** (tk_posts.csv + tk_comments.csv):
- Posts: TK-specific metrics (views, trends if present)
- Comments: comment frequency and user patterns
- Data structure differences from FB/IG

**For each platform generate**:
- Missing value analysis (platform-specific patterns)
- Distribution analysis of engagement metrics
- Correlation analysis within platform
- Outlier detection
- Data quality report

### 3. **Missingness Analysis**
- Calculate and visualize missing value percentages across all columns
- Show which columns have missing data
- Heatmap of missingness patterns for each platform
- Create summary statistics table

### 4. **Feature Distribution Analysis**
- Univariate distributions (histograms/KDE for numeric, bar plots for categorical)
- Identify skewness and kurtosis for numeric features
- Show value counts for categorical columns
- Flag potential data entry errors or anomalies

### 5. **Correlation Analysis**
- Correlation matrix (Pearson) for numeric features per platform
- Heatmap visualization with proper coloring
- Identify strong correlations (> 0.7 or < -0.7)
- Comment on which feature pairs are highly correlated

### 6. **Outlier Detection**
- IQR-based outlier detection for numeric columns
- Box plots showing outliers per platform
- Count and percentage of outliers per column
- Flag columns with extreme outliers

### 7. **Data Type & Cardinality Analysis**
- Verify correct data types for each column
- Identify potential type mismatches
- Check cardinality (unique value counts)
- Flag categorical columns with too many unique values (potential IDs)
- Identify potential boolean columns treated as strings/integers

### 8. **Duplicate Records**
- Check for complete duplicate rows per platform
- Check for duplicates on key columns (if identifiable)
- Display count and percentage of duplicates
- Provide guidance on handling

### 9. **Class Imbalance** (if applicable)
- Identify target/label columns
- Check class distribution
- Calculate imbalance ratios
- Visualize with bar plots

### 10. **Cross-Platform Analysis (Camihawke Only)**

After individual platform validation:
- **Schema comparison**: Which columns are common across FB/IG/TK?
- **Metric comparison**: Normalize and compare engagement metrics across platforms
- **User behavior comparison**: How do users interact differently on each platform?
- **Content patterns**: Compare posting frequency and engagement by platform
- **Temporal patterns**: Compare timing of posts/comments across platforms
- **Visualization**: Side-by-side comparisons of key metrics

### 11. **Summary Report**
- Key findings and data quality issues identified by platform and cross-platform
- Risk assessment (critical vs minor issues)
- Recommendations for data cleaning/preprocessing
- Platform-specific insights and differences

Use professional visualization styles with proper labels, titles, and legends. Include both visual and numeric outputs for clarity.
