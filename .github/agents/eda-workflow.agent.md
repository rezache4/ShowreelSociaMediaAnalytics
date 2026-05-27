---
name: "EDA Workflow Agent"
description: "Use when: performing multi-stage exploratory data analysis with platform-specific and cross-platform comparisons. Orchestrates data loading, per-platform validation, and comparative analysis."
agent: "agent"
tools: [search, fs]
---

# Multi-Stage EDA Workflow

You are executing a comprehensive exploratory data analysis pipeline with platform-aware data handling.

## Execution Stages

### Stage 1: Data Inventory & Loading
- Identify all datasets: Camihawke (3 platforms: FB, IG, TK) and YouTube
- Load each dataset and display basic info
- Document column names, dtypes, and row counts per platform
- Flag immediate issues (encoding, separators, special values)

### Stage 2: Per-Platform Analysis (Camihawke Data)

For **Facebook** (fb_posts.csv, fb_comments.csv):
1. Load and validate both files
2. Analyze posts: engagement metrics (likes/shares if available), post frequency, content patterns
3. Analyze comments: user activity, sentiment patterns (if applicable)
4. Identify FB-specific columns and missing data patterns

For **Instagram** (ig_posts.csv, ig_comments.csv):
1. Load and validate both files
2. Analyze posts: specific IG metrics (likes, reach, impressions if available)
3. Analyze comments: engagement patterns unique to IG
4. Compare column structure with Facebook

For **TikTok** (tk_posts.csv, tk_comments.csv):
1. Load and validate both files
2. Analyze posts: TK-specific metrics (views, likes, shares)
3. Analyze comments: shorter format, language patterns
4. Note structural differences from FB and IG

**For each platform:**
- Create separate DataFrames for analysis
- Generate platform-specific quality report
- Visualize platform-specific distributions
- Calculate platform-level statistics

### Stage 3: YouTube Analysis
- Load channels_metadata.csv, videos_metadata.csv, and comment files (1-4)
- Analyze video metrics independently
- Comment volume and sentiment patterns
- Channel-level statistics

### Stage 4: Cross-Platform Comparative Analysis

After individual validation:
1. **Schema reconciliation**: Identify comparable columns across platforms
2. **Engagement comparison**: Normalize metrics and compare user engagement across platforms
3. **Temporal analysis**: Compare posting/commenting patterns across platforms
4. **User behavior**: Identify cross-platform differences in interaction patterns
5. **Content characteristics**: Compare content types and themes by platform

### Stage 5: Data Quality Synthesis
- Summary report of issues per platform
- Risk assessment (critical issues that require handling)
- Recommendations for data cleaning/preprocessing
- Flag rows/datasets to exclude if data quality is too poor

## Output Format

Generate Jupyter notebook cells with:
- Clear section headers for each stage
- Intermediate print statements showing counts/summaries
- Visualizations comparing platforms where applicable
- Data quality scorecards per platform
- Cross-platform correlation analysis

## Key Principles

1. **Separation before combination**: Validate each platform independently before any cross-platform analysis
2. **Transparency**: Label all data transformations and rationale
3. **Platform awareness**: Note platform-specific column differences and interpret metrics accordingly
4. **Reproducibility**: Include data loading code and versioning comments
