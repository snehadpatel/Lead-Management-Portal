# Tableau Public Integration Guide

## Step 1: Download Export Files
The following CSV files are ready for import into Tableau Public:

1. **leads_for_bi.csv** — Lead scoring data with engagement metrics
2. **investor_clusters_for_bi.csv** — Investor behavior and cluster assignments
3. **sentiment_for_bi.csv** — Social sentiment data
4. **summary_metrics.csv** — KPI metrics for dashboard

## Step 2: Import to Tableau Public
1. Go to [Tableau Public](https://public.tableau.com/)
2. Create a new workbook
3. Connect to Text File → Select the CSV files
4. Build visualizations using the imported data

## Step 3: Recommended Visualizations

### Lead Scoring Dashboard
- **Conversion Probability Distribution** (Histogram)
- **Engagement Tier Breakdown** (Pie Chart)
- **Lead Source Performance** (Bar Chart)
- **Conversion by Engagement Score** (Scatter Plot)

### Investor Clustering Dashboard
- **Cluster Distribution** (Pie Chart)
- **Persona Radar Chart** (Multiple dimensions)
- **Behavior Heatmap** (Correlation matrix)

### Sentiment Analysis Dashboard
- **Sentiment Distribution** (Donut Chart)
- **Sentiment Over Time** (Line Chart, if timestamp available)

## Step 4: Publish & Embed
1. Publish workbook to Tableau Public
2. Get embed URL from Share → Embed Code
3. Set environment variable:
   ```bash
   export LUME_TABLEAU_EMBED_URL="your_embed_url"
   ```
4. Restart Streamlit app to see embedded dashboard

## Data Dictionary

### leads_for_bi.csv
| Column | Description |
|--------|-------------|
| totalvisits | Number of website visits |
| total_time_spent_on_website | Time in seconds |
| page_views_per_visit | Average pages per visit |
| engagement_score | Calculated engagement metric |
| engagement_tier | Low/Medium/High |
| conversion_probability | ML model prediction (0-1) |
| predicted_conversion | 0 or 1 |

### investor_clusters_for_bi.csv
| Column | Description |
|--------|-------------|
| ProfManage | Preference for professional management (0-10) |
| Diversification | Diversification preference (0-10) |
| Affordability | Affordability importance (0-10) |
| Liquidity | Liquidity preference (0-10) |
| Growth | Growth preference (0-10) |
| Trustworthiness | Trust in financial institutions (0-10) |
| Technology | Tech-savviness (0-10) |
| cluster_id | Assigned cluster (0-3) |
| persona | Human-readable cluster label |

### sentiment_for_bi.csv
| Column | Description |
|--------|-------------|
| sentence/text | Input text |
| sentiment | True label |
| predicted_sentiment | ML model prediction |

## Tips for Effective Visualizations

1. **Use calculated fields** for custom metrics
2. **Create filters** for interactive exploration
3. **Add tooltips** for detailed information on hover
4. **Use color consistently** across dashboards
5. **Add reference lines** for thresholds (e.g., 0.85 for hot leads)

## Troubleshooting

**File too large for Tableau Public?**
- Tableau Public has a 10M row limit
- Sample data using the provided CSVs (already sampled)
- Or use Tableau Desktop with Tableau Server

**Data not refreshing?**
- Re-export from this script after model retraining
- Re-import CSV in Tableau Public
- Republish the workbook
