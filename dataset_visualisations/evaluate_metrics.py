import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io

plt.style.use('dark_background')
sns.set_palette("husl")

DATA_DIR = '../datasets/'
OUT_DIR = './'

print(f"Starting Metrics Evaluation for Lume AI Datasets in: {DATA_DIR}")

# 1. Lead Scoring Target Imbalance
print("Evaluating Lead Scoring Metrics...")
try:
    lead_df = pd.read_csv(f"{DATA_DIR}structured/leads/lead_scoring/Lead Scoring.csv")
    converted = lead_df['Converted'].value_counts()
    plt.figure(figsize=(8, 6))
    converted.plot(kind='pie', labels=['Not Converted (0)', 'Converted (1)'], autopct='%1.1f%%', colors=['#ff6b6b', '#4ecdc4'])
    plt.title('Lead Conversion Distribution (Target Imbalance)')
    plt.ylabel('')
    plt.savefig(os.path.join(OUT_DIR, 'lead_conversion_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Baseline Conversion Rate: {converted[1] / lead_df.shape[0] * 100:.2f}%\n")
except Exception as e:
    print(f"Failed to load Lead Scoring: {e}")

# 2. MF Investor Behavior Heatmap
print("Evaluating Investor Behavior Metrics...")
try:
    mf_df = pd.read_excel(f"{DATA_DIR}structured/leads/mf_investor_behavior/MF_Behavior.xlsx")
    behavior_cols = ['ProfManage', 'Diversification', 'Affordability', 'Liquidity', 'Growth', 'Trustworthiness', 'Technology']
    corr_matrix = mf_df[behavior_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Investor Behavioral Trait Correlation')
    plt.savefig(os.path.join(OUT_DIR, 'investor_behavior_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"Failed to load MF Behavior: {e}")

# 3. Macro NIFTY Trend
print("Evaluating NIFTY 50 Macro Metrics...")
try:
    nifty_df = pd.read_csv(f"{DATA_DIR}structured/stock_prices/nifty50_index/nse_nifty50_historical_merged.csv")
    nifty_df['Date'] = pd.to_datetime(nifty_df['Date'])
    nifty_df.sort_values('Date', inplace=True)
    plt.figure(figsize=(12, 6))
    plt.plot(nifty_df['Date'], nifty_df['Close'], color='#feca57', linewidth=2)
    plt.title('NIFTY 50 Index Growth')
    plt.xlabel('Date')
    plt.ylabel('Closing Price (INR)')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(OUT_DIR, 'nifty50_macro_trend.png'), dpi=300, bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"Failed to load NIFTY 50 Data: {e}")

with open(os.path.join(OUT_DIR, 'METRICS_EVALUATION_REPORT.md'), 'w') as f:
    f.write("# Dataset Metrics Evaluation Report\n\nGenerated evaluation matrices and plots covering the class imbalance ratio in our Leads dataset, trait correlations from Investor K-Means features, and a macro growth indicator for time-series LSTMs.")

print("All metrics generated successfully.")
