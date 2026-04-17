import wbgapi as wb
import pandas as pd
import os

# Set up directories
BASE_DIR = "/Users/snehapatel/BigData"
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
os.makedirs(DATASETS_DIR, exist_ok=True)

def fetch_macro_data():
    print("Fetching World Bank data for India (IND)...")
    
    # Indicators:
    # FP.CPI.TOTL.ZG = Inflation, consumer prices (annual %)
    # NY.GDP.MKTP.KD.ZG = GDP growth (annual %)
    # FR.INR.RINR = Real interest rate (%)
    
    indicators = ['FP.CPI.TOTL.ZG', 'NY.GDP.MKTP.KD.ZG', 'FR.INR.RINR']
    
    try:
        # Fetch data for India for the last 20 years
        df = wb.data.DataFrame(indicators, 'IND', time=range(2003, 2025))
        
        # Reset index to make the series a column, and rename indicators for clarity
        df = df.reset_index()
        
        # Melt the dataframe from wide to long format
        df_melted = pd.melt(df, id_vars=['series'], var_name='Year', value_name='Value')
        
        # Clean up the Year column (remove 'YR' prefix)
        df_melted['Year'] = df_melted['Year'].str.replace('YR', '').astype(int)
        
        # Pivot so each indicator is a column
        df_final = df_melted.pivot(index='Year', columns='series', values='Value').reset_index()
        
        # Rename columns to human-readable names
        df_final.rename(columns={
            'FP.CPI.TOTL.ZG': 'Inflation_Annual_Pct',
            'NY.GDP.MKTP.KD.ZG': 'GDP_Growth_Annual_Pct',
            'FR.INR.RINR': 'Real_Interest_Rate_Pct'
        }, inplace=True)
        
        out_file = os.path.join(DATASETS_DIR, "macro_indicators_wb.csv")
        df_final.to_csv(out_file, index=False)
        print(f"Successfully saved World Bank macro data to {out_file}")
        
    except Exception as e:
        print(f"Error fetching data from World Bank API: {e}")

if __name__ == "__main__":
    fetch_macro_data()
