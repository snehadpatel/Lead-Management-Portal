import pandas as pd
import numpy as np
import os
import random

# For deterministic mock conversions
np.random.seed(42)

class TableauExportEngine:
    def __init__(self):
        self.data_dir = '../datasets/structured/'
        self.out_dir = '../tableau_exports/'
        
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
            print(f"Created export directory: {self.out_dir}")

    def generate_distributor_view(self):
        """
        Creates the 'Fund Distributor Dashboard' flat wide-table CSV.
        Fuses the Leads table with mock XGBoost conversion probabilities and mapped recommended funds.
        """
        print("Generating Distributor Dashboard CSV...")
        try:
            leads_df = pd.read_csv(f"{self.data_dir}leads/lead_scoring/Lead Scoring.csv")
            # We use a manageable sample if it's too huge, but Tableau handles millions easily. 
            # We'll export the full thing with AI synthetic columns added
            
            # --- AI SYNTHETIC GENERATION (Mocking XGBoost outputs for Demo) ---
            # Create a mock Probability score (higher for actually Converted ones)
            leads_df['AI_Conversion_Probability'] = np.where(
                leads_df['Converted'] == 1, 
                np.random.uniform(0.70, 0.99, size=len(leads_df)),
                np.random.uniform(0.10, 0.45, size=len(leads_df))
            )
            
            # Tier classification based on Probability
            conditions = [
                (leads_df['AI_Conversion_Probability'] >= 0.8),
                (leads_df['AI_Conversion_Probability'] >= 0.5),
                (leads_df['AI_Conversion_Probability'] < 0.5)
            ]
            choices = ['Tier 1: Hot', 'Tier 2: Nurture', 'Tier 3: Cold']
            leads_df['Lead_Tier'] = np.select(conditions, choices, default='Tier 3: Cold')
            
            # Assigning a Recommended Fund Category (Equity for high time on site, Debt for low)
            leads_df['Recommended_Asset_Class'] = np.where(
                leads_df['Total Time Spent on Website'] > 1000, 
                'Equity Scheme', 
                'Debt Scheme'
            )
            
            # Drop heavy useless columns
            cols_to_keep = ['Prospect ID', 'Lead Origin', 'Lead Source', 'Total Time Spent on Website', 'Last Activity', 'Converted', 'AI_Conversion_Probability', 'Lead_Tier', 'Recommended_Asset_Class']
            tableau_df = leads_df[cols_to_keep].copy()
            
            export_path = os.path.join(self.out_dir, 'distributor_dashboard.csv')
            tableau_df.to_csv(export_path, index=False)
            print(f"✅ Successfully exported Distributor view [{tableau_df.shape[0]} rows] to: {export_path}")
            
        except Exception as e:
            print(f"Error generating Distributor view: {e}")

    def generate_investor_view(self):
        """
        Creates the 'Investor Persona Dashboard' CSV.
        Fuses Investor Behavioral data with K-Means clustering labels.
        """
        print("Generating Investor Dashboard CSV...")
        try:
            investor_df = pd.read_excel(f"{self.data_dir}leads/mf_investor_behavior/MF_Behavior.xlsx")
            
            # --- AI SYNTHETIC GENERATION (Mocking K-Means outputs for Demo) ---
            # We use a deterministic logic rule set that mimics our actual clusters
            def assign_persona(row):
                if row['Growth'] > 5 and row['Risk'] if 'Risk' in row else row['Growth'] > 5:
                    return "Cluster 0: Aggressive Growth"
                elif row['Liquidity'] > 5:
                    return "Cluster 1: Liquid Needers"
                elif row['Trustworthiness'] > 6:
                    return "Cluster 2: Brand Loyalists"
                else:
                    return "Cluster 3: Conservative Preservers"
                    
            investor_df['KMeans_Persona_Cluster'] = investor_df.apply(assign_persona, axis=1)
            
            # Map Persona to Risk Tolerance
            risk_map = {
                "Cluster 0: Aggressive Growth": "High Risk",
                "Cluster 1: Liquid Needers": "Low Risk",
                "Cluster 2: Brand Loyalists": "Medium Risk",
                "Cluster 3: Conservative Preservers": "Low Risk"
            }
            investor_df['Risk_Tolerance'] = investor_df['KMeans_Persona_Cluster'].map(risk_map)
            
            export_path = os.path.join(self.out_dir, 'investor_dashboard.csv')
            investor_df.to_csv(export_path, index=False)
            print(f"✅ Successfully exported Investor view [{investor_df.shape[0]} rows] to: {export_path}")
            
        except Exception as e:
            print(f"Error generating Investor view: {e}")

if __name__ == "__main__":
    engine = TableauExportEngine()
    engine.generate_distributor_view()
    engine.generate_investor_view()
    print("\nTableau Data Engineering Pipeline Complete.")
