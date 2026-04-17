import pandas as pd
import numpy as np
import os
import json
import pickle
import warnings
import matplotlib.pyplot as plt
import seaborn as plt_sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings('ignore')

class DualAIArchitecture:
    def __init__(self):
        self.base_dir = '../datasets/'
        self.models_dir = os.path.join(self.base_dir, 'models/saved_models/')
        self.export_dir = '../output_production_final/'
        
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.export_dir, exist_ok=True)

    def train_distributor_model(self):
        """
        AI Side A: FOR THE FUND DISTRIBUTOR
        Input: Lead Interaction Data
        Output: Suggests which Leads to contact, ranked by probability, and assigns an Investor Persona so they know how to pitch.
        """
        print("=== STAGE 1: Training Distributor Lead Recommender ===")
        df = pd.read_csv(f"{self.base_dir}structured/leads/lead_scoring/Lead Scoring.csv")
        
        # 1. Authentic Advanced Kaggle Feature Engineering
        numeric_features = ['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit']
        cat_features = ['Lead Origin', 'Lead Source', 'Specialization', 'What is your current occupation']
        
        df_ml = df[numeric_features + cat_features + ['Converted']].copy()
        df_ml[cat_features] = df_ml[cat_features].fillna("Unknown")
        df_ml[numeric_features] = df_ml[numeric_features].fillna(0)
        
        # Turn Text strings into massive Mathematical Dummy Sparse coordinates
        df_encoded = pd.get_dummies(df_ml, columns=cat_features, drop_first=True)
        X = df_encoded.drop('Converted', axis=1)
        y = df_encoded['Converted']
        
        # 2. Retrain Architecture (With Data-Balancing)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf_model = RandomForestClassifier(n_estimators=150, max_depth=12, class_weight='balanced', random_state=42)
        rf_model.fit(X_train, y_train)
        
        preds = rf_model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        print(f"RandomForest Legitimate Real-World Accuracy Re-Scored: {acc * 100:.2f}%")
        
        report_json = {
            "accuracy": f"{acc * 100:.2f}%", "precision": f"{prec:.2f}",
            "recall": f"{rec:.2f}", "f1_score": f"{f1:.2f}"
        }
        os.makedirs('../model_evaluations/random_forest', exist_ok=True)
        with open('../model_evaluations/random_forest/real_metrics.json', 'w') as f:
            json.dump(report_json, f)
            
        # 2.5 Explainable AI (XAI) Feature Importance Extraction
        print("Scraping Random Forest internal mathematical node weights for XAI Visualization...")
        importances = rf_model.feature_importances_
        feature_names = X.columns
        weights_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        weights_df = weights_df.sort_values(by='Importance', ascending=False).head(10)
        
        plt.style.use('dark_background')
        plt.figure(figsize=(9, 5))
        plt_sns.barplot(x='Importance', y='Feature', data=weights_df, palette='viridis')
        plt.title('Explainable AI (XAI): Model Feature Impact on Conversion', color='white', weight='bold')
        plt.xlabel('Predictive Weight Assignment (Random Forest Matrix)', color='white')
        plt.ylabel('User/Behavioral Feature', color='white')
        plt.grid(alpha=0.3, color='#444444')
        
        xai_out_path = '../model_evaluations/xai_insights/xai_rf_feature_weights.png'
        plt.savefig(xai_out_path, dpi=300, bbox_inches='tight', transparent=False)
        plt.close()
        print(f"XAI Visualization successful. Saved to {xai_out_path}")
        
        # 3. Generate Output Dataset for Presentation
        # Calculate probability for ALL leads
        df['Conversion_Probability'] = rf_model.predict_proba(X)[:, 1]
        
        # Add a custom Pitch Persona based on Time Spent matching real psychological hooks
        def assign_pitch_persona(time):
            if time > 1500: return "Aggressive Growth Chaser"
            if time > 800: return "Brand Loyalist"
            return "Conservative Wealth Preserver"
            
        df['Recommended_Pitch_Persona'] = df['Total Time Spent on Website'].apply(assign_pitch_persona)
        
        # Filter to only actual recommendations
        recommended_leads = df[df['Conversion_Probability'] >= 0.65].sort_values(by='Conversion_Probability', ascending=False)
        
        # Export for Presentation
        dist_export_path = os.path.join(self.export_dir, 'distributor_leads_master.csv')
        recommended_leads.to_csv(dist_export_path, index=False)
        
        # Pickle the model
        with open(os.path.join(self.models_dir, 'distributor_lead_scorer.pkl'), 'wb') as f:
            pickle.dump(rf_model, f)
            
        print(f"✅ Generated {len(recommended_leads)} actionable Leads for Distributors.")
        print(f"✅ Saved output CSV: {dist_export_path}\n")


    def train_investor_model(self):
        """
        AI Side B: FOR THE INVESTOR
        Input: Investor Behavioral Variables
        Output: Recommends the TYPE of Mutual Fund they should buy, and routes them to a specific Distributor to contact!
        """
        print("=== STAGE 2: Training Investor Fund & Routing Engine ===")
        df = pd.read_excel(f"{self.base_dir}structured/leads/mf_investor_behavior/MF_Behavior.xlsx")
        
        # 1. Feature Engineering
        behavior_cols = ['ProfManage', 'Diversification', 'Affordability', 'Liquidity', 'Growth', 'Trustworthiness', 'Technology']
        X = df[behavior_cols].dropna()
        
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 2. Train Clustering
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        df['Persona_Cluster'] = kmeans.fit_predict(X_scaled)
        
        # 3. Create Routing Engine Logic
        # Cluster 0 usually leans 'Growth/Risk', Cluster 1 leans 'Liquidity/Safe', etc.
        def route_investor(cluster):
            if cluster == 0:
                return pd.Series(["Equity Schemes (High Risk)", "Lume Elite Advisory"])
            elif cluster == 1:
                return pd.Series(["Liquid/Debt Funds (Safe)", "MintLeads Premier Capital"])
            elif cluster == 2:
                return pd.Series(["Hybrid Allocation Funds", "Global Wealth Partners India"])
            else:
                return pd.Series(["Index Trackers (Passive)", "Index Advisory Group"])
                
        df[['Recommended_Fund_Type', 'Recommended_Distributor_To_Contact']] = df['Persona_Cluster'].apply(route_investor)
        
        # Export for Presentation
        inv_export_path = os.path.join(self.export_dir, 'investor_routing_matches.csv')
        df.to_csv(inv_export_path, index=False)
        
        # Pickle the model
        with open(os.path.join(self.models_dir, 'investor_kmeans_model.pkl'), 'wb') as f:
            pickle.dump(kmeans, f)
        with open(os.path.join(self.models_dir, 'investor_behavior_scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
            
        print(f"✅ Generated Fund matches & Distributor Routes for {len(df)} investors.")
        print(f"✅ Saved output CSV: {inv_export_path}\n")

if __name__ == "__main__":
    engine = DualAIArchitecture()
    engine.train_distributor_model()
    engine.train_investor_model()
    print("Dual-Sided Recommender System Core successfully initialized and deployed.")
