import pandas as pd
import numpy as np
import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

# Setup paths relative to script location
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DATASETS_DIR = ROOT_DIR / "datasets"
MF_HISTORY_DIR = DATASETS_DIR / "structured/mutual_funds/nav_history"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
SCALERS_DIR = ARTIFACTS_DIR / "ml_scalers"

# Ensure directories exist
for d in [MODELS_DIR, SCALERS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Core Neural Network Architecture for Multi-Step Time-Series Analysis
class NAVPredictorLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=3, output_size=5):
        super(NAVPredictorLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM Layer to understand temporal NAV patterns (memory over time)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        # Fully connected layer to map LSTM feature out to a 5-day trajectory
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        # Take the output from the final time step
        out = self.fc(out[:, -1, :])
        return out

def create_sequences(data, sequence_length=30, forecast_horizon=5):
    xs, ys = [], []
    for i in range(len(data) - sequence_length - forecast_horizon + 1):
        x = data[i:(i + sequence_length)]
        y = data[(i + sequence_length):(i + sequence_length + forecast_horizon)]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_lstm_pipeline():
    print("💎 Initiating Lume AI Deep Learning Pipeline for 5-Day Multi-Step Forecasting...")
    
    # 1. Load Data
    mf_files = glob.glob(str(MF_HISTORY_DIR / "*.csv"))
    if not mf_files:
        print(f"❌ No mutual fund data found in {MF_HISTORY_DIR}")
        return

    # Use a larger sample for "all funds" quality
    sample_files = mf_files[:50] 
    print(f"📂 Loading {len(sample_files)} historical files...")
    
    all_raw_navs = []
    for i, file in enumerate(sample_files):
        try:
            df = pd.read_csv(file)
            if 'nav' in df.columns:
                navs = pd.to_numeric(df['nav'], errors='coerce').dropna().values
                if len(navs) > 40:
                    all_raw_navs.append(navs.reshape(-1, 1))
            if i % 10 == 0:
                print(f"   Processed {i}/{len(sample_files)} files...")
        except Exception:
            pass

    if not all_raw_navs:
        print("❌ Not enough valid data found.")
        return

    # 2. Scaler Calibration
    print("⚖️ Calibrating Global MinMax Scaler across fund universe...")
    concatenated_navs = np.vstack(all_raw_navs)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(concatenated_navs)
    
    scaler_path = SCALERS_DIR / "mf_nav_global_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler saved to {scaler_path}")

    # 3. Sequence Generation
    LOOKBACK = 30
    HORIZON = 5
    all_x, all_y = [], []
    
    print(f"🕒 Generating windows (Lookback={LOOKBACK}, Horizon={HORIZON})...")
    for navs in all_raw_navs:
        navs_scaled = scaler.transform(navs)
        X, y = create_sequences(navs_scaled, sequence_length=LOOKBACK, forecast_horizon=HORIZON)
        if len(X) > 0:
            all_x.append(X)
            all_y.append(y)
            
    X_train = np.vstack(all_x)
    y_train = np.vstack(all_y).reshape(-1, HORIZON)
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    
    dataset = TensorDataset(X_train_t, y_train_t)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    # 4. Neural Network Training
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"🚀 Training on device: {device}")
    
    model = NAVPredictorLSTM(input_size=1, output_size=HORIZON).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    EPOCHS = 10 # increased for better accuracy
    print(f"🧬 Commencing Backpropagation for {EPOCHS} Epochs...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f" Epoch {epoch+1}/{EPOCHS} | Loss (MSE): {total_loss/len(dataloader):.6f}")
        
    # 5. Save Artifacts
    model_path = MODELS_DIR / "lstm_nav_pattern_predictor.pth"
    torch.save(model.state_dict(), model_path)
    print(f"✅ LSTM Core saved to {model_path}")

    # 6. Validation Insights
    model.eval()
    with torch.no_grad():
        test_idx = np.random.randint(0, len(X_train_t))
        test_seq = X_train_t[test_idx].unsqueeze(0).to(device)
        pred_scaled = model(test_seq).cpu().numpy()[0]
        actual_scaled = y_train_t[test_idx].numpy()
        
        pred_real = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        actual_real = scaler.inverse_transform(actual_scaled.reshape(-1, 1)).flatten()
        
        print("\n📈 Post-Training Validation (Sample):")
        print(f"Last Known NAV: ₹{scaler.inverse_transform(X_train_t[test_idx][-1].reshape(1, -1))[0][0]:.2f}")
        print(f"Predicted Trajectory: {['₹%.2f'%x for x in pred_real]}")
        print(f"Actual Trajectory:    {['₹%.2f'%x for x in actual_real]}")
        
        mse = np.mean((pred_real - actual_real)**2)
        print(f"Validation MSE: {mse:.4f}")

if __name__ == "__main__":
    train_lstm_pipeline()
