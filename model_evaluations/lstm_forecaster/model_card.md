# Model Card: PyTorch Multi-Horizon LSTM Forecaster

## 1. Overview
- **Model Architecture**: Multi-Layer LSTM with Monte Carlo Dropout (2 layers, 64 hidden units, dropout=0.2)
- **Task**: 5-Day Multi-Step NAV Forecast (Lookback = 30 days, Horizon = 5 days)
- **Primary Asset**: Mutual Fund Scheme `100119` (Merged with NIFTY 50 trading calendar)

## 2. Zero-Data-Leakage Design
- **Splits**: Strictly chronological
  - Train: 2023-02-21 to 2025-03-27 (514 trading days)
  - Validation: 2025-03-28 to 2025-09-08 (110 trading days)
  - Test: 2025-09-09 to 2026-02-20 (111 trading days)
- **Scaling**: `MinMaxScaler` fitted solely on the Train split. Out-of-sample evaluation strictly on unseen Test split.

## 3. Real Performance Benchmarks (Unseen Test Holdout)
| Model | RMSE (₹) | MAE (₹) | MAPE (%) | Directional Accuracy (%) |
|---|---|---|---|---|
| **PyTorch LSTM** | **₹10.91** | **₹9.83** | **1.84%** | **46.75%** |
| Naive Last-Value | ₹3.76 | ₹2.98 | 0.56% | 0.0% |
| 5-Day Moving Average | ₹4.13 | ₹3.33 | 0.63% | 61.04% |

## 4. Uncertainty & Limitations
- **Epistemic Uncertainty**: Quantified via $T=20$ Monte Carlo Dropout evaluations to produce empirical 90% confidence bands.
- **Product Honesty**: Not investment advice. Mutual fund NAV trajectories reflect market-wide drift and systematic index shocks.
