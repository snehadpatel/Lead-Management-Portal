# Model Card: Multivariate BiLSTM with Temporal Multihead Attention

## Overview
- **Model Name**: BiLSTMAttentionQuantileForecaster
- **Version**: Phase 2 (Time-Series Engine)
- **Target**: 5-day multi-step forward NAV trajectory for AMFI mutual funds
- **Quantiles**: P10, P50 (median), P90 via multi-task Pinball Loss
- **Lookback Window**: 30 trading days

## Features
- **NAV Daily Return**: $(NAV_t - NAV_{t-1}) / NAV_{t-1}$
- **NIFTY 50 Benchmark Close & Daily Return**
- **NIFTY 50 10-Day Rolling Realized Volatility**
- **NIFTY 50 Traded Volume**
- **14-Day Relative Strength Index (RSI)** on NAV
- **20-Day SMA Deviation**: $(NAV_t - SMA_{20}) / SMA_{20}$
- **Financial Sentiment Flag** (default 0.0, stubbed for Phase 4)

## Architecture & Mathematical Formulation
- **Encoder**: 2-layer Bidirectional LSTM (hidden_dim=64, total=128 per step)
- **Attention**: `nn.MultiheadAttention` with 4 heads over time axis
- **Quantile Offsets**: Monotonic positive parameterization:
  $$P_{10} = P_{50} - \text{softplus}(\Delta_{10})$$
  $$P_{90} = P_{50} + \text{softplus}(\Delta_{90})$$
  Guarantees $P_{10} \le P_{50} \le P_{90}$ without quantile crossing.
- **Objective Function**: Pinball Loss across $q \in \{0.10, 0.50, 0.90\}$

## Evaluation Results (Unseen Holdout Test Set)
- **Full Model**:
  - Test RMSE: ₹10.5496
  - Test MAE: ₹9.6783
  - Directional Accuracy: 46.75%
  - Quantile Coverage Rate (QCR): 100.0% (Target: 80.0%)
  - Mean Interval Width: ₹42.07
- **Baselines Comparison**:
  - Naive Last-Value: RMSE ₹3.7565 | Coverage 91.69%
  - ARIMA(3, 1, 0): RMSE ₹3.8963 | Coverage 89.35%
  - Univariate Vanilla LSTM: RMSE ₹10.91 | Coverage 74.5%

## Limitations & Honesty
- In periods of macroeconomic regime shifts (e.g. monetary policy rate revisions), uncertainty intervals naturally widen.
- The model is an analytical decision support tool; forecasts carry intrinsic market risk.
- Disclaimer: Educational and analytical tool only. Not investment advice.
