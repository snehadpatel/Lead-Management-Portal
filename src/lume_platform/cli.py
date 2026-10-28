"""
Lume AI Unified Command-Line Interface (CLI).

Usage:
  python -m lume_platform.cli train --engine [all|lstm|leads|clustering|sentiment]
  python -m lume_platform.cli forecast --scheme 100119
  python -m lume_platform.cli eval
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Lume AI Platform CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train AI/DL models")
    train_parser.add_argument(
        "--engine",
        choices=["all", "bilstm", "lstm", "leads", "clustering", "sentiment"],
        default="all",
        help="Which model engine to train (default: all)",
    )
    train_parser.add_argument(
        "--scheme",
        default="100119",
        help="Scheme code for time-series forecasting (default: 100119)",
    )

    # Forecast command
    forecast_parser = subparsers.add_parser("forecast", help="Run forecast inference for a scheme")
    forecast_parser.add_argument(
        "--scheme",
        default="100119",
        help="Scheme code to forecast",
    )
    forecast_parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of historical lookback days to input",
    )

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Show current model evaluation metrics")

    args = parser.parse_args()

    if args.command == "train":
        if args.engine == "all":
            from lume_platform.ml.training import train_all
            print("🚀 Training all Lume AI models (including PyTorch LSTM)...")
            results = train_all(include_lstm=True)
            print("\nSaved artifacts:")
            print(json.dumps(results, indent=2))
        elif args.engine == "bilstm":
            from lume_platform.ml.phase2_trainer import train_phase2_engine
            print(f"🚀 Training Phase 2 BiLSTM+Attention Forecaster for scheme {args.scheme}...")
            res = train_phase2_engine(primary_scheme=args.scheme)
            print("\nFull Model Metrics:")
            print(json.dumps(res["full_model"], indent=2))
        elif args.engine == "lstm":
            from lume_platform.ml.lstm_trainer import train_lstm_forecaster
            print(f"🚀 Training PyTorch LSTM Forecaster for scheme {args.scheme}...")
            res = train_lstm_forecaster(scheme_code=args.scheme)
            print("\nMetrics:")
            print(json.dumps(res["metrics"], indent=2))
        elif args.engine == "leads":
            from lume_platform.ml.training import train_lead_classifier
            p = train_lead_classifier()
            print(f"✅ Lead classifier trained: {p}")
        elif args.engine == "clustering":
            from lume_platform.ml.training import train_investor_kmeans_and_dbscan
            p = train_investor_kmeans_and_dbscan()
            print(f"✅ Investor clustering trained: {p}")
        elif args.engine == "sentiment":
            from lume_platform.ml.training import train_sentiment_nlp
            p = train_sentiment_nlp()
            print(f"✅ Sentiment model trained: {p}")

    elif args.command == "forecast":
        from lume_platform.data.timeseries_dataset import load_nav_history
        from lume_platform.ml.forecaster import LUMEForecaster

        forecaster = LUMEForecaster()
        df = load_nav_history(args.scheme)
        history = df["nav"].tail(args.days).tolist()
        result = forecaster.forecast(history)
        print(f"\n🔮 5-Day Forecast for Scheme {args.scheme} (Source: {result.get('model_source')}):")
        print(f"   Last NAV: ₹{result.get('historical_last'):.2f}")
        if result.get("p10") and result.get("p50") and result.get("p90"):
            print(f"   P10 (Lower Quantile): {result.get('p10')}")
            print(f"   P50 (Median Forecast): {result.get('p50')}")
            print(f"   P90 (Upper Quantile): {result.get('p90')}")
        else:
            print(f"   Predicted Trajectory: {result.get('forecast_trajectory')}")
            print(f"   90% Confidence Lower: {result.get('lower_bound')}")
            print(f"   90% Confidence Upper: {result.get('upper_bound')}")
        print(f"   Naive Baseline:       {result.get('baseline_trajectory')}")
        if result.get("attention_weights"):
            attn = result["attention_weights"]
            top_step = int(np.argmax(attn))
            print(f"   Temporal Attention:   Peak weight ({max(attn):.4f}) at step t-{len(attn) - top_step}")
        if result.get("quantile_coverage"):
            print(f"   Quantile Calibration: {result.get('quantile_coverage')}")
        print(f"   Trend: {result.get('trend')} | Confidence: {result.get('confidence_score')}%")
        print(f"   Metric: {result.get('accuracy_metric')}")
        print(f"   Disclaimer: {result.get('disclaimer')}")

    elif args.command == "eval":
        from lume_platform.config import EVAL_DIR
        print("📋 Lume AI Model Evaluation Cards:")
        for metric_file in EVAL_DIR.glob("**/*.json"):
            try:
                with open(metric_file) as f:
                    data = json.load(f)
                rel = metric_file.relative_to(EVAL_DIR)
                print(f"\n--- {rel} ---")
                print(json.dumps(data, indent=2))
            except Exception:
                pass
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
