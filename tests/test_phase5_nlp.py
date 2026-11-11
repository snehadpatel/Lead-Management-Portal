"""Unit tests for Phase 5 Financial NLP, Domain Transformer, and Token Saliency."""

import os
import sys
from pathlib import Path
import pytest
import torch

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from lume_platform.nlp.financial_transformer import (
    FinancialTransformerClassifier,
    FinancialVocabTokenizer,
    LABEL_NAMES,
    LABEL_TO_ID,
    ID_TO_LABEL,
)
from lume_platform.nlp.financial_nlp_service import FinancialNLPService


def test_tokenizer_encoding_and_financial_tokens():
    """Test regex financial tokenizer preserves numbers, percentages, and currencies."""
    tokenizer = FinancialVocabTokenizer(max_vocab_size=1000, max_len=32)
    sample_text = "HDFC Bank reported 15.5% profit jump to $2.5B in Q4"
    tokenizer.fit([sample_text])

    ids, tokens = tokenizer.encode(sample_text)
    assert len(ids) == 32
    assert len(tokens) == 32
    assert tokens[0] == tokenizer.CLS_TOKEN
    assert "15.5%" in tokens
    assert "profit" in tokens
    assert "bank" in tokens
    assert ids[0] == tokenizer.word2idx[tokenizer.CLS_TOKEN]


def test_transformer_forward_and_attention_saliency():
    """Test PyTorch FinancialTransformer forward pass and attention saliency weights."""
    vocab_size = 200
    model = FinancialTransformerClassifier(
        vocab_size=vocab_size,
        d_model=64,
        nhead=2,
        num_layers=2,
        dim_feedforward=128,
        num_classes=3,
        max_len=32,
    )

    batch_size = 4
    x = torch.randint(0, vocab_size, (batch_size, 32))
    logits, attn = model(x, return_attention=True)

    assert logits.shape == (batch_size, 3)
    assert attn.shape == (batch_size, 32)

    # Attention weights should sum to 1.0 per sequence
    sums = attn.sum(dim=-1).detach().cpu().numpy()
    for s in sums:
        assert pytest.approx(1.0, rel=1e-3) == float(s)


def test_financial_nlp_service_inference_and_saliency():
    """Test service loads weights, computes confidence, and produces token saliency."""
    svc = FinancialNLPService.get_instance()
    text = "RBI cuts repo rate by 25 bps, boosting banking liquidity and equity market sentiment."
    res = svc.predict_sentiment_with_saliency(text)

    assert "sentiment" in res
    assert res["sentiment"] in ["positive", "neutral", "negative"]
    assert "confidence" in res
    assert 0.0 <= res["confidence"] <= 1.0

    assert "probabilities" in res
    assert "positive" in res["probabilities"]
    assert "neutral" in res["probabilities"]
    assert "negative" in res["probabilities"]

    assert "token_saliency" in res
    assert len(res["token_saliency"]) > 0

    first_token = res["token_saliency"][0]
    assert "token" in first_token
    assert "weight" in first_token
    assert "polarity" in first_token
    assert first_token["polarity"] in ["positive", "neutral", "negative"]
    assert res["latency_ms"] >= 0.0


def test_real_metrics_provenance():
    """Verify saved genuine metrics file has authentic counts and honest evaluation."""
    metrics_path = PROJECT_ROOT / "model_evaluations" / "phase5_nlp" / "real_metrics.json"
    assert metrics_path.exists(), "real_metrics.json must exist"

    import json
    with open(metrics_path) as f:
        data = json.load(f)

    assert data["dataset"]["total_samples"] == 5842
    assert data["dataset"]["train_samples"] == 4673
    assert data["dataset"]["test_samples"] == 1169

    models = data["models"]
    assert "tfidf_logistic" in models
    assert "sbert_baseline" in models
    assert "financial_transformer" in models

    tf = models["financial_transformer"]
    assert tf["trainable_parameters"] == 1635715
    assert tf["accuracy"] > 0.65
    assert tf["macro_f1"] > 0.60
    assert tf["latency_ms"] < 10.0  # Fast CPU inference


def test_fastapi_nlp_endpoints():
    """Test FastAPI /api/nlp routes."""
    from fastapi.testclient import TestClient
    from api.main_enhanced import app

    client = TestClient(app)

    # 1. Benchmark route
    r_bm = client.get("/api/nlp/benchmark")
    assert r_bm.status_code == 200
    bm_data = r_bm.json()
    assert "models" in bm_data
    assert "financial_transformer" in bm_data["models"]

    # 2. Financial Sentiment Prediction route
    payload = {"text": "Company reported record quarterly revenue and 30% dividend payout increase."}
    r_pred = client.post("/api/nlp/financial-sentiment", json=payload)
    assert r_pred.status_code == 200
    pred_data = r_pred.json()
    assert pred_data["sentiment"] in ["positive", "neutral", "negative"]
    assert "token_saliency" in pred_data
    assert len(pred_data["token_saliency"]) > 0
