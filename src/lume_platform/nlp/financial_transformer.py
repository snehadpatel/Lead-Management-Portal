"""Financial Domain Transformer with Multi-Head Self-Attention and Token Saliency."""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LABEL_NAMES = ["negative", "neutral", "positive"]
LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}
ID_TO_LABEL = {0: "negative", 1: "neutral", 2: "positive"}


class FinancialVocabTokenizer:
    """Lightweight, regex-based tokenizer preserving financial tokens ($%, decimals, tickers)."""

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    CLS_TOKEN = "<s>"
    SEP_TOKEN = "</s>"

    def __init__(self, max_vocab_size: int = 8000, max_len: int = 64) -> None:
        self.max_vocab_size = max_vocab_size
        self.max_len = max_len
        self.word2idx: Dict[str, int] = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.CLS_TOKEN: 2,
            self.SEP_TOKEN: 3,
        }
        self.idx2word: Dict[int, str] = {idx: word for word, idx in self.word2idx.items()}
        self.word_counts: Dict[str, int] = {}

    @staticmethod
    def _tokenize_text(text: str) -> List[str]:
        # Preserve $1.50, 15%, numbers, and words
        tokens = re.findall(r"\$?\d+(?:\.\d+)?%?|\w+|[^\w\s]", text.lower())
        return tokens

    def fit(self, texts: List[str]) -> "FinancialVocabTokenizer":
        counts: Dict[str, int] = {}
        for text in texts:
            for token in self._tokenize_text(text):
                counts[token] = counts.get(token, 0) + 1
        self.word_counts = counts

        # Sort by frequency
        sorted_tokens = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        for token, _ in sorted_tokens:
            if len(self.word2idx) >= self.max_vocab_size:
                break
            if token not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[token] = idx
                self.idx2word[idx] = token
        return self

    def encode(self, text: str) -> Tuple[List[int], List[str]]:
        raw_tokens = self._tokenize_text(text)
        tokens = [self.CLS_TOKEN] + raw_tokens[: self.max_len - 2] + [self.SEP_TOKEN]
        input_ids = [self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in tokens]
        # Pad to max_len
        padded_ids = input_ids + [0] * (self.max_len - len(input_ids))
        padded_tokens = tokens + [self.PAD_TOKEN] * (self.max_len - len(tokens))
        return padded_ids, padded_tokens

    def save(self, filepath: str) -> None:
        data = {
            "max_vocab_size": self.max_vocab_size,
            "max_len": self.max_len,
            "word2idx": self.word2idx,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "FinancialVocabTokenizer":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(max_vocab_size=data["max_vocab_size"], max_len=data["max_len"])
        tok.word2idx = data["word2idx"]
        tok.idx2word = {int(idx): word for word, idx in tok.word2idx.items()}
        return tok


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (batch_size, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class FinancialTransformerClassifier(nn.Module):
    """Multi-Head Self-Attention Transformer with explicit Attention-Pooling for Token Saliency."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        num_classes: int = 3,
        max_len: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Attention Pooling head for explainability (Token Saliency)
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.Tanh(),
            nn.Linear(64, 1, bias=False),
        )

        # Classification MLP
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(
        self, input_ids: torch.Tensor, return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # input_ids: (batch_size, seq_len)
        padding_mask = input_ids == 0  # (batch_size, seq_len) True for pad tokens

        embeddings = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoder(embeddings)

        # Pass through Transformer encoder blocks
        encoded = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        # Attention pooling: compute attention weight for each token
        attn_scores = self.attention_pool(encoded).squeeze(-1)  # (batch_size, seq_len)
        attn_scores = attn_scores.masked_fill(padding_mask, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch_size, seq_len)

        # Weighted sum across sequence dimension
        pooled = torch.bmm(attn_weights.unsqueeze(1), encoded).squeeze(1)  # (batch_size, d_model)

        # Class logits
        logits = self.classifier(pooled)

        if return_attention:
            return logits, attn_weights
        return logits, None

