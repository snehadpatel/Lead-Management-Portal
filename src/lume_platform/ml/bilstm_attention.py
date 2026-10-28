"""
Bidirectional LSTM with Temporal Multihead Attention for Multi-Quantile Forecasting.

Architectural Design:
1. Bidirectional LSTM encoding over past sequence (lookback=30).
2. Temporal Multihead Attention over hidden states across the time axis.
   Weights reveal which past market dates/shocks dominate the forecast.
3. Residual connection + LayerNorm.
4. Non-crossing Monotonic Quantile Output Head:
   P50 = median_head(h)
   P10 = P50 - softplus(delta_10(h))
   P90 = P50 + softplus(delta_90(h))
   Guarantees P10 <= P50 <= P90 by construction.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMAttentionQuantileForecaster(nn.Module):
    """
    Multivariate BiLSTM with Multihead Temporal Attention and Quantile Heads.
    Supports ablations:
      - use_attention=False (BiLSTM only)
      - bidirectional=False (Unidirectional LSTM)
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        horizon: int = 5,
        dropout: float = 0.2,
        bidirectional: bool = True,
        use_attention: bool = True,
        quantiles: Sequence[float] = (0.10, 0.50, 0.90),
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.horizon = horizon
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.quantiles = list(quantiles)

        num_dirs = 2 if bidirectional else 1
        lstm_out_dim = hidden_dim * num_dirs
        self.lstm_out_dim = lstm_out_dim

        # 1. Recurrent Encoder
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 2. Temporal Multihead Attention Layer
        if self.use_attention:
            # Ensure embed_dim is divisible by num_heads
            if lstm_out_dim % num_heads != 0:
                raise ValueError(
                    f"lstm_out_dim ({lstm_out_dim}) must be divisible by num_heads ({num_heads})"
                )
            self.attention = nn.MultiheadAttention(
                embed_dim=lstm_out_dim,
                num_heads=num_heads,
                batch_first=True,
                dropout=dropout,
            )
            self.layer_norm = nn.LayerNorm(lstm_out_dim)
        else:
            self.attention = None
            self.layer_norm = None

        self.dropout = nn.Dropout(dropout)

        # 3. Dense representation head
        self.fc_dense = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 4. Non-crossing Quantile Output Heads (P50, delta_P10, delta_P90)
        # Median (P50) prediction
        self.head_p50 = nn.Linear(hidden_dim, horizon)

        # Lower bound offset (P50 - P10 >= 0)
        self.head_delta_p10 = nn.Linear(hidden_dim, horizon)

        # Upper bound offset (P90 - P50 >= 0)
        self.head_delta_p90 = nn.Linear(hidden_dim, horizon)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Tensor of shape (batch, seq_len, input_dim)
            return_attention: If True, returns attention weights along with predictions.

        Returns:
            quantiles: Tensor of shape (batch, horizon, 3) where dim -1 corresponds to [P10, P50, P90]
            attn_weights: Tensor of shape (batch, seq_len) if use_attention and return_attention, else None.
        """
        batch_size, seq_len, _ = x.shape

        # LSTM encoding: (batch, seq_len, lstm_out_dim)
        lstm_out, _ = self.lstm(x)

        attn_weights = None
        if self.use_attention:
            # Temporal Attention: Q=lstm_out, K=lstm_out, V=lstm_out
            # average_attn_weights=True yields (batch, seq_len, seq_len)
            attn_out, raw_weights = self.attention(
                lstm_out, lstm_out, lstm_out, average_attn_weights=True
            )
            # Residual connection + norm
            h = self.layer_norm(lstm_out + attn_out)

            # Pooling over time axis: use last step query or mean attention over targets
            # Context representation: focus on last timestep query representation
            context = h[:, -1, :]  # (batch, lstm_out_dim)

            # Attention weights assigned to each historical timestep (by the final decision step)
            if return_attention:
                # raw_weights shape: (batch, seq_len, seq_len)
                # Take weights of final step looking back at all past steps: (batch, seq_len)
                attn_weights = raw_weights[:, -1, :]
        else:
            # No attention: take final hidden step
            context = lstm_out[:, -1, :]
            if return_attention:
                # Uniform weights fallback
                attn_weights = torch.ones(batch_size, seq_len, device=x.device) / seq_len

        feat = self.fc_dense(self.dropout(context))  # (batch, hidden_dim)

        # Monotonic non-crossing quantile heads
        p50 = self.head_p50(feat)  # (batch, horizon)
        delta_p10 = F.softplus(self.head_delta_p10(feat))  # (batch, horizon) >= 0
        delta_p90 = F.softplus(self.head_delta_p90(feat))  # (batch, horizon) >= 0

        p10 = p50 - delta_p10
        p90 = p50 + delta_p90

        # Stack into (batch, horizon, 3) where [0]=P10, [1]=P50, [2]=P90
        quantiles = torch.stack([p10, p50, p90], dim=-1)

        return quantiles, attn_weights
