"""Positional encodings: Sinusoidal (absolute, additive) and RoPE (rotary, applied to Q/K).

Sinusoidal is added to token embeddings before the first layer (C1, C3, C4, C5).
RoPE replaces it in C2: no absolute encoding is added; instead Q and K are rotated
inside every self-attention block. Cross-attention is left unrotated because query
and key positions live on different sequences and share no coordinate system.
"""

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Classic fixed sin/cos encoding from Vaswani et al. (2017), 'Attention Is
    All You Need' (NeurIPS).

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.1):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model) token embeddings; returns embeddings + PE."""
        if x.size(1) > self.pe.size(0):
            raise ValueError(
                f"sequence length {x.size(1)} exceeds max_len {self.pe.size(0)}"
            )
        return self.dropout(x + self.pe[: x.size(1)].to(dtype=x.dtype))


class RotaryPositionalEmbedding(nn.Module):
    """RoPE (Su et al., 2021), 'RoFormer: Enhanced Transformer with Rotary
    Position Embedding' (arXiv:2104.09864); half-split convention.

    The head dimension is split in two halves (x1, x2) and each position m is
    rotated by angles m * theta_i with theta_i = base^(-2i/head_dim):
        out = [x1*cos - x2*sin, x2*cos + x1*sin]
    Relative offsets then appear directly in the attention dot products.
    """

    def __init__(self, head_dim: int, max_len: int = 4096, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head_dim")
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        t = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # (max_len, head_dim/2)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    @staticmethod
    def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        half = x.size(-1) // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """q, k: (B, H, L, head_dim) from the same sequence (self-attention)."""
        L = q.size(-2)
        if L > self.cos.size(0):
            raise ValueError(
                f"sequence length {L} exceeds RoPE max_len {self.cos.size(0)}"
            )
        cos = self.cos[:L].to(dtype=q.dtype)  # (L, head_dim/2), broadcasts over (B, H)
        sin = self.sin[:L].to(dtype=q.dtype)
        return self._rotate(q, cos, sin), self._rotate(k, cos, sin)
