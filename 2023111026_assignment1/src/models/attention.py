"""Attention modules from scratch: Scaled Dot-Product, Multi-Head (MHA),
and Grouped-Query (GQA).

No nn.MultiheadAttention / F.scaled_dot_product_attention anywhere — the
softmax(QK^T / sqrt(d_k))V computation is written out explicitly.

Mask convention (boolean, True = BLOCKED), broadcast against (B, H, Lq, Lk):
  * key-padding mask:  (B, 1, 1, Lk)
  * causal mask:       (1, 1, Lq, Lk) upper-triangular
Blocked positions are filled with the dtype's most negative value (not -inf,
so a fully-masked row degrades to a uniform distribution instead of NaN).
"""

import math

import torch
import torch.nn as nn

from .positional import RotaryPositionalEmbedding


class ScaledDotProductAttention(nn.Module):
    """Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V."""

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,  # (B, H, Lq, d_k)
        k: torch.Tensor,  # (B, H, Lk, d_k)
        v: torch.Tensor,  # (B, H, Lk, d_v)
        mask: torch.Tensor | None = None,  # bool, True = blocked
    ) -> torch.Tensor:
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # (B, H, Lq, Lk)
        if mask is not None:
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        return torch.matmul(attn, v)  # (B, H, Lq, d_v)


class MultiHeadAttention(nn.Module):
    """Standard MHA: n_heads independent heads over d_model / n_heads dims each.

    If a RoPE module is supplied it is applied to Q and K (self-attention only;
    callers pass rope=None for cross-attention).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        rope: RotaryPositionalEmbedding | None = None,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(dropout)
        self.rope = rope

    def _split(self, x: torch.Tensor, n_heads: int) -> torch.Tensor:
        B, L, _ = x.shape
        return x.view(B, L, n_heads, -1).transpose(1, 2)  # (B, H, L, head_dim)

    def forward(
        self,
        query: torch.Tensor,  # (B, Lq, d_model)
        key: torch.Tensor,  # (B, Lk, d_model)
        value: torch.Tensor,  # (B, Lk, d_model)
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self._split(self.q_proj(query), self.n_heads)
        k = self._split(self.k_proj(key), self.n_heads)
        v = self._split(self.v_proj(value), self.n_heads)
        if self.rope is not None:
            q, k = self.rope(q, k)
        out = self.attention(q, k, v, mask)  # (B, H, Lq, head_dim)
        B, _, Lq, _ = out.shape
        out = out.transpose(1, 2).contiguous().view(B, Lq, -1)
        return self.o_proj(out)


class GroupedQueryAttention(nn.Module):
    """GQA (Ainslie et al., 2023, EMNLP): n_heads query heads share n_kv_heads
    K/V heads.

    K/V projections output only n_kv_heads * head_dim features and are
    repeated (n_heads // n_kv_heads) times to face every query head, cutting
    K/V projection parameters and K/V memory traffic.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        dropout: float = 0.1,
        rope: RotaryPositionalEmbedding | None = None,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.o_proj = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(dropout)
        self.rope = rope

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, Lq, _ = query.shape
        Lk = key.size(1)
        q = self.q_proj(query).view(B, Lq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, Lk, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = (
            self.v_proj(value)
            .view(B, Lk, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        if self.rope is not None:
            q, k = self.rope(q, k)
        repeat = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(repeat, dim=1)  # (B, n_heads, Lk, head_dim)
        v = v.repeat_interleave(repeat, dim=1)
        out = self.attention(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(B, Lq, -1)
        return self.o_proj(out)
