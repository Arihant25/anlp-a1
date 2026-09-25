"""Transformer assembly built from the from-scratch modules in this package.

Contains the position-wise FeedForward network, Pre-LN encoder/decoder layers,
the Encoder/Decoder stacks (which operate on already-embedded inputs so the
BLT global transformer can reuse them over patch representations), and the
Seq2SeqTransformer used by configurations C1-C4.

Pre-LN placement: every sublayer computes x + Dropout(Sublayer(Norm(x))),
with one final Norm at the top of each stack.
"""

import math

import torch
import torch.nn as nn

from .attention import (
    GroupedQueryAttention,
    MultiHeadAttention,
)
from .norm import LayerNorm, RMSNorm
from .positional import RotaryPositionalEmbedding, SinusoidalPositionalEncoding


def init_transformer_weights(module: nn.Module) -> None:
    """Xavier for linears, N(0, d^-1/2) for embeddings.

    PyTorch's default N(0,1) embedding init is catastrophic here: combined with
    the sqrt(d_model) embedding scale it produces std~16 activations that (a)
    drown the unit-amplitude sinusoidal positional encoding — destroying the
    positional information this task depends on — and (b) yield hugely
    over-confident initial logits through the tied output head. N(0, d^-1/2)
    makes the scaled embeddings unit-variance, matching the PE and giving
    O(1) initial logits.
    """
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=m.embedding_dim**-0.5)
            if m.padding_idx is not None:
                with torch.no_grad():
                    m.weight[m.padding_idx].fill_(0)


def make_norm(kind: str, d_model: int) -> nn.Module:
    if kind == "layernorm":
        return LayerNorm(d_model)
    if kind == "rmsnorm":
        return RMSNorm(d_model)
    raise ValueError(f"unknown norm kind: {kind}")


def make_self_attention(
    attn_type: str,
    d_model: int,
    n_heads: int,
    dropout: float,
    n_kv_heads: int | None = None,
    rope: RotaryPositionalEmbedding | None = None,
) -> nn.Module:
    if attn_type == "mha":
        return MultiHeadAttention(d_model, n_heads, dropout, rope=rope)
    if attn_type == "gqa":
        if n_kv_heads is None:
            raise ValueError("GQA requires n_kv_heads")
        return GroupedQueryAttention(d_model, n_heads, n_kv_heads, dropout, rope=rope)
    raise ValueError(f"unknown attention type: {attn_type}")


class FeedForward(nn.Module):
    """Position-wise FFN: Linear -> GELU -> Dropout -> Linear."""

    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float,
        norm: str = "layernorm",
        attn_type: str = "mha",
        n_kv_heads: int | None = None,
        rope: RotaryPositionalEmbedding | None = None,
    ):
        super().__init__()
        self.self_attn = make_self_attention(
            attn_type, d_model, n_heads, dropout, n_kv_heads, rope
        )
        self.ffn = FeedForward(d_model, ffn_dim, dropout)
        self.norm1 = make_norm(norm, d_model)
        self.norm2 = make_norm(norm, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.dropout(self.self_attn(h, h, h, mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class DecoderLayer(nn.Module):
    """Causal self-attention + cross-attention to encoder memory + FFN.

    Cross-attention is always MHA without RoPE: the ablated components
    (attention mechanism in C3, RoPE in C2) concern self-attention, and query/key
    positions of different sequences share no coordinate system for rotation.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float,
        norm: str = "layernorm",
        attn_type: str = "mha",
        n_kv_heads: int | None = None,
        rope: RotaryPositionalEmbedding | None = None,
    ):
        super().__init__()
        self.self_attn = make_self_attention(
            attn_type, d_model, n_heads, dropout, n_kv_heads, rope
        )
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout, rope=None)
        self.ffn = FeedForward(d_model, ffn_dim, dropout)
        self.norm1 = make_norm(norm, d_model)
        self.norm2 = make_norm(norm, d_model)
        self.norm3 = make_norm(norm, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        self_mask: torch.Tensor | None = None,
        cross_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.dropout(self.self_attn(h, h, h, self_mask))
        x = x + self.dropout(self.cross_attn(self.norm2(x), memory, memory, cross_mask))
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x


class Encoder(nn.Module):
    """Stack of encoder layers over already-embedded inputs (B, L, d_model)."""

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float,
        norm: str = "layernorm",
        attn_type: str = "mha",
        n_kv_heads: int | None = None,
        rope: RotaryPositionalEmbedding | None = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            EncoderLayer(
                d_model, n_heads, ffn_dim, dropout, norm, attn_type, n_kv_heads, rope
            )
            for _ in range(n_layers)
        )
        self.final_norm = make_norm(norm, d_model)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.final_norm(x)


class Decoder(nn.Module):
    """Stack of decoder layers over already-embedded inputs (B, L, d_model)."""

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float,
        norm: str = "layernorm",
        attn_type: str = "mha",
        n_kv_heads: int | None = None,
        rope: RotaryPositionalEmbedding | None = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            DecoderLayer(
                d_model, n_heads, ffn_dim, dropout, norm, attn_type, n_kv_heads, rope
            )
            for _ in range(n_layers)
        )
        self.final_norm = make_norm(norm, d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        self_mask: torch.Tensor | None = None,
        cross_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, self_mask, cross_mask)
        return self.final_norm(x)


def causal_mask(length: int, device: torch.device) -> torch.Tensor:
    """(1, 1, L, L) bool mask, True above the diagonal (blocked future positions)."""
    return torch.triu(
        torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1
    )[None, None]


def padding_mask(ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """(B, 1, 1, L) bool key-padding mask, True at pad positions (blocked)."""
    return (ids == pad_id)[:, None, None, :]


class Seq2SeqTransformer(nn.Module):
    """Encoder-decoder transformer for the tokenized configurations C1-C4.

    Source: cipher byte tokens (each 8-bit group of the binary string as one
    token, vocab 256 + pad). Target: BPE subword ids. The target embedding is
    weight-tied with the output projection.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_pad_id: int,
        tgt_pad_id: int,
        d_model: int = 256,
        n_heads: int = 8,
        enc_layers: int = 4,
        dec_layers: int = 4,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        pos_encoding: str = "sinusoidal",  # 'sinusoidal' | 'rope'
        attn_type: str = "mha",  # 'mha' | 'gqa'
        n_kv_heads: int | None = None,
        norm: str = "layernorm",  # 'layernorm' | 'rmsnorm'
        max_len: int = 4096,
    ):
        super().__init__()
        self.d_model = d_model
        self.src_pad_id = src_pad_id
        self.tgt_pad_id = tgt_pad_id
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=src_pad_id)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=tgt_pad_id)

        if pos_encoding == "sinusoidal":
            self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len, dropout)
            rope = None
        elif pos_encoding == "rope":
            self.pos_enc = None  # RoPE fully replaces the absolute encoding
            rope = RotaryPositionalEmbedding(d_model // n_heads, max_len)
        else:
            raise ValueError(f"unknown pos_encoding: {pos_encoding}")
        self.embed_dropout = nn.Dropout(dropout)

        self.encoder = Encoder(
            enc_layers,
            d_model,
            n_heads,
            ffn_dim,
            dropout,
            norm,
            attn_type,
            n_kv_heads,
            rope,
        )
        self.decoder = Decoder(
            dec_layers,
            d_model,
            n_heads,
            ffn_dim,
            dropout,
            norm,
            attn_type,
            n_kv_heads,
            rope,
        )
        init_transformer_weights(self)
        self.generator = nn.Linear(d_model, tgt_vocab_size, bias=False)
        self.generator.weight = self.tgt_embed.weight  # weight tying

    def _embed(self, ids: torch.Tensor, table: nn.Embedding) -> torch.Tensor:
        # sqrt(d_model) scaling (Vaswani et al., 2017) keeps embedding magnitudes
        # comparable to the unit-scale sinusoidal PE being added on top.
        x = table(ids) * math.sqrt(self.d_model)
        if self.pos_enc is not None:
            return self.pos_enc(x)
        return self.embed_dropout(x)

    def encode(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        src_mask = padding_mask(src, self.src_pad_id)
        memory = self.encoder(self._embed(src, self.src_embed), src_mask)
        return memory, src_mask

    def decode(
        self, tgt_in: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor
    ) -> torch.Tensor:
        self_mask = causal_mask(tgt_in.size(1), tgt_in.device) | padding_mask(
            tgt_in, self.tgt_pad_id
        )
        h = self.decoder(
            self._embed(tgt_in, self.tgt_embed), memory, self_mask, src_mask
        )
        return self.generator(h)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        """src: (B, Ls) byte tokens; tgt_in: (B, Lt) BPE ids (bos-prefixed).
        Returns logits (B, Lt, tgt_vocab)."""
        memory, src_mask = self.encode(src)
        return self.decode(tgt_in, memory, src_mask)
