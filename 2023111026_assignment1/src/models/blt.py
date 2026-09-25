"""Simplified Byte Latent Transformer (BLT) modules for configuration C5.

After Pagnoni et al. (2024), 'Byte Latent Transformer: Patches Scale Better
Than Tokens' (arXiv:2412.09871), simplified as the assignment allows.

Token-free pipeline:

  cipher bytes (every 8 bits of the binary sequence grouped into one value
  0-255, which is the underlying data rather than its text encoding)
      -> LocalEncoder: byte embedding + windowed local layers + cross-attention
         pooling into entropy-segmented patches. Boundaries come from a byte
         n-gram entropy model (see dataset.ByteEntropyModel): a new patch starts
         wherever the next byte is hard to predict, so patches are variable
         length as in the paper rather than a fixed stride.
      -> global Encoder stack (shared classes from src/models/__init__.py)
  plaintext bytes (BOS-shifted)
      -> pooled into patches of size patch_tgt, shifted right by one patch
         with a learned start patch, run through the global Decoder stack
         (causal at patch level, cross-attending to encoder memory)
      -> LocalDecoder: causal windowed byte layers, each cross-attending to
         the global decoder states under a patch-causal mask, then a byte head.

Patch-causality: with the start-patch shift, global state g_j summarizes target
input positions <= j*patch_tgt - 1, so byte position i may attend states
j <= (i+1) // patch_tgt without ever seeing its own or future bytes.

Byte vocabulary: 256 raw byte values + PAD/BOS/EOS specials.
"""

import math

import torch
import torch.nn as nn

from . import (
    Decoder,
    Encoder,
    FeedForward,
    causal_mask,
    init_transformer_weights,
    make_norm,
)
from .attention import MultiHeadAttention
from .positional import SinusoidalPositionalEncoding

BYTE_PAD = 256
BYTE_BOS = 257
BYTE_EOS = 258
BYTE_VOCAB = 259


def _pad_to_multiple(
    x: torch.Tensor, multiple: int, value: float = 0.0
) -> torch.Tensor:
    """Pad dim=1 of (B, L, ...) up to the next multiple."""
    L = x.size(1)
    remainder = L % multiple
    if remainder == 0:
        return x
    pad_len = multiple - remainder
    shape = list(x.shape)
    shape[1] = pad_len
    return torch.cat([x, x.new_full(shape, value)], dim=1)


class LocalTransformerLayer(nn.Module):
    """Pre-LN transformer layer with block-local self-attention.

    The sequence is split into non-overlapping windows of `window` positions and
    attention runs independently inside each window, so memory is O(L * window)
    instead of O(L^2) — essential for ~21k-byte cipher inputs. With causal=True
    the mask is causal within each window (never leaks future positions; blocks
    simply see no earlier windows directly — the global path carries long-range
    context).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float,
        window: int,
        causal: bool,
    ):
        super().__init__()
        self.window = window
        self.causal = causal
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, ffn_dim, dropout)
        self.norm1 = make_norm("layernorm", d_model)
        self.norm2 = make_norm("layernorm", d_model)
        self.dropout = nn.Dropout(dropout)

    def _local_attention(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D); pad_mask: (B, L) bool, True = pad."""
        B, L, D = x.shape
        W = self.window
        xp = _pad_to_multiple(x, W)
        mp = (
            _pad_to_multiple(pad_mask[..., None].to(x.dtype), W, value=1.0)
            .squeeze(-1)
            .bool()
        )
        nb = xp.size(1) // W
        xb = xp.view(B * nb, W, D)
        mask = mp.view(B * nb, 1, 1, W)  # key padding per window
        if self.causal:
            mask = mask | causal_mask(W, x.device)
        out = self.attn(xb, xb, xb, mask)
        return out.view(B, nb * W, D)[:, :L]

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self._local_attention(self.norm1(x), pad_mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class PatchPooling(nn.Module):
    """Pool a variable-length group of byte vectors into one patch vector by
    cross-attention, as in the BLT paper's patch encoder.

    A learned query attends over exactly the bytes belonging to a patch, so
    patches of any length collapse to one vector and the result depends on the
    order and identity of those bytes. Order matters here: an earlier version
    used a mean, which is permutation-invariant, and the model could not
    recover a character from what amounted to a bag of its bits.

    Membership arrives as a (B, K, L) boolean mask rather than a patch size,
    because entropy-based segmentation gives patches no fixed length.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.query, std=0.02)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm = make_norm("layernorm", d_model)

    def forward(
        self, x: torch.Tensor, member: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, L, D) byte vectors; member: (B, K, L) bool, True = in patch.

        Returns (patches (B, K, D), patch_pad_mask (B, K) bool for empty patches).
        """
        B, K, L = member.shape
        D = x.size(-1)
        empty = ~member.any(dim=-1)  # (B, K) patches with no bytes

        # One attention call over the flattened batch of patches: every patch is
        # a query of length 1 attending to the whole sequence, masked to its own
        # bytes. Fully-masked rows are harmless because ScaledDotProductAttention
        # fills blocked scores with the dtype minimum rather than -inf.
        q = self.query.to(x.dtype).expand(B * K, 1, D)
        kv = x.unsqueeze(1).expand(B, K, L, D).reshape(B * K, L, D)
        mask = (~member).reshape(B * K, 1, 1, L)
        pooled = self.attn(q, kv, kv, mask).view(B, K, D)
        pooled = self.norm(pooled).masked_fill(empty.unsqueeze(-1), 0.0)
        return pooled, empty


def fixed_patch_member(pad_mask: torch.Tensor, patch: int) -> torch.Tensor:
    """(B, K, L) membership for contiguous fixed-size patches, pads excluded.

    Used on the target side only. Entropy segmentation applies to the source,
    where the paper defines it; the decoder's patches must line up with the
    patch-causal mask, so they stay a fixed stride.
    """
    B, L = pad_mask.shape
    K = (L + patch - 1) // patch
    pos = torch.arange(L, device=pad_mask.device)
    ks = torch.arange(K, device=pad_mask.device)
    member = (pos // patch).unsqueeze(0) == ks.unsqueeze(1)  # (K, L)
    return member.unsqueeze(0).expand(B, K, L) & (~pad_mask).unsqueeze(1)


class LocalEncoder(nn.Module):
    """Byte embedding + local windowed layers + entropy-patch pooling."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float,
        n_layers: int = 2,
        window: int = 128,
        max_len: int = 4096,
    ):
        super().__init__()
        self.embed = nn.Embedding(BYTE_VOCAB, d_model, padding_idx=BYTE_PAD)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            LocalTransformerLayer(
                d_model, n_heads, ffn_dim, dropout, window, causal=False
            )
            for _ in range(n_layers)
        )
        self.norm = make_norm("layernorm", d_model)
        self.pooling = PatchPooling(d_model)

    def forward(
        self, src_bytes: torch.Tensor, patch_member: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """src_bytes: (B, L) byte ids; patch_member: (B, K, L) bool.
        Returns (patches (B, K, D), patch_pad_mask (B, K))."""
        pad_mask = src_bytes == BYTE_PAD
        # sqrt(d) scaling for the same reason as the tokenized model: keeps the
        # (d^-1/2-initialized) embeddings at unit scale next to the sinusoidal PE
        x = self.pos_enc(self.embed(src_bytes) * math.sqrt(self.embed.embedding_dim))
        for layer in self.layers:
            x = layer(x, pad_mask)
        return self.pooling(self.norm(x), patch_member & (~pad_mask).unsqueeze(1))


class LocalDecoderLayer(nn.Module):
    """Causal windowed byte self-attention + two cross-attentions + FFN.

    Each byte cross-attends (a) the target-side global decoder states under the
    patch-causal mask and (b) the encoder's source-patch memory. (b) exists
    because this task is a positional cipher: without a direct byte->source
    path, the only route for source information is the pooled-target-patch
    chain, which empirically never learns the alignment (the 80-epoch C5 run
    converged to a source-free language model). Source memory is fully visible
    — it carries no target information, so causality is unaffected.
    """

    def __init__(
        self, d_model: int, n_heads: int, ffn_dim: int, dropout: float, window: int
    ):
        super().__init__()
        self.local = LocalTransformerLayer(
            d_model, n_heads, ffn_dim, dropout, window, causal=True
        )
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm_cross = make_norm("layernorm", d_model)
        self.cross_mem = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm_mem = make_norm("layernorm", d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor,
        global_states: torch.Tensor,
        cross_mask: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.local(x, pad_mask)
        x = x + self.dropout(
            self.cross_attn(
                self.norm_cross(x), global_states, global_states, cross_maskp
            )
        )
        x = x + self.dropout(
            self.cross_mem(self.norm_mem(x), memory, memory, memory_mask)
        )
        return x


class LocalDecoder(nn.Module):
    """Byte-level decoder refining global patch states into next-byte logits."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float,
        n_layers: int = 2,
        window: int = 128,
        patch: int = 4,
        max_len: int = 4096,
    ):
        super().__init__()
        self.patch = patch
        self.embed = nn.Embedding(BYTE_VOCAB, d_model, padding_idx=BYTE_PAD)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            LocalDecoderLayer(d_model, n_heads, ffn_dim, dropout, window)
            for _ in range(n_layers)
        )
        self.norm = make_norm("layernorm", d_model)
        self.head = nn.Linear(d_model, BYTE_VOCAB, bias=False)
        self.head.weight = self.embed.weight  # weight tying

    def embed_bytes(self, tgt_in: torch.Tensor) -> torch.Tensor:
        return self.pos_enc(self.embed(tgt_in) * math.sqrt(self.embed.embedding_dim))

    def patch_causal_cross_mask(
        self, Lt: int, K: int, device: torch.device
    ) -> torch.Tensor:
        """(1, 1, Lt, K) bool: byte position i may attend global slot j iff
        j <= (i + 1) // patch (see module docstring); True = blocked."""
        i = torch.arange(Lt, device=device)[:, None]
        j = torch.arange(K, device=device)[None, :]
        return (j > (i + 1) // self.patch)[None, None]

    def forward(
        self,
        x: torch.Tensor,
        tgt_pad_mask: torch.Tensor,
        global_states: torch.Tensor,
        global_pad_mask: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        """x: (B, Lt, D) embedded target bytes; global_states: (B, K, D);
        memory: (B, Ks, D) encoder source-patch states."""
        Lt, K = x.size(1), global_states.size(1)
        cross_mask = (
            self.patch_causal_cross_mask(Lt, K, x.device)
            | global_pad_mask[:, None, None, :]
        )
        for layer in self.layers:
            x = layer(x, tgt_pad_mask, global_states, cross_mask, memory, memory_mask)
        return self.head(self.norm(x))


class BLTModel(nn.Module):
    """Full token-free seq2seq model for C5.

    The global transformer is the exact same Encoder/Decoder stack as C1
    (sinusoidal positions at patch level, MHA, LayerNorm) — only the
    tokenization boundary changes, as required by the ablation design.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        enc_layers: int = 4,
        dec_layers: int = 4,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        local_layers: int = 2,
        local_window: int = 128,
        patch_tgt: int = 4,
        max_src_bytes: int = 4096,
        max_tgt_bytes: int = 4096,
    ):
        super().__init__()
        self.patch_tgt = patch_tgt
        self.tgt_pooling = PatchPooling(d_model)
        self.local_encoder = LocalEncoder(
            d_model,
            n_heads,
            ffn_dim,
            dropout,
            local_layers,
            local_window,
            max_src_bytes,
        )
        self.local_decoder = LocalDecoder(
            d_model,
            n_heads,
            ffn_dim,
            dropout,
            local_layers,
            local_window,
            patch_tgt,
            max_tgt_bytes,
        )
        # Entropy patches have no fixed size, so in the worst case every byte
        # is its own patch: size the patch-level encoding for that bound.
        self.patch_pos_enc = SinusoidalPositionalEncoding(
            d_model, max_src_bytes + 8, dropout
        )
        self.global_encoder = Encoder(
            enc_layers,
            d_model,
            n_heads,
            ffn_dim,
            dropout,
            norm="layernorm",
            attn_type="mha",
        )
        self.global_decoder = Decoder(
            dec_layers,
            d_model,
            n_heads,
            ffn_dim,
            dropout,
            norm="layernorm",
            attn_type="mha",
        )
        self.start_patch = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.start_patch, std=0.02)

        init_transformer_weights(self)
        # The xavier pass above also hit the output head, which shares storage
        # with the byte embedding table — restore the embedding-scale init on
        # the shared tables (the tie itself is unaffected: same tensor).
        for emb in (self.local_encoder.embed, self.local_decoder.embed):
            nn.init.normal_(emb.weight, mean=0.0, std=emb.embedding_dim**-0.5)
            with torch.no_grad():
                emb.weight[emb.padding_idx].fill_(0)

    def encode(
        self, src_bytes: torch.Tensor, patch_member: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (memory (B, Ks, D), memory key-padding mask (B, 1, 1, Ks))."""
        patches, patch_pad = self.local_encoder(src_bytes, patch_member)
        memory = self.global_encoder(
            self.patch_pos_enc(patches), patch_pad[:, None, None, :]
        )
        return memory, patch_pad[:, None, None, :]

    def decode(
        self, tgt_in: torch.Tensor, memory: torch.Tensor, memory_mask: torch.Tensor
    ) -> torch.Tensor:
        tgt_pad_mask = tgt_in == BYTE_PAD
        x = self.local_decoder.embed_bytes(tgt_in)  # (B, Lt, D)

        # Global decoder input: pooled target patches shifted right by one slot
        # behind a learned start patch (keeps every byte's view strictly causal).
        tgt_patches, tgt_patch_pad = self.tgt_pooling(
            x, fixed_patch_member(tgt_pad_mask, self.patch_tgt)
        )
        B, K, D = tgt_patches.shape
        start = self.start_patch.to(tgt_patches.dtype).expand(B, 1, D)
        g_in = torch.cat([start, tgt_patches[:, :-1]], dim=1)  # (B, K, D)
        g_pad = torch.cat([tgt_patch_pad.new_zeros(B, 1), tgt_patch_pad[:, :-1]], dim=1)
        self_mask = causal_mask(K, tgt_in.device) | g_pad[:, None, None, :]
        g = self.global_decoder(
            self.patch_pos_enc(g_in), memory, self_mask, memory_mask
        )

        return self.local_decoder(x, tgt_pad_mask, g, g_pad, memory, memory_mask)

    def forward(
        self,
        src_bytes: torch.Tensor,
        tgt_in: torch.Tensor,
        patch_member: torch.Tensor,
    ) -> torch.Tensor:
        """src_bytes: (B, Ls) cipher byte values; tgt_in: (B, Lt) BOS-shifted
        plaintext bytes; patch_member: (B, Ks, Ls) entropy patch membership.
        Returns next-byte logits (B, Lt, BYTE_VOCAB)."""
        memory, memory_mask = self.encode(src_bytes, patch_member)
        return self.decode(tgt_in, memory, memory_mask)
