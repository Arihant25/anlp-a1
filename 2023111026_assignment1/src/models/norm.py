"""Normalization modules written from scratch: LayerNorm and RMSNorm.

Both are used in the Pre-LN placement (normalize the residual-branch input,
i.e. x + Sublayer(Norm(x))); the block wiring lives in src/models/__init__.py.
Statistics are computed in fp32 for stability under bf16 autocast.
"""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """y = (x - mean) / sqrt(var + eps) * gamma + beta, over the last dimension."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        y = (x - mean) * torch.rsqrt(var + self.eps)
        return (y * self.gamma.float() + self.beta.float()).to(dtype)


class RMSNorm(nn.Module):
    """y = x / sqrt(mean(x^2) + eps) * gamma.

    Zhang & Sennrich (2019), 'Root Mean Square Layer Normalization' (NeurIPS).

    No mean-centering and no bias: cheaper than LayerNorm and empirically
    on par in transformers (used by Llama/T5).
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        y = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (y * self.gamma.float()).to(dtype)
