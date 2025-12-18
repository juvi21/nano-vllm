import torch
from torch import nn
import torch.nn.functional as F

try:
    from flashinfer import silu_and_mul
except Exception:  # pragma: no cover
    silu_and_mul = None


class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if silu_and_mul is not None and x.is_cuda and x.dtype in {torch.float16, torch.bfloat16}:
            return silu_and_mul(x)
        x1, x2 = x.chunk(2, -1)
        return F.silu(x1) * x2
