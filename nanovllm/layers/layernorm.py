import torch
from torch import nn

try:
    from flashinfer import fused_add_rmsnorm, rmsnorm
except Exception:  # pragma: no cover
    fused_add_rmsnorm = None
    rmsnorm = None


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        use_flashinfer = (
            rmsnorm is not None
            and x.is_cuda
            and x.dtype in {torch.float16, torch.bfloat16}
            and self.weight.dtype == x.dtype
        )
        if use_flashinfer:
            if residual is None:
                return rmsnorm(x, self.weight, self.eps)
            if fused_add_rmsnorm is not None and x.ndim == 2 and residual.ndim == 2:
                fused_add_rmsnorm(x, residual, self.weight, self.eps)
                return x, residual
            residual_out = residual + x
            return rmsnorm(residual_out, self.weight, self.eps), residual_out

        orig_dtype = x.dtype
        if residual is None:
            x_f = x.float()
            var = x_f.pow(2).mean(dim=-1, keepdim=True)
            x_f.mul_(torch.rsqrt(var + self.eps))
            return x_f.to(orig_dtype).mul_(self.weight)

        x_f = x.float().add_(residual.float())
        residual_out = x_f.to(orig_dtype)
        var = x_f.pow(2).mean(dim=-1, keepdim=True)
        x_f.mul_(torch.rsqrt(var + self.eps))
        return x_f.to(orig_dtype).mul_(self.weight), residual_out
