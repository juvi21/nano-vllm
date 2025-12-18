import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

        self.fp8_enabled = False
        self.register_buffer("_fp8_weight_t", torch.tensor([]), persistent=False)
        self.register_buffer("_fp8_scale_b", torch.tensor([]), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def enable_fp8(self):
        self.fp8_enabled = True
        self._prepare_fp8_weights()

    def _prepare_fp8_weights(self):
        if not self.fp8_enabled or not self.weight.is_cuda:
            return
        if self.weight.numel() == 0:
            return
        fp8_dtype = torch.float8_e4m3fn
        fp8_max = float(torch.finfo(fp8_dtype).max)

        w = self.weight
        # Rowwise weight scaling (per output feature) for scaled_mm's scale_b: (1, N).
        max_abs = w.abs().amax(dim=1).float()
        scale_b = (max_abs / fp8_max).clamp_min(1e-6).unsqueeze(0).contiguous()  # [1, out]
        w_scaled = w / scale_b.squeeze(0).to(dtype=w.dtype).unsqueeze(1)
        w_fp8 = w_scaled.to(fp8_dtype)
        self._fp8_weight_t = w_fp8.t()  # [in, out], column-major view
        self._fp8_scale_b = scale_b

    def _fp8_forward(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        # Uses torch._scaled_mm with rowwise scaling.
        if bias is None:
            bias = self.bias
        if (
            not self.fp8_enabled
            or not x.is_cuda
            or x.ndim != 2
            or x.dtype not in {torch.float16, torch.bfloat16}
            or torch.cuda.get_device_capability(x.device)[0] < 10
        ):
            return F.linear(x, self.weight, bias)

        self._prepare_fp8_weights()
        if self._fp8_weight_t.numel() == 0:
            return F.linear(x, self.weight, bias)

        k = x.size(1)
        if k % 16 != 0 or self._fp8_weight_t.size(0) != k:
            return F.linear(x, self.weight, self.bias)

        fp8_dtype = torch.float8_e4m3fn
        fp8_max = float(torch.finfo(fp8_dtype).max)

        max_abs = x.abs().amax(dim=1, keepdim=True).float()
        scale_a = (max_abs / fp8_max).clamp_min(1e-6).contiguous()  # [M, 1], fp32
        x_fp8 = (x / scale_a.to(dtype=x.dtype)).to(fp8_dtype)
        return torch._scaled_mm(
            x_fp8,
            self._fp8_weight_t,
            scale_a,
            self._fp8_scale_b,
            bias=bias,
            out_dtype=x.dtype,
        )


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fp8_enabled:
            return self._fp8_forward(x)
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias if self.tp_rank == 0 else None
        if self.fp8_enabled:
            y = self._fp8_forward(x, bias=bias)
        else:
            y = F.linear(x, self.weight, bias)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
