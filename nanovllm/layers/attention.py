from functools import lru_cache
import torch
from torch import nn
import triton
import triton.language as tl

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
except Exception:  # pragma: no cover
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None

try:
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
    )
except Exception:  # pragma: no cover
    BatchDecodeWithPagedKVCacheWrapper = None
    BatchPrefillWithPagedKVCacheWrapper = None

from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    tok = tl.program_id(0)
    blk = tl.program_id(1)
    slot = tl.load(slot_mapping_ptr + tok)
    if slot == -1:
        return

    offs = blk * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs < D

    key_offsets = tok * key_stride + offs
    value_offsets = tok * value_stride + offs
    key = tl.load(key_ptr + key_offsets, mask=mask)
    value = tl.load(value_ptr + value_offsets, mask=mask)

    cache_offsets = slot * D + offs
    tl.store(k_cache_ptr + cache_offsets, key, mask=mask, cache_modifier=".cs")
    tl.store(v_cache_ptr + cache_offsets, value, mask=mask, cache_modifier=".cs")


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    BLOCK_D = 256
    grid = (N, triton.cdiv(D, BLOCK_D))
    store_kvcache_kernel[grid](
        key,
        key.stride(0),
        value,
        value.stride(0),
        k_cache,
        v_cache,
        slot_mapping,
        D,
        BLOCK_D=BLOCK_D,
        num_warps=4,
    )


@lru_cache(maxsize=None)
def _get_flashinfer_wrappers(device_index: int, *, use_tensor_cores: bool):
    if BatchDecodeWithPagedKVCacheWrapper is None or BatchPrefillWithPagedKVCacheWrapper is None:
        raise RuntimeError(
            "flashinfer-python is required for attn_backend=flashinfer. Install `flashinfer-python` and `flashinfer-cubin`."
        )

    device = torch.device(f"cuda:{device_index}")
    float_workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
        float_workspace_buffer,
        kv_layout="NHD",
        backend="fa2",
    )
    decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
        float_workspace_buffer,
        kv_layout="NHD",
        use_tensor_cores=use_tensor_cores,
    )

    # Reuse int workspace to avoid extra allocations (matches mini-sglang pattern).
    try:  # pragma: no cover
        int_workspace_buffer = prefill_wrapper._int_workspace_buffer
        decode_wrapper._int_workspace_buffer = int_workspace_buffer
    except Exception:
        pass

    return prefill_wrapper, decode_wrapper


def _sdpa_varlen_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
) -> torch.Tensor:
    # Fallback for warmup path before kv-cache is allocated.
    # Assumes fixed-length sequences (as used by ModelRunner.warmup_model()).
    bs = cu_seqlens_q.numel() - 1
    if bs <= 0:
        return q.new_empty((0, num_qo_heads, head_dim))
    if int(max_seqlen_q) * bs != q.size(0):
        raise RuntimeError(
            "SDPA fallback only supports fixed-length warmup batches; got ragged prefill."
        )

    q_ = q.view(bs, max_seqlen_q, num_qo_heads, head_dim).transpose(1, 2)  # [B, Hq, T, D]
    k_ = k.view(bs, max_seqlen_q, num_kv_heads, head_dim).transpose(1, 2)  # [B, Hk, T, D]
    v_ = v.view(bs, max_seqlen_q, num_kv_heads, head_dim).transpose(1, 2)

    if num_qo_heads != num_kv_heads:
        g = num_qo_heads // num_kv_heads
        k_ = k_.repeat_interleave(g, dim=1)
        v_ = v_.repeat_interleave(g, dim=1)

    o = torch.nn.functional.scaled_dot_product_attention(q_, k_, v_, is_causal=True, scale=scale)
    o = o.transpose(1, 2).contiguous().view(-1, num_qo_heads, head_dim)
    return o


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        backend = context.attn_backend
        if backend == "auto":
            backend = "flashinfer" if torch.cuda.get_device_capability(q.device)[0] >= 10 else "flashattn"

        k_cache, v_cache = self.k_cache, self.v_cache
        if backend == "flashinfer":
            # Warmup runs before kv-cache allocation; use SDPA to avoid hard dependency on paged cache.
            if not (k_cache.numel() and v_cache.numel()):
                if not context.is_prefill:
                    raise RuntimeError("flashinfer backend requires kv-cache for decode.")
                return _sdpa_varlen_fallback(
                    q,
                    k,
                    v,
                    num_qo_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim,
                    scale=self.scale,
                    cu_seqlens_q=context.cu_seqlens_q,
                    max_seqlen_q=context.max_seqlen_q,
                )

            if context.slot_mapping is not None and context.slot_mapping.numel():
                store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

            prefill_wrapper, decode_wrapper = _get_flashinfer_wrappers(
                q.device.index, use_tensor_cores=(self.num_heads // self.num_kv_heads) >= 4
            )

            if not context.fi_planned:
                if context.is_prefill:
                    assert context.fi_qo_indptr is not None
                    assert context.fi_kv_indptr is not None
                    assert context.fi_kv_indices is not None
                    assert context.fi_kv_last_page_len is not None
                    assert context.fi_seq_lens is not None
                    assert context.fi_seq_lens_q is not None
                    prefill_wrapper.plan(
                        qo_indptr=context.fi_qo_indptr,
                        paged_kv_indptr=context.fi_kv_indptr,
                        paged_kv_indices=context.fi_kv_indices,
                        paged_kv_last_page_len=context.fi_kv_last_page_len,
                        num_qo_heads=self.num_heads,
                        num_kv_heads=self.num_kv_heads,
                        head_dim_qk=self.head_dim,
                        page_size=k_cache.size(1),
                        causal=True,
                        pos_encoding_mode="NONE",
                        sm_scale=self.scale,
                        q_data_type=q.dtype,
                        kv_data_type=k_cache.dtype,
                        non_blocking=True,
                        seq_lens=context.fi_seq_lens,
                        seq_lens_q=context.fi_seq_lens_q,
                    )
                else:
                    assert context.fi_kv_indptr is not None
                    assert context.fi_kv_indices is not None
                    assert context.fi_kv_last_page_len is not None
                    assert context.fi_seq_lens is not None
                    decode_wrapper.plan(
                        indptr=context.fi_kv_indptr,
                        indices=context.fi_kv_indices,
                        last_page_len=context.fi_kv_last_page_len,
                        num_qo_heads=self.num_heads,
                        num_kv_heads=self.num_kv_heads,
                        head_dim=self.head_dim,
                        page_size=k_cache.size(1),
                        pos_encoding_mode="NONE",
                        sm_scale=self.scale,
                        q_data_type=q.dtype,
                        kv_data_type=k_cache.dtype,
                        non_blocking=True,
                        seq_lens=context.fi_seq_lens,
                    )
                context.fi_planned = True

            wrapper = prefill_wrapper if context.is_prefill else decode_wrapper
            return wrapper.run(q=q, paged_kv_cache=(k_cache, v_cache))

        if backend == "flashattn":
            if flash_attn_varlen_func is None or flash_attn_with_kvcache is None:
                raise RuntimeError("flash-attn is not installed but attn_backend=flashattn was requested.")

            if k_cache.numel() and v_cache.numel() and context.slot_mapping is not None and context.slot_mapping.numel():
                store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
            if context.is_prefill:
                if context.block_tables is not None:  # prefix cache
                    k, v = k_cache, v_cache
                o = flash_attn_varlen_func(
                    q,
                    k,
                    v,
                    max_seqlen_q=context.max_seqlen_q,
                    cu_seqlens_q=context.cu_seqlens_q,
                    max_seqlen_k=context.max_seqlen_k,
                    cu_seqlens_k=context.cu_seqlens_k,
                    softmax_scale=self.scale,
                    causal=True,
                    block_table=context.block_tables,
                )
            else:  # decode
                o = flash_attn_with_kvcache(
                    q.unsqueeze(1),
                    k_cache,
                    v_cache,
                    cache_seqlens=context.context_lens,
                    block_table=context.block_tables,
                    softmax_scale=self.scale,
                    causal=True,
                )
            return o

        raise ValueError(f"Unsupported attention backend: {backend}")
