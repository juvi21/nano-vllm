from dataclasses import dataclass
import torch


@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    attn_backend: str = "auto"

    # FlashInfer paged-kv metadata (page_size = kvcache_block_size)
    fi_qo_indptr: torch.Tensor | None = None  # CPU int32, shape [bs+1]
    fi_kv_indptr: torch.Tensor | None = None  # CPU int32, shape [bs+1]
    fi_kv_indices: torch.Tensor | None = None  # GPU int32, shape [fi_kv_indptr[-1]]
    fi_kv_last_page_len: torch.Tensor | None = None  # CPU int32, shape [bs]
    fi_seq_lens: torch.Tensor | None = None  # CPU int32, shape [bs]
    fi_seq_lens_q: torch.Tensor | None = None  # CPU int32, shape [bs]
    fi_prefill_wrapper: object | None = None
    fi_decode_wrapper: object | None = None
    fi_planned: bool = False

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(
    is_prefill,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    context_lens=None,
    block_tables=None,
    attn_backend: str = "auto",
    fi_qo_indptr=None,
    fi_kv_indptr=None,
    fi_kv_indices=None,
    fi_kv_last_page_len=None,
    fi_seq_lens=None,
    fi_seq_lens_q=None,
    fi_prefill_wrapper=None,
    fi_decode_wrapper=None,
    fi_planned: bool = False,
):
    ctx = _CONTEXT
    ctx.is_prefill = is_prefill
    ctx.cu_seqlens_q = cu_seqlens_q
    ctx.cu_seqlens_k = cu_seqlens_k
    ctx.max_seqlen_q = max_seqlen_q
    ctx.max_seqlen_k = max_seqlen_k
    ctx.slot_mapping = slot_mapping
    ctx.context_lens = context_lens
    ctx.block_tables = block_tables
    ctx.attn_backend = attn_backend
    ctx.fi_qo_indptr = fi_qo_indptr
    ctx.fi_kv_indptr = fi_kv_indptr
    ctx.fi_kv_indices = fi_kv_indices
    ctx.fi_kv_last_page_len = fi_kv_last_page_len
    ctx.fi_seq_lens = fi_seq_lens
    ctx.fi_seq_lens_q = fi_seq_lens_q
    ctx.fi_prefill_wrapper = fi_prefill_wrapper
    ctx.fi_decode_wrapper = fi_decode_wrapper
    ctx.fi_planned = fi_planned

def reset_context():
    ctx = _CONTEXT
    ctx.is_prefill = False
    ctx.cu_seqlens_q = None
    ctx.cu_seqlens_k = None
    ctx.max_seqlen_q = 0
    ctx.max_seqlen_k = 0
    ctx.slot_mapping = None
    ctx.context_lens = None
    ctx.block_tables = None
    ctx.attn_backend = "auto"
    ctx.fi_qo_indptr = None
    ctx.fi_kv_indptr = None
    ctx.fi_kv_indices = None
    ctx.fi_kv_last_page_len = None
    ctx.fi_seq_lens = None
    ctx.fi_seq_lens_q = None
    ctx.fi_prefill_wrapper = None
    ctx.fi_decode_wrapper = None
    ctx.fi_planned = False
