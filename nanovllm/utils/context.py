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
):
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill=is_prefill,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        attn_backend=attn_backend,
        fi_qo_indptr=fi_qo_indptr,
        fi_kv_indptr=fi_kv_indptr,
        fi_kv_indices=fi_kv_indices,
        fi_kv_last_page_len=fi_kv_last_page_len,
        fi_seq_lens=fi_seq_lens,
        fi_seq_lens_q=fi_seq_lens_q,
    )

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
