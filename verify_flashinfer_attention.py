import argparse

import torch
import torch.nn.functional as F

import flashinfer


def _expand_kv_for_gqa(x: torch.Tensor, num_qo_heads: int) -> torch.Tensor:
    # x: [T, Hk, D] -> [Hq, T, D] for SDPA
    t, h_kv, d = x.shape
    if num_qo_heads == h_kv:
        return x.permute(1, 0, 2).contiguous()
    g = num_qo_heads // h_kv
    return x.permute(1, 0, 2).repeat_interleave(g, dim=0).contiguous()


def _reference_prefill(
    q: torch.Tensor,
    kv_pages_k: torch.Tensor,
    kv_pages_v: torch.Tensor,
    page_indices: list[int],
    last_page_len: int,
    *,
    q_start_pos: int,
    num_qo_heads: int,
    scale: float,
) -> torch.Tensor:
    # q: [q_len, Hq, D]
    # kv_pages_*: [max_pages, page_size, Hk, D]
    page_size = kv_pages_k.size(1)
    num_pages = len(page_indices)
    kv_len = (num_pages - 1) * page_size + last_page_len

    pages_k = kv_pages_k[torch.tensor(page_indices, device=kv_pages_k.device)]
    pages_v = kv_pages_v[torch.tensor(page_indices, device=kv_pages_v.device)]
    k_full = pages_k.reshape(num_pages * page_size, pages_k.size(2), pages_k.size(3))[:kv_len]
    v_full = pages_v.reshape(num_pages * page_size, pages_v.size(2), pages_v.size(3))[:kv_len]

    # SDPA expects [B, H, T, D]
    q_ = q.transpose(0, 1).unsqueeze(0)  # [1, Hq, q_len, D]
    k_ = _expand_kv_for_gqa(k_full, num_qo_heads).unsqueeze(0)  # [1, Hq, kv_len, D]
    v_ = _expand_kv_for_gqa(v_full, num_qo_heads).unsqueeze(0)

    # Causal mask for q positions within the sequence.
    # Query tokens correspond to absolute positions [q_start_pos, ..., q_start_pos+q_len-1].
    q_len = q.size(0)
    mask = torch.ones((q_len, kv_len), device=q.device, dtype=torch.bool).tril(diagonal=q_start_pos)
    out = F.scaled_dot_product_attention(q_, k_, v_, attn_mask=mask, scale=scale)
    return out.squeeze(0).transpose(0, 1).contiguous()  # [q_len, Hq, D]


def _reference_decode(
    q: torch.Tensor,
    kv_pages_k: torch.Tensor,
    kv_pages_v: torch.Tensor,
    page_indices: list[int],
    last_page_len: int,
    *,
    num_qo_heads: int,
    scale: float,
) -> torch.Tensor:
    # q: [Hq, D] (single token)
    page_size = kv_pages_k.size(1)
    num_pages = len(page_indices)
    kv_len = (num_pages - 1) * page_size + last_page_len

    pages_k = kv_pages_k[torch.tensor(page_indices, device=kv_pages_k.device)]
    pages_v = kv_pages_v[torch.tensor(page_indices, device=kv_pages_v.device)]
    k_full = pages_k.reshape(num_pages * page_size, pages_k.size(2), pages_k.size(3))[:kv_len]
    v_full = pages_v.reshape(num_pages * page_size, pages_v.size(2), pages_v.size(3))[:kv_len]

    q_ = q.unsqueeze(0).unsqueeze(2)  # [1, Hq, 1, D]
    k_ = _expand_kv_for_gqa(k_full, num_qo_heads).unsqueeze(0)
    v_ = _expand_kv_for_gqa(v_full, num_qo_heads).unsqueeze(0)
    # The decode query corresponds to the *last* token position, so it's allowed to attend to the
    # full KV length. SDPA's `is_causal=True` assumes query positions start from 0, so use an
    # explicit all-true mask instead.
    mask = torch.ones((1, kv_len), device=q.device, dtype=torch.bool)
    out = F.scaled_dot_product_attention(q_, k_, v_, attn_mask=mask, is_causal=False, scale=scale)
    return out.squeeze(0).squeeze(1)  # [Hq, D]


@torch.inference_mode()
def main() -> None:
    p = argparse.ArgumentParser(description="Sanity-check FlashInfer paged-kv vs SDPA.")
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    args = p.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = torch.device("cuda")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # Small shapes.
    page_size = 16
    head_dim = 64
    num_kv_heads = 4
    num_qo_heads = 8  # GQA ratio = 2
    scale = head_dim**-0.5

    # Two sequences with different lengths.
    # Each sequence is represented by pages (block ids) + last_page_len.
    seq_pages = [[2, 5], [1, 3, 7]]
    last_lens = [11, 4]
    seq_lens = [(len(p) - 1) * page_size + ll for p, ll in zip(seq_pages, last_lens)]

    max_pages = 8
    k_cache = torch.randn((max_pages, page_size, num_kv_heads, head_dim), device=device, dtype=dtype)
    v_cache = torch.randn((max_pages, page_size, num_kv_heads, head_dim), device=device, dtype=dtype)

    workspace = torch.zeros(128 * 1024 * 1024, device=device, dtype=torch.uint8)
    prefill = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, kv_layout="NHD", backend="fa2")
    decode = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, kv_layout="NHD")

    # Prefill test: q_len < kv_len (simulate prefix cache hit).
    q_lens = [3, 5]
    qo_indptr = torch.tensor([0, q_lens[0], q_lens[0] + q_lens[1]], dtype=torch.int32, pin_memory=True)
    kv_indptr = torch.tensor([0, len(seq_pages[0]), len(seq_pages[0]) + len(seq_pages[1])], dtype=torch.int32, pin_memory=True)
    kv_indices = torch.tensor([*seq_pages[0], *seq_pages[1]], dtype=torch.int32, device=device)
    kv_last = torch.tensor(last_lens, dtype=torch.int32, pin_memory=True)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, pin_memory=True)
    seq_lens_q_t = torch.tensor(q_lens, dtype=torch.int32, pin_memory=True)

    q_tokens = torch.randn((sum(q_lens), num_qo_heads, head_dim), device=device, dtype=dtype)
    prefill.plan(
        qo_indptr=qo_indptr,
        paged_kv_indptr=kv_indptr,
        paged_kv_indices=kv_indices,
        paged_kv_last_page_len=kv_last,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        page_size=page_size,
        causal=True,
        sm_scale=scale,
        q_data_type=dtype,
        kv_data_type=dtype,
        non_blocking=True,
        seq_lens=seq_lens_t,
        seq_lens_q=seq_lens_q_t,
    )
    out = prefill.run(q=q_tokens, paged_kv_cache=(k_cache, v_cache))

    # Reference prefill outputs per sequence (queries are the last q_len tokens).
    ref_chunks = []
    off = 0
    for pages, last_len, q_len, kv_len in zip(seq_pages, last_lens, q_lens, seq_lens):
        q_chunk = q_tokens[off : off + q_len]
        q_start = kv_len - q_len
        ref = _reference_prefill(
            q_chunk,
            k_cache,
            v_cache,
            pages,
            last_len,
            q_start_pos=q_start,
            num_qo_heads=num_qo_heads,
            scale=scale,
        )
        ref_chunks.append(ref)
        off += q_len
    ref_out = torch.cat(ref_chunks, dim=0)

    max_err = (out - ref_out).abs().max().item()
    print(f"prefill max_abs_err={max_err:.3e}")
    assert max_err < (5e-2 if dtype == torch.bfloat16 else 1e-2)

    # Decode test: single query token per seq at the last position.
    indptr = kv_indptr
    indices = kv_indices
    last_page_len = kv_last
    decode.plan(
        indptr=indptr,
        indices=indices,
        last_page_len=last_page_len,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        sm_scale=scale,
        q_data_type=dtype,
        kv_data_type=dtype,
        non_blocking=True,
        seq_lens=seq_lens_t,
    )
    q_decode = torch.randn((len(seq_pages), num_qo_heads, head_dim), device=device, dtype=dtype)
    out_d = decode.run(q=q_decode, paged_kv_cache=(k_cache, v_cache))

    ref_d = []
    for q1, pages, last_len in zip(q_decode, seq_pages, last_lens):
        ref_d.append(_reference_decode(q1, k_cache, v_cache, pages, last_len, num_qo_heads=num_qo_heads, scale=scale))
    ref_d = torch.stack(ref_d, dim=0)
    max_err_d = (out_d - ref_d).abs().max().item()
    print(f"decode max_abs_err={max_err_d:.3e}")
    assert max_err_d < (5e-2 if dtype == torch.bfloat16 else 1e-2)

    print("OK")


if __name__ == "__main__":
    main()
