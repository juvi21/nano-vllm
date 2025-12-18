import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import torch


def _env_info(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    info: dict[str, Any] = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info.update(
            {
                "device_name": torch.cuda.get_device_name(0),
                "device_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
                "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
                "total_mem_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
            }
        )
    if extra:
        info.update(extra)
    return info


def _make_token_prompts(*, num_seqs: int, prompt_len: int, seed: int) -> list[list[int]]:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return (
        torch.randint(low=0, high=10_000, size=(num_seqs, prompt_len), dtype=torch.int64, generator=g)
        .tolist()
    )


def _sample_prompts() -> list[str]:
    return [
        "Write a short haiku about NVIDIA Blackwell B200.",
        "Explain in 2 sentences what KV-cache is for LLM decoding.",
    ]


def _write_json(out: str, payload: dict[str, Any]) -> None:
    if not out:
        return
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")


@torch.inference_mode()
def bench_nanovllm(
    *,
    model: str,
    num_seqs: int,
    prompt_len: int,
    max_tokens: int,
    max_model_len: int,
    max_num_batched_tokens: int,
    attn_backend: str,
    seed: int,
) -> dict[str, Any]:
    from nanovllm import LLM, SamplingParams

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    t_init0 = perf_counter()
    llm = LLM(
        model,
        enforce_eager=False,
        max_model_len=max_model_len,
        max_num_seqs=num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        attn_backend=attn_backend,
    )
    init_time_s = perf_counter() - t_init0

    warmup_num_seqs = min(num_seqs, 8)
    warmup_prompt_len = min(prompt_len, 128)
    warmup_max_tokens = min(max_tokens, 8)
    warmup_prompts = _make_token_prompts(num_seqs=warmup_num_seqs, prompt_len=warmup_prompt_len, seed=seed + 1)
    warmup_sps = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=warmup_max_tokens)
        for _ in range(warmup_num_seqs)
    ]
    llm.generate(warmup_prompts, warmup_sps, use_tqdm=False)
    torch.cuda.synchronize()

    prompt_token_ids = _make_token_prompts(num_seqs=num_seqs, prompt_len=prompt_len, seed=seed + 2)
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=max_tokens) for _ in range(num_seqs)]
    for p, sp in zip(prompt_token_ids, sampling_params):
        llm.add_request(p, sp)

    prefill_tokens = 0
    prefill_time_s = 0.0
    decode_tokens = 0
    decode_time_s = 0.0

    t_total0 = perf_counter()
    while not llm.is_finished():
        t0 = perf_counter()
        _outputs, n = llm.step()
        dt = perf_counter() - t0
        if n > 0:
            prefill_tokens += n
            prefill_time_s += dt
        else:
            decode_tokens += -n
            decode_time_s += dt
    total_time_s = perf_counter() - t_total0

    def _safe_div(num: float, den: float) -> float:
        return float(num / den) if den > 0 else 0.0

    sample_sps = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=64) for _ in _sample_prompts()]
    sample_outputs = llm.generate(_sample_prompts(), sample_sps, use_tqdm=False)

    return {
        "engine": "nano-vllm",
        "env": _env_info(),
        "config": {
            "model": model,
            "num_seqs": num_seqs,
            "prompt_len": prompt_len,
            "max_tokens": max_tokens,
            "max_model_len": max_model_len,
            "max_num_batched_tokens": max_num_batched_tokens,
            "attn_backend": attn_backend,
            "seed": seed,
        },
        "timing": {"init_time_s": float(init_time_s), "total_time_s": float(total_time_s)},
        "results": {
            "prefill_tokens": int(prefill_tokens),
            "prefill_time_s": float(prefill_time_s),
            "prefill_tok_s": _safe_div(prefill_tokens, prefill_time_s),
            "decode_tokens": int(decode_tokens),
            "decode_time_s": float(decode_time_s),
            "decode_tok_s": _safe_div(decode_tokens, decode_time_s),
            "total_tok_s": _safe_div(prefill_tokens + decode_tokens, total_time_s),
            "output_tok_s": _safe_div(decode_tokens, total_time_s),
        },
        "samples": [
            {"prompt": p, "text": o["text"]} for p, o in zip(_sample_prompts(), sample_outputs)
        ],
    }


@torch.inference_mode()
def bench_minisgl(
    *,
    model: str,
    num_seqs: int,
    prompt_len: int,
    max_tokens: int,
    max_model_len: int,
    max_num_batched_tokens: int,
    seed: int,
) -> dict[str, Any]:
    from minisgl.core import SamplingParams
    from minisgl.llm import LLM

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    t_init0 = perf_counter()
    llm = LLM(
        model,
        max_running_req=num_seqs,
        max_seq_len_override=max_model_len,
        max_extend_tokens=max_num_batched_tokens,
        cuda_graph_max_bs=min(num_seqs, 256),
    )
    init_time_s = perf_counter() - t_init0

    llm.generate(["Benchmark"], SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=8))
    torch.cuda.synchronize()

    prompt_token_ids = _make_token_prompts(num_seqs=num_seqs, prompt_len=prompt_len, seed=seed + 2)
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=max_tokens) for _ in range(num_seqs)]

    t0 = perf_counter()
    _outputs = llm.generate(prompt_token_ids, sampling_params)
    total_time_s = perf_counter() - t0

    total_tokens = num_seqs * (prompt_len + max_tokens)
    output_tokens = num_seqs * max_tokens

    sample_sps = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=64) for _ in _sample_prompts()]
    sample_outputs = llm.generate(_sample_prompts(), sample_sps)

    return {
        "engine": "mini-sglang",
        "env": _env_info(extra={"minisgl_version": getattr(__import__("minisgl"), "__version__", "unknown")}),
        "config": {
            "model": model,
            "num_seqs": num_seqs,
            "prompt_len": prompt_len,
            "max_tokens": max_tokens,
            "max_model_len": max_model_len,
            "max_num_batched_tokens": max_num_batched_tokens,
            "seed": seed,
            "page_size": 1,
        },
        "timing": {"init_time_s": float(init_time_s), "total_time_s": float(total_time_s)},
        "results": {
            "total_tokens": int(total_tokens),
            "output_tokens": int(output_tokens),
            "total_tok_s": float(total_tokens / total_time_s),
            "output_tok_s": float(output_tokens / total_time_s),
        },
        "samples": [
            {"prompt": p, "text": o["text"]} for p, o in zip(_sample_prompts(), sample_outputs)
        ],
    }


@torch.inference_mode()
def bench_vllm(
    *,
    model: str,
    num_seqs: int,
    prompt_len: int,
    max_tokens: int,
    max_model_len: int,
    max_num_batched_tokens: int,
    seed: int,
) -> dict[str, Any]:
    from vllm import LLM, SamplingParams

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    t_init0 = perf_counter()
    llm = LLM(
        model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=max_model_len,
        max_num_seqs=num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        gpu_memory_utilization=0.9,
        enforce_eager=False,
        seed=seed,
        disable_log_stats=True,
    )
    init_time_s = perf_counter() - t_init0

    llm.generate(["Benchmark"], SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=8), use_tqdm=False)

    prompt_token_ids = _make_token_prompts(num_seqs=num_seqs, prompt_len=prompt_len, seed=seed + 2)
    prompts = [{"prompt_token_ids": p} for p in prompt_token_ids]
    sp = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=max_tokens, seed=seed)

    t0 = perf_counter()
    _outputs = llm.generate(prompts, sp, use_tqdm=False)
    total_time_s = perf_counter() - t0

    total_tokens = num_seqs * (prompt_len + max_tokens)
    output_tokens = num_seqs * max_tokens

    sample_outputs = llm.generate(
        _sample_prompts(),
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=64, seed=seed),
        use_tqdm=False,
    )
    samples = []
    for p, o in zip(_sample_prompts(), sample_outputs):
        samples.append({"prompt": p, "text": o.outputs[0].text})

    return {
        "engine": "vllm",
        "env": _env_info(extra={"vllm_version": getattr(__import__("vllm"), "__version__", "unknown")}),
        "config": {
            "model": model,
            "num_seqs": num_seqs,
            "prompt_len": prompt_len,
            "max_tokens": max_tokens,
            "max_model_len": max_model_len,
            "max_num_batched_tokens": max_num_batched_tokens,
            "seed": seed,
        },
        "timing": {"init_time_s": float(init_time_s), "total_time_s": float(total_time_s)},
        "results": {
            "total_tokens": int(total_tokens),
            "output_tokens": int(output_tokens),
            "total_tok_s": float(total_tokens / total_time_s),
            "output_tok_s": float(output_tokens / total_time_s),
        },
        "samples": samples,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark nano-vllm vs mini-sglang vs vLLM (B200-friendly).")
    p.add_argument("--engine", type=str, required=True, choices=["nanovllm", "minisgl", "vllm"])
    p.add_argument("--model", type=str, default="/root/huggingface/Qwen3-0.6B")
    p.add_argument("--num-seqs", type=int, default=256)
    p.add_argument("--prompt-len", type=int, default=1024)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--max-num-batched-tokens", type=int, default=16384)
    p.add_argument("--attn-backend", type=str, default="flashinfer", help="nano-vllm only: auto|flashinfer|flashattn")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    if args.engine == "nanovllm":
        payload = bench_nanovllm(
            model=args.model,
            num_seqs=args.num_seqs,
            prompt_len=args.prompt_len,
            max_tokens=args.max_tokens,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_num_batched_tokens,
            attn_backend=args.attn_backend,
            seed=args.seed,
        )
    elif args.engine == "minisgl":
        payload = bench_minisgl(
            model=args.model,
            num_seqs=args.num_seqs,
            prompt_len=args.prompt_len,
            max_tokens=args.max_tokens,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_num_batched_tokens,
            seed=args.seed,
        )
    else:
        payload = bench_vllm(
            model=args.model,
            num_seqs=args.num_seqs,
            prompt_len=args.prompt_len,
            max_tokens=args.max_tokens,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_num_batched_tokens,
            seed=args.seed,
        )

    print(json.dumps(payload, indent=2))
    _write_json(args.out, payload)


if __name__ == "__main__":
    main()

