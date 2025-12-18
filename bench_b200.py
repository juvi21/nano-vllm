import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from nanovllm import LLM, SamplingParams


def _env_info() -> dict[str, Any]:
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
    return info


@torch.inference_mode()
def run_benchmark(
    *,
    model: str,
    num_seqs: int,
    prompt_len: int,
    max_tokens: int,
    max_model_len: int,
    max_num_batched_tokens: int,
    enforce_eager: bool,
    attn_backend: str,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    llm = LLM(
        model,
        enforce_eager=enforce_eager,
        max_model_len=max_model_len,
        max_num_seqs=num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        attn_backend=attn_backend,
    )

    # Warmup: run a small (but representative) request to trigger compilation/JIT paths.
    warmup_num_seqs = min(num_seqs, 8)
    warmup_prompt_len = min(prompt_len, 128)
    warmup_max_tokens = min(max_tokens, 8)
    warmup_prompts = torch.randint(
        low=0, high=10_000, size=(warmup_num_seqs, warmup_prompt_len), dtype=torch.int64
    ).tolist()
    warmup_sps = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=warmup_max_tokens)
        for _ in range(warmup_num_seqs)
    ]
    llm.generate(warmup_prompts, warmup_sps, use_tqdm=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    prompt_token_ids = torch.randint(low=0, high=10_000, size=(num_seqs, prompt_len), dtype=torch.int64).tolist()
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

    return {
        "env": _env_info(),
        "config": {
            "model": model,
            "num_seqs": num_seqs,
            "prompt_len": prompt_len,
            "max_tokens": max_tokens,
            "max_model_len": max_model_len,
            "max_num_batched_tokens": max_num_batched_tokens,
            "enforce_eager": enforce_eager,
            "attn_backend": attn_backend,
            "seed": seed,
        },
        "results": {
            "prefill_tokens": int(prefill_tokens),
            "prefill_time_s": float(prefill_time_s),
            "prefill_tok_s": _safe_div(prefill_tokens, prefill_time_s),
            "decode_tokens": int(decode_tokens),
            "decode_time_s": float(decode_time_s),
            "decode_tok_s": _safe_div(decode_tokens, decode_time_s),
            "total_time_s": float(total_time_s),
            "total_tok_s": _safe_div(prefill_tokens + decode_tokens, total_time_s),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Nano-vLLM benchmark runner (B200/SM100 friendly).")
    p.add_argument("--model", type=str, default="/root/huggingface/Qwen3-0.6B")
    p.add_argument("--num-seqs", type=int, default=256)
    p.add_argument("--prompt-len", type=int, default=1024)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-num-batched-tokens", type=int, default=16384)
    p.add_argument("--enforce-eager", action="store_true", help="Disable CUDA graph path.")
    p.add_argument("--attn-backend", type=str, default="auto", help="auto|flashinfer|flashattn")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="", help="Optional JSON output path.")
    args = p.parse_args()

    result = run_benchmark(
        model=args.model,
        num_seqs=args.num_seqs,
        prompt_len=args.prompt_len,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enforce_eager=args.enforce_eager,
        attn_backend=args.attn_backend,
        seed=args.seed,
    )

    print(json.dumps(result, indent=2))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
