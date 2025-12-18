import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _run_one(engine: str, args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    engine_out = out_dir / f"compare_{engine}.json"

    cmd = [
        sys.executable,
        "bench_engines.py",
        "--engine",
        engine,
        "--model",
        args.model,
        "--num-seqs",
        str(args.num_seqs),
        "--prompt-len",
        str(args.prompt_len),
        "--max-tokens",
        str(args.max_tokens),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--seed",
        str(args.seed),
        "--out",
        str(engine_out),
    ]
    if engine == "nanovllm":
        cmd += ["--attn-backend", args.attn_backend]
    subprocess.run(cmd, check=True)
    return json.loads(engine_out.read_text())


def main() -> None:
    p = argparse.ArgumentParser(description="Run all engine benches and aggregate results.")
    p.add_argument("--model", type=str, default="/root/huggingface/Qwen3-0.6B")
    p.add_argument("--num-seqs", type=int, default=256)
    p.add_argument("--prompt-len", type=int, default=1024)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--max-num-batched-tokens", type=int, default=16384)
    p.add_argument("--attn-backend", type=str, default="flashinfer", help="nano-vllm only")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="results/compare_latest.json")
    args = p.parse_args()

    results = {}
    for engine in ["nanovllm", "minisgl", "vllm"]:
        results[engine] = _run_one(engine, args)

    payload = {
        "config": {
            "model": args.model,
            "num_seqs": args.num_seqs,
            "prompt_len": args.prompt_len,
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "attn_backend": args.attn_backend,
            "seed": args.seed,
        },
        "results": results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
