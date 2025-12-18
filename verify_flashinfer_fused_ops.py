import torch

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.rotary_embedding import RotaryEmbedding, apply_rotary_emb


def _max_abs_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item()) if a.numel() else 0.0


@torch.inference_mode()
def verify_rope() -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    head_dim = 64
    max_pos = 4096
    n = 8192

    rope = RotaryEmbedding(
        head_size=head_dim,
        rotary_dim=head_dim,
        max_position_embeddings=max_pos,
        base=1_000_000,
    ).to(device)

    positions = torch.randint(0, max_pos, (n,), device=device, dtype=torch.int64)
    q = torch.randn(n, 16, head_dim, device=device, dtype=dtype)
    k = torch.randn(n, 8, head_dim, device=device, dtype=dtype)

    q_fi, k_fi = rope(positions, q.clone(), k.clone())

    cos_sin = rope.cos_sin_cache[positions]  # [N, D*2], fp32
    cos, sin = cos_sin.chunk(2, dim=-1)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_ref = apply_rotary_emb(q, cos, sin)
    k_ref = apply_rotary_emb(k, cos, sin)

    q_err = _max_abs_err(q_fi, q_ref)
    k_err = _max_abs_err(k_fi, k_ref)
    print(f"rope max_abs_err(q)={q_err:.3e} max_abs_err(k)={k_err:.3e}")


@torch.inference_mode()
def verify_rmsnorm() -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    n, h = 8192, 1024
    eps = 1e-6

    mod = RMSNorm(h, eps=eps).to(device=device, dtype=dtype)
    mod.weight.copy_(torch.randn_like(mod.weight))

    x = torch.randn(n, h, device=device, dtype=dtype)

    # No residual
    y = mod(x)
    x_f = x.float()
    var = x_f.pow(2).mean(dim=-1, keepdim=True)
    y_ref = (x_f * torch.rsqrt(var + eps)).to(dtype) * mod.weight
    err = _max_abs_err(y, y_ref)
    print(f"rmsnorm(no_residual) max_abs_err={err:.3e}")

    # With residual: prefer fused_add_rmsnorm path (2D)
    x_in = torch.randn(n, h, device=device, dtype=dtype)
    residual_in = torch.randn(n, h, device=device, dtype=dtype)
    x_out, residual_out = mod(x_in.clone(), residual_in.clone())

    residual_ref = (x_in + residual_in).to(dtype)
    tmp = residual_ref.float()
    var = tmp.pow(2).mean(dim=-1, keepdim=True)
    y_ref = (tmp * torch.rsqrt(var + eps)).to(dtype) * mod.weight

    err_y = _max_abs_err(x_out, y_ref)
    err_res = _max_abs_err(residual_out, residual_ref)
    print(f"rmsnorm(with_residual) max_abs_err(y)={err_y:.3e} max_abs_err(res)={err_res:.3e}")


@torch.inference_mode()
def verify_silu_and_mul() -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    n, h = 8192, 3072

    mod = SiluAndMul().to(device)
    x = torch.randn(n, 2 * h, device=device, dtype=dtype)
    y = mod(x)

    x1, x2 = x.chunk(2, dim=-1)
    y_ref = torch.nn.functional.silu(x1) * x2
    err = _max_abs_err(y, y_ref)
    print(f"silu_and_mul max_abs_err={err:.3e}")


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")

    verify_rope()
    verify_rmsnorm()
    verify_silu_and_mul()
    print("OK")


if __name__ == "__main__":
    main()

