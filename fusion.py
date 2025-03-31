import torch
import triton
import triton.language as tl
import triton.testing


@triton.jit
def fused_gemm_bn_relu(
    A, B, C,
    gamma, beta, mean, var,
    M, N, K, eps,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(A + offs_m[:, None] * K + offs_k[None, :], mask=a_mask, other=0.0)
        b = tl.load(B + offs_k[:, None] * N + offs_n[None, :], mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

    # BatchNorm parameters
    mean_v = tl.load(mean + offs_n, mask=offs_n < N, other=0.0)
    var_v = tl.load(var + offs_n, mask=offs_n < N, other=0.0)
    gamma_v = tl.load(gamma + offs_n, mask=offs_n < N, other=1.0)
    beta_v = tl.load(beta + offs_n, mask=offs_n < N, other=0.0)

    norm = (acc - mean_v[None, :]) / tl.sqrt(var_v[None, :] + eps)
    out = tl.maximum(norm * gamma_v[None, :] + beta_v[None, :], 0)

    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C + offs_m[:, None] * N + offs_n[None, :], out, mask=c_mask)


def run_benchmark(M=1024, N=1024, K=1024):
    print(f"\n🚀 Benchmarking GEMM + BatchNorm + ReLU — M={M}, N={N}, K={K}")
    torch.manual_seed(0)

    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    gamma = torch.randn(N, device="cuda", dtype=torch.float32)
    beta = torch.randn(N, device="cuda", dtype=torch.float32)
    mean = torch.randn(N, device="cuda", dtype=torch.float32)
    var = torch.rand(N, device="cuda", dtype=torch.float32)

    def torch_impl():
        out = torch.matmul(a, b)
        out = (out - mean) / torch.sqrt(var + 1e-5)
        out = out * gamma + beta
        return torch.relu(out)

    def triton_impl():
        c = torch.empty((M, N), device="cuda", dtype=torch.float32)
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
        fused_gemm_bn_relu[grid](
            a, b, c,
            gamma, beta, mean, var,
            M, N, K, 1e-5,
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
        )
        return c

    ms_torch = triton.testing.do_bench(torch_impl)
    ms_triton = triton.testing.do_bench(triton_impl)

    print(f"✅ PyTorch     : {ms_torch:.3f} ms")
    print(f"✅ Triton Fused: {ms_triton:.3f} ms")


if __name__ == "__main__":
    run_benchmark()