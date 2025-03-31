import torch
import triton
import triton.language as tl

DEVICE = torch.device('cuda')
@triton.jit
def tile_scale_kernel(X_ptr, Y_ptr, M, N, stride_xm, stride_xn, stride_ym, stride_yn,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)  # 행 방향 타일 ID
    pid_n = tl.program_id(1)  # 열 방향 타일 ID

    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    # Block pointer 생성 (row-major layout)
    x_block_ptr = tl.make_block_ptr(
        base=X_ptr,
        shape=(M, N),
        strides=(stride_xm, stride_xn),
        offsets=(off_m, off_n),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(0, 1)  # row-major
    )

    y_block_ptr = tl.make_block_ptr(
        base=Y_ptr,
        shape=(M, N),
        strides=(stride_ym, stride_yn),
        offsets=(off_m, off_n),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(0, 1)
    )

    # 타일 단위로 로드 및 연산
    x = tl.load(x_block_ptr, boundary_check=(0, 1))
    y = x * 2.0
    tl.store(y_block_ptr, y, boundary_check=(0, 1))

@triton.jit
def scale_kernel(X_ptr, Y_ptr, BLOCKSIZE: tl.constexpr):
    pid = tl.program_id(0)  # 행 방향 타일 ID

    off = pid * BLOCKSIZE + tl.arange(0, BLOCKSIZE)

    # 타일 단위로 로드 및 연산
    x = tl.load(X_ptr, boundary_check=(0))
    y = x * 2.0
    tl.store(Y_ptr, y, boundary_check=(0))
    
# ----------------------------------
# 실행 예제
# ----------------------------------
def scale(x: torch.Tensor, y: torch.Tensor):
    # We need to preallocate the output.
    n_elements = y.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    scale_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


# 1. 입력 행렬 (M x N)
M, N = 512, 512
x = torch.randn((M, N), device='cuda', dtype=torch.float32)
y = torch.empty_like(x)

# 2. Block size 설정 (tile 크기)
BLOCK_M = 32
BLOCK_N = 64
BLOCKSIZE = BLOCK_M * BLOCK_N
# 3. Grid 설정: (ceil-div 방식으로 타일 수 지정)
grid = (
    triton.cdiv(M*N, BLOCK_M*BLOCK_N),
)
grid_tile = (
    triton.cdiv(M, BLOCK_M),
    triton.cdiv(N, BLOCK_N)
)

# 4. 커널 실행
tile_scale_kernel[grid](
    x, y, M, N,
    x.stride(0), x.stride(1),
    y.stride(0), y.stride(1),
    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(12, 28, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='vector-add-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(size, provider):
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: scale_kernel[grid](x,y,BLOCKSIZE), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: tile_scale_kernel[grid](x, y, M, N, x.stride(0), x.stride(1), y.stride(0), y.stride(1), BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N), quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


benchmark.run(print_data=True, save_path="./test2")
