import torch

import triton
import triton.language as tl
from triton.compiler import ASTSource, compile
import inspect


DEVICE = torch.device('cuda')
import pdb

@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               ):

    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y

    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    # We need to preallocate the output.
    output = torch.empty_like(x).to(DEVICE)
    assert x.device == y.device and output.device == x.device, \
    f"Device mismatch: x.device={x.device}, y.device={y.device}, output.device={output.device}, expected={DEVICE}"
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


source = ASTSource(
    fn=add_kernel,
    signature={
        "x_ptr": "*fp32",
        "y_ptr": "*fp32",
        "output_ptr": "*fp32",
        "n_elements": "i32",
        "BLOCK_SIZE": "constexpr",
    },
    constexprs={(4,): 1024},
)
# 원래에 있었던 것들
# attrs={(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}
compiled = compile(source)

# print("=== TTIR (SSA IR) ===")
# print(compiled.asm['ttir'])

