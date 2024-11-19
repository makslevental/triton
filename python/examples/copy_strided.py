import numpy as np
import torch
from numpy.random import RandomState

import triton
import triton.language as tl
import triton.compiler as tc


@triton.jit
def copy_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    output = input
    # tl.store(output_ptr + offsets, output, mask=mask)
    tl.amd_buffer_store(output_ptr, offsets, output)


BLOCK_SIZE = 64

kernel = triton.compile(
    triton.compiler.ASTSource(
        fn=copy_kernel,
        signature={
            "input_ptr": "*fp32",
            "output_ptr": "*fp32",
            "n_elements": "i32",
        },
        constants={"BLOCK_SIZE": BLOCK_SIZE},
    )
)
print(kernel.asm["ttgir"])

rs = RandomState(17)
inp_cpu = rs.randint(-100, 100, (BLOCK_SIZE,)).astype(np.float32)

inp = torch.tensor(inp_cpu, device="cuda")
outp = torch.zeros((BLOCK_SIZE,), device="cuda", dtype=torch.float32)

kernel[(1, 1, 1)](inp, outp, BLOCK_SIZE)

np.testing.assert_equal(inp_cpu, outp.cpu().numpy())
