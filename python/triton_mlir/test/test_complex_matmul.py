from pathlib import Path

import pytest
from triton_mlir.ir import Module, WalkOrder

from python.triton_mlir.test.util import (
    print_prolog,
    generic_print_walk_callback,
    print_epilog,
    OUTPUT_BUF,
)

from triton_mlir.extras.testing import mlir_ctx as ctx, filecheck, MLIRContext

pytest.mark.usefixtures("ctx")

try:
    import triton
    import triton.language as tl
except ImportError as e:
    pass


def make_triton_jitted():

    @triton.jit
    def matmul_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        bias_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_bias,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        SPLIT_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        BIAS: tl.constexpr,
        EVEN_K: tl.constexpr,
        GRID_MN: tl.constexpr,
        NUM_XCDS: tl.constexpr,
    ):

        tl.assume(stride_am > 0)
        tl.assume(stride_ak > 0)
        tl.assume(stride_bk > 0)
        tl.assume(stride_bn > 0)
        tl.assume(stride_cm > 0)
        tl.assume(stride_cn > 0)
        tl.assume(stride_bias > 0)
        tl.assume(K >= 1)
        tl.assume(K >= BLOCK_SIZE_K * SPLIT_K)

        pid = tl.program_id(axis=0)
        pid_z = tl.program_id(1)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

        if NUM_XCDS != 1:
            ## pid remapping on xcds
            # Number of pids per XCD in the new arrangement
            pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
            # Compute current XCD and local pid within the XCD
            xcd = pid % NUM_XCDS
            local_pid = pid // NUM_XCDS
            # Calculate new pid based on the new grouping
            pid = xcd * pids_per_xcd + local_pid

        if GROUP_SIZE_M == 1:
            pid_m = pid // num_pid_n
            pid_n = pid % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m

        tl.assume(pid_m > 0)
        tl.assume(pid_n > 0)

        if SPLIT_K == 1:
            offs_k = tl.arange(0, BLOCK_SIZE_K)
        else:
            offs_k = pid_z * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        if BIAS:
            bias_ptrs = bias_ptr + offs_am * stride_bias
            bias = tl.load(bias_ptrs, mask=offs_am < M, other=0.0)
        acc_dtype = tl.float32 if a_ptr.type.element_ty != tl.int8 else tl.int32
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        k = tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)
        tl.assume(k >= 1)
        for k in range(0, k):
            if EVEN_K:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
            else:
                a = tl.load(
                    a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0
                )
                b = tl.load(
                    b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0
                )
            accumulator += tl.dot(a, b)
            a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk
        c = accumulator.to(c_ptr.type.element_ty)
        if BIAS:
            c += bias[:, None]
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        if SPLIT_K == 1:
            tl.store(c_ptrs, c, mask=c_mask)
        else:
            tl.atomic_add(c_ptrs, c, mask=c_mask)

    kernel = triton.compile(
        triton.compiler.ASTSource(
            fn=matmul_kernel,
            signature={
                "a_ptr": "*fp32",
                "b_ptr": "*fp32",
                "c_ptr": "*fp32",
                "bias_ptr": "*fp32",
                "M": "i32",
                "N": "i32",
                "K": "i32",
                "stride_am": "i32",
                "stride_ak": "i32",
                "stride_bk": "i32",
                "stride_bn": "i32",
                "stride_cm": "i32",
                "stride_cn": "i32",
                "stride_bias": "i32",
                "BLOCK_SIZE_M": "constexpr",
                "BLOCK_SIZE_N": "constexpr",
                "BLOCK_SIZE_K": "constexpr",
                "SPLIT_K": "constexpr",
                "GROUP_SIZE_M": "constexpr",
                "BIAS": "constexpr",
                "EVEN_K": "constexpr",
                "GRID_MN": "constexpr",
                "NUM_XCDS": "constexpr",
            },
            constexprs={
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
                "SPLIT_K": 1,
                "GROUP_SIZE_M": 8,
                "BIAS": 0,
                "EVEN_K": 1,
                "GRID_MN": 64,
                "NUM_XCDS": 4,
            },
        )
    )

    with open("complex_matmul.ttir", "w") as f:
        f.write(kernel.asm["ttir"])


HERE = Path(__file__).parent


def test_dump_kernel(ctx):

    # make_triton_jitted()

    with open(HERE / "complex_matmul.ttir") as f:
        mod = Module.parse(f.read())
    assert mod.operation.verify()

    print_prolog()
    mod.operation.walk(generic_print_walk_callback, WalkOrder.PRE_ORDER)
    print_epilog()
    OUTPUT_BUF.seek(0)
    with open(HERE / "complex_matmul.py", "w") as f:
        f.write(OUTPUT_BUF.read())
    OUTPUT_BUF.seek(0)

    # with open("complex_matmul.ttgir", "w") as f:
    #     f.write(kernel.asm["ttgir"])
