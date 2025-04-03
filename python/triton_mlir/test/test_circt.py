#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from textwrap import dedent

import numpy as np
import pytest
from triton_mlir.compiler import unwrap_c_module_op, unwrap_c_op
from triton_mlir.dialects import tt, verif, ttpp, smt, builtin
from triton_mlir.extras.context import mlir_mod_ctx

# this needs to be below the triton_mlir_bindings
from triton_mlir.extras.dialects.ext import arith
from triton_mlir.extras.dialects.ext import scf
from triton_mlir.extras.dialects.ext.func import func

# noinspection PyUnresolvedReferences
from triton_mlir.extras.testing import mlir_ctx as ctx, filecheck, MLIRContext
from triton_mlir.extras.util import find_ops
from triton_mlir.ir import InsertionPoint
from triton_mlir.passmanager import PassManager
from triton_mlir.types import T

pytest.mark.usefixtures("ctx")


def test_vadd(ctx):
    @func(emit=True)
    def kernel_0123(v0: T.int32, v1: T.int32):
        c32 = arith.constant(64, T.int32)
        v2 = v0 * c32
        v3 = arith.addi(v0, v2)

    print("before")
    print(ctx.module)
    pm = PassManager.parse("builtin.module(convert-arith-to-smt)")
    pm.run(ctx.module.operation)
    print("after")
    print(ctx.module)


autotune_configs = [
    {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 16,
        "GROUP_SIZE_M": 1,
        "WAVES_PER_EU": 2,
        "NUM_WARPS": 4,
    },
    {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 16,
        "GROUP_SIZE_M": 4,
        "WAVES_PER_EU": 2,
        "NUM_WARPS": 4,
    },
    {
        "BLOCK_SIZE_M": 256,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 16,
        "GROUP_SIZE_M": 4,
        "WAVES_PER_EU": 2,
        "NUM_WARPS": 8,
    },
    {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 1,
        "WAVES_PER_EU": 2,
        "NUM_WARPS": 8,
    },
    {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
        "WAVES_PER_EU": 3,
        "NUM_WARPS": 4,
    },
    {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 1,
        "WAVES_PER_EU": 8,
        "NUM_WARPS": 4,
    },
    {
        "BLOCK_SIZE_M": 8,
        "BLOCK_SIZE_N": 8,
        "BLOCK_SIZE_K": 4,
        "GROUP_SIZE_M": 1,
        "WAVES_PER_EU": 8,
        "NUM_WARPS": 4,
    },
]


@pytest.mark.parametrize("autotune_config", autotune_configs[:1])
def test_verify(ctx, autotune_config):
    BLOCK_SIZE_M = autotune_config["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = autotune_config["BLOCK_SIZE_N"]
    BLOCK_SIZE_K = autotune_config["BLOCK_SIZE_K"]
    GROUP_SIZE_M = autotune_config["GROUP_SIZE_M"]

    @tt.jit(emit=True)
    def matmul_kernel_3(
        a_ptr: +T.f32,
        b_ptr: +T.f32,
        c_ptr: +T.f32,
        M: T.i32,
        N: T.i32,
        K: T.i32,
        stride_am: T.i32,
        stride_ak: T.i32,
        stride_bk: T.i32,
        stride_bn: T.i32,
        stride_cm: T.i32,
        stride_cn: T.i32,
    ):
        BLOCK_SIZE_M_ = arith.constant(BLOCK_SIZE_M, T.i32)
        BLOCK_SIZE_N_ = arith.constant(BLOCK_SIZE_N, T.i32)
        BLOCK_SIZE_K_ = arith.constant(BLOCK_SIZE_K, T.i32)
        GROUP_SIZE_M_ = arith.constant(GROUP_SIZE_M, T.i32)

        pid = tt.get_program_id(axis=0)
        one = arith.constant(1, T.i32)

        @verif.contract(inputs=[pid, pid], outputs=[T.i32, T.i32])
        def pid_m_pid_n():
            smt.assert_(builtin.unrealized_conversion_cast([smt.bool_t()], [pid > 1]))

            @smt.forall(bound_var_types=[smt.bv_t(N.type.width)], bound_var_names=["N"])
            def forall(n):
                num_pid_n = arith.ceildivsi(N, BLOCK_SIZE_N_)
                num_pid_in_group = GROUP_SIZE_M * num_pid_n
                group_id = pid / num_pid_in_group
                first_pid_m = group_id * GROUP_SIZE_M
                pid_m = first_pid_m + ((pid % num_pid_in_group) % GROUP_SIZE_M)
                pid_n = (pid % num_pid_in_group) / GROUP_SIZE_M
                lhs = builtin.unrealized_conversion_cast([smt.bool_t()], [pid_m > 1])
                rhs = builtin.unrealized_conversion_cast([smt.bool_t()], [pid_n > 1])
                smt.yield_([smt.and_([lhs, rhs])])

        pid_m, pid_n = pid_m_pid_n

        offs_am = (pid_m * BLOCK_SIZE_M + tt.make_range(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tt.make_range(0, BLOCK_SIZE_N)) % N
        offs_k = tt.make_range(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        accum = arith.constant(np.full([BLOCK_SIZE_M, BLOCK_SIZE_N], 0.0, np.float32))
        stop = arith.ceildivsi(K, BLOCK_SIZE_K_)
        for k, [accum, a_ptrs, b_ptrs], _ in scf.range_(
            0, stop, 1, iter_args=[accum, a_ptrs, b_ptrs]
        ):
            a = tt.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tt.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            accum += tt.dot(a, b)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
            accum, *_ = scf.yield_(accum, a_ptrs, b_ptrs)

        c = accum

        offs_cm = pid_m * BLOCK_SIZE_M + tt.make_range(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tt.make_range(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tt.store(c_ptrs, c, mask=c_mask)

    print("before")
    print(ctx.module)
    print(ctx.module.operation.verify())

    pm = PassManager.parse("builtin.module(lower-contracts)")
    pm.run(ctx.module.operation)

    print("after")
    print(ctx.module)

    pm = PassManager.parse(
        "builtin.module(convert-arith-to-smt,reconcile-unrealized-casts)"
    )
    pm.run(ctx.module.operation)
    print(ctx.module)

    formals = find_ops(
        ctx.module.operation, lambda o: isinstance(o.opview, verif.FormalOp)
    )
    assert len(formals) == 1
    formal = formals[0]

    symbols = find_ops(
        formal.operation, lambda o: isinstance(o.opview, verif.SymbolicValueOp)
    )
    syms_to_smt_types = []
    for s in symbols:
        uses = list(s.result.uses)
        assert len(uses) == 1
        use = uses[0]
        assert isinstance(use.owner, builtin.UnrealizedConversionCastOp)
        syms_to_smt_types.append((s, use.owner.result))

    @builtin.module
    def mod():
        @smt.solver
        def solver(block):
            _forall = None
            for op in formal.body.blocks[0].operations:
                block.append(op)
                if isinstance(op, smt.ForallOp):
                    _forall = op
                elif isinstance(op, tt.GetProgramIdOp):
                    pid = smt.declare_fun(smt.bv_t(32), name_prefix="pid")
                    op.result.replace_all_uses_with(pid)
                    op.erase()

            for i, arg in enumerate(_forall.body.blocks[0].arguments):
                s, r = syms_to_smt_types[i]
                r.replace_all_uses_with(arg)
                r.owner.erase()
                s.erase()

            smt.assert_(_forall.result)

            smt.check()

            smt.yield_([])

    pm = PassManager.parse("builtin.module(canonicalize,symbol-dce)")
    pm.run(mod.operation)
    print(mod)
    print(mod.operation.verify())

    smtlib_query = smt.export_smtlib(unwrap_c_op(mod.operation))
    correct = dedent(
        """\
    ; solver scope 0
    (declare-const pid (_ BitVec 32))
    (assert (let ((tmp (bvsgt pid #x00000001)))
            tmp))
    (declare-const tmp_0 (_ BitVec 32))
    (declare-const tmp_1 (_ BitVec 32))
    (declare-const tmp_2 (_ BitVec 32))
    (assert (let ((tmp_3 (forall ((N (_ BitVec 32)))
                         (let ((tmp_4 (bvneg #x00000001)))
                                 (let ((tmp_5 (bvadd N #x00000100)))
                                 (let ((tmp_6 (bvadd tmp_5 tmp_4)))
                                 (let ((tmp_7 (bvsdiv tmp_6 #x00000100)))
                                 (let ((tmp_8 (= #x00000100 #x00000000)))
                                 (let ((tmp_9 (ite tmp_8 tmp_0 tmp_7)))
                                 (let ((tmp_10 (bvsrem pid tmp_9)))
                                 (let ((tmp_11 (= tmp_9 #x00000000)))
                                 (let ((tmp_12 (ite tmp_11 tmp_2 tmp_10)))
                                 (let ((tmp_13 (bvsgt tmp_12 #x00000001)))
                                 (let ((tmp_14 (bvsdiv pid tmp_9)))
                                 (let ((tmp_15 (= tmp_9 #x00000000)))
                                 (let ((tmp_16 (ite tmp_15 tmp_1 tmp_14)))
                                 (let ((tmp_17 (bvadd tmp_16 #x00000000)))
                                 (let ((tmp_18 (bvsgt tmp_17 #x00000001)))
                                 (let ((tmp_19 (and tmp_18 tmp_13)))
                                 tmp_19)))))))))))))))))))
            tmp_3))
    (check-sat)
    (reset)
    """
    )
    assert correct == smtlib_query

    import z3

    s = z3.Solver()
    s.from_string(smtlib_query)
    assert s.check() == z3.sat


if __name__ == "__main__":
    with mlir_mod_ctx(allow_unregistered_dialects=True) as ctx:
        test_verify(ctx, autotune_configs[0])
