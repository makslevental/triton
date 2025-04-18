import numpy as np
import pytest
from triton_mlir.compiler import (
    HIPBackend,
    unwrap_c_module_op,
    tritonir,
    llvm,
    make_backend,
)
from triton_mlir.dialects import tt, linalg
from triton_mlir.dialects.tt import (
    PointerType,
    broadcast,
    splat,
    load,
    dot,
    addptr,
    expand_dims,
    store,
    get_program_id,
    make_range,
)
from triton_mlir.extras.context import mlir_mod_ctx
from triton_mlir.dialects.builtin import module
from triton_mlir.dialects.gpu import MappingId
from triton_mlir.dialects.transform import (
    get_parent_op,
    apply_cse,
    apply_licm,
    any_op_t,
)
from triton_mlir.dialects.transform.extras import named_sequence, apply_patterns
from triton_mlir.ir import UnitAttr, StringAttr
from triton_mlir.extras.dialects.ext.transform import (
    match,
    tile_to_scf_for,
    tile_to_scf_forall,
    split_handle,
    include,
    transform_op_t,
    transform_any_op_t,
    get_producer_of_operand,
    get_consumers_of_result,
)
from triton_mlir.extras.dialects.ext import transform
from triton_mlir.extras.runtime.passes import run_pipeline, Pipeline
from triton_mlir.dialects.transform.structured import MatchInterfaceEnum
from triton_mlir.extras.dialects.ext.gpu import block_attr, thread_attr


# this needs to be below the triton_mlir_bindings
from triton_mlir.extras.dialects.ext import arith

# noinspection PyUnresolvedReferences
from triton_mlir.extras.testing import mlir_ctx as ctx, filecheck, MLIRContext
from triton_mlir.ir import ArrayAttr, AffineMap
from triton_mlir.types import T
from triton_mlir.passmanager import PassManager


pytest.mark.usefixtures("ctx")


def test_inline_mod_linalg_generic_transform(ctx):
    M, K, N = 512, 512, 512
    BS_M = 64
    BS_N = 64
    BS_K = 64
    GS_M = 1

    @tt.jit(
        arg_attrs=ArrayAttr.parse(
            "[{tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, {}, {}, {}, {}, {}, {}, {}, {}, {}]"
        ),
        noinline=False,
        sym_name="matmul_kernel_2",
        sym_visibility="public",
    )
    def matmul_kernel_2(
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
        BS_M_ = arith.constant(BS_M, T.i32)
        BS_N_ = arith.constant(BS_N, T.i32)
        BS_K_ = arith.constant(BS_K, T.i32)
        GS_M_ = arith.constant(GS_M, T.i32)

        pid = get_program_id(axis=0)

        num_pid_m = arith.ceildivsi(M, BS_M_)
        num_pid_n = arith.ceildivsi(N, BS_N_)
        num_pid_in_group = GS_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GS_M
        group_size_m = arith.minsi(num_pid_m - first_pid_m, GS_M_)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        v15 = splat(pid_m * BS_M, (BS_M,)) + make_range(start=0, end=BS_M)
        offs_am = v15 % splat(M, (BS_M,))

        v20 = splat(pid_n * BS_N, (BS_N,)) + make_range(start=0, end=BS_N)
        offs_bn = v20 % splat(N, (BS_N,))

        offs_k = make_range(start=0, end=BS_K)

        v26 = expand_dims(offs_am, axis=1) * splat(stride_am, (BS_M, 1))
        v29 = expand_dims(offs_k, axis=0) * splat(stride_ak, (1, BS_K))
        v32 = broadcast(v26, (BS_M, BS_K)) + broadcast(v29, (BS_M, BS_K))
        v33 = splat(a_ptr, (BS_M, BS_K))
        a_ptrs = addptr(v33, offset=v32)

        v37 = expand_dims(offs_k, axis=1) * splat(stride_bk, (BS_K, 1))
        v40 = expand_dims(offs_bn, axis=0) * splat(stride_bn, (1, BS_N))
        v43 = broadcast(v37, (BS_K, BS_N)) + broadcast(v40, (BS_K, BS_N))
        v44 = splat(b_ptr, (BS_K, BS_N))
        b_ptrs = addptr(v44, offset=v43)

        a_ptr_incr = splat(stride_ak * BS_K, (BS_M, BS_K))
        b_ptr_incr = splat(stride_bk * BS_K, (BS_K, BS_N))

        accum = arith.constant(np.full([BS_M, BS_N], 0.0, np.float32))
        cst_0 = arith.constant(np.full([BS_K, BS_N], 0.0, np.float32))
        cst_1 = arith.constant(np.full([BS_M, BS_K], 0.0, np.float32))
        stop = arith.ceildivsi(K, BS_K_)
        id_map_1 = AffineMap.get_identity(2)

        @linalg.generic(
            [a_ptrs],
            [b_ptrs],
            [id_map_1, id_map_1],
            [linalg.IteratorType.parallel, linalg.IteratorType.parallel],
        )
        def f(a_ptr, b_ptr):
            a_ptrs = splat(a_ptr, (BS_M, BS_K))
            b_ptrs = splat(b_ptr, (BS_K, BS_N))
            a_mask = broadcast(
                expand_dims(offs_k, axis=0) < splat(K - 1 * BS_K, (1, BS_K)),
                (BS_M, BS_K),
            )
            a = load(a_ptrs, mask=a_mask, other=cst_1)
            b_mask = broadcast(
                expand_dims(offs_k, axis=1) < splat(K - 1 * BS_K, (BS_K, 1)),
                (BS_K, BS_N),
            )
            b = load(b_ptrs, mask=b_mask, other=cst_0)
            accum = dot(a, b)

            return b_ptr

        offs_cm = expand_dims(v15, axis=1)

        c_ptr = splat(c_ptr, (BS_M, 1))
        c_ptr = addptr(c_ptr, offset=splat(stride_cm, (BS_M, 1)) * offs_cm)
        c_ptr = broadcast(c_ptr, (BS_M, BS_N))

        offs_cn = expand_dims(v20, axis=0)
        v60 = splat(stride_cn, (1, BS_N)) * offs_cn
        c_ptrs = addptr(c_ptr, offset=broadcast(v60, (BS_M, BS_N)))

        v68 = broadcast(offs_cm < splat(M, (BS_M, 1)), (BS_M, BS_N))
        v69 = broadcast(offs_cn < splat(N, (1, BS_N)), (BS_M, BS_N))
        c_mask = v68 & v69

        # store(c_ptrs, value=c, mask=c_mask)

    @module(attrs={"transform.target_tag": StringAttr.get("payload")})
    def payload():
        matmul_kernel_2.emit(force=True)

    @module(attrs={"transform.with_named_sequence": UnitAttr.get()})
    def mod_transform():
        @named_sequence(
            "cleanup",
            [any_op_t()],
            [],
            arg_attrs=[{"transform.readonly": UnitAttr.get()}],
        )
        def cleanup(target: any_op_t()):
            top_func = match(target, ["func.func"])

            @apply_patterns(top_func)
            def pats():
                transform.apply_patterns.linalg.tiling_canonicalization()
                # transform.apply_patterns.iree.fold_fill_into_pad
                transform.apply_patterns.scf.for_loop_canonicalization()
                transform.apply_patterns.canonicalization()

            all_loops = match(target, interface=MatchInterfaceEnum.LoopLikeInterface)
            apply_licm(all_loops)
            apply_cse(top_func)

        @named_sequence(
            "main", [any_op_t()], [], arg_attrs=[{"transform.readonly": UnitAttr.get()}]
        )
        def main(variant_op: any_op_t()):
            gen = match(variant_op, ops=["linalg.generic"])
            # First level tile to forall with tile_sizes [16, 64].
            tiled_matmul, (forall,) = tile_to_scf_forall(
                gen,
                [16, 64],
                mapping=[
                    thread_attr(MappingId.DimY),
                    thread_attr(MappingId.DimX),
                ],
            )
            # Run canonicalization to fold fill with pack and unpack operations.
            include("cleanup", [variant_op])

    assert ctx.module.operation.verify()
    print("\nBEFORE\n")
    print(ctx.module)
    mod = run_pipeline(
        ctx.module,
        Pipeline().transform_interpreter(
            entry_point="main", debug_payload_root_tag="payload"
        ),
    )
    print("\nAFTER\n")
    print(mod)


if __name__ == "__main__":
    with mlir_mod_ctx(allow_unregistered_dialects=True) as ctx:
        test_inline_mod_linalg_generic_transform(ctx)
