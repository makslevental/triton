import ctypes
import json
import math
from pathlib import Path

import numpy as np
from triton_mlir import types as _types
from triton_mlir._mlir_libs._triton import unwrap_c_module_op
from triton_mlir.extras.ast.canonicalize import canonicalize
from triton_mlir.extras.context import RAIIMLIRContextModule
from triton_mlir.dialects import (
    tt as ttpp,
    ttg,
    scf,
    llvm,
    _tt_ops_gen as tt,
    amdgpu,
    rocdl,
)
from triton_mlir.dialects.arith import IntegerOverflowFlags
from triton_mlir.ir import ArrayAttr, Type, Attribute
from triton_mlir.extras.dialects.ext import arith
from triton_mlir.extras.dialects import ext
import triton_mlir.extras.dialects.ext.scf as scf_dialect
from triton_mlir.extras import types as T
from hip import hip

from util import chip_check, backend_, hip_check

ctx = RAIIMLIRContextModule()

ctx.module.operation.attributes["ttg.num-ctas"] = Attribute.parse("1 : i32")
ctx.module.operation.attributes["ttg.num-warps"] = Attribute.parse("8 : i32")
ctx.module.operation.attributes["ttg.target"] = Attribute.parse('"hip:gfx1100"')
ctx.module.operation.attributes["ttg.threads-per-warp"] = Attribute.parse("32 : i32")

# fmt: off

blocked = ttg.BlockedEncodingAttr.get(size_per_thread=[1, 8], threads_per_warp__=[4, 8], warps_per_cta__=[8, 1], order=[1, 0])
blocked1 = ttg.BlockedEncodingAttr.get(size_per_thread=[8, 1], threads_per_warp__=[8, 4], warps_per_cta__=[1, 8], order=[0, 1])
shared = ttg.SwizzledSharedEncodingAttr.get(vec=4, per_phase=1, max_phase=16, order=[1, 0])
shared1 = ttg.SwizzledSharedEncodingAttr.get(vec=1, per_phase=1, max_phase=1, order=[0, 1])
mma = Attribute.parse('#ttg.amd_wmma<{version = 1, isTranspose = false, warpsPerCTA = [2, 4]}>')
smem = ttg.SharedMemorySpaceAttr.get()
dot0 = ttg.DotOperandEncodingAttr.get(op_idx=0, parent=mma, k_width=16)
dot1 = ttg.DotOperandEncodingAttr.get(op_idx=1, parent=mma, k_width=16)

@ttpp.jit(arg_attrs=ArrayAttr.parse('[{tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}]'), function_type=T.function(inputs=[ttpp.ptr(T.f16(), 1), ttpp.ptr(T.f16(), 1), ttpp.ptr(T.f16(), 1), T.i32(), T.i32(), T.i32(), T.i32(), T.i32(), T.i32(), T.i32(), T.i32(), T.i32(), T.i32()], results=[]), noinline=False, sym_name='matmul_kernel', sym_visibility='public')
def matmul_kernel(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12):
    cst = arith.constant(np.full([64, 64], 0.0, np.float32), T.tensor(64, 64, T.f32(), encoding=mma))
    true = arith.constant(True, T.bool())
    c2_i32 = arith.constant(2, T.i32())
    c1_i32 = arith.constant(1, T.i32())
    c32_i32 = arith.constant(32, T.i32())
    c63_i32 = arith.constant(63, T.i32())
    c64_i32 = arith.constant(64, T.i32())
    c128_i32 = arith.constant(128, T.i32())
    c192_i32 = arith.constant(192, T.i32())
    c4_i32 = arith.constant(4, T.i32())
    c256_i32 = arith.constant(256, T.i32())
    c255_i32 = arith.constant(255, T.i32())
    c8_i32 = arith.constant(8, T.i32())
    c0_i32 = arith.constant(0, T.i32())
    v0 = tt.get_program_id(axis=0)
    v1 = arith.remsi(lhs=v0, rhs=c8_i32)
    v3 = arith.muli(lhs=v1, rhs=c2_i32)
    v2 = arith.divsi(lhs=v0, rhs=c8_i32)
    v4 = arith.addi(lhs=v3, rhs=v2)
    v5 = arith.addi(lhs=arg4, rhs=c255_i32)
    v6 = arith.divsi(lhs=v5, rhs=c256_i32)
    v7 = arith.muli(lhs=v6, rhs=c4_i32)
    v8 = arith.divsi(lhs=v4, rhs=v7)
    v9 = arith.muli(lhs=v8, rhs=c4_i32)
    v40 = arith.remsi(lhs=v4, rhs=v7)
    v10 = arith.addi(lhs=arg3, rhs=c255_i32)
    v11 = arith.divsi(lhs=v10, rhs=c256_i32)
    v12 = arith.subi(lhs=v11, rhs=v9)
    v13 = arith.minsi(lhs=v12, rhs=c4_i32)
    v14 = arith.remsi(lhs=v4, rhs=v13)
    v15 = arith.addi(lhs=v9, rhs=v14)
    v16 = arith.muli(lhs=v15, rhs=c256_i32)

    v17 = tt.splat(result=T.tensor(256, T.i32(), encoding=ttg.SliceEncodingAttr.get(dim=1, parent=blocked)), src=v16)
    v18 = tt.make_range(result=T.tensor(256, T.i32(), encoding=ttg.SliceEncodingAttr.get(dim=1, parent=blocked)), start=0, end=256)
    v19 = arith.addi(lhs=v17, rhs=v18)
    v20 = tt.splat(result=T.tensor(256, T.i32(), encoding=ttg.SliceEncodingAttr.get(dim=1, parent=blocked)), src=arg3)
    v21 = arith.remsi(lhs=v19, rhs=v20)
    v22 = tt.make_range(result=T.tensor(64, T.i32(), encoding=ttg.SliceEncodingAttr.get(dim=0, parent=blocked)), start=0, end=64)
    v23 = tt.expand_dims(src=v21, axis=1)
    v24 = tt.splat(result=T.tensor(256, 1, T.i32(), encoding=blocked), src=arg6)
    v25 = arith.muli(lhs=v23, rhs=v24)
    v26 = tt.broadcast(result=T.tensor(256, 64, T.i32(), encoding=blocked), src=v25)
    v27 = tt.expand_dims(src=v22, axis=0)
    v28 = tt.broadcast(result=T.tensor(256, 64, T.i32(), encoding=blocked), src=v27)
    v29 = arith.addi(lhs=v26, rhs=v28)
    v30 = tt.addptr(result=ttpp.ptr(T.f16(), 1), ptr=arg0, offset=c64_i32)
    v31 = arith.addi(lhs=arg5, rhs=c63_i32)
    v32 = arith.divsi(lhs=v31, rhs=c64_i32)
    v33 = arith.cmpi(predicate=4, lhs=v32, rhs=c0_i32)
    v34 = tt.splat(result=T.tensor(256, 64, T.bool(), encoding=blocked), src=v33)
    # GR[0]
    v35 = amdgpu.buffer_load(result=T.tensor(256, 64, T.f16(), encoding=blocked), ptr=arg0, offsets=v29, stride=arg6, cache=1, mask=v34)
    v35.owner.attributes['OpIdx'] = amdgpu.OpIdxAttr.get(0)
    v36 = arith.cmpi(predicate=4, lhs=v32, rhs=c1_i32)
    v37 = tt.splat(result=T.tensor(256, 64, T.bool(), encoding=blocked), src=v36)
    v38 = amdgpu.buffer_load(result=T.tensor(256, 64, T.f16(), encoding=blocked), ptr=v30, offsets=v29, stride=arg6, cache=1, mask=v37)
    v38.owner.attributes['OpIdx'] = amdgpu.OpIdxAttr.get(0)
    v39 = tt.make_range(result=T.tensor(64, T.i32(), encoding=ttg.SliceEncodingAttr.get(dim=1, parent=blocked1)), start=0, end=64)
    v41 = arith.divsi(lhs=v40, rhs=v13)
    v42 = arith.muli(lhs=v41, rhs=c256_i32)
    v43 = tt.splat(result=T.tensor(256, T.i32(), encoding=ttg.SliceEncodingAttr.get(dim=0, parent=blocked1)), src=v42)
    v44 = tt.make_range(result=T.tensor(256, T.i32(), encoding=ttg.SliceEncodingAttr.get(dim=0, parent=blocked1)), start=0, end=256)
    v45 = arith.addi(lhs=v43, rhs=v44)
    v46 = tt.splat(result=T.tensor(256, T.i32(), encoding=ttg.SliceEncodingAttr.get(dim=0, parent=blocked1)), src=arg4)
    v47 = arith.remsi(lhs=v45, rhs=v46)
    v48 = tt.expand_dims(src=v39, axis=1)
    v49 = tt.broadcast(result=T.tensor(64, 256, T.i32(), encoding=blocked1), src=v48)
    v50 = tt.expand_dims(src=v47, axis=0)
    v51 = tt.splat(result=T.tensor(1, 256, T.i32(), encoding=blocked1), src=arg7)
    v52 = arith.muli(lhs=v50, rhs=v51)
    v53 = tt.broadcast(result=T.tensor(64, 256, T.i32(), encoding=blocked1), src=v52)
    v54 = arith.addi(lhs=v49, rhs=v53)
    v55 = tt.addptr(result=ttpp.ptr(T.f16(), 1), ptr=arg1, offset=c64_i32)
    v56 = tt.splat(result=T.tensor(64, 256, T.bool(), encoding=blocked1), src=v33)
    # GR[1]
    v57 = amdgpu.buffer_load(result=T.tensor(64, 256, T.f16(), encoding=blocked1), ptr=arg1, offsets=v54, stride=arg7, cache=1, mask=v56)
    v57.owner.attributes['OpIdx'] = amdgpu.OpIdxAttr.get(1)
    v58 = tt.splat(result=T.tensor(64, 256, T.bool(), encoding=blocked1), src=v36)
    v59 = amdgpu.buffer_load(result=T.tensor(64, 256, T.f16(), encoding=blocked1), ptr=v55, offsets=v54, stride=arg7, cache=1, mask=v58)
    v59.owner.attributes['OpIdx'] = amdgpu.OpIdxAttr.get(1)
    v60 = arith.cmpi(predicate=4, lhs=v15, rhs=c0_i32)
    llvm.intr_assume(cond=v60, op_bundle_operands=[], op_bundle_sizes=[])
    v61 = arith.cmpi(predicate=4, lhs=v41, rhs=c0_i32)
    llvm.intr_assume(cond=v61, op_bundle_operands=[], op_bundle_sizes=[])
    v62 = tt.make_range(result=T.tensor(256, T.i32(), encoding=ttg.SliceEncodingAttr.get(dim=1, parent=mma)), start=0, end=256)
    v63 = tt.make_range(result=T.tensor(256, T.i32(), encoding=ttg.SliceEncodingAttr.get(dim=0, parent=mma)), start=0, end=256)
    v64 = tt.splat(result=T.tensor(256, T.i32(), encoding=ttg.SliceEncodingAttr.get(dim=1, parent=mma)), src=v16)
    v65 = arith.addi(lhs=v64, rhs=v62)
    v66 = tt.splat(result=T.tensor(256, T.i32(), encoding=ttg.SliceEncodingAttr.get(dim=0, parent=mma)), src=v42)
    v67 = arith.addi(lhs=v66, rhs=v63)
    v68 = ttg.local_alloc(result=ttg.MemDescType.get(shape=[1, 256, 64], element_type=T.f16(), encoding=shared, memory_space=smem, mutable_memory=True, alloc_shape=[1, 256, 64]))
    v69 = ttg.local_alloc(result=ttg.MemDescType.get(shape=[1, 64, 256], element_type=T.f16(), encoding=shared1, memory_space=smem, mutable_memory=True, alloc_shape=[1, 64, 256]))
    # LW[0]
    v70 = ttg.memdesc_subview(result=ttg.MemDescType.get(shape=[256, 64], element_type=T.f16(), encoding=shared, memory_space=smem, mutable_memory=True, alloc_shape=[256, 64]), src=v68, offsets=[c0_i32, c0_i32, c0_i32])
    ttg.local_store_4 = ttg.local_store(src=v35, dst=v70)
    ttg.local_store_4.attributes['OpIdx'] = amdgpu.OpIdxAttr.get(0)
    v71 = ttg.memdesc_subview(result=ttg.MemDescType.get(shape=[64, 256], element_type=T.f16(), encoding=shared1, memory_space=smem, mutable_memory=True, alloc_shape=[64, 256]), src=v69, offsets=[c0_i32, c0_i32, c0_i32])
    ttg.local_store_5 = ttg.local_store(src=v57, dst=v71)
    ttg.local_store_5.attributes['OpIdx'] = amdgpu.OpIdxAttr.get(1)

    # LR[0]
    # A[MK] = 64x32
    a00_68 = ttg.memdesc_subview(result=ttg.MemDescType.get(shape=[64, 32], element_type=T.f16(), encoding=shared, memory_space=smem, mutable_memory=True, alloc_shape=[64, 32]), src=v68, offsets=[c0_i32, c0_i32, c0_i32])
    a10_68 = ttg.memdesc_subview(result=ttg.MemDescType.get(shape=[64, 32], element_type=T.f16(), encoding=shared, memory_space=smem, mutable_memory=True, alloc_shape=[64, 32]), src=v68, offsets=[c0_i32, c64_i32, c0_i32])
    a20_68 = ttg.memdesc_subview(result=ttg.MemDescType.get(shape=[64, 32], element_type=T.f16(), encoding=shared, memory_space=smem, mutable_memory=True, alloc_shape=[64, 32]), src=v68, offsets=[c0_i32, c128_i32, c0_i32])
    a30_68 = ttg.memdesc_subview(result=ttg.MemDescType.get(shape=[64, 32], element_type=T.f16(), encoding=shared, memory_space=smem, mutable_memory=True, alloc_shape=[64, 32]), src=v68, offsets=[c0_i32, c192_i32, c0_i32])
    a01_68 = ttg.memdesc_subview(result=ttg.MemDescType.get(shape=[64, 32], element_type=T.f16(), encoding=shared, memory_space=smem, mutable_memory=True, alloc_shape=[64, 32]), src=v68, offsets=[c0_i32, c0_i32, c32_i32])
    a11_68 = ttg.memdesc_subview(result=ttg.MemDescType.get(shape=[64, 32], element_type=T.f16(), encoding=shared, memory_space=smem, mutable_memory=True, alloc_shape=[64, 32]), src=v68, offsets=[c0_i32, c64_i32, c32_i32])
    a21_68 = ttg.memdesc_subview(result=ttg.MemDescType.get(shape=[64, 32], element_type=T.f16(), encoding=shared, memory_space=smem, mutable_memory=True, alloc_shape=[64, 32]), src=v68, offsets=[c0_i32, c128_i32, c32_i32])
    a31_68 = ttg.memdesc_subview(result=ttg.MemDescType.get(shape=[64, 32], element_type=T.f16(), encoding=shared, memory_space=smem, mutable_memory=True, alloc_shape=[64, 32]), src=v68, offsets=[c0_i32, c192_i32, c32_i32])
    # B[NK] = 32x64
    b00_69 = ttg.memdesc_subview(result=ttg.MemDescType.get(shape=[32, 64], element_type=T.f16(), encoding=shared1, memory_space=smem, mutable_memory=True, alloc_shape=[32, 64]), src=v69, offsets=[c0_i32, c0_i32, c0_i32])
    b10_69 = ttg.memdesc_subview(result=ttg.MemDescType.get(shape=[32, 64], element_type=T.f16(), encoding=shared1, memory_space=smem, mutable_memory=True, alloc_shape=[32, 64]), src=v69, offsets=[c0_i32, c0_i32, c64_i32])
    b20_69 = ttg.memdesc_subview(result=ttg.MemDescType.get(shape=[32, 64], element_type=T.f16(), encoding=shared1, memory_space=smem, mutable_memory=True, alloc_shape=[32, 64]), src=v69, offsets=[c0_i32, c0_i32, c128_i32])
    b30_69 = ttg.memdesc_subview(result=ttg.MemDescType.get(shape=[32, 64], element_type=T.f16(), encoding=shared1, memory_space=smem, mutable_memory=True, alloc_shape=[32, 64]), src=v69, offsets=[c0_i32, c0_i32, c192_i32])
    b01_69 = ttg.memdesc_subview(result=ttg.MemDescType.get(shape=[32, 64], element_type=T.f16(), encoding=shared1, memory_space=smem, mutable_memory=True, alloc_shape=[32, 64]), src=v69, offsets=[c0_i32, c32_i32, c0_i32])
    b11_69 = ttg.memdesc_subview(result=ttg.MemDescType.get(shape=[32, 64], element_type=T.f16(), encoding=shared1, memory_space=smem, mutable_memory=True, alloc_shape=[32, 64]), src=v69, offsets=[c0_i32, c32_i32, c64_i32])
    b21_69 = ttg.memdesc_subview(result=ttg.MemDescType.get(shape=[32, 64], element_type=T.f16(), encoding=shared1, memory_space=smem, mutable_memory=True, alloc_shape=[32, 64]), src=v69, offsets=[c0_i32, c32_i32, c128_i32])
    b31_69 = ttg.memdesc_subview(result=ttg.MemDescType.get(shape=[32, 64], element_type=T.f16(), encoding=shared1, memory_space=smem, mutable_memory=True, alloc_shape=[32, 64]), src=v69, offsets=[c0_i32, c32_i32, c192_i32])

    a00_70 = ttg.local_load(result=T.tensor(64, 32, T.f16(), encoding=dot0), src=a00_68)
    b00_71 = ttg.local_load(result=T.tensor(32, 64, T.f16(), encoding=dot1), src=b00_69)
    a10_70 = ttg.local_load(result=T.tensor(64, 32, T.f16(), encoding=dot0), src=a10_68)
    b10_71 = ttg.local_load(result=T.tensor(32, 64, T.f16(), encoding=dot1), src=b10_69)

    v74 = arith.subi(lhs=v32, rhs=c2_i32)
    for (
        arg10,
        [
            arg11,
            arg12,
            arg13,
            arg14,
            arg15,
            arg16,
            arg17,
            arg18,
            arg19,
            arg20,
            arg21,
            arg22,
            arg23,
            arg24,
            arg25,
            arg26,
            arg27,
            arg28,
            arg29,
            arg30,
            arg31,
            arg32,
            arg33,
            arg34,
            arg35,
        ],
        [
            v71_0,
            v71_1,
            v71_2,
            v71_3,
            v71_4,
            v71_5,
            v71_6,
            v71_7,
            v71_8,
            v71_9,
            v71_10,
            v71_11,
            v71_12,
            v71_13,
            v71_14,
            v71_15,
            v71_16,
            v71_17,
            v71_18,
            v71_19,
            v71_20,
            v71_21,
            v71_22,
            v71_23,
            v71_24,
        ],
    ) in scf.for_(
        c0_i32,
        v74,
        c1_i32,
        iter_args=[
            cst,
            c0_i32,
            v38,
            v59,
            a00_70,
            b00_71,
            v30,
            v55,
            cst,
            cst,
            cst,
            cst,
            cst,
            cst,
            cst,
            cst,
            cst,
            cst,
            cst,
            cst,
            cst,
            cst,
            cst,
            a10_70,
            b10_71,
        ],
    ):
        c00 = tt.dot(a=arg15, b=arg16, c=arg11, input_precision=2, max_num_imprecise_acc=0)
        rocdl.sched_group_barrier(mask=8, size=8, group_id=0)
        c01 = tt.dot(a=arg15, b=arg35, c=arg19, input_precision=2, max_num_imprecise_acc=0)
        rocdl.sched_group_barrier(mask=8, size=8, group_id=0)
        rocdl.sched_barrier(mask=0)

        b20_71 = ttg.local_load(result=T.tensor(32, 64, T.f16(), encoding=dot1), src=b20_69)
        b30_71 = ttg.local_load(result=T.tensor(32, 64, T.f16(), encoding=dot1), src=b30_69)
        c10 = tt.dot(a=arg34, b=arg16, c=arg22, input_precision=2, max_num_imprecise_acc=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=256, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=256, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=256, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=256, size=1, group_id=0)
        rocdl.sched_barrier(mask=0)

        c11 = tt.dot(a=arg34, b=arg35, c=arg23, input_precision=2, max_num_imprecise_acc=0)
        rocdl.sched_group_barrier(mask=8, size=8, group_id=0)
        rocdl.sched_barrier(mask=0)

        c02 = tt.dot(a=arg15, b=b20_71, c=arg20, input_precision=2, max_num_imprecise_acc=0)
        rocdl.sched_group_barrier(mask=8, size=8, group_id=0)
        c03 = tt.dot(a=arg15, b=b30_71, c=arg21, input_precision=2, max_num_imprecise_acc=0)
        rocdl.sched_group_barrier(mask=8, size=8, group_id=0)
        rocdl.sched_barrier(mask=0)

        a20_70 = ttg.local_load(result=T.tensor(64, 32, T.f16(), encoding=dot0), src=a20_68)
        a30_70 = ttg.local_load(result=T.tensor(64, 32, T.f16(), encoding=dot0), src=a30_68)
        c12 = tt.dot(a=arg34, b=b20_71, c=arg24, input_precision=2, max_num_imprecise_acc=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=256, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=256, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=256, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=256, size=1, group_id=0)
        rocdl.sched_barrier(mask=0)

        c13 = tt.dot(a=arg34, b=b30_71, c=arg25, input_precision=2, max_num_imprecise_acc=0)
        rocdl.sched_group_barrier(mask=8, size=8, group_id=0)
        rocdl.sched_barrier(mask=0)

        c20 = tt.dot(a=a20_70, b=arg16, c=arg26, input_precision=2, max_num_imprecise_acc=0)
        c21 = tt.dot(a=a20_70, b=arg35, c=arg27, input_precision=2, max_num_imprecise_acc=0)
        c30 = tt.dot(a=a30_70, b=arg16, c=arg30, input_precision=2, max_num_imprecise_acc=0)
        c31 = tt.dot(a=a30_70, b=arg35, c=arg31, input_precision=2, max_num_imprecise_acc=0)
        rocdl.sched_barrier(mask=1030)

        c22 = tt.dot(a=a20_70, b=b20_71, c=arg28, input_precision=2, max_num_imprecise_acc=0)
        rocdl.sched_barrier(mask=1030)

        a01_70 = ttg.local_load(result=T.tensor(64, 32, T.f16(), encoding=dot0), src=a01_68)
        b01_71 = ttg.local_load(result=T.tensor(32, 64, T.f16(), encoding=dot1), src=b01_69)
        c23 = tt.dot(a=a20_70, b=b30_71, c=arg29, input_precision=2, max_num_imprecise_acc=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=256, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=256, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=256, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=256, size=1, group_id=0)
        rocdl.sched_barrier(mask=1030)

        b11_71 = ttg.local_load(result=T.tensor(32, 64, T.f16(), encoding=dot1), src=b11_69)
        a11_70 = ttg.local_load(result=T.tensor(64, 32, T.f16(), encoding=dot0), src=a11_68)
        c32 = tt.dot(a=a30_70, b=b20_71, c=arg32, input_precision=2, max_num_imprecise_acc=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=256, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=256, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=256, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=256, size=1, group_id=0)
        rocdl.sched_barrier(mask=1030)

        c33 = tt.dot(a=a30_70, b=b30_71, c=arg33, input_precision=2, max_num_imprecise_acc=0)
        rocdl.sched_group_barrier(mask=8, size=8, group_id=0)
        rocdl.sched_barrier(mask=1030)

        d00 = tt.dot(a=a01_70, b=b01_71, c=c00, input_precision=2, max_num_imprecise_acc=0)
        rocdl.sched_group_barrier(mask=8, size=8, group_id=0)
        rocdl.sched_barrier(mask=1030)

        b21_71 = ttg.local_load(result=T.tensor(32, 64, T.f16(), encoding=dot1), src=b21_69)
        b31_71 = ttg.local_load(result=T.tensor(32, 64, T.f16(), encoding=dot1), src=b31_69)
        d01 = tt.dot(a=a01_70, b=b11_71, c=c01, input_precision=2, max_num_imprecise_acc=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=256, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=256, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=256, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=256, size=1, group_id=0)
        rocdl.sched_barrier(mask=1030)

        a21_70 = ttg.local_load(result=T.tensor(64, 32, T.f16(), encoding=dot0), src=a21_68)
        a31_70 = ttg.local_load(result=T.tensor(64, 32, T.f16(), encoding=dot0), src=a31_68)
        d10 = tt.dot(a=a11_70, b=b01_71, c=c10, input_precision=2, max_num_imprecise_acc=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=256, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=256, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=256, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=256, size=1, group_id=0)
        rocdl.sched_barrier(mask=1030)

        d11 = tt.dot(a=a11_70, b=b11_71, c=c11, input_precision=2, max_num_imprecise_acc=0)
        rocdl.sched_group_barrier(mask=8, size=8, group_id=0)
        rocdl.sched_barrier(mask=1030)

        w102 = ttg.memdesc_subview(result=ttg.MemDescType.get(shape=[256, 64], element_type=T.f16(), encoding=shared, memory_space=smem, mutable_memory=True, alloc_shape=[256, 64]), src=v68, offsets=[c0_i32, c0_i32, c0_i32])
        ttg.local_store_6 = ttg.local_store(src=arg13, dst=w102)
        ttg.local_store_6.attributes['OpIdx'] = amdgpu.OpIdxAttr.get(0)
        g96 = tt.addptr(result=ttpp.ptr(T.f16(), 1), ptr=arg17, offset=c64_i32)
        g97 = amdgpu.buffer_load(result=T.tensor(256, 64, T.f16(), encoding=blocked), ptr=g96, offsets=v29, cache=1)
        g97.owner.attributes['OpIdx'] = amdgpu.OpIdxAttr.get(0)
        d02 = tt.dot(a=a01_70, b=b21_71, c=c02, input_precision=2, max_num_imprecise_acc=0)
        d03 = tt.dot(a=a01_70, b=b31_71, c=c03, input_precision=2, max_num_imprecise_acc=0)
        d12 = tt.dot(a=a11_70, b=b21_71, c=c12, input_precision=2, max_num_imprecise_acc=0)
        d13 = tt.dot(a=a11_70, b=b31_71, c=c13, input_precision=2, max_num_imprecise_acc=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=512, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=32, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=512, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=32, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=512, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=32, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=512, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=32, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=512, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=32, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=512, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=32, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=512, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=32, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=512, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=32, size=1, group_id=0)
        rocdl.sched_barrier(mask=1030)

        w103 = ttg.memdesc_subview(result=ttg.MemDescType.get(shape=[64, 256], element_type=T.f16(), encoding=shared1, memory_space=smem, mutable_memory=True, alloc_shape=[64, 256]), src=v69, offsets=[c0_i32, c0_i32, c0_i32])
        ttg.local_store_8 = ttg.local_store(src=arg14, dst=w103)
        ttg.local_store_8.attributes['OpIdx'] = amdgpu.OpIdxAttr.get(1)
        g98 = tt.addptr(result=ttpp.ptr(T.f16(), 1), ptr=arg18, offset=c64_i32)
        g104 = amdgpu.buffer_load(result=T.tensor(64, 256, T.f16(), encoding=blocked1), ptr=g98, offsets=v54, cache=1)
        g104.owner.attributes['OpIdx'] = amdgpu.OpIdxAttr.get(1)
        d20 = tt.dot(a=a21_70, b=b01_71, c=c20, input_precision=2, max_num_imprecise_acc=0)
        d21 = tt.dot(a=a21_70, b=b11_71, c=c21, input_precision=2, max_num_imprecise_acc=0)
        d30 = tt.dot(a=a31_70, b=b01_71, c=c30, input_precision=2, max_num_imprecise_acc=0)
        d31 = tt.dot(a=a31_70, b=b11_71, c=c31, input_precision=2, max_num_imprecise_acc=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=512, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=32, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=512, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=32, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=512, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=32, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=512, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=32, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=512, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=32, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=512, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=32, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=512, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=32, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=512, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=32, size=1, group_id=0)
        rocdl.sched_barrier(mask=0)

        d22 = tt.dot(a=a21_70, b=b21_71, c=c22, input_precision=2, max_num_imprecise_acc=0)
        rocdl.sched_group_barrier(mask=8, size=8, group_id=0)
        rocdl.sched_barrier(mask=0)

        a0 = ttg.local_load(result=T.tensor(64, 32, T.f16(), encoding=dot0), src=a00_68)
        b0 = ttg.local_load(result=T.tensor(32, 64, T.f16(), encoding=dot1), src=b00_69)
        d23 = tt.dot(a=a21_70, b=b31_71, c=c23, input_precision=2, max_num_imprecise_acc=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=256, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=256, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=256, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=256, size=1, group_id=0)
        rocdl.sched_barrier(mask=0)

        a1 = ttg.local_load(result=T.tensor(64, 32, T.f16(), encoding=dot0), src=a10_68)
        b1 = ttg.local_load(result=T.tensor(32, 64, T.f16(), encoding=dot1), src=b10_69)
        d32 = tt.dot(a=a31_70, b=b21_71, c=c32, input_precision=2, max_num_imprecise_acc=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=256, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=256, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=256, size=1, group_id=0)
        rocdl.sched_group_barrier(mask=8, size=2, group_id=0)
        rocdl.sched_group_barrier(mask=256, size=1, group_id=0)
        rocdl.sched_barrier(mask=0)

        d33 = tt.dot(a=a31_70, b=b31_71, c=c33, input_precision=2, max_num_imprecise_acc=0)
        rocdl.sched_group_barrier(mask=8, size=8, group_id=0)
        rocdl.sched_barrier(mask=0)

        scf.yield_(results_=[d00, arg12, g97, g104, a0, b0, g96, g98, d01, d02, d03, d10, d11, d12, d13, d20, d21, d22, d23, d30, d31, d32, d33, a1, b1])

    eg00 = tt.dot(a=v71_4, b=v71_5, c=v71_0, input_precision=2, max_num_imprecise_acc=0)
    eg01 = tt.dot(a=v71_4, b=v71_24, c=v71_8, input_precision=2, max_num_imprecise_acc=0)
    eg10 = tt.dot(a=v71_23, b=v71_5, c=v71_11, input_precision=2, max_num_imprecise_acc=0)
    eg11 = tt.dot(a=v71_23, b=v71_24, c=v71_12, input_precision=2, max_num_imprecise_acc=0)
    rocdl.sched_barrier(mask=0)

    eb20_71 = ttg.local_load(result=T.tensor(32, 64, T.f16(), encoding=dot1), src=b20_69)
    eg02 = tt.dot(a=v71_4, b=eb20_71, c=v71_9, input_precision=2, max_num_imprecise_acc=0)
    eb30_71 = ttg.local_load(result=T.tensor(32, 64, T.f16(), encoding=dot1), src=b30_69)
    eg03 = tt.dot(a=v71_4, b=eb30_71, c=v71_10, input_precision=2, max_num_imprecise_acc=0)
    eg12 = tt.dot(a=v71_23, b=eb20_71, c=v71_13, input_precision=2, max_num_imprecise_acc=0)
    eg13 = tt.dot(a=v71_23, b=eb30_71, c=v71_14, input_precision=2, max_num_imprecise_acc=0)
    rocdl.sched_barrier(mask=0)

    ea20_70 = ttg.local_load(result=T.tensor(64, 32, T.f16(), encoding=dot0), src=a20_68)
    eg20 = tt.dot(a=ea20_70, b=v71_5, c=v71_15, input_precision=2, max_num_imprecise_acc=0)
    eg21 = tt.dot(a=ea20_70, b=v71_24, c=v71_16, input_precision=2, max_num_imprecise_acc=0)
    ea30_70 = ttg.local_load(result=T.tensor(64, 32, T.f16(), encoding=dot0), src=a30_68)
    eg30 = tt.dot(a=ea30_70, b=v71_5, c=v71_19, input_precision=2, max_num_imprecise_acc=0)
    eg31 = tt.dot(a=ea30_70, b=v71_24, c=v71_20, input_precision=2, max_num_imprecise_acc=0)
    rocdl.sched_barrier(mask=0)

    eg22 = tt.dot(a=ea20_70, b=eb20_71, c=v71_17, input_precision=2, max_num_imprecise_acc=0)
    eg23 = tt.dot(a=ea20_70, b=eb30_71, c=v71_18, input_precision=2, max_num_imprecise_acc=0)
    eg32 = tt.dot(a=ea30_70, b=eb20_71, c=v71_21, input_precision=2, max_num_imprecise_acc=0)
    eg33 = tt.dot(a=ea30_70, b=eb30_71, c=v71_22, input_precision=2, max_num_imprecise_acc=0)
    rocdl.sched_barrier(mask=0)

    eb01_71 = ttg.local_load(result=T.tensor(32, 64, T.f16(), encoding=dot1), src=b01_69)
    ea01_70 = ttg.local_load(result=T.tensor(64, 32, T.f16(), encoding=dot0), src=a01_68)
    ed00 = tt.dot(a=ea01_70, b=eb01_71, c=eg00, input_precision=2, max_num_imprecise_acc=0)
    eb11_71 = ttg.local_load(result=T.tensor(32, 64, T.f16(), encoding=dot1), src=b11_69)
    ed01 = tt.dot(a=ea01_70, b=eb11_71, c=eg01, input_precision=2, max_num_imprecise_acc=0)
    ea11_70 = ttg.local_load(result=T.tensor(64, 32, T.f16(), encoding=dot0), src=a11_68)
    ed10 = tt.dot(a=ea11_70, b=eb01_71, c=eg10, input_precision=2, max_num_imprecise_acc=0)
    ed11 = tt.dot(a=ea11_70, b=eb11_71, c=eg11, input_precision=2, max_num_imprecise_acc=0)
    rocdl.sched_barrier(mask=0)

    eb21_71 = ttg.local_load(result=T.tensor(32, 64, T.f16(), encoding=dot1), src=b21_69)
    ed02 = tt.dot(a=ea01_70, b=eb21_71, c=eg02, input_precision=2, max_num_imprecise_acc=0)
    eb31_71 = ttg.local_load(result=T.tensor(32, 64, T.f16(), encoding=dot1), src=b31_69)
    ed03 = tt.dot(a=ea01_70, b=eb31_71, c=eg03, input_precision=2, max_num_imprecise_acc=0)
    ed12 = tt.dot(a=ea11_70, b=eb21_71, c=eg12, input_precision=2, max_num_imprecise_acc=0)
    ed13 = tt.dot(a=ea11_70, b=eb31_71, c=eg13, input_precision=2, max_num_imprecise_acc=0)
    rocdl.sched_barrier(mask=0)

    ea21_70 = ttg.local_load(result=T.tensor(64, 32, T.f16(), encoding=dot0), src=a21_68)
    ed20 = tt.dot(a=ea21_70, b=eb01_71, c=eg20, input_precision=2, max_num_imprecise_acc=0)
    ed21 = tt.dot(a=ea21_70, b=eb11_71, c=eg21, input_precision=2, max_num_imprecise_acc=0)
    ea31_70 = ttg.local_load(result=T.tensor(64, 32, T.f16(), encoding=dot0), src=a31_68)
    ed30 = tt.dot(a=ea31_70, b=eb01_71, c=eg30, input_precision=2, max_num_imprecise_acc=0)
    ed31 = tt.dot(a=ea31_70, b=eb11_71, c=eg31, input_precision=2, max_num_imprecise_acc=0)
    rocdl.sched_barrier(mask=0)

    ed22 = tt.dot(a=ea21_70, b=eb21_71, c=eg22, input_precision=2, max_num_imprecise_acc=0)
    ed23 = tt.dot(a=ea21_70, b=eb31_71, c=eg23, input_precision=2, max_num_imprecise_acc=0)
    ed32 = tt.dot(a=ea31_70, b=eb21_71, c=eg32, input_precision=2, max_num_imprecise_acc=0)
    ed33 = tt.dot(a=ea31_70, b=eb31_71, c=eg33, input_precision=2, max_num_imprecise_acc=0)
    rocdl.sched_barrier(mask=0)

    v72 = ttg.memdesc_subview(result=ttg.MemDescType.get(shape=[256, 64], element_type=T.f16(), encoding=shared, memory_space=smem, mutable_memory=True, alloc_shape=[256, 64]), src=v68, offsets=[c0_i32, c0_i32, c0_i32])
    ttg.local_store_10 = ttg.local_store(src=v71_2, dst=v72)
    ttg.local_store_10.attributes['OpIdx'] = amdgpu.OpIdxAttr.get(0)
    v73 = ttg.memdesc_subview(result=ttg.MemDescType.get(shape=[64, 256], element_type=T.f16(), encoding=shared1, memory_space=smem, mutable_memory=True, alloc_shape=[64, 256]), src=v69, offsets=[c0_i32, c0_i32, c0_i32])
    ttg.local_store_11 = ttg.local_store(src=v71_3, dst=v73)
    ttg.local_store_11.attributes['OpIdx'] = amdgpu.OpIdxAttr.get(1)
    rocdl.sched_barrier(mask=0)

    efb00 = ttg.local_load(result=T.tensor(32, 64, T.f16(), encoding=dot1), src=b00_69)
    efa00 = ttg.local_load(result=T.tensor(64, 32, T.f16(), encoding=dot0), src=a00_68)
    ef00 = tt.dot(a=efa00, b=efb00, c=ed00, input_precision=2, max_num_imprecise_acc=0)
    efb10 = ttg.local_load(result=T.tensor(32, 64, T.f16(), encoding=dot1), src=b10_69)
    ef01 = tt.dot(a=efa00, b=efb10, c=ed01, input_precision=2, max_num_imprecise_acc=0)
    efb20 = ttg.local_load(result=T.tensor(32, 64, T.f16(), encoding=dot1), src=b20_69)
    ef02 = tt.dot(a=efa00, b=efb20, c=ed02, input_precision=2, max_num_imprecise_acc=0)
    efb30 = ttg.local_load(result=T.tensor(32, 64, T.f16(), encoding=dot1), src=b30_69)
    ef03 = tt.dot(a=efa00, b=efb30, c=ed03, input_precision=2, max_num_imprecise_acc=0)
    rocdl.sched_barrier(mask=0)

    efa10 = ttg.local_load(result=T.tensor(64, 32, T.f16(), encoding=dot0), src=a10_68)
    ef10 = tt.dot(a=efa10, b=efb00, c=ed10, input_precision=2, max_num_imprecise_acc=0)
    ef11 = tt.dot(a=efa10, b=efb10, c=ed11, input_precision=2, max_num_imprecise_acc=0)
    ef12 = tt.dot(a=efa10, b=efb20, c=ed12, input_precision=2, max_num_imprecise_acc=0)
    ef13 = tt.dot(a=efa10, b=efb30, c=ed13, input_precision=2, max_num_imprecise_acc=0)
    rocdl.sched_barrier(mask=0)

    efa20 = ttg.local_load(result=T.tensor(64, 32, T.f16(), encoding=dot0), src=a20_68)
    ef20 = tt.dot(a=efa20, b=efb00, c=ed20, input_precision=2, max_num_imprecise_acc=0)
    ef21 = tt.dot(a=efa20, b=efb10, c=ed21, input_precision=2, max_num_imprecise_acc=0)
    ef22 = tt.dot(a=efa20, b=efb20, c=ed22, input_precision=2, max_num_imprecise_acc=0)
    ef23 = tt.dot(a=efa20, b=efb30, c=ed23, input_precision=2, max_num_imprecise_acc=0)
    rocdl.sched_barrier(mask=0)

    efa30 = ttg.local_load(result=T.tensor(64, 32, T.f16(), encoding=dot0), src=a30_68)
    ef30 = tt.dot(a=efa30, b=efb00, c=ed30, input_precision=2, max_num_imprecise_acc=0)
    ef31 = tt.dot(a=efa30, b=efb10, c=ed31, input_precision=2, max_num_imprecise_acc=0)
    ef32 = tt.dot(a=efa30, b=efb20, c=ed32, input_precision=2, max_num_imprecise_acc=0)
    ef33 = tt.dot(a=efa30, b=efb30, c=ed33, input_precision=2, max_num_imprecise_acc=0)
    rocdl.sched_barrier(mask=0)

    efb01 = ttg.local_load(result=T.tensor(32, 64, T.f16(), encoding=dot1), src=b01_69)
    efa01 = ttg.local_load(result=T.tensor(64, 32, T.f16(), encoding=dot0), src=a01_68)
    e00_86 = tt.dot(a=efa01, b=efb01, c=ef00, input_precision=2, max_num_imprecise_acc=0)
    efb11 = ttg.local_load(result=T.tensor(32, 64, T.f16(), encoding=dot1), src=b11_69)
    e01_86 = tt.dot(a=efa01, b=efb11, c=ef01, input_precision=2, max_num_imprecise_acc=0)
    efa11 = ttg.local_load(result=T.tensor(64, 32, T.f16(), encoding=dot0), src=a11_68)
    e10_86 = tt.dot(a=efa11, b=efb01, c=ef10, input_precision=2, max_num_imprecise_acc=0)
    e11_86 = tt.dot(a=efa11, b=efb11, c=ef11, input_precision=2, max_num_imprecise_acc=0)
    efb21 = ttg.local_load(result=T.tensor(32, 64, T.f16(), encoding=dot1), src=b21_69)
    e02_86 = tt.dot(a=efa01, b=efb21, c=ef02, input_precision=2, max_num_imprecise_acc=0)
    efb31 = ttg.local_load(result=T.tensor(32, 64, T.f16(), encoding=dot1), src=b31_69)
    e03_86 = tt.dot(a=efa01, b=efb31, c=ef03, input_precision=2, max_num_imprecise_acc=0)
    e12_86 = tt.dot(a=efa11, b=efb21, c=ef12, input_precision=2, max_num_imprecise_acc=0)
    e13_86 = tt.dot(a=efa11, b=efb31, c=ef13, input_precision=2, max_num_imprecise_acc=0)
    efa21 = ttg.local_load(result=T.tensor(64, 32, T.f16(), encoding=dot0), src=a21_68)
    e20_86 = tt.dot(a=efa21, b=efb01, c=ef20, input_precision=2, max_num_imprecise_acc=0)
    e21_86 = tt.dot(a=efa21, b=efb11, c=ef21, input_precision=2, max_num_imprecise_acc=0)
    efa31 = ttg.local_load(result=T.tensor(64, 32, T.f16(), encoding=dot0), src=a31_68)
    e30_86 = tt.dot(a=efa31, b=efb01, c=ef30, input_precision=2, max_num_imprecise_acc=0)
    e31_86 = tt.dot(a=efa31, b=efb11, c=ef31, input_precision=2, max_num_imprecise_acc=0)
    e22_86 = tt.dot(a=efa21, b=efb21, c=ef22, input_precision=2, max_num_imprecise_acc=0)
    e23_86 = tt.dot(a=efa21, b=efb31, c=ef23, input_precision=2, max_num_imprecise_acc=0)
    e32_86 = tt.dot(a=efa31, b=efb21, c=ef32, input_precision=2, max_num_imprecise_acc=0)
    e33_86 = tt.dot(a=efa31, b=efb31, c=ef33, input_precision=2, max_num_imprecise_acc=0)
    ttg.local_dealloc(src=v68)
    ttg.local_dealloc(src=v69)

    # finish all dots first and now all a,b vgpr operands are done, global and local addresses are done
    rocdl.sched_barrier(mask=1030)

    ################################################
    ## Store Addresses
    ################################################

    # epilogue address; same for whole wave?
    e88 = arith.muli(lhs=arg8, rhs=v16)
    e91 = tt.addptr(result=ttpp.ptr(T.f16(), 1), ptr=arg2, offset=e88)
    ea = tt.addptr(result=ttpp.ptr(T.f16(), 1), ptr=e91, offset=v42)

    # epilogue index
    e0_89 = tt.splat(result=T.tensor(64, 1, T.i32(), encoding=mma), src=arg8)
    e1_89 = tt.splat(result=T.tensor(64, 1, T.i32(), encoding=mma), src=arg8)
    e2_89 = tt.splat(result=T.tensor(64, 1, T.i32(), encoding=mma), src=arg8)
    e3_89 = tt.splat(result=T.tensor(64, 1, T.i32(), encoding=mma), src=arg8)

    # extract slice of dims
    v62 = tt.make_range(result=T.tensor(256, T.i32(), encoding=ttg.SliceEncodingAttr.get(dim=1, parent=mma)), start=0, end=256)
    v64 = tt.expand_dims(src=v62, axis=1)
    v74 = amdgpu.extract_slice(result=T.tensor(64, 1, T.i32(), encoding=mma), source=v64, static_offsets=Attribute.parse('array<i64: 0, 0>'))
    v75 = amdgpu.extract_slice(result=T.tensor(64, 1, T.i32(), encoding=mma), source=v64, static_offsets=Attribute.parse('array<i64: 64, 0>'))
    v76 = amdgpu.extract_slice(result=T.tensor(64, 1, T.i32(), encoding=mma), source=v64, static_offsets=Attribute.parse('array<i64: 128, 0>'))
    v77 = amdgpu.extract_slice(result=T.tensor(64, 1, T.i32(), encoding=mma), source=v64, static_offsets=Attribute.parse('array<i64: 192, 0>'))

    e0_90 = arith.muli(lhs=e0_89, rhs=v74)
    e1_90 = arith.muli(lhs=e1_89, rhs=v75)
    e2_90 = arith.muli(lhs=e2_89, rhs=v76)
    e3_90 = arith.muli(lhs=e3_89, rhs=v77)

    e0_92 = tt.broadcast(result=T.tensor(64, 64, T.i32(), encoding=mma), src=e0_90)
    e1_92 = tt.broadcast(result=T.tensor(64, 64, T.i32(), encoding=mma), src=e1_90)
    e2_92 = tt.broadcast(result=T.tensor(64, 64, T.i32(), encoding=mma), src=e2_90)
    e3_92 = tt.broadcast(result=T.tensor(64, 64, T.i32(), encoding=mma), src=e3_90)

    # extract slice of dims
    v63 = tt.make_range(result=T.tensor(256, T.i32(), encoding=ttg.SliceEncodingAttr.get(dim=0, parent=mma)), start=0, end=256)
    v65 = tt.expand_dims(src=v63, axis=0)
    v78 = amdgpu.extract_slice(result=T.tensor(1, 64, T.i32(), encoding=mma), source=v65, static_offsets=Attribute.parse('array<i64: 0, 0>'))
    v79 = amdgpu.extract_slice(result=T.tensor(1, 64, T.i32(), encoding=mma), source=v65, static_offsets=Attribute.parse('array<i64: 0, 64>'))
    v80 = amdgpu.extract_slice(result=T.tensor(1, 64, T.i32(), encoding=mma), source=v65, static_offsets=Attribute.parse('array<i64: 0, 128>'))
    v81 = amdgpu.extract_slice(result=T.tensor(1, 64, T.i32(), encoding=mma), source=v65, static_offsets=Attribute.parse('array<i64: 0, 192>'))
    
    e0_93 = tt.broadcast(result=T.tensor(64, 64, T.i32(), encoding=mma), src=v78)
    e1_93 = tt.broadcast(result=T.tensor(64, 64, T.i32(), encoding=mma), src=v79)
    e2_93 = tt.broadcast(result=T.tensor(64, 64, T.i32(), encoding=mma), src=v80)
    e3_93 = tt.broadcast(result=T.tensor(64, 64, T.i32(), encoding=mma), src=v81)
    
    #################################################
    ## Blocked Stores
    #################################################

    # store 0 - no spills
    rocdl.sched_barrier(mask=1030)

    ei00 = arith.addi(lhs=e0_92, rhs=e0_93)
    ec00 = arith.truncf(out=T.tensor(64, 64, T.f16(), encoding=mma), in_=e00_86)
    amdgpu.buffer_store(value=ec00, ptr=ea, offsets=ei00, cache=1)

    ei01 = arith.addi(lhs=e0_92, rhs=e1_93)
    ec01 = arith.truncf(out=T.tensor(64, 64, T.f16(), encoding=mma), in_=e01_86)
    amdgpu.buffer_store(value=ec01, ptr=ea, offsets=ei01, cache=1)

    ei02 = arith.addi(lhs=e0_92, rhs=e2_93)
    ec02 = arith.truncf(out=T.tensor(64, 64, T.f16(), encoding=mma), in_=e02_86)
    amdgpu.buffer_store(value=ec02, ptr=ea, offsets=ei02, cache=1)

    ei03 = arith.addi(lhs=e0_92, rhs=e3_93)
    ec03 = arith.truncf(out=T.tensor(64, 64, T.f16(), encoding=mma), in_=e03_86)
    amdgpu.buffer_store(value=ec03, ptr=ea, offsets=ei03, cache=1)

    # store 1 - no spills
    rocdl.sched_barrier(mask=1030)

    ei10 = arith.addi(lhs=e1_92, rhs=e0_93)
    ec10 = arith.truncf(out=T.tensor(64, 64, T.f16(), encoding=mma), in_=e10_86)
    amdgpu.buffer_store(value=ec10, ptr=ea, offsets=ei10, cache=1)

    ei11 = arith.addi(lhs=e1_92, rhs=e1_93)
    ec11 = arith.truncf(out=T.tensor(64, 64, T.f16(), encoding=mma), in_=e11_86)
    amdgpu.buffer_store(value=ec11, ptr=ea, offsets=ei11, cache=1)

    ei12 = arith.addi(lhs=e1_92, rhs=e2_93)
    ec12 = arith.truncf(out=T.tensor(64, 64, T.f16(), encoding=mma), in_=e12_86)
    amdgpu.buffer_store(value=ec12, ptr=ea, offsets=ei12, cache=1)

    ei13 = arith.addi(lhs=e1_92, rhs=e3_93)
    ec13 = arith.truncf(out=T.tensor(64, 64, T.f16(), encoding=mma), in_=e13_86)
    amdgpu.buffer_store(value=ec13, ptr=ea, offsets=ei13, cache=1)


    # store 2 - no spills
    rocdl.sched_barrier(mask=1030)

    ei20 = arith.addi(lhs=e2_92, rhs=e0_93)
    ec20 = arith.truncf(out=T.tensor(64, 64, T.f16(), encoding=mma), in_=e20_86)
    amdgpu.buffer_store(value=ec20, ptr=ea, offsets=ei20, cache=1)

    ei21 = arith.addi(lhs=e2_92, rhs=e1_93)
    ec21 = arith.truncf(out=T.tensor(64, 64, T.f16(), encoding=mma), in_=e21_86)
    amdgpu.buffer_store(value=ec21, ptr=ea, offsets=ei21, cache=1)

    ei22 = arith.addi(lhs=e2_92, rhs=e2_93)
    ec22 = arith.truncf(out=T.tensor(64, 64, T.f16(), encoding=mma), in_=e22_86)
    amdgpu.buffer_store(value=ec22, ptr=ea, offsets=ei22, cache=1)

    ei23 = arith.addi(lhs=e2_92, rhs=e3_93)
    ec23 = arith.truncf(out=T.tensor(64, 64, T.f16(), encoding=mma), in_=e23_86)
    amdgpu.buffer_store(value=ec23, ptr=ea, offsets=ei23, cache=1)

    # store 3 - no spills
    rocdl.sched_barrier(mask=1030)

    ei30 = arith.addi(lhs=e3_92, rhs=e0_93)
    ec30 = arith.truncf(out=T.tensor(64, 64, T.f16(), encoding=mma), in_=e30_86)
    amdgpu.buffer_store(value=ec30, ptr=ea, offsets=ei30, cache=1)

    ei31 = arith.addi(lhs=e3_92, rhs=e1_93)
    ec31 = arith.truncf(out=T.tensor(64, 64, T.f16(), encoding=mma), in_=e31_86)
    amdgpu.buffer_store(value=ec31, ptr=ea, offsets=ei31, cache=1)

    ei32 = arith.addi(lhs=e3_92, rhs=e2_93)
    ec32 = arith.truncf(out=T.tensor(64, 64, T.f16(), encoding=mma), in_=e32_86)
    amdgpu.buffer_store(value=ec32, ptr=ea, offsets=ei32, cache=1)

    ei33 = arith.addi(lhs=e3_92, rhs=e3_93)
    ec33 = arith.truncf(out=T.tensor(64, 64, T.f16(), encoding=mma), in_=e33_86)
    amdgpu.buffer_store(value=ec33, ptr=ea, offsets=ei33, cache=1)

    # start mask
    # v102 = tt.splat(result=T.tensor(256, 1, T.i32(), encoding=mma), src=arg3)
    # v90 = tt.expand_dims(src=v65, axis=1)
    # v103 = arith.cmpi(predicate=2, lhs=v90, rhs=v102)
    # v104 = tt.splat(result=T.tensor(1, 256, T.i32(), encoding=mma), src=arg4)
    # v96 = tt.expand_dims(src=v67, axis=0)
    # v105 = arith.cmpi(predicate=2, lhs=v96, rhs=v104)
    # v106 = tt.broadcast(result=T.tensor(256, 256, T.bool(), encoding=mma), src=v103)
    # v107 = tt.broadcast(result=T.tensor(256, 256, T.bool(), encoding=mma), src=v105)
    # v108 = arith.andi(lhs=v106, rhs=v107)
    # end mask
    # amdgpu.buffer_store(value=v89, ptr=v100, offsets=v101, stride=arg8, cache=1, mask=v108)

# fmt: on

matmul_kernel.emit()
ctx.module.operation.verify()


def mod_str():
    return str(ctx.module)


def launch(
    function,
    gridX,
    gridY,
    gridZ,
    stream,
    warp_size,
    num_warps,
    shared_memory,
    *args,
):
    from triton_mlir import chip

    from hip._util.types import DeviceArray

    params = [None] * len(args)
    addresses = [None] * len(args)
    for i, p in enumerate(args):
        if isinstance(p, DeviceArray):
            addresses[i] = params[i] = p.createRef().as_c_void_p()
        elif isinstance(p, int):
            params[i] = ctypes.c_int32(p)
            addresses[i] = ctypes.addressof(params[i])
        else:
            raise NotImplementedError(f"{p=} not supported with {p=}")

    global_scratch = chip.hipDeviceptr_t()
    addresses += [ctypes.addressof(global_scratch)]
    c_args = (ctypes.c_void_p * len(addresses))(*addresses)
    function = ctypes.cast(function, chip.hipFunction_t)
    stream = ctypes.cast(stream, chip.hipStream_t)
    chip_check(
        chip.hipModuleLaunchKernel(
            function,
            gridX,
            gridY,
            gridZ,
            warp_size * num_warps,
            1,
            1,
            shared_memory,
            stream,
            c_args,
            None,
        )
    )


ttgir_mod = unwrap_c_module_op(ctx.module.operation)

HERE = Path(__file__).parent

backend = backend_()
options_dict = json.load(open(HERE / "matmul_kernel.json"))
options = backend.parse_options(options_dict)

hsaco, metadata = backend.compile(
    ttgir_mod,
    ttir=False,
    ttgir=False,
    options=options,
    dump_ir=True,
    ir_dump_dir=Path(__file__).parent / "david",
    dump_file_prefix="0",
)

module = hip_check(hip.hipModuleLoadData(hsaco))
function = hip_check(
    hip.hipModuleGetFunction(module, metadata["name"].encode())
).as_c_void_p()

# kernel launch
M, K, N = 1024, 1024, 1024
BLOCK_SIZE_M = BLOCK_SIZE_N = 256

a_h = np.random.rand(M, K).astype(np.float16)
b_h = np.random.rand(K, N).T.astype(np.float16)
c_h = -3 * np.ones((M, N), dtype=np.float16)

a_num_bytes = a_h.size * a_h.itemsize
b_num_bytes = b_h.size * b_h.itemsize
c_num_bytes = c_h.size * c_h.itemsize

a_d = hip_check(hip.hipMalloc(a_num_bytes)).configure(typestr="float16", shape=(M, K))
b_d = hip_check(hip.hipMalloc(b_num_bytes)).configure(typestr="float16", shape=(K, N))
c_d = hip_check(hip.hipMalloc(c_num_bytes)).configure(typestr="float16", shape=(M, N))

hip_check(hip.hipMemcpy(a_d, a_h, a_num_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
hip_check(hip.hipMemcpy(b_d, b_h, b_num_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
hip_check(hip.hipMemcpy(c_d, c_h, c_num_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))

gridX = math.ceil(M / BLOCK_SIZE_M) * math.ceil(N / BLOCK_SIZE_N)
gridY = 1
gridZ = 1
warp_size = options.warp_size
num_warps = options.num_warps
shared_memory = options_dict["shared"]
stream = 0

launch(
    function,
    gridX,
    gridY,
    gridZ,
    stream,
    warp_size,
    num_warps,
    shared_memory,
    a_d,
    b_d,
    c_d,
    M,
    N,
    K,
    a_h.strides[0] // a_h.itemsize,
    b_h.strides[1] // b_h.itemsize,
    c_h.strides[0] // c_h.itemsize,
    0,
    0,
    0,
    0,
)

correct = a_h @ b_h
assert np.allclose(c_h, -3.0)
assert not np.allclose(correct, c_h)
hip_check(hip.hipMemcpy(c_h, c_d, c_num_bytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
if not np.allclose(c_h, correct, atol=5e-3, rtol=1e-2):
    print(c_h)
    # print(correct)
    total_wrong = np.sum((c_h - correct) != 0)
    print(f"{total_wrong=}")
    assert (total_wrong / M * N) < 0.005
