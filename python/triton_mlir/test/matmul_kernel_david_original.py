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
    cst = arith.constant(np.full([256, 256], 0.0, np.float32), T.tensor(256, 256, T.f32(), encoding=mma))
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
    v2 = arith.divsi(lhs=v0, rhs=c8_i32)
    v3 = arith.muli(lhs=v1, rhs=c2_i32)
    v4 = arith.addi(lhs=v3, rhs=v2)
    v5 = arith.addi(lhs=arg4, rhs=c255_i32)
    v6 = arith.divsi(lhs=v5, rhs=c256_i32)
    v7 = arith.muli(lhs=v6, rhs=c4_i32)
    v8 = arith.divsi(lhs=v4, rhs=v7)
    v9 = arith.muli(lhs=v8, rhs=c4_i32)
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
    v40 = arith.remsi(lhs=v4, rhs=v7)
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

    v72 = ttg.local_load(result=T.tensor(256, 64, T.f16(), encoding=dot0), src=v70)
    v73 = ttg.local_load(result=T.tensor(64, 256, T.f16(), encoding=dot1), src=v71)
    v74 = arith.subi(lhs=v32, rhs=c2_i32)
    for arg13, [arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21], [v75_0, v75_1, v75_2, v75_3, v75_4, v75_5, v75_6, v75_7] in scf.for_(c0_i32, v74, c1_i32, iter_args=[cst, v30, v55, c0_i32, v38, v59, v72, v73]):
        v114 = tt.addptr(result=ttpp.ptr(T.f16(), 1), ptr=arg15, offset=c64_i32)
        v115 = amdgpu.buffer_load(result=T.tensor(256, 64, T.f16(), encoding=blocked), ptr=v114, offsets=v29, stride=arg6, cache=1)
        v115.owner.attributes['OpIdx'] = amdgpu.OpIdxAttr.get(0)
        v116 = tt.addptr(result=ttpp.ptr(T.f16(), 1), ptr=arg16, offset=c64_i32)
        v109 = arith.addi(lhs=arg17, rhs=c1_i32)
        v110 = arith.cmpi(predicate=2, lhs=v109, rhs=c1_i32)
        v111 = arith.select(condition=v110, true_value=v109, false_value=c0_i32)
        v112 = ttg.memdesc_subview(result=ttg.MemDescType.get(shape=[256, 64], element_type=T.f16(), encoding=shared, memory_space=smem, mutable_memory=True, alloc_shape=[256, 64]), src=v68, offsets=[v111, c0_i32, c0_i32])
        ttg.local_store_6 = ttg.local_store(src=arg18, dst=v112)
        ttg.local_store_6.attributes['OpIdx'] = amdgpu.OpIdxAttr.get(0)
        v113 = ttg.memdesc_subview(result=ttg.MemDescType.get(shape=[64, 256], element_type=T.f16(), encoding=shared1, memory_space=smem, mutable_memory=True, alloc_shape=[64, 256]), src=v69, offsets=[v111, c0_i32, c0_i32])
        ttg.local_store_7 = ttg.local_store(src=arg19, dst=v113)
        ttg.local_store_7.attributes['OpIdx'] = amdgpu.OpIdxAttr.get(1)
        v117 = amdgpu.buffer_load(result=T.tensor(64, 256, T.f16(), encoding=blocked1), ptr=v116, offsets=v54, stride=arg7, cache=1)
        v117.owner.attributes['OpIdx'] = amdgpu.OpIdxAttr.get(1)
        v118 = tt.dot(a=arg20, b=arg21, c=arg14, input_precision=2, max_num_imprecise_acc=0)
        v119 = ttg.local_load(result=T.tensor(256, 64, T.f16(), encoding=dot0), src=v112)
        v120 = ttg.local_load(result=T.tensor(64, 256, T.f16(), encoding=dot1), src=v113)
        scf.yield_(results_=[v118, v114, v116, v111, v115, v117, v119, v120])
    v76 = arith.cmpi(predicate=5, lhs=v32, rhs=c1_i32)
    v77 = arith.cmpi(predicate=5, lhs=v32, rhs=c2_i32)
    v78 = arith.addi(lhs=v75_3, rhs=c1_i32)
    v79 = arith.cmpi(predicate=2, lhs=v78, rhs=c1_i32)
    v80 = arith.select(condition=v79, true_value=v78, false_value=c0_i32)
    v81 = ttg.memdesc_subview(result=ttg.MemDescType.get(shape=[256, 64], element_type=T.f16(), encoding=shared, memory_space=smem, mutable_memory=True, alloc_shape=[256, 64]), src=v68, offsets=[v80, c0_i32, c0_i32])
    ttg.local_store_10 = ttg.local_store(src=v75_4, dst=v81)
    ttg.local_store_10.attributes['OpIdx'] = amdgpu.OpIdxAttr.get(0)
    v82 = ttg.memdesc_subview(result=ttg.MemDescType.get(shape=[64, 256], element_type=T.f16(), encoding=shared1, memory_space=smem, mutable_memory=True, alloc_shape=[64, 256]), src=v69, offsets=[v80, c0_i32, c0_i32])
    ttg.local_store_11 = ttg.local_store(src=v75_5, dst=v82)
    ttg.local_store_11.attributes['OpIdx'] = amdgpu.OpIdxAttr.get(1)
    @ext.scf.if_(v76, results=[T.tensor(256, 256, T.f32(), encoding=mma)])
    def v83():                
        v109 = tt.dot(a=v75_6, b=v75_7, c=v75_0, input_precision=2, max_num_imprecise_acc=0)
        return v109
    @ext.scf.else_(v83)
    def v83_else():                
        return v75_0
    v84 = ttg.local_load(result=T.tensor(256, 64, T.f16(), encoding=dot0), src=v81)
    v85 = ttg.local_load(result=T.tensor(64, 256, T.f16(), encoding=dot1), src=v82)
    v86 = arith.select(condition=v76, true_value=v83, false_value=v75_0)
    @ext.scf.if_(v77, results=[T.tensor(256, 256, T.f32(), encoding=mma)])
    def v87():                
        v109 = tt.dot(a=v84, b=v85, c=v86, input_precision=2, max_num_imprecise_acc=0)
        return v109
    @ext.scf.else_(v87)
    def v87_else():                
        return v86
    v88 = arith.select(condition=v77, true_value=v87, false_value=v86)
    ttg.local_dealloc(src=v68)
    ttg.local_dealloc(src=v69)
    v89 = arith.truncf(out=T.tensor(256, 256, T.f16(), encoding=mma), in_=v88)
    v92 = arith.muli(lhs=arg8, rhs=v16)
    v93 = tt.splat(result=T.tensor(256, 1, T.i32(), encoding=mma), src=arg8)
    v91 = tt.expand_dims(src=v62, axis=1)
    v94 = arith.muli(lhs=v93, rhs=v91)
    v95 = tt.addptr(result=ttpp.ptr(T.f16(), 1), ptr=arg2, offset=v92)
    v97 = tt.broadcast(result=T.tensor(256, 256, T.i32(), encoding=mma), src=v94)
    v98 = tt.expand_dims(src=v63, axis=0)
    v99 = tt.broadcast(result=T.tensor(256, 256, T.i32(), encoding=mma), src=v98)
    v100 = tt.addptr(result=ttpp.ptr(T.f16(), 1), ptr=v95, offset=v42)
    v101 = arith.addi(lhs=v99, rhs=v97)
    # start mask
    v102 = tt.splat(result=T.tensor(256, 1, T.i32(), encoding=mma), src=arg3)
    v90 = tt.expand_dims(src=v65, axis=1)
    v103 = arith.cmpi(predicate=2, lhs=v90, rhs=v102)
    v104 = tt.splat(result=T.tensor(1, 256, T.i32(), encoding=mma), src=arg4)
    v96 = tt.expand_dims(src=v67, axis=0)
    v105 = arith.cmpi(predicate=2, lhs=v96, rhs=v104)
    v106 = tt.broadcast(result=T.tensor(256, 256, T.bool(), encoding=mma), src=v103)
    v107 = tt.broadcast(result=T.tensor(256, 256, T.bool(), encoding=mma), src=v105)
    v108 = arith.andi(lhs=v106, rhs=v107)
    # end mask
    amdgpu.buffer_store(value=v89, ptr=v100, offsets=v101, stride=arg8, cache=1, mask=v108)

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
    assert np.sum((c_h - correct) != 0) < 0.005
# print(c_h)
