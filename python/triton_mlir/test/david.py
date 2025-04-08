import ctypes
import json
import math
from pathlib import Path
from textwrap import dedent

import numpy as np
from hip import hip

from triton_mlir.extras.ast.canonicalize import canonicalize
from triton_mlir.extras.context import RAIIMLIRContextModule, mlir_mod, mlir_mod_ctx
from triton_mlir.extras.dialects.ext import memref, scf, arith, gpu, llvm
from triton_mlir.dialects import index as index_dialect
from triton_mlir.ir import InsertionPoint, IntegerAttr, UnitAttr, Attribute
import triton_mlir.extras.types as T
from triton_mlir.compiler import unwrap_c_module_op

from triton_mlir.dialects._amdgpu_enum_gen import (
    _tritonamdgpu_schedhintvariantattr,
    SchedHint,
)

# noinspection PyUnresolvedReferences
from util import (
    hip_check,
    hip_synchronize,
    backend_,
    chip_check,
    print_prolog,
    generic_print_walk_callback,
    WalkOrder,
    print_epilog,
    OUTPUT_BUF,
)


def time_to_gflops(time_ms, N):
    return 1e-6 * (N * N * N * 2 + 3 * N * N) // time_ms


# just so it doesn't get DCE'd by black/reformat
# TypeError: 'mlir._mlir_libs._mlir.ir.BlockArgument' object is not subscriptable
_ = memref


props = hip.hipDeviceProp_t()
hip_check(hip.hipGetDeviceProperties(props, 0))
arch = props.gcnArchName.decode()

HERE = Path(__file__).parent

backend = backend_()
options_dict = json.load(open(HERE / "matmul_kernel.json"))
options = backend.parse_options(options_dict)


# ctx = RAIIMLIRContextModule()
with open(HERE / "matmul_kernel_david.ttgir") as f:
    src = f.read()
with mlir_mod_ctx(src) as ctx:
    print_prolog()
    ctx.module.operation.walk(generic_print_walk_callback, WalkOrder.PRE_ORDER)
    print_epilog()
    OUTPUT_BUF.seek(0)
    with open(HERE / "matmul_kernel_david_ttgir.py", "w") as f:
        f.write(OUTPUT_BUF.read())
    OUTPUT_BUF.seek(0)

# noinspection PyUnresolvedReferences
import matmul_kernel_david_ttgir

print(matmul_kernel_david_ttgir.ctx.module)
assert str(matmul_kernel_david_ttgir.ctx.module) == src


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
BLOCK_SIZE_M = BLOCK_SIZE_N = 128

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
