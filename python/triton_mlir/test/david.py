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
    print_attr_alias,
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
print_prolog()
with mlir_mod_ctx(src) as ctx:
    for line in src.splitlines():
        if line.startswith("#") and not line.startswith("#loc"):
            print_attr_alias(line)
    ctx.module.operation.walk(generic_print_walk_callback, WalkOrder.PRE_ORDER)
    print_epilog()
    OUTPUT_BUF.seek(0)
    with open(HERE / "matmul_kernel_david_ttgir.py", "w") as f:
        f.write(OUTPUT_BUF.read())
    OUTPUT_BUF.seek(0)

# noinspection PyUnresolvedReferences
import matmul_kernel_david_ttgir

s = str(matmul_kernel_david_ttgir.ctx.module)

if s != src:
    print(s)
    print(src)
    assert False
