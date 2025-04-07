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
from util import hip_check, launch_kernel, hip_synchronize, backend_


def time_to_gflops(time_ms, N):
    return 1e-6 * (N * N * N * 2 + 3 * N * N) // time_ms


# just so it doesn't get DCE'd by black/reformat
# TypeError: 'mlir._mlir_libs._mlir.ir.BlockArgument' object is not subscriptable
_ = memref


props = hip.hipDeviceProp_t()
hip_check(hip.hipGetDeviceProperties(props, 0))
arch = props.gcnArchName.decode()
# arch = "gfx942"

WAVES_PER_EU = 1
NUM_WARPS = 4
backend = backend_()
options = backend.parse_options(
    {"arch": arch, "waves_per_eu": WAVES_PER_EU, "num_warps": NUM_WARPS}
)

M, K, N = 1024, 1024, 1024


with open("/home/mlevental/dev_projects/triton/SWDEV-512461/1_lp_original.ttgir") as f:
    src = f.read()


with mlir_mod_ctx(src=src) as ctx:
    print(_tritonamdgpu_schedhintvariantattr(SchedHint.none, ctx.context))
    print(ctx.module)

    ttgir_mod = unwrap_c_module_op(ctx.module.operation)
    # metadata = {}
    # llvm_mod = backend.make_llir(ttgir_mod, metadata, options)
    hsaco, metadata = backend.compile(
        ttgir_mod,
        ttir=False,
        ttgir=False,
        options=options,
        dump_ir=True,
        ir_dump_dir=Path(__file__).parent / "david",
        dump_file_prefix="0",
    )