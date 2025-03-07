import array
import math
import random

import numpy as np
import pytest
from triton_mlir.extras.context import mlir_mod_ctx

# this needs to be below the triton_mlir_bindings
from triton_mlir.extras.dialects.ext import arith

# noinspection PyUnresolvedReferences
from triton_mlir.extras.testing import mlir_ctx as ctx, filecheck, MLIRContext

from triton_mlir.dialects import tt, ttpp
from triton_mlir.types import T
from triton_mlir.compiler import (
    HIPBackend,
    unwrap_c_module_op,
    tritonir,
    llvm,
    HIPBackend,
)

# noinspection PyUnresolvedReferences
from util import hip_bindings_not_installed, hip_check, backend, backend_

pytest.mark.usefixtures("backend")
pytest.mark.usefixtures("ctx")


if hip_bindings_not_installed():
    pytest.skip(allow_module_level=True)

from hip import hip


# https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html
def test_run_vector_add_bare(ctx: MLIRContext, backend: HIPBackend):
    BLOCK_SIZE = 64

    @tt.jit
    def vector_add(
        x: +T.float32, y: +T.float32, output: +T.float32, n_elements: T.int32
    ):
        v0 = tt.get_program_id(axis=tt.ProgramIDDim.X)
        c32 = arith.constant(BLOCK_SIZE, T.int32)
        v1 = v0 * c32
        v2 = tt.make_range(0, BLOCK_SIZE)
        v3 = tt.splat(v1, (BLOCK_SIZE,))
        v4 = arith.addi(v3, v2)
        v5 = tt.splat(n_elements, (BLOCK_SIZE,))
        v6 = arith.cmpi("slt", v4, v5)
        v7 = tt.splat(x, (BLOCK_SIZE,))
        v8 = tt.addptr(v7, v4)
        v9 = tt.load(
            v8,
            v6,
            cache=tt.CacheModifier.NONE,
            evict=tt.EvictionPolicy.NORMAL,
            is_volatile=False,
        )
        v10 = tt.splat(y, (BLOCK_SIZE,))
        v11 = tt.addptr(v10, v4)
        v12 = tt.load(
            v11,
            v6,
            cache=tt.CacheModifier.NONE,
            evict=tt.EvictionPolicy.NORMAL,
            is_volatile=False,
        )
        v13 = arith.addf(v9, v12)
        v14 = tt.splat(output, (BLOCK_SIZE,))
        v15 = tt.addptr(v14, v4)
        tt.store(v15, v13, v6)
        tt.return_(srcs=[])

    vector_add.emit()
    ctx.module.operation.verify()
    triton_mod = unwrap_c_module_op(ctx.module.operation)
    hsaco, metadata = backend.compile(triton_mod, {"arch": backend.target.arch})
    assert metadata.get("shared") == 0

    props = hip.hipDeviceProp_t()
    hip_check(hip.hipGetDeviceProperties(props, 0))
    module = hip_check(hip.hipModuleLoadData(hsaco))
    kernel = hip_check(hip.hipModuleGetFunction(module, b"vector_add"))

    # kernel launch

    ## inputs
    n_elements = 128
    x_h = array.array("f", [random.random() for i in range(0, n_elements)])
    y_h = array.array("f", [random.random() for i in range(0, n_elements)])
    output_h = array.array("f", [0 for i in range(0, n_elements)])

    num_bytes = x_h.itemsize * len(x_h)
    x_d = hip_check(hip.hipMalloc(num_bytes))
    y_d = hip_check(hip.hipMalloc(num_bytes))
    output_d = hip_check(hip.hipMalloc(num_bytes))
    print(f"{hex(int(x_d))=}")

    ## upload host data
    hip_check(
        hip.hipMemcpy(x_d, x_h, num_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice)
    )
    hip_check(
        hip.hipMemcpy(y_d, y_h, num_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice)
    )
    hip_check(
        hip.hipMemcpy(
            output_d, output_h, num_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice
        )
    )

    ## expected result
    output_expected = [x + y for x, y in zip(x_h, y_h)]

    block = hip.dim3(x=BLOCK_SIZE)
    grid = hip.dim3(math.ceil(n_elements / block.x))

    ## launch
    hip_check(
        hip.hipModuleLaunchKernel(
            kernel,
            *grid,
            *block,
            sharedMemBytes=0,
            stream=None,
            kernelParams=None,
            extra=(x_d, y_d, output_d, n_elements),
        )
    )

    # copy result back
    hip_check(
        hip.hipMemcpy(
            output_h, output_d, num_bytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost
        )
    )

    for i, output_h_i in enumerate(output_h):
        if not math.isclose(output_h_i, output_expected[i], rel_tol=1e-6):
            raise RuntimeError(
                f"values do not match, {output_h[i]=} vs. {output_expected[i]=}, {i=}"
            )

    hip_check(hip.hipFree(x_d))
    hip_check(hip.hipFree(y_d))
    hip_check(hip.hipFree(output_d))

    hip_check(hip.hipModuleUnload(module))


# https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html
def test_run_vector_add_np(ctx: MLIRContext, backend: HIPBackend):
    BLOCK_SIZE = 64

    @tt.jit
    def vector_add(
        x: +T.float32, y: +T.float32, output: +T.float32, n_elements: T.int32
    ):
        v0 = tt.get_program_id(axis=tt.ProgramIDDim.X)
        tt.print_(prefix=" pid: ", hex=False, args=[v0], is_signed=[1])
        c32 = arith.constant(BLOCK_SIZE, T.int32)
        v1 = v0 * c32
        v2 = tt.make_range(0, BLOCK_SIZE)
        v3 = tt.splat(v1, (BLOCK_SIZE,))
        v4 = arith.addi(v3, v2)
        v5 = tt.splat(n_elements, (BLOCK_SIZE,))
        v6 = arith.cmpi("slt", v4, v5)
        v7 = tt.splat(x, (BLOCK_SIZE,))
        v8 = tt.addptr(v7, v4)
        v9 = tt.load(
            v8,
            v6,
            cache=tt.CacheModifier.NONE,
            evict=tt.EvictionPolicy.NORMAL,
            is_volatile=False,
        )
        v10 = tt.splat(y, (BLOCK_SIZE,))
        v11 = tt.addptr(v10, v4)
        v12 = tt.load(
            v11,
            v6,
            cache=tt.CacheModifier.NONE,
            evict=tt.EvictionPolicy.NORMAL,
            is_volatile=False,
        )
        v13 = arith.addf(v9, v12)
        v14 = tt.splat(output, (BLOCK_SIZE,))
        v15 = tt.addptr(v14, v4)
        tt.store(v15, v13, v6)
        tt.return_(srcs=[])

    vector_add.emit()
    ctx.module.operation.verify()
    triton_mod = unwrap_c_module_op(ctx.module.operation)
    hsaco, metadata = backend.compile(triton_mod, {"arch": backend.target.arch})
    assert metadata.get("shared") == 0

    props = hip.hipDeviceProp_t()
    hip_check(hip.hipGetDeviceProperties(props, 0))
    module = hip_check(hip.hipModuleLoadData(hsaco))
    kernel = hip_check(hip.hipModuleGetFunction(module, b"vector_add"))

    # kernel launch

    ## inputs
    n_elements = 128
    x_h = np.random.rand(n_elements).astype(dtype=np.float32)
    y_h = np.random.rand(n_elements).astype(dtype=np.float32)
    output_h = np.zeros((n_elements,)).astype(dtype=np.float32)

    # device vectors
    num_bytes = n_elements * x_h.itemsize
    x_d = hip_check(hip.hipMalloc(num_bytes))
    y_d = hip_check(hip.hipMalloc(num_bytes))
    output_d = hip_check(hip.hipMalloc(num_bytes))

    # copy input data to device
    hip_check(
        hip.hipMemcpy(x_d, x_h, num_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice)
    )
    hip_check(
        hip.hipMemcpy(y_d, y_h, num_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice)
    )
    hip_check(
        hip.hipMemcpy(
            output_d, output_h, num_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice
        )
    )

    block = hip.dim3(x=BLOCK_SIZE)
    grid = hip.dim3(math.ceil(n_elements / block.x))

    ## launch
    hip_check(
        hip.hipModuleLaunchKernel(
            kernel,
            *grid,
            *block,
            sharedMemBytes=0,
            stream=None,
            kernelParams=None,
            extra=(x_d, y_d, output_d, n_elements),
        )
    )

    # copy result back
    hip_check(
        hip.hipMemcpy(
            output_h, output_d, num_bytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost
        )
    )

    assert np.allclose(output_h, x_h + y_h)

    hip_check(hip.hipFree(x_d))
    hip_check(hip.hipFree(y_d))
    hip_check(hip.hipFree(output_d))

    hip_check(hip.hipModuleUnload(module))


# https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html
def test_run_vector_add_np_sure(ctx: MLIRContext, backend: HIPBackend):
    BLOCK_SIZE = 64

    @tt.jit
    def vector_add(
        x_ptr: +T.float32,
        y_ptr: +T.float32,
        output_ptr: +T.float32,
        n_elements: T.int32,
    ):
        pid = tt.get_program_id(axis=tt.ProgramIDDim.X)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tt.make_range(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tt.load(x_ptr + offsets, mask)
        y = tt.load(y_ptr + offsets, mask)
        output = x + y
        tt.store(output_ptr + offsets, output, mask)

        tt.return_(srcs=[])

    vector_add.emit()
    ctx.module.operation.verify()
    triton_mod = unwrap_c_module_op(ctx.module.operation)
    hsaco, metadata = backend.compile(triton_mod, {"arch": backend.target.arch})
    assert metadata.get("shared") == 0

    props = hip.hipDeviceProp_t()
    hip_check(hip.hipGetDeviceProperties(props, 0))
    module = hip_check(hip.hipModuleLoadData(hsaco))
    kernel = hip_check(hip.hipModuleGetFunction(module, b"vector_add"))

    # kernel launch

    ## inputs
    n_elements = 128
    x_h = np.random.rand(n_elements).astype(dtype=np.float32)
    y_h = np.random.rand(n_elements).astype(dtype=np.float32)
    output_h = np.zeros((n_elements,)).astype(dtype=np.float32)

    # device vectors
    num_bytes = n_elements * x_h.itemsize
    x_d = hip_check(hip.hipMalloc(num_bytes))
    y_d = hip_check(hip.hipMalloc(num_bytes))
    output_d = hip_check(hip.hipMalloc(num_bytes))

    # copy input data to device
    hip_check(
        hip.hipMemcpy(x_d, x_h, num_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice)
    )
    hip_check(
        hip.hipMemcpy(y_d, y_h, num_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice)
    )
    hip_check(
        hip.hipMemcpy(
            output_d, output_h, num_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice
        )
    )

    block = hip.dim3(x=BLOCK_SIZE)
    grid = hip.dim3(math.ceil(n_elements / block.x))

    ## launch
    hip_check(
        hip.hipModuleLaunchKernel(
            kernel,
            *grid,
            *block,
            sharedMemBytes=0,
            stream=None,
            kernelParams=None,
            extra=(x_d, y_d, output_d, n_elements),
        )
    )

    # copy result back
    hip_check(
        hip.hipMemcpy(
            output_h, output_d, num_bytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost
        )
    )

    assert np.allclose(output_h, x_h + y_h)

    hip_check(hip.hipFree(x_d))
    hip_check(hip.hipFree(y_d))
    hip_check(hip.hipFree(output_d))

    hip_check(hip.hipModuleUnload(module))


if __name__ == "__main__":
    backend = backend_()
    with mlir_mod_ctx(allow_unregistered_dialects=True) as ctx:
        test_run_vector_add_bare(ctx, backend)
    with mlir_mod_ctx(allow_unregistered_dialects=True) as ctx:
        test_run_vector_add_np(ctx, backend)
