import array
import math
import random
from pathlib import Path

import numpy as np
import pytest
from triton_mlir.extras.context import mlir_mod_ctx

# this needs to be below the triton_mlir_bindings
from triton_mlir.extras.dialects.ext import arith

# noinspection PyUnresolvedReferences
from triton_mlir.extras.testing import mlir_ctx as ctx, filecheck, MLIRContext

from triton_mlir.dialects import tt
from triton_mlir.types import T
from triton_mlir.compiler import HIPBackend, unwrap_c_module_op, tritonir, llvm

from util import hip_bindings_not_installed, hip_check, backend

pytest.mark.usefixtures("backend")
pytest.mark.usefixtures("ctx")


if hip_bindings_not_installed():
    pytest.skip(allow_module_level=True)

from hip import hip, hiprtc


def test_smoke():
    props = hip.hipDeviceProp_t()
    hip_check(hip.hipGetDeviceProperties(props, 0))

    source = b"""\
    extern "C" __global__ void print_tid() {
      printf("tid: %d\\n", (int) threadIdx.x);
    }
    """

    prog = hip_check(hiprtc.hiprtcCreateProgram(source, b"print_tid", 0, [], []))

    props = hip.hipDeviceProp_t()
    hip_check(hip.hipGetDeviceProperties(props, 0))
    arch = props.gcnArchName

    print(f"Compiling kernel for {arch}")

    cflags = [b"--offload-arch=" + arch]
    (err,) = hiprtc.hiprtcCompileProgram(prog, len(cflags), cflags)
    if err != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
        log_size = hip_check(hiprtc.hiprtcGetProgramLogSize(prog))
        log = bytearray(log_size)
        hip_check(hiprtc.hiprtcGetProgramLog(prog, log))
        raise RuntimeError(log.decode())
    code_size = hip_check(hiprtc.hiprtcGetCodeSize(prog))
    code = bytearray(code_size)
    hip_check(hiprtc.hiprtcGetCode(prog, code))
    module = hip_check(hip.hipModuleLoadData(code))
    kernel = hip_check(hip.hipModuleGetFunction(module, b"print_tid"))
    #
    hip_check(
        hip.hipModuleLaunchKernel(
            kernel,
            *(1, 1, 1),  # grid
            *(32, 1, 1),  # block
            sharedMemBytes=0,
            stream=None,
            kernelParams=None,
            extra=None,
        )
    )

    hip_check(hip.hipDeviceSynchronize())
    hip_check(hip.hipModuleUnload(module))
    hip_check(hiprtc.hiprtcDestroyProgram(prog.createRef()))

    print("ok")


# https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html
@pytest.mark.parametrize("arch", ["gfx1100"])
def test_run_vector_add_bare(ctx, backend, arch):
    BLOCK_SIZE = 64

    @tt.jit
    def vector_add(
        x: +T.float32, y: +T.float32, output: +T.float32, n_elements: T.int32
    ):
        v0 = tt.get_program_id(axis=tt.ProgramIDDim.X)
        c32 = arith.constant(BLOCK_SIZE, T.int32)
        v1 = v0 * c32
        v2 = tt.arange(0, BLOCK_SIZE)
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
    hsaco, metadata = backend.compile(triton_mod, {"arch": arch})
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
@pytest.mark.parametrize("arch", ["gfx1100"])
def test_run_vector_add_np(ctx, backend, arch):
    BLOCK_SIZE = 64

    @tt.jit
    def vector_add(
        x: +T.float32, y: +T.float32, output: +T.float32, n_elements: T.int32
    ):
        v0 = tt.get_program_id(axis=tt.ProgramIDDim.X)
        tt.print_(prefix=" pid: ", hex=False, args=[v0], is_signed=[1])
        c32 = arith.constant(BLOCK_SIZE, T.int32)
        v1 = v0 * c32
        v2 = tt.arange(0, BLOCK_SIZE)
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
    hsaco, metadata = backend.compile(triton_mod, {"arch": arch})
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
    test_smoke()
    with mlir_mod_ctx(allow_unregistered_dialects=True) as ctx:
        test_run_vector_add_bare(ctx)
    with mlir_mod_ctx(allow_unregistered_dialects=True) as ctx:
        test_run_vector_add_np(ctx)
