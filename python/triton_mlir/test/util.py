import os

import pytest

from triton_mlir.compiler import HIPBackend, make_backend, ENV_OR_DEFAULT_ARCH


def hip_check(call_result):
    from hip import hip

    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    return result


def hip_bindings_not_installed():
    try:
        from hip import hip

        props = hip.hipDeviceProp_t()
        hip_check(hip.hipGetDeviceProperties(props, 0))

        # don't skip
        return False

    except:
        # skip
        return True


@pytest.fixture
def backend(request) -> HIPBackend:
    arch = None
    if hasattr(request.node, "callspec"):
        arch = request.node.callspec.params.get("arch")
    if arch is None:
        arch = ENV_OR_DEFAULT_ARCH

    warp_size = 32 if "gfx10" in arch or "gfx11" in arch or "gfx12" in arch else 64
    return make_backend(arch, warp_size)
