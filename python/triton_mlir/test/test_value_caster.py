import pytest
from triton_mlir.dialects.ttpp import T
from triton_mlir.extras.context import MLIRContext

# this needs to be below the triton_mlir_bindings
from triton_mlir.extras.dialects.ext.tensor import empty


# noinspection PyUnresolvedReferences
from triton_mlir.extras.testing import mlir_ctx as ctx, filecheck, MLIRContext

pytest.mark.usefixtures("ctx")


def test_value_caster(ctx: MLIRContext):
    t = empty(10, 10, T.float32)
    assert repr(t) == "Tensor(%0 = tensor.empty() : tensor<10x10xf32>)"

    t = empty(10, 10, +T.float32)
    assert repr(t) == "TritonTensor(%1 = tensor.empty() : tensor<10x10x!tt.ptr<f32>>)"
