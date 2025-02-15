from triton_mlir._mlir_libs._triton._eudsl.mlir import (
    unwrap_c_context,
    wrap_context,
    unwrap_c_type,
    wrap_type,
    unwrap_c_attribute,
    wrap_attribute,
    MLIRContext,
    SmallVector,
    Attribute,
    Float32Type,
    Type,
)
from triton_mlir.ir import (
    Context as c_Context,
    Attribute as c_Attribute,
    Type as c_Type,
)

# noinspection PyUnresolvedReferences
from triton_mlir.dialects.tt import splat, arange, addptr, load, store
from triton_mlir.dialects.ttg import SwizzledSharedEncodingAttr, CTALayoutAttr
from triton_mlir.extras.context import mlir_mod_ctx

# noinspection PyUnresolvedReferences
from triton_mlir.extras.testing import mlir_ctx as ctx, filecheck
from triton_mlir.types import T


# this needs to be below the triton_mlir_bindings

# pytest.mark.usefixtures("ctx")


def test_wrap_unrap_mlir_c(ctx):
    unwrapped_ctx = unwrap_c_context(ctx.context)
    assert isinstance(unwrapped_ctx, MLIRContext)
    wrapped_ctx = wrap_context(unwrapped_ctx)
    assert isinstance(wrapped_ctx, c_Context)

    swizzled = SwizzledSharedEncodingAttr.get(
        vec=8,
        per_phase=1,
        max_phase=8,
        order=[0, 1],
        cta_layout=CTALayoutAttr.get_default(unwrapped_ctx, rank=2),
    )
    assert (
        str(swizzled)
        == "#ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>"
    )
    wrapped_attr = wrap_attribute(swizzled)
    assert (
        str(wrapped_attr)
        == "#ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>"
    )
    assert isinstance(wrapped_attr, c_Attribute)
    unwrapped_attr = unwrap_c_attribute(wrapped_attr)
    assert isinstance(unwrapped_attr, Attribute)

    f32_ty = Float32Type.get(unwrapped_ctx)
    wrapped_type = wrap_type(f32_ty)
    assert str(f32_ty) == "f32"
    assert isinstance(wrapped_type, c_Type)
    unwrapped_type = unwrap_c_type(wrapped_type)
    assert isinstance(unwrapped_type, Type)


if __name__ == "__main__":
    with mlir_mod_ctx(allow_unregistered_dialects=True) as ctx:
        test_wrap_unrap_mlir_c(ctx)
