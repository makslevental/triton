import pytest
from triton_mlir.ir import (
    Context as c_Context,
    Attribute as c_Attribute,
    Type as c_Type,
)

# noinspection PyUnresolvedReferences
from triton_mlir.dialects.tt import splat, make_range, addptr, load, store
from triton_mlir.dialects.ttg import SwizzledSharedEncodingAttr, CTALayoutAttr

# noinspection PyUnresolvedReferences
from triton_mlir.extras.testing import mlir_ctx as ctx, filecheck
from triton_mlir.types import T


# this needs to be below the triton_mlir_bindings

pytest.mark.usefixtures("ctx")


def test_SwizzledSharedEncodingAttr(ctx):
    swizzled = SwizzledSharedEncodingAttr.get(
        vec=8,
        per_phase=1,
        max_phase=8,
        order=[0, 1],
        cta_layout=CTALayoutAttr.get(
            ct_as_per_cga=[1, 2], cta_split_num=[1, 2], cta_order=[0, 1]
        ),
    )
    assert (
        str(swizzled)
        == "#ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], CTAsPerCGA = [1, 2], CTASplitNum = [1, 2], CTAOrder = [0, 1]}>"
    )

    assert swizzled.vec == 8
    assert swizzled.per_phase == 1
    assert swizzled.max_phase == 8
    assert swizzled.order == [0, 1]
