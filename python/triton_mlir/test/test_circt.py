#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

# this needs to be below the triton_mlir_bindings
from triton_mlir.extras.dialects.ext import arith
from triton_mlir.extras.dialects.ext.func import func

# noinspection PyUnresolvedReferences
from triton_mlir.extras.testing import mlir_ctx as ctx, filecheck, MLIRContext

from triton_mlir.dialects import tt
from triton_mlir.types import T
from triton_mlir.passmanager import PassManager

# noinspection PyUnresolvedReferences
from triton_mlir.dialects.tt import splat, arange, addptr, load, store

from triton_mlir.passes import TritonPipeline

pytest.mark.usefixtures("ctx")


def test_vadd(ctx):
    @func(emit=True)
    def kernel_0123(v0: T.int32, v1: T.int32):
        c32 = arith.constant(64, T.int32)
        v2 = v0 * c32
        v3 = arith.addi(v0, v2)


    print("before")
    print(ctx.module)
    p = TritonPipeline().convert_arith_to_smt()
    pm = PassManager.parse(p.materialize())
    pm.run(ctx.module.operation)
    print("after")
    print(ctx.module)
