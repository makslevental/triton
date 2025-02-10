import threading

import llvm
import pytest
from triton_mlir.extras.context import mlir_mod_ctx

# this needs to be below the triton_mlir_bindings
from triton_mlir.extras.dialects.ext import arith

# noinspection PyUnresolvedReferences
from triton_mlir.extras.testing import mlir_ctx as ctx, filecheck, MLIRContext

from triton_mlir.dialects import tt, ttpp
from triton_mlir.dialects.ttpp import T
from triton_mlir.runtime import make_ttir, make_ttgir, parse_options, make_llir

# noinspection PyUnresolvedReferences
from triton_mlir.dialects.ttpp import splat, arange, addptr, load, store
from llvm import print_module_to_string

pytest.mark.usefixtures("ctx")


def test_make_ttir(ctx):
    @ttpp.jit
    def kernel_0123(
        arg0: +T.float32, arg1: +T.float32, arg2: +T.float32, arg3: T.int32
    ):
        v0 = tt.get_program_id(axis=tt.ProgramIDDim.X)
        c32 = arith.constant(64, T.int32)
        v1 = v0 * c32
        v2 = arange(0, 64)
        v3 = splat(v1, (64,))
        v4 = arith.addi(v3, v2)
        v5 = splat(arg3, (64,))
        v6 = arith.cmpi("slt", v4, v5)
        v7 = splat(arg0, (64,))
        v8 = addptr(v7, v4)
        v9 = load(
            v8,
            v6,
            cache=tt.CacheModifier.NONE,
            evict=tt.EvictionPolicy.NORMAL,
            is_volatile=False,
        )
        v10 = splat(arg1, (64,))
        v11 = addptr(v10, v4)
        v12 = load(
            v11,
            v6,
            cache=tt.CacheModifier.NONE,
            evict=tt.EvictionPolicy.NORMAL,
            is_volatile=False,
        )
        v13 = arith.addf(v9, v12)
        v14 = splat(arg2, (64,))
        v15 = addptr(v14, v4)
        store(v15, v13, v6)

    kernel_0123.emit()
    ctx.module.operation.verify()
    mod = make_ttir(ctx.module)
    print(mod)


def test_make_ttgir(ctx):
    @ttpp.jit
    def kernel_0123(
        arg0: +T.float32, arg1: +T.float32, arg2: +T.float32, arg3: T.int32
    ):
        v0 = tt.get_program_id(axis=tt.ProgramIDDim.X)
        c32 = arith.constant(64, T.int32)
        v1 = v0 * c32
        v2 = arange(0, 64)
        v3 = splat(v1, (64,))
        v4 = arith.addi(v3, v2)
        v5 = splat(arg3, (64,))
        v6 = arith.cmpi("slt", v4, v5)
        v7 = splat(arg0, (64,))
        v8 = addptr(v7, v4)
        v9 = load(
            v8,
            v6,
            cache=tt.CacheModifier.NONE,
            evict=tt.EvictionPolicy.NORMAL,
            is_volatile=False,
        )
        v10 = splat(arg1, (64,))
        v11 = addptr(v10, v4)
        v12 = load(
            v11,
            v6,
            cache=tt.CacheModifier.NONE,
            evict=tt.EvictionPolicy.NORMAL,
            is_volatile=False,
        )
        v13 = arith.addf(v9, v12)
        v14 = splat(arg2, (64,))
        v15 = addptr(v14, v4)
        store(v15, v13, v6)

    kernel_0123.emit()
    ctx.module.operation.verify()
    mod = make_ttir(ctx.module)
    options = parse_options("gfx1100")
    mod = make_ttgir(mod, options)
    print(mod)


def test_make_llir(ctx):
    @ttpp.jit
    def kernel_0123(
        arg0: +T.float32, arg1: +T.float32, arg2: +T.float32, arg3: T.int32
    ):
        v0 = tt.get_program_id(axis=tt.ProgramIDDim.X)
        c32 = arith.constant(64, T.int32)
        v1 = v0 * c32
        v2 = arange(0, 64)
        v3 = splat(v1, (64,))
        v4 = arith.addi(v3, v2)
        v5 = splat(arg3, (64,))
        v6 = arith.cmpi("slt", v4, v5)
        v7 = splat(arg0, (64,))
        v8 = addptr(v7, v4)
        v9 = load(
            v8,
            v6,
            cache=tt.CacheModifier.NONE,
            evict=tt.EvictionPolicy.NORMAL,
            is_volatile=False,
        )
        v10 = splat(arg1, (64,))
        v11 = addptr(v10, v4)
        v12 = load(
            v11,
            v6,
            cache=tt.CacheModifier.NONE,
            evict=tt.EvictionPolicy.NORMAL,
            is_volatile=False,
        )
        v13 = arith.addf(v9, v12)
        v14 = splat(arg2, (64,))
        v15 = addptr(v14, v4)
        store(v15, v13, v6)

    kernel_0123.emit()
    ctx.module.operation.verify()
    mod = make_ttir(ctx.module)
    options = parse_options("gfx1100")
    mod = make_ttgir(mod, options)
    llvm_mod = make_llir(mod, options)
    print(print_module_to_string(llvm_mod))

    llvm.dispose_module(llvm_mod)


if __name__ == "__main__":
    with mlir_mod_ctx(allow_unregistered_dialects=True) as ctx:
        test_make_llir(ctx)
