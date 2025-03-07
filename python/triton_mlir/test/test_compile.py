from textwrap import dedent

import pytest
from triton_mlir.extras.context import mlir_mod_ctx

# this needs to be below the triton_mlir_bindings
from triton_mlir.extras.dialects.ext import arith

# noinspection PyUnresolvedReferences
from triton_mlir.extras.testing import mlir_ctx as ctx, filecheck, MLIRContext

pytest.mark.usefixtures("ctx")

from triton_mlir.dialects import tt
from triton_mlir.types import T
from triton_mlir.compiler import (
    HIPBackend,
    unwrap_c_module_op,
    tritonir,
    llvm,
    make_backend,
)


def test_smoke_test(ctx):
    src = dedent(
        """\
        module {
          %0 = tensor.empty() : tensor<16x16x!tt.ptr<f32>>
          %1 = tensor.empty() : tensor<16x16xi32>
          %2 = tt.addptr %0, %1 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
        }
        """
    )
    mod = ctx.module.parse(src)
    # TypeError: Expected an MLIR object (got <triton_mlir._mlir_libs._mlir.ir.Module object at 0x795602bdc090>).
    # llvm-install\include\mlir\Bindings\Python\NanobindAdaptors.h:49
    # triton_mod = unwrap_c_module_op(mod)
    triton_mod = unwrap_c_module_op(mod.operation)
    assert isinstance(triton_mod, tritonir.module)


def test_make_ttir(ctx):
    @tt.jit
    def kernel_0123(
        arg0: +T.float32, arg1: +T.float32, arg2: +T.float32, arg3: T.int32
    ):
        v0 = tt.get_program_id(axis=tt.ProgramIDDim.X)
        c32 = arith.constant(64, T.int32)
        v1 = v0 * c32
        v2 = tt.make_range(0, 64)
        v3 = tt.splat(v1, (64,))
        v4 = arith.addi(v3, v2)
        v5 = tt.splat(arg3, (64,))
        v6 = arith.cmpi("slt", v4, v5)
        v7 = tt.splat(arg0, (64,))
        v8 = tt.addptr(v7, v4)
        v9 = tt.load(
            v8,
            v6,
            cache=tt.CacheModifier.NONE,
            evict=tt.EvictionPolicy.NORMAL,
            is_volatile=False,
        )
        v10 = tt.splat(arg1, (64,))
        v11 = tt.addptr(v10, v4)
        v12 = tt.load(
            v11,
            v6,
            cache=tt.CacheModifier.NONE,
            evict=tt.EvictionPolicy.NORMAL,
            is_volatile=False,
        )
        v13 = arith.addf(v9, v12)
        v14 = tt.splat(arg2, (64,))
        v15 = tt.addptr(v14, v4)
        tt.store(v15, v13, v6)
        tt.return_(srcs=[])

    kernel_0123.emit()
    assert ctx.module.operation.verify()
    triton_mod = unwrap_c_module_op(ctx.module.operation)
    assert isinstance(triton_mod, tritonir.module)
    backend = make_backend("gfx1100", 32)
    ttir_mod = backend.make_ttir(triton_mod)
    assert isinstance(triton_mod, tritonir.module)
    assert ttir_mod.verify()


def test_make_ttgir(ctx):
    @tt.jit
    def kernel_0123(
        arg0: +T.float32, arg1: +T.float32, arg2: +T.float32, arg3: T.int32
    ):
        v0 = tt.get_program_id(axis=tt.ProgramIDDim.X)
        c32 = arith.constant(64, T.int32)
        v1 = v0 * c32
        v2 = tt.make_range(0, 64)
        v3 = tt.splat(v1, (64,))
        v4 = arith.addi(v3, v2)
        v5 = tt.splat(arg3, (64,))
        v6 = arith.cmpi("slt", v4, v5)
        v7 = tt.splat(arg0, (64,))
        v8 = tt.addptr(v7, v4)
        v9 = tt.load(
            v8,
            v6,
            cache=tt.CacheModifier.NONE,
            evict=tt.EvictionPolicy.NORMAL,
            is_volatile=False,
        )
        v10 = tt.splat(arg1, (64,))
        v11 = tt.addptr(v10, v4)
        v12 = tt.load(
            v11,
            v6,
            cache=tt.CacheModifier.NONE,
            evict=tt.EvictionPolicy.NORMAL,
            is_volatile=False,
        )
        v13 = arith.addf(v9, v12)
        v14 = tt.splat(arg2, (64,))
        v15 = tt.addptr(v14, v4)
        tt.store(v15, v13, v6)
        tt.return_(srcs=[])

    kernel_0123.emit()
    ctx.module.operation.verify()

    triton_mod = unwrap_c_module_op(ctx.module.operation)
    assert isinstance(triton_mod, tritonir.module)
    backend = make_backend("gfx1100", 32)
    ttir_mod = backend.make_ttir(triton_mod)
    assert isinstance(ttir_mod, tritonir.module)
    assert ttir_mod.verify()
    options = backend.parse_options({})
    ttgir_mod = backend.make_ttgir(ttir_mod, options)
    assert isinstance(ttgir_mod, tritonir.module)
    assert ttgir_mod.verify()


def test_make_llir(ctx):
    @tt.jit
    def kernel_0123(
        arg0: +T.float32, arg1: +T.float32, arg2: +T.float32, arg3: T.int32
    ):
        v0 = tt.get_program_id(axis=tt.ProgramIDDim.X)
        c32 = arith.constant(64, T.int32)
        v1 = v0 * c32
        v2 = tt.make_range(0, 64)
        v3 = tt.splat(v1, (64,))
        v4 = arith.addi(v3, v2)
        v5 = tt.splat(arg3, (64,))
        v6 = arith.cmpi("slt", v4, v5)
        v7 = tt.splat(arg0, (64,))
        v8 = tt.addptr(v7, v4)
        v9 = tt.load(
            v8,
            v6,
            cache=tt.CacheModifier.NONE,
            evict=tt.EvictionPolicy.NORMAL,
            is_volatile=False,
        )
        v10 = tt.splat(arg1, (64,))
        v11 = tt.addptr(v10, v4)
        v12 = tt.load(
            v11,
            v6,
            cache=tt.CacheModifier.NONE,
            evict=tt.EvictionPolicy.NORMAL,
            is_volatile=False,
        )
        v13 = arith.addf(v9, v12)
        v14 = tt.splat(arg2, (64,))
        v15 = tt.addptr(v14, v4)
        tt.store(v15, v13, v6)
        tt.return_(srcs=[])

    kernel_0123.emit()
    ctx.module.operation.verify()

    triton_mod = unwrap_c_module_op(ctx.module.operation)
    assert isinstance(triton_mod, tritonir.module)

    backend = make_backend("gfx1100", 32)
    ttir_mod = backend.make_ttir(triton_mod)
    assert isinstance(ttir_mod, tritonir.module)
    assert ttir_mod.verify()
    options = backend.parse_options({})
    ttgir_mod = backend.make_ttgir(ttir_mod, options)
    assert isinstance(ttgir_mod, tritonir.module)
    assert ttgir_mod.verify()

    metadata = {}
    llvm_mod = backend.make_llir(ttgir_mod, metadata, options)
    assert isinstance(llvm_mod, llvm.module)
    assert metadata.get("shared") == 0
    assert llvm_mod.verify()


def test_make_amdgcn(ctx):
    @tt.jit
    def kernel_0123(
        arg0: +T.float32, arg1: +T.float32, arg2: +T.float32, arg3: T.int32
    ):
        v0 = tt.get_program_id(axis=tt.ProgramIDDim.X)
        c32 = arith.constant(64, T.int32)
        v1 = v0 * c32
        v2 = tt.make_range(0, 64)
        v3 = tt.splat(v1, (64,))
        v4 = arith.addi(v3, v2)
        v5 = tt.splat(arg3, (64,))
        v6 = arith.cmpi("slt", v4, v5)
        v7 = tt.splat(arg0, (64,))
        v8 = tt.addptr(v7, v4)
        v9 = tt.load(
            v8,
            v6,
            cache=tt.CacheModifier.NONE,
            evict=tt.EvictionPolicy.NORMAL,
            is_volatile=False,
        )
        v10 = tt.splat(arg1, (64,))
        v11 = tt.addptr(v10, v4)
        v12 = tt.load(
            v11,
            v6,
            cache=tt.CacheModifier.NONE,
            evict=tt.EvictionPolicy.NORMAL,
            is_volatile=False,
        )
        v13 = arith.addf(v9, v12)
        v14 = tt.splat(arg2, (64,))
        v15 = tt.addptr(v14, v4)
        tt.store(v15, v13, v6)
        tt.return_(srcs=[])

    kernel_0123.emit()
    ctx.module.operation.verify()

    triton_mod = unwrap_c_module_op(ctx.module.operation)
    assert isinstance(triton_mod, tritonir.module)

    backend = make_backend("gfx1100", 32)

    ttir_mod = backend.make_ttir(triton_mod)
    assert isinstance(ttir_mod, tritonir.module)
    assert ttir_mod.verify()
    options = backend.parse_options({})
    ttgir_mod = backend.make_ttgir(ttir_mod, options)
    assert isinstance(ttgir_mod, tritonir.module)
    assert ttgir_mod.verify()

    metadata = {}
    llvm_mod = backend.make_llir(ttgir_mod, metadata, options)
    assert isinstance(llvm_mod, llvm.module)
    assert metadata.get("shared") == 0
    assert llvm_mod.verify()

    amdgcn = backend.make_amdgcn(str(llvm_mod), metadata, options)
    assert len(amdgcn)
    print(amdgcn)


def test_make_hsaco(ctx):
    @tt.jit
    def kernel_0123(
        arg0: +T.float32, arg1: +T.float32, arg2: +T.float32, arg3: T.int32
    ):
        v0 = tt.get_program_id(axis=tt.ProgramIDDim.X)
        c32 = arith.constant(64, T.int32)
        v1 = v0 * c32
        v2 = tt.make_range(0, 64)
        v3 = tt.splat(v1, (64,))
        v4 = arith.addi(v3, v2)
        v5 = tt.splat(arg3, (64,))
        v6 = arith.cmpi("slt", v4, v5)
        v7 = tt.splat(arg0, (64,))
        v8 = tt.addptr(v7, v4)
        v9 = tt.load(
            v8,
            v6,
            cache=tt.CacheModifier.NONE,
            evict=tt.EvictionPolicy.NORMAL,
            is_volatile=False,
        )
        v10 = tt.splat(arg1, (64,))
        v11 = tt.addptr(v10, v4)
        v12 = tt.load(
            v11,
            v6,
            cache=tt.CacheModifier.NONE,
            evict=tt.EvictionPolicy.NORMAL,
            is_volatile=False,
        )
        v13 = arith.addf(v9, v12)
        v14 = tt.splat(arg2, (64,))
        v15 = tt.addptr(v14, v4)
        tt.store(v15, v13, v6)
        tt.return_(srcs=[])

    kernel_0123.emit()
    ctx.module.operation.verify()

    triton_mod = unwrap_c_module_op(ctx.module.operation)
    assert isinstance(triton_mod, tritonir.module)

    backend = make_backend("gfx1100", 32)

    ttir_mod = backend.make_ttir(triton_mod)
    assert isinstance(ttir_mod, tritonir.module)
    assert ttir_mod.verify()
    options = backend.parse_options({})
    ttgir_mod = backend.make_ttgir(ttir_mod, options)
    assert isinstance(ttgir_mod, tritonir.module)
    assert ttgir_mod.verify()

    metadata = {}
    llvm_mod = backend.make_llir(ttgir_mod, metadata, options)
    assert isinstance(llvm_mod, llvm.module)
    assert metadata.get("shared") == 0
    assert llvm_mod.verify()

    amdgcn = backend.make_amdgcn(str(llvm_mod), metadata, options)
    assert len(amdgcn)
    hsaco = backend.make_hsaco(amdgcn, options)
    assert len(hsaco)
    assert "kernel_0123" in str(hsaco)


def test_compile(ctx):
    @tt.jit
    def kernel_0123(
        arg0: +T.float32, arg1: +T.float32, arg2: +T.float32, arg3: T.int32
    ):
        v0 = tt.get_program_id(axis=tt.ProgramIDDim.X)
        c32 = arith.constant(64, T.int32)
        v1 = v0 * c32
        v2 = tt.make_range(0, 64)
        v3 = tt.splat(v1, (64,))
        v4 = arith.addi(v3, v2)
        v5 = tt.splat(arg3, (64,))
        v6 = arith.cmpi("slt", v4, v5)
        v7 = tt.splat(arg0, (64,))
        v8 = tt.addptr(v7, v4)
        v9 = tt.load(
            v8,
            v6,
            cache=tt.CacheModifier.NONE,
            evict=tt.EvictionPolicy.NORMAL,
            is_volatile=False,
        )
        v10 = tt.splat(arg1, (64,))
        v11 = tt.addptr(v10, v4)
        v12 = tt.load(
            v11,
            v6,
            cache=tt.CacheModifier.NONE,
            evict=tt.EvictionPolicy.NORMAL,
            is_volatile=False,
        )
        v13 = arith.addf(v9, v12)
        v14 = tt.splat(arg2, (64,))
        v15 = tt.addptr(v14, v4)
        tt.store(v15, v13, v6)
        tt.return_(srcs=[])

    kernel_0123.emit()
    ctx.module.operation.verify()

    triton_mod = unwrap_c_module_op(ctx.module.operation)
    arch = "gfx1100"
    backend = make_backend("gfx1100", 32)
    hsaco, metadata = backend.compile(triton_mod, {"arch": arch})
    assert len(hsaco)
    assert "kernel_0123" in str(hsaco)
    assert metadata.get("shared") == 0


if __name__ == "__main__":
    with mlir_mod_ctx(allow_unregistered_dialects=True) as ctx:
        test_make_ttir(ctx)
    with mlir_mod_ctx(allow_unregistered_dialects=True) as ctx:
        test_make_ttgir(ctx)
    with mlir_mod_ctx(allow_unregistered_dialects=True) as ctx:
        test_make_llir(ctx)
    with mlir_mod_ctx(allow_unregistered_dialects=True) as ctx:
        test_make_amdgcn(ctx)
    with mlir_mod_ctx(allow_unregistered_dialects=True) as ctx:
        test_make_hsaco(ctx)
