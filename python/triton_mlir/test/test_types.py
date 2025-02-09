import pytest
from triton_mlir.ir import Type
from triton_mlir.dialects.tensor import empty
from triton_mlir.dialects.ttpp import PointerType, T, is_ptr
from triton_mlir.extras.context import MLIRContext
from triton_mlir.extras.testing import mlir_ctx as ctx

pytest.mark.usefixtures("ctx")


def test_ptr_type(ctx: MLIRContext):
    p_f32 = PointerType.of_pointee_type(T.f32)
    assert (
        (+T.f16).typeid
        == p_f32.typeid
        == (+T.f64).typeid
        == p_f32.typeid
        == (+T.bf16).typeid
        != T.f32.typeid
    )

    assert is_ptr(Type.parse(f"!tt.ptr<f32>"))
    assert is_ptr(Type.parse(f"!tt.ptr<bf16>"))

    assert is_ptr(+T.f16)
    assert is_ptr(p_f32)
    assert is_ptr(+T.f64)
    assert is_ptr(+T.bf16)


def test_tensor_ptrs(ctx: MLIRContext):
    t_f32_ptr_t = T.tensor(10, 10, +T.f32)
    assert repr(t_f32_ptr_t) == "RankedTensorType(tensor<10x10x!tt.ptr<f32>>)"
    tt = empty((10, 10), +T.f32)
    assert tt.type == t_f32_ptr_t
    assert tt.type.typeid == t_f32_ptr_t.typeid

    assert t_f32_ptr_t.typeid == tt.type.typeid
    assert not t_f32_ptr_t.typeid == tt.dtype.typeid

    ctx.module.operation.verify()


def test_plus_ptrs(ctx: MLIRContext):
    p_i16 = +T.int16
    p_i32 = +T.int32
    p_i64 = +T.int64

    p_f16 = +T.f16
    p_f32 = +T.f32
    p_f64 = +T.f64

    pp_i16 = +T.int16
    pp_i32 = +T.int32
    pp_i64 = +T.int64

    assert pp_i16 == p_i16
    assert pp_i32 == p_i32
    assert pp_i64 == p_i64

    pp_f16 = +T.f16
    pp_f32 = +T.f32
    pp_f64 = +T.f64

    assert pp_f16 == p_f16
    assert pp_f32 == p_f32
    assert pp_f64 == p_f64

    x = +++++++++T.int64
    assert (
        str(x)
        == "!tt.ptr<!tt.ptr<!tt.ptr<!tt.ptr<!tt.ptr<!tt.ptr<!tt.ptr<!tt.ptr<!tt.ptr<i64>>>>>>>>>"
    )
