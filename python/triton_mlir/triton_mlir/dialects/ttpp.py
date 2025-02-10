from typing import Optional, Sequence

import triton_mlir.extras.dialects.ext.tensor
from triton_mlir._mlir_libs._mlir import register_value_caster
from triton_mlir.dialects import tt, arith
from triton_mlir.dialects._ods_common import (
    get_op_result_or_value,
    get_default_loc_context,
    get_op_result_or_op_results,
)
from triton_mlir.dialects.arith import _is_integer_like_type
from triton_mlir.dialects.tt import (
    FuncOp,
    ReturnOp,
    CallOp,
    get_ptr_type_typeid,
    PointerType,
)
from triton_mlir.extras import types as _T
from triton_mlir.extras.dialects.ext.arith import Scalar, constant, _binary_op
from triton_mlir.extras.dialects.ext.func import FuncBase
from triton_mlir.extras.dialects.ext.tensor import Tensor
from triton_mlir.extras.util import (
    make_maybe_no_args_decorator,
    get_user_code_loc,
)
from triton_mlir.ir import (
    Attribute,
    IntegerAttr,
    Value,
    RankedTensorType,
    register_attribute_builder,
    Context,
    IntegerType,
    Type,
    AttrBuilder,
    ShapedType,
)


def is_ptr(o: Type | Value):
    if not isinstance(o, (Type, Value)):
        return False
    if isinstance(o, TritonPointer):
        return True
    if isinstance(o, Value):
        o = o.type
    if ShapedType.isinstance(o):
        o = ShapedType(o).element_type
    if o.typeid == get_ptr_type_typeid():
        return True
    return False


class _classproperty:
    def __init__(self, func):
        self.fget = func

    def __get__(self, instance, owner):
        return self.fget(owner)


class _PlusPtr(Type):
    def __pos__(self):
        return PointerType.of_pointee_type(self)


class T:
    @_classproperty
    def i16(cls):
        return _PlusPtr(_T.i16())

    @_classproperty
    def i32(cls):
        return _PlusPtr(_T.i32())

    @_classproperty
    def i64(cls):
        return _PlusPtr(_T.i64())

    @_classproperty
    def f16(cls):
        return _PlusPtr(_T.f16())

    @_classproperty
    def f32(cls):
        return _PlusPtr(_T.f32())

    @_classproperty
    def f64(cls):
        return _PlusPtr(_T.f64())

    @_classproperty
    def bf16(cls):
        return _PlusPtr(_T.bf16())

    # matches python/triton/language/core.py
    @_classproperty
    def void(cls):
        return _T.none()

    @_classproperty
    def int1(cls):
        return _T.bool()

    # note that triton thinks these are signed but they're actually signless
    @_classproperty
    def int8(cls):
        return _T.i8()

    @_classproperty
    def int16(cls):
        return _PlusPtr(_T.i16())

    @_classproperty
    def int32(cls):
        return _PlusPtr(_T.i32())

    @_classproperty
    def int64(cls):
        return _PlusPtr(_T.i64())

    @_classproperty
    def float16(cls):
        return _PlusPtr(_T.f16())

    @_classproperty
    def float32(cls):
        return _PlusPtr(_T.f32())

    @_classproperty
    def float64(cls):
        return _PlusPtr(_T.f64())

    @_classproperty
    def bfloat16(cls):
        return _T.bf16()

    tensor = lambda *args, **kwargs: _T.tensor(*args, **kwargs)


@make_maybe_no_args_decorator
def jit(
    f,
    *,
    sym_visibility=None,
    arg_attrs=None,
    res_attrs=None,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()
    return FuncBase(
        body_builder=f,
        func_op_ctor=FuncOp,
        return_op_ctor=ReturnOp,
        call_op_ctor=CallOp,
        sym_visibility=sym_visibility,
        arg_attrs=arg_attrs,
        res_attrs=res_attrs,
        loc=loc,
        ip=ip,
    )


def arange(start, end, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    result_type = _T.tensor(end - start, _T.i32())
    return TritonTensor(tt.make_range(result_type, start, end, loc=loc, ip=ip))


def splat(src: Value, sizes: tuple[int], *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    result_type = _T.tensor(*sizes, src.type)
    return TritonTensor(tt.splat(result_type, src, loc=loc, ip=ip))


def zeros(shape: Sequence[int], dtype: Optional[Type] = None):
    if dtype is None:
        dtype = _T.f32()
    return TritonTensor(constant(0, RankedTensorType.get(shape, dtype)))


def broadcast(shape: list[int], src: Tensor, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return TritonTensor(
        tt.broadcast(RankedTensorType.get(shape, src.dtype), src, loc=loc, ip=ip)
    )


class ExpandDimsOp(tt.ExpandDimsOp):
    OPERATION_NAME = "tt.expand_dims"

    _ODS_REGIONS = (0, True)

    def __init__(self, src, axis, *, loc=None, ip=None):
        operands = []
        input_type = RankedTensorType(src.type)
        input_shape = input_type.shape
        input_shape.insert(axis, 1)
        results = [RankedTensorType.get(input_shape, input_type.element_type)]
        attributes = {}
        regions = None
        operands.append(get_op_result_or_value(src))
        _ods_context = get_default_loc_context(loc)
        attributes["axis"] = (
            axis
            if (
                issubclass(type(axis), Attribute) or not AttrBuilder.contains("I32Attr")
            )
            else AttrBuilder.get("I32Attr")(axis, context=_ods_context)
        )
        _ods_successors = None
        super(ExpandDimsOp.__base__, self).__init__(
            self.build_generic(
                attributes=attributes,
                results=results,
                operands=operands,
                successors=_ods_successors,
                regions=regions,
                loc=loc,
                ip=ip,
            )
        )


def expand_dims(src: Value, axis, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return TritonTensor(
        maybe_cast_triton_tensor(
            get_op_result_or_op_results(ExpandDimsOp(src, axis, loc=loc, ip=ip))
        )
    )


def broadcast_binary(lhs: Tensor, rhs: Tensor) -> tuple[Tensor, Tensor]:
    lhs_shape = lhs.shape
    rhs_shape = rhs.shape

    if len(lhs_shape) < len(rhs_shape):
        # Add new axes to lhs
        for dim in range(len(lhs_shape), len(rhs_shape)):
            lhs = expand_dims(lhs, 0)
            lhs_shape = lhs.shape
    elif len(rhs_shape) < len(lhs_shape):
        # Add new axes to rhs
        for dim in range(len(rhs_shape), len(lhs_shape)):
            rhs = expand_dims(rhs, 0)
            rhs_shape = rhs.shape
    assert len(rhs_shape) == len(lhs_shape)

    ret_shape = []
    for i, left in enumerate(lhs_shape):
        right = rhs_shape[i]
        if left == 1:
            ret_shape.append(right)
        elif right == 1:
            ret_shape.append(left)
        elif left == right:
            ret_shape.append(left)
        else:
            raise ValueError(
                "Cannot make_shape_compatible: incompatible dimensions "
                "at index " + str(i) + ": " + str(left) + " and " + str(right)
            )
    if lhs_shape != ret_shape:
        lhs = broadcast(ret_shape, lhs)
    if rhs_shape != ret_shape:
        rhs = broadcast(ret_shape, rhs)
    return TritonTensor(lhs), TritonTensor(rhs)


def _extract_slice(
    ten: "TritonTensor",
    idx,
) -> "TritonTensor":
    indexer = triton_mlir.dialects.ext.tensor._indices_to_indexer(idx, ten.shape)
    out = ten

    if indexer.is_full():
        out = out
    elif indexer.is_constant():
        out = triton_mlir.dialects.ext.tensor.extract_slice(
            out,
            static_offsets=indexer.static_offsets(),
            static_sizes=indexer.static_sizes(),
            static_strides=indexer.static_strides(),
        )
    else:
        raise ValueError(f"non-constant indices not supported {indexer}")

    # This adds newaxis/None dimensions.
    return TritonTensor(expand_dims(out, indexer.newaxis_dims))


class TritonTensor(Tensor):
    def coerce(self, other) -> tuple["TritonTensor", "TritonTensor"]:
        if not (
            isinstance(other, (TritonPointer, TritonScalar))
            or isinstance(other, Tensor)
        ):
            self, other = super().coerce(other)

        if isinstance(other, (TritonPointer, TritonScalar)):
            assert self.has_static_shape()
            other = splat(other, self.shape)

        if isinstance(other, Tensor) and self.shape != other.shape:
            self, other = broadcast_binary(self, other)

        return self, other

    def __add__(self, other: Tensor | Value, *, loc=None):
        if loc is None:
            loc = get_user_code_loc()
        if isinstance(other, Tensor) and self.shape != other.shape:
            self, other = broadcast_binary(self, other)
        if is_ptr(self):
            return addptr(self, other, loc=loc)

        return TritonTensor(super().__add__(other))

    def __lt__(self, other: Tensor | Value, *, loc=None):
        if loc is None:
            loc = get_user_code_loc()
        return TritonTensor(
            _binary_op(self, other, op="cmp", predicate="lt", signedness="s", loc=loc)
        )

    def __getitem__(self, idx):
        if is_ptr(self):
            return load(self, idx)
        if not isinstance(idx, (tuple, list)):
            idx = [idx]
        if not self.has_rank():
            raise ValueError("only ranked tensor slicing/indexing supported")

        if idx is None:
            return expand_dims(self, 0)
        if idx == Ellipsis or idx == slice(None):
            return self
        if isinstance(idx, tuple) and all(i == slice(None) for i in idx):
            return self
        if isinstance(idx, tuple) and all(i == slice(None) or i is None for i in idx):
            nones = [i for i, n in enumerate(idx) if n is None]
            assert len(nones), f"only one newaxis supported {idx=}"
            return expand_dims(self, nones[0])

        idx = list((idx,) if isinstance(idx, int) else idx)
        for i, d in enumerate(idx):
            if isinstance(d, int):
                idx[i] = constant(d, index=True)

        if all(isinstance(d, Scalar) for d in idx) and len(idx) == len(self.shape):
            raise ValueError("scalar indexing not supported")
        else:
            return _extract_slice(self, tuple(idx))

    def __setitem__(self, mask, value, *, loc=None):
        if loc is None:
            loc = get_user_code_loc()
        tt.store(self, value, mask=mask, loc=loc)


@register_value_caster(get_ptr_type_typeid())
class TritonPointer(Scalar):
    def __add__(self, other: Scalar | Tensor, *, loc=None):
        if isinstance(other, Tensor):
            assert _is_integer_like_type(other.dtype)
            return addptr(self, other)
        else:
            return super().__add__(other)


@register_value_caster(IntegerType.static_typeid, replace=True)
class TritonScalar(Scalar):
    def coerce(self, other) -> tuple["Tensor", "Tensor"]:
        if isinstance(other, Tensor):
            assert other.has_static_shape()
            return splat(self, other.shape), other

        return super().coerce(other)


@register_value_caster(RankedTensorType.static_typeid, replace=True)
def maybe_cast_triton_tensor(val: Value):
    if is_ptr(val):
        return TritonTensor(val)
    return Tensor(val)


def program_id(axis, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return TritonScalar(tt.get_program_id(axis, loc=loc, ip=ip))


def num_programs(axis, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return TritonScalar(tt.get_num_programs(axis, loc=loc, ip=ip))


def cdiv(lhs, rhs, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    lhs, rhs = lhs.coerce(rhs)
    # return (lhs + rhs - 1) // rhs
    return arith.ceildivsi(lhs, rhs, loc=loc, ip=ip)


def load(
    ptr: TritonTensor,
    mask: TritonTensor,
    cache=tt.CacheModifier.NONE,
    evict=tt.EvictionPolicy.NORMAL,
    is_volatile=False,
    *,
    other=None,
    boundary_check=None,
    padding=None,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()
    if isinstance(other, (int, float, bool)):
        if RankedTensorType.isinstance(ptr.type):
            other = constant(other, type=ptr.type.element_type.pointee_type)
        else:
            other = constant(other, type=ptr.type.pointee_type)
    if other is not None:
        if isinstance(other, Scalar):
            other = splat(other, ptr.shape)
        if ptr.shape != other.shape:
            other = broadcast(ptr.shape, other, loc=loc, ip=ip)
    if ptr.shape != mask.shape:
        mask = broadcast(ptr.shape, mask, loc=loc, ip=ip)

    return TritonTensor(
        tt.load(
            ptr,
            cache=cache,
            evict=evict,
            is_volatile=is_volatile,
            mask=mask,
            other=other,
            boundary_check=boundary_check,
            padding=padding,
            loc=loc,
            ip=ip,
        )
    )


def store(
    ptr: TritonTensor,
    value: TritonTensor,
    mask: TritonTensor,
    *,
    boundary_check=None,
    cache=None,
    evict=None,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()
    if ptr.shape != value.shape:
        ptr = broadcast(value.shape, ptr, loc=loc, ip=ip)
    return tt.store(
        ptr,
        value,
        mask=mask,
        boundary_check=boundary_check,
        cache=cache,
        evict=evict,
        loc=loc,
        ip=ip,
    )


def dot(
    a: TritonTensor,
    b: TritonTensor,
    *,
    c: TritonTensor = None,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()

    if c is None:
        c = constant(0, type=a.dtype)
    if isinstance(c, Scalar):
        assert a.has_static_shape()
        c = splat(c, a.shape)

    return tt.dot(
        a,
        b,
        c,
        loc=loc,
        ip=ip,
    )


def addptr(
    ptr: TritonPointer | TritonTensor,
    offset: Tensor | TritonTensor | int,
    *,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()
    if isinstance(offset, int):
        offset = constant(offset, type=ptr.type.pointee_type)
    if isinstance(offset, (Tensor, TritonTensor)) and not isinstance(
        ptr, (Tensor, TritonTensor)
    ):
        assert offset.has_static_shape()
        ptr = splat(ptr, offset.shape)
    if isinstance(offset, Scalar):
        assert ptr.has_static_shape()
        offset = splat(offset, ptr.shape)
    if ShapedType.isinstance(ptr.type) and ShapedType.isinstance(offset.type):
        assert (
            ptr.shape == offset.shape
        ), f"'tt.addptr' op all non-scalar operands/results must have the same shape and base type: {ptr=} {offset=}"
    result_type = ptr.type
    return TritonTensor(tt.addptr(result_type, ptr, offset, loc=loc, ip=ip))
