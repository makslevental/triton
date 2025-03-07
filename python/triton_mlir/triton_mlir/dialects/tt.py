#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional, Sequence

# noinspection PyUnresolvedReferences
from ._tt_enum_gen import *
from ._tt_ops_gen import *

# noinspection PyUnresolvedReferences
from .._mlir_libs._triton.tt import *

from . import amdgpu
from . import arith
from ..extras import types as _T
from ..extras.dialects.ext.arith import Scalar, constant
from ..extras.dialects.ext.func import FuncBase
from ..extras.dialects.ext.tensor import Tensor
from ..extras.util import make_maybe_no_args_decorator, get_user_code_loc
from ..ir import Value, RankedTensorType, Type, ShapedType, BoolAttr, TypeAttr


@make_maybe_no_args_decorator
def jit(
    f,
    *,
    sym_visibility=None,
    sym_name=None,
    arg_attrs=None,
    res_attrs=None,
    function_type=None,
    noinline=None,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()
    func_attrs = {}
    if noinline is not None:
        assert isinstance(noinline, bool)
        func_attrs["noinline"] = BoolAttr.get(noinline)
    if function_type is not None and isinstance(function_type, TypeAttr):
        function_type = function_type.value
    return FuncBase(
        body_builder=f,
        func_op_ctor=FuncOp,
        return_op_ctor=None,
        call_op_ctor=CallOp,
        sym_visibility=sym_visibility,
        sym_name=sym_name,
        arg_attrs=arg_attrs,
        res_attrs=res_attrs,
        func_attrs=func_attrs,
        function_type=function_type,
        loc=loc,
        ip=ip,
    )


_make_range = make_range


def make_range(start, end, *, result=None, encoding=None, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    if result is None:
        result = RankedTensorType.get((end - start,), _T.i32(), encoding=encoding)
    return _make_range(result, start, end, loc=loc, ip=ip)


_splat = splat


def splat(
    src: Value,
    sizes: tuple[int] = None,
    *,
    result=None,
    encoding=None,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()
    if result is None:
        result = RankedTensorType.get(sizes, src.type, encoding=encoding)
    return _splat(result, src, loc=loc, ip=ip)


def zeros(shape: Sequence[int], dtype: Optional[Type] = None):
    if dtype is None:
        dtype = _T.f32()
    return constant(0, RankedTensorType.get(shape, dtype))


_broadcast = broadcast


def broadcast(
    src: Tensor,
    shape: list[int] = None,
    *,
    result=None,
    encoding=None,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()
    if result is None:
        result = RankedTensorType.get(shape, src.dtype, encoding=encoding)
    return _broadcast(result, src, loc=loc, ip=ip)


_expand_dims = expand_dims


def expand_dims(src: Value, axis, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return _expand_dims(src, axis, loc=loc, ip=ip)


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
        lhs = broadcast(lhs, ret_shape, encoding=lhs.type.encoding)
    if rhs_shape != ret_shape:
        rhs = broadcast(rhs, ret_shape, encoding=rhs.type.encoding)
    return lhs, rhs


def program_id(axis, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return get_program_id(axis, loc=loc, ip=ip)


def num_programs(axis, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return get_num_programs(axis, loc=loc, ip=ip)


def cdiv(lhs, rhs, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    lhs, rhs = lhs.coerce(rhs)
    return arith.ceildivsi(lhs, rhs, loc=loc, ip=ip)


_load = load


def load(
    ptr,
    mask,
    cache=CacheModifier.NONE,
    evict=EvictionPolicy.NORMAL,
    is_volatile=False,
    *,
    other=None,
    boundary_check=None,
    padding=None,
    amdgpu_op_idx=None,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()
    if isinstance(other, (int, float, bool)):
        if RankedTensorType.isinstance(ptr.type):
            other = constant(
                other,
                type=RankedTensorType.get(
                    ptr.type.shape,
                    ptr.type.element_type.pointee_type,
                    encoding=ptr.type.encoding,
                ),
            )
        else:
            other = constant(other, type=ptr.type.pointee_type)
    if other is not None:
        if isinstance(other, Scalar):
            other = splat(other, ptr.shape)
        if ptr.shape != other.shape:
            other = broadcast(other, ptr.shape, loc=loc, ip=ip)
    if ptr.shape != mask.shape:
        mask = broadcast(mask, ptr.shape, loc=loc, ip=ip)

    l = _load(
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
    if amdgpu_op_idx is not None:
        l.owner.attributes["OpIdx"] = amdgpu.OpIdxAttr.get(amdgpu_op_idx)
    return l


_store = store


def store(
    ptr,
    value,
    mask,
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
        ptr = broadcast(ptr, value.shape, loc=loc, ip=ip)
    return _store(
        ptr,
        value,
        mask=mask,
        boundary_check=boundary_check,
        cache=cache,
        evict=evict,
        loc=loc,
        ip=ip,
    )


_dot = dot


def dot(
    a,
    b,
    *,
    c=None,
    input_precision=None,
    max_num_imprecise_acc=None,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()

    if c is None:
        c = constant(0, type=a.dtype)
    if isinstance(c, Scalar):
        assert len(a.type.shape) == len(b.type.shape) == 2
        assert a.type.has_static_shape and b.type.has_static_shape
        c = splat(c, (a.type.shape[0], b.type.shape[1]))

    return _dot(
        a,
        b,
        c,
        input_precision=input_precision,
        max_num_imprecise_acc=max_num_imprecise_acc,
        loc=loc,
        ip=ip,
    )


_addptr = addptr


def addptr(
    ptr,
    offset: Tensor | int,
    *,
    result=None,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()
    if isinstance(offset, int):
        offset = constant(offset, type=ptr.type.pointee_type)
    if isinstance(offset, (Tensor,)) and not isinstance(ptr, (Tensor,)):
        assert offset.has_static_shape()
        ptr = splat(ptr, offset.shape)
    if isinstance(offset, Scalar):
        assert ptr.has_static_shape()
        offset = splat(offset, ptr.shape)
    if ShapedType.isinstance(ptr.type) and ShapedType.isinstance(offset.type):
        assert (
            ptr.shape == offset.shape
        ), f"'addptr' op all non-scalar operands/results must have the same shape and base type: {ptr=} {offset=}"
    if result is None:
        result = ptr.type
    return _addptr(result, ptr, offset, loc=loc, ip=ip)


def ptr(pointee_type, address_space=1):
    return PointerType.of_pointee_type(pointee_type, address_space)
