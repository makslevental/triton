#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional, Sequence

# noinspection PyUnresolvedReferences
from ._tt_enum_gen import *
from ._tt_ops_gen import *
# noinspection PyUnresolvedReferences
from .._mlir_libs._triton.tt import *

from . import arith
from ..extras import types as _T
from ..extras.dialects.ext.arith import Scalar, constant
from ..extras.dialects.ext.func import FuncBase
from ..extras.dialects.ext.tensor import Tensor
from ..extras.util import make_maybe_no_args_decorator, get_user_code_loc
from ..ir import Value, RankedTensorType, Type, ShapedType


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
    return make_range(result_type, start, end, loc=loc, ip=ip)


_splat = splat


def splat(src: Value, sizes: tuple[int], *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    result_type = _T.tensor(*sizes, src.type)
    return _splat(result_type, src, loc=loc, ip=ip)


def zeros(shape: Sequence[int], dtype: Optional[Type] = None):
    if dtype is None:
        dtype = _T.f32()
    return constant(0, RankedTensorType.get(shape, dtype))


_broadcast = broadcast


def broadcast(shape: list[int], src: Tensor, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return _broadcast(RankedTensorType.get(shape, src.dtype), src, loc=loc, ip=ip)


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
        lhs = broadcast(ret_shape, lhs)
    if rhs_shape != ret_shape:
        rhs = broadcast(ret_shape, rhs)
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

    return _load(
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
        ptr = broadcast(value.shape, ptr, loc=loc, ip=ip)
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

    return _dot(
        a,
        b,
        c,
        loc=loc,
        ip=ip,
    )


_addptr = addptr


def addptr(
    ptr,
    offset: Tensor | int,
    *,
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
    result_type = ptr.type
    return _addptr(result_type, ptr, offset, loc=loc, ip=ip)
