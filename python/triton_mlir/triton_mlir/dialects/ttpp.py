# noinspection PyUnresolvedReferences
from ..extras.dialects.ext import tensor
from .arith import _is_integer_like_type
from .tt import (
    expand_dims,
    splat,
    broadcast_binary,
    addptr,
    store,
    load,
)
from .tt import get_ptr_type_typeid
from .ttg import BlockedEncodingAttr, SliceEncodingAttr, SwizzledSharedEncodingAttr
from ..extras.dialects.ext.arith import Scalar, constant, _binary_op
from ..extras.dialects.ext.tensor import Tensor
from ..extras.util import get_user_code_loc
from ..ir import (
    Value,
    RankedTensorType,
    IntegerType,
    Type,
    ShapedType,
    register_value_caster,
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


def _extract_slice(
    ten: "TritonTensor",
    idx,
) -> "TritonTensor":
    indexer = tensor._indices_to_indexer(idx, ten.shape)
    out = ten

    if indexer.is_full():
        out = out
    elif indexer.is_constant():
        out = tensor.extract_slice(
            out,
            static_offsets=indexer.static_offsets(),
            static_sizes=indexer.static_sizes(),
            static_strides=indexer.static_strides(),
        )
    else:
        raise ValueError(f"non-constant indices not supported {indexer}")

    # This adds newaxis/None dimensions.
    return TritonTensor(expand_dims(out, indexer.newaxis_dims))


@register_value_caster(RankedTensorType.static_typeid, replace=True)
class TritonTensor(Tensor):
    def coerce(self, other) -> tuple["TritonTensor", "TritonTensor"]:
        if not (
            isinstance(other, (TritonPointer, TritonScalar))
            or isinstance(other, Tensor)
        ):
            self, other = super().coerce(other)

        if isinstance(other, (TritonPointer, TritonScalar)):
            assert self.has_static_shape()
            other = splat(other, result=self.type)

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

    def __and__(self, other: Tensor | Value, *, loc=None):
        if isinstance(other, Tensor) and self.shape != other.shape:
            self, other = broadcast_binary(self, other)

        return TritonTensor(super().__and__(other))

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
        encoding = None
        if isinstance(
            idx[-1],
            (BlockedEncodingAttr, SliceEncodingAttr, SwizzledSharedEncodingAttr),
        ):
            encoding = idx.pop()

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
            if encoding is not None:
                assert isinstance(encoding, SliceEncodingAttr)
                assert encoding.dim == nones[0]
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
        store(self, value, mask=mask, loc=loc)


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
            return splat(self, result=other.type), other

        return super().coerce(other)
