from functools import partial
from typing import Optional

from .dialects.tt import PointerType
from .extras import types as _T
from .ir import Type, StringAttr, UnrankedTensorType, RankedTensorType


class _classproperty:
    def __init__(self, func):
        self.fget = func

    def __get__(self, instance, owner):
        return self.fget(owner)


class _PlusPtr(Type):
    def __pos__(self):
        return PointerType.of_pointee_type(self)


def tensor(*shape, element_type: Type = None, encoding: Optional[str] = None):
    if isinstance(encoding, str):
        encoding = StringAttr.get(encoding)
    if not shape or (len(shape) == 1 and isinstance(shape[-1], Type)):
        if encoding is not None:
            raise ValueError("UnrankedTensorType does not support encoding.")
        return _T._shaped(
            *shape, element_type=element_type, type_constructor=UnrankedTensorType.get
        )
    return _T._shaped(
        *shape,
        element_type=element_type,
        type_constructor=partial(RankedTensorType.get, encoding=encoding),
    )

_T.tensor = tensor

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

    tensor = lambda *args, **kwargs: tensor(*args, **kwargs)
    function = lambda *args, **kwargs: _T.function(*args, **kwargs)
