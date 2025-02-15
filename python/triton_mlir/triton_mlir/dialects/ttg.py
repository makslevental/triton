#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# noinspection PyUnresolvedReferences
from ._ttg_ops_gen import *

# noinspection PyUnresolvedReferences
from ._ttg_enum_gen import *

# noinspection PyUnresolvedReferences
from .._mlir_libs._triton._eudsl.ttg import *
from .._mlir_libs._triton._eudsl.mlir import SmallVector, unwrap_c_context
from ..ir import Context


_SwizzledSharedEncodingAttr = SwizzledSharedEncodingAttr


class SwizzledSharedEncodingAttr(_SwizzledSharedEncodingAttr):
    @staticmethod
    def get(vec, per_phase, max_phase, order, cta_layout):
        unwrapped_ctx = unwrap_c_context(Context.current)
        return _SwizzledSharedEncodingAttr.get(
            unwrapped_ctx,
            vec=vec,
            per_phase=per_phase,
            max_phase=max_phase,
            order=SmallVector["uint32"](order),
            cta_layout=cta_layout,
        )
