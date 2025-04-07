#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ._smt_ops_gen import *

# noinspection PyUnresolvedReferences
from .._mlir_libs._triton.smt import *

from ..extras.util import get_user_code_loc, make_maybe_no_args_decorator
from ..ir import InsertionPoint, Type


@make_maybe_no_args_decorator
def solver(
    body_builder,
    *,
    results=None,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()
    if results is None:
        results = []

    solver_op = SolverOp(results, [], loc=loc, ip=ip)
    block = solver_op.bodyRegion.blocks.append()
    with InsertionPoint(block):
        body_builder(block)
    return solver_op


def bool_t():
    return Type.parse("!smt.bool")


def bv_t(size):
    return Type.parse(f"!smt.bv<{size}>")


@make_maybe_no_args_decorator
def forall(
    body_builder,
    num_patterns=0,
    bound_var_types=None,
    *,
    weight=None,
    no_pattern=None,
    bound_var_names=None,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()
    if bound_var_names is None:
        bound_var_names = []
    if bound_var_types is None:
        bound_var_types = []
    assert len(bound_var_names) == len(bound_var_types)

    forall = ForallOp(
        bool_t(), num_patterns, boundVarNames=bound_var_names, loc=loc, ip=ip
    )
    block = forall.body.blocks.append(*bound_var_types)
    with InsertionPoint(block):
        body_builder(*list(block.arguments))


def check(results=None, *, loc=None, ip=None):
    if results is None:
        results = []
    c = CheckOp(results_=results, loc=loc, ip=ip)
    block = c.satRegion.blocks.append()
    with InsertionPoint(block):
        yield_([])
    block = c.unsatRegion.blocks.append()
    with InsertionPoint(block):
        yield_([])
    block = c.unknownRegion.blocks.append()
    with InsertionPoint(block):
        yield_([])
    return c
