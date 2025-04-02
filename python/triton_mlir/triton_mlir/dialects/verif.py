#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..extras.util import get_user_code_loc, make_maybe_no_args_decorator
from ._verif_ops_gen import *
from ..ir import InsertionPoint


@make_maybe_no_args_decorator
def contract(
    body_builder,
    *,
    inputs=None,
    outputs=None,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()
    if inputs is None:
        inputs = []
    if outputs is None:
        outputs = [i.type for i in inputs]

    contract_op = ContractOp(outputs, inputs, loc=loc, ip=ip)
    contract_op.body.blocks.append()
    with InsertionPoint(contract_op.body.blocks[0]):
        body_builder()
    if len(outputs) > 1:
        return contract_op.results
    elif len(outputs) == 1:
        return contract_op.results[0]
    else:
        return contract_op
