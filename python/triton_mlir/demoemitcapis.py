#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#  Copyright (c) 2025.
import argparse
from pathlib import Path

from eudsl_tblgen import (
    RecordKeeper,
    collect_all_defs,
    collect_all_attr_or_type_defs,
)
from eudsl_tblgen.mlir import (
    emit_decls_defns_nbclasses,
    CClassKind,
)


def emit_attrs_or_types(kind, rk, triton_mlir_root):
    all_defs = collect_all_attr_or_type_defs(collect_all_defs(rk))
    decls, defns, nbclasses = emit_decls_defns_nbclasses(kind, all_defs)

    attr_or_type = "attr" if kind == CClassKind.ATTRIBUTE else "type"

    attr_decls = open(triton_mlir_root / f"{attr_or_type}_decls.h.inc", "w")
    attr_defns = open(triton_mlir_root / f"{attr_or_type}_defns.cpp.inc", "w")
    attr_nbclasses = open(triton_mlir_root / f"{attr_or_type}_nbclasses.cpp.inc", "w")
    for d in decls:
        if "LinearLayout" in d:
            continue
        print(d, file=attr_decls)
    for d in defns:
        if "LinearLayout" in d:
            continue
        print(d, file=attr_defns)
    for hdecls, hdefns, n in nbclasses:
        if "LinearLayout" in n:
            continue
        for h in hdecls:
            print(h, file=attr_decls)
        for h in hdefns:
            print(h, file=attr_defns)

        print(n, file=attr_nbclasses)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--triton-src-root", type=Path)
    args.add_argument("--mlir-include-path", type=Path)
    args = args.parse_args()

    attr_defs_rk = RecordKeeper().parse_td(
        str(
            args.triton_src_root
            / "include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td"
        ),
        [str(args.triton_src_root / "include"), str(args.mlir_include_path)],
    )
    type_defs_rk = RecordKeeper().parse_td(
        str(
            args.triton_src_root
            / "include/triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"
        ),
        [str(args.triton_src_root / "include"), str(args.mlir_include_path)],
    )

    emit_attrs_or_types(
        CClassKind.ATTRIBUTE,
        attr_defs_rk,
        args.triton_src_root / "python/triton_mlir/triton_mlir",
    )
    emit_attrs_or_types(
        CClassKind.TYPE,
        type_defs_rk,
        args.triton_src_root / "python/triton_mlir/triton_mlir",
    )
