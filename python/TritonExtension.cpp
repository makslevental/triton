//===- TritonExtension.cpp - Extension module -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "triton-c/Dialects.h"

using namespace mlir::python::adaptors;

PYBIND11_MODULE(_site_initialize_0, m) {
  //===--------------------------------------------------------------------===//
  // triton dialect
  //===--------------------------------------------------------------------===//
  m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle_triton = mlirGetDialectHandle__triton__();
        MlirDialectHandle handle_triton_gpu =
            mlirGetDialectHandle__triton_gpu__();
        MlirDialectHandle handle_triton_amd_gpu =
            mlirGetDialectHandle__triton_amd_gpu__();
        mlirDialectHandleRegisterDialect(handle_triton, context);
        mlirDialectHandleRegisterDialect(handle_triton_gpu, context);
        mlirDialectHandleRegisterDialect(handle_triton_amd_gpu, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle_triton, context);
          mlirDialectHandleLoadDialect(handle_triton_gpu, context);
          mlirDialectHandleLoadDialect(handle_triton_amd_gpu, context);
        }
      },
      py::arg("context") = py::none(), py::arg("load") = true);
}
