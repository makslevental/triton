//===- site_initialize_0.cpp - Extension module ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "triton-c/Dialects.h"

#include <nanobind/nanobind.h>

using namespace mlir::python;
using namespace mlir::python::nanobind_adaptors;
using namespace nanobind::literals;

NB_MODULE(_site_initialize_0, m) {
  m.def("register_dialects", [](MlirDialectRegistry registry) {
    tritonMlirRegisterTritonDialectsAndPasses(registry);
  });
}
