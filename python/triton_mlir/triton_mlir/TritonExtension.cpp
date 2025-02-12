//===- TritonExtension.cpp - Extension module -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "triton-c/Dialects.h"

#include <nanobind/nanobind.h>

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_TritonExtension, m) {
  nb::set_leak_warnings(false);

  m.def("get_ptr_type_typeid",
        []() { return tritonMlirPointerTypeGetTypeID(); });
  mlir::python::nanobind_adaptors::mlir_type_subclass(
      m, "PointerType", tritonMlirTypeIsAPointerType,
      tritonMlirPointerTypeGetTypeID)
      .def_staticmethod(
          "of_pointee_type",
          [](MlirType pointeeType, int addressSpace) {
            return tritonMlirPointerTypeOfPointeeType(pointeeType,
                                                      addressSpace);
          },
          "pointee_type"_a, "address_space"_a = 1)
      .def("__pos__",
           [](MlirType self) {
             return tritonMlirPointerTypeOfPointeeType(self,
                                                       /*addressSpace*/ 1);
           })
      .def_property_readonly("pointee_type", [](MlirType self) {
        return tritonMlirPointerTypeGetPointeeType(self);
      });

  populateTritonExtension(m.ptr());
}
