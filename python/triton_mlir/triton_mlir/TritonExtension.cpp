//===- TritonExtension.cpp - Extension module -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Target/LLVMIR.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "triton-c/Dialects.h"

#include <nanobind/nanobind.h>

using namespace mlir::python;
using namespace mlir::python::nanobind_adaptors;
namespace nb = nanobind;
using namespace nb::literals;

static const std::string kNanobindCapsuleAttr = "ptr";
static const std::string kNanobindCapsuleName = "nb_handle";
static const std::string kEUDSLLLVMPyImportName = "llvm.eudslllvm_ext";
static const std::string kEUDSLLLVMPyFromCapsuleAttr = "from_capsule";

static LLVMContextRef
LLVMContextRefFromApiObject(const nb::object &llvmCtxApiObject) {
  nb::capsule obj = llvmCtxApiObject.attr(kNanobindCapsuleAttr.c_str());
  void *ptr = PyCapsule_GetPointer(obj.ptr(), kNanobindCapsuleName.c_str());
  auto ctx = static_cast<LLVMContextRef>(ptr);
  return ctx;
}

NB_MODULE(_TritonExtension, m) {
  nb::set_leak_warnings(false);

  m.def("get_ptr_type_typeid",
        []() { return tritonMlirPointerTypeGetTypeID(); });
  mlir_type_subclass(m, "PointerType", tritonMlirTypeIsAPointerType,
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

  m.def("has_matrix_core_feature", [](const std::string &arch) {
    MlirStringRef a = mlirStringRefCreate(arch.data(), arch.size());
    return hasMatrixCoreFeature(a);
  });

  m.def("translate_to_llvmir",
        [](MlirModule module, const nb::object &llvmCtxApiObject) {
          MlirOperation operation = mlirModuleGetOperation(module);
          auto mod = mlirTranslateModuleToLLVMIR(
              operation, LLVMContextRefFromApiObject(llvmCtxApiObject));
          return nanobind::module_::import_(kEUDSLLLVMPyImportName.c_str())
              .attr("ModuleRef")
              .attr(kEUDSLLLVMPyFromCapsuleAttr.c_str())(
                  nb::capsule(mod, kNanobindCapsuleName.c_str()))
              .release();
        });
}
