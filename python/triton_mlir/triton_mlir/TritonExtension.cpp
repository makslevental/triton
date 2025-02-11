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
#include <numeric>

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
  return static_cast<LLVMContextRef>(ptr);
}

static nb::handle LLVMModuleApiObjectFromRef(LLVMModuleRef mod) {
  return nanobind::module_::import_(kEUDSLLLVMPyImportName.c_str())
      .attr("ModuleRef")
      .attr(kEUDSLLLVMPyFromCapsuleAttr.c_str())(
          nb::capsule(mod, kNanobindCapsuleName.c_str()))
      .release();
}

static LLVMModuleRef
LLVMModuleRefFromApiObject(const nb::object &llvmModuleApiObject) {
  nb::capsule obj = llvmModuleApiObject.attr(kNanobindCapsuleAttr.c_str());
  void *ptr = PyCapsule_GetPointer(obj.ptr(), kNanobindCapsuleName.c_str());
  return static_cast<LLVMModuleRef>(ptr);
}

static nb::handle LLVMTargetMachineApiObjectFromRef(LLVMTargetMachineRef mod) {
  return nanobind::module_::import_(kEUDSLLLVMPyImportName.c_str())
      .attr("TargetMachineRef")
      .attr(kEUDSLLLVMPyFromCapsuleAttr.c_str())(
          nb::capsule(mod, kNanobindCapsuleName.c_str()))
      .release();
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

  auto llvm = m.def_submodule("llvm");

  llvm.def(
      "has_matrix_core_feature",
      [](const std::string &arch) {
        MlirStringRef a = mlirStringRefCreate(arch.data(), arch.size());
        return hasMatrixCoreFeature(a);
      },
      "arch"_a);

  llvm.def(
      "translate_to_llvmir",
      [](MlirModule module) {
        auto mod = translateToLLVMIR(module);
        return LLVMModuleApiObjectFromRef(mod);
      },
      "module"_a);

  llvm.def("add_module_flag", [](const nb::object &llvmModuleApiObject,
                                 const std::string &key, uint32_t val) {
    addModuleFlag(LLVMModuleRefFromApiObject(llvmModuleApiObject),
                  mlirStringRefCreateFromCString(key.c_str()), val);
  });

  nb::class_<IsaVersion>(llvm, "AMDIsaVersion")
      .def_ro("major", &IsaVersion::Major)
      .def_ro("minor", &IsaVersion::Minor)
      .def_ro("stepping", &IsaVersion::Stepping);

  llvm.def(
      "get_isa_version",
      [](const std::string &arch) {
        return getIsaVersion(mlirStringRefCreateFromCString(arch.c_str()));
      },
      "arch"_a);

  llvm.def("get_calling_conv_amdgpu_kernel",
           []() { return getCallingConvAMDGPUKernel(); });

  nb::enum_<LLVMOptimizationLevel>(llvm, "OptimizationLevel")
      .value("O0", O0)
      .value("O1", O1)
      .value("O2", O2)
      .value("O3", O3)
      .value("Os", Os)
      .value("Oz", Oz);

  llvm.def("init_all_targets", initAllLLVMTargets);

  llvm.def(
      "create_target_machine",
      [](const std::string &targetTriple, const std::string &arch,
         bool enableFPFusion, const std::string &features,
         bool disableLLVMOpt) {
        LLVMTargetMachineRef tm = createTargetMachine(
            mlirStringRefCreateFromCString(targetTriple.c_str()),
            mlirStringRefCreateFromCString(arch.c_str()), enableFPFusion,
            mlirStringRefCreateFromCString(features.c_str()), disableLLVMOpt);
        return LLVMTargetMachineApiObjectFromRef(tm);
      },
      "target_triple"_a, "arch"_a, "enable_fp_fusion"_a, "features"_a,
      "disable_llvm_opt"_a);

  llvm.def(
      "optimize_module",
      [](const nb::object &llvmModuleApiObject,
         const std::vector<std::string> &flagList, const std::string &arch,
         const std::string &features, LLVMOptimizationLevel optLevel,
         bool enableFPFusion, bool llvmIREnableDump,
         bool enableAddressSanitizer, bool disableLLVMOpt) {
        const char *const delim = ",";
        std::string flags;
        if (!flagList.empty())
          flags = std::accumulate(
              std::next(flagList.begin()), flagList.end(), flagList[0],
              [&delim](const std::string &a, const std::string &b) {
                return a + delim + b;
              });
        auto mod =
            optimizeModule(LLVMModuleRefFromApiObject(llvmModuleApiObject),
                           mlirStringRefCreateFromCString(flags.c_str()),
                           mlirStringRefCreateFromCString(arch.c_str()),
                           mlirStringRefCreateFromCString(features.c_str()),
                           optLevel, enableFPFusion, llvmIREnableDump,
                           enableAddressSanitizer, disableLLVMOpt);
        return LLVMModuleApiObjectFromRef(mod);
      },
      "llvm_module"_a, "flags"_a, "arch"_a, "features"_a, "opt_level"_a,
      "enable_fp_fusion"_a, "llvm_ir_enable_dump"_a,
      "enable_address_sanitizer"_a, "disable_llvm_opt"_a = false);

  llvm.def(
      "cleanup_bitcode_metadata", [](const nb::object &llvmModuleApiObject) {
        cleanupBitcodeMetadata(LLVMModuleRefFromApiObject(llvmModuleApiObject));
      });
}
