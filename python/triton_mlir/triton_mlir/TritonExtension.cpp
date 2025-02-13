//===- TritonExtension.cpp - Extension module -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TritonAMDGPUToLLVM/TargetUtils.h"
#include "amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "amd/include/TritonAMDGPUToLLVM/Passes.h"
#include "amd/include/TritonAMDGPUTransforms/Passes.h"
#include "lib/Target/LLVMIR/LLVMPasses.h"
#include "lld/Common/Driver.h"
#include "mlir-c/Target/LLVMIR.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "nvidia/include/NVGPUToLLVM/Passes.h"
#include "nvidia/include/TritonNVIDIAGPUToLLVM/Passes.h"
#include "proton/dialect/include/Dialect/Proton/IR/Dialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "triton/Target/LLVMIR/Passes.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/PassTimingInfo.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/TargetParser.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizer.h"

#include <iostream>
#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/unique_ptr.h>
#include <numeric>

namespace mlir::triton::AMD {
enum class ISAFamily;
}
namespace mlir::test {
void registerTestAliasPass();
void registerTestAlignmentPass();
void registerTestAllocationPass();
void registerTestMembarPass();
} // namespace mlir::test

namespace llvm {
class OptimizationLevel;
}
using namespace mlir::python;
using namespace mlir::python::nanobind_adaptors;
namespace nb = nanobind;
using namespace nb::literals;

namespace {
void registerTritonDialectsAndPasses(mlir::DialectRegistry &registry) {
  // clang-format off
  registry
      .insert<
              mlir::DLTIDialect,
              mlir::LLVM::LLVMDialect,
              mlir::NVVM::NVVMDialect,
              mlir::ROCDL::ROCDLDialect,
              mlir::acc::OpenACCDialect,
              mlir::affine::AffineDialect,
              mlir::amx::AMXDialect,
              // mlir::amdgpu::AMDGPUDialect,
              mlir::arith::ArithDialect,
              mlir::arm_neon::ArmNeonDialect,
              mlir::arm_sme::ArmSMEDialect,
              mlir::arm_sve::ArmSVEDialect,
              mlir::async::AsyncDialect,
              mlir::bufferization::BufferizationDialect,
              mlir::cf::ControlFlowDialect,
              mlir::complex::ComplexDialect,
              mlir::emitc::EmitCDialect,
              mlir::func::FuncDialect,
              mlir::gpu::GPUDialect,
              mlir::index::IndexDialect,
              mlir::irdl::IRDLDialect,
              mlir::linalg::LinalgDialect,
              mlir::math::MathDialect,
              mlir::memref::MemRefDialect,
              mlir::mesh::MeshDialect,
              mlir::ml_program::MLProgramDialect,
              mlir::mpi::MPIDialect,
              // mlir::nvgpu::NVGPUDialect,
              mlir::omp::OpenMPDialect,
              mlir::pdl::PDLDialect,
              mlir::pdl_interp::PDLInterpDialect,
              mlir::polynomial::PolynomialDialect,
              mlir::ptr::PtrDialect,
              mlir::quant::QuantDialect,
              mlir::scf::SCFDialect,
              mlir::shape::ShapeDialect,
              mlir::sparse_tensor::SparseTensorDialect,
              mlir::spirv::SPIRVDialect,
              mlir::tensor::TensorDialect,
              mlir::tosa::TosaDialect,
              mlir::transform::TransformDialect,

              mlir::triton::TritonDialect,
              mlir::triton::amdgpu::TritonAMDGPUDialect,
              mlir::triton::gpu::TritonGPUDialect,
              mlir::triton::nvgpu::NVGPUDialect,
              mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect,
              mlir::triton::proton::ProtonDialect,

              mlir::ub::UBDialect,
              mlir::vector::VectorDialect,
              mlir::x86vector::X86VectorDialect,
              mlir::xegpu::XeGPUDialect
  >();
  // clang-format on

  mlir::arith::registerConvertArithToLLVMInterface(registry);
  mlir::registerConvertComplexToLLVMInterface(registry);
  mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
  mlir::func::registerAllExtensions(registry);
  mlir::tensor::registerAllExtensions(registry);
  mlir::registerConvertFuncToLLVMInterface(registry);
  mlir::index::registerConvertIndexToLLVMInterface(registry);
  mlir::registerConvertMathToLLVMInterface(registry);
  mlir::registerConvertMemRefToLLVMInterface(registry);
  mlir::registerConvertNVVMToLLVMInterface(registry);
  mlir::registerConvertOpenMPToLLVMInterface(registry);
  mlir::ub::registerConvertUBToLLVMInterface(registry);
  mlir::registerConvertAMXToLLVMInterface(registry);
  mlir::gpu::registerConvertGpuToLLVMInterface(registry);
  mlir::NVVM::registerConvertGpuToNVVMInterface(registry);

  // Register all transform dialect extensions.
  mlir::affine::registerTransformDialectExtension(registry);
  mlir::bufferization::registerTransformDialectExtension(registry);
  mlir::dlti::registerTransformDialectExtension(registry);
  mlir::func::registerTransformDialectExtension(registry);
  mlir::gpu::registerTransformDialectExtension(registry);
  mlir::linalg::registerTransformDialectExtension(registry);
  // mlir::nvgpu::registerTransformDialectExtension(registry);
  // memreftransforms depends on upstream nvgpu...
  // mlir::memref::registerTransformDialectExtension(registry);
  mlir::scf::registerTransformDialectExtension(registry);
  mlir::sparse_tensor::registerTransformDialectExtension(registry);
  mlir::tensor::registerTransformDialectExtension(registry);
  mlir::transform::registerDebugExtension(registry);
  mlir::transform::registerIRDLExtension(registry);
  mlir::transform::registerLoopExtension(registry);
  mlir::transform::registerPDLExtension(registry);
  mlir::vector::registerTransformDialectExtension(registry);

  mlir::registerAllToLLVMIRTranslations(registry);

  mlir::registerAllPasses();
  mlir::registerLLVMDIScope();
  mlir::registerTritonNvidiaGPUPasses();
  mlir::registerTritonPasses();
  mlir::test::registerTestAliasPass();
  mlir::test::registerTestAlignmentPass();
  mlir::test::registerTestAllocationPass();
  mlir::test::registerTestMembarPass();
  mlir::triton::gpu::registerTritonGPUPasses();
  mlir::triton::registerAllocateSharedMemoryPass();
  mlir::triton::registerConvertNVGPUToLLVMPass();
  mlir::triton::registerConvertTritonGPUToLLVMPass();
  mlir::triton::registerConvertTritonToTritonGPUPass();
  mlir::triton::registerTritonGPUGlobalScratchAllocationPass();

  // TritonAMDGPUToLLVM passes
  mlir::triton::registerConvertTritonAMDGPUToLLVM();
  mlir::triton::registerConvertBuiltinFuncToLLVM();
  mlir::triton::registerDecomposeUnsupportedAMDConversions();
  mlir::triton::registerOptimizeAMDLDSUsage();

  // TritonAMDGPUTransforms passes
  mlir::registerTritonAMDGPUAccelerateMatmul();
  mlir::registerTritonAMDGPUOptimizeEpilogue();
  mlir::registerTritonAMDGPUReorderInstructions();
  mlir::registerTritonAMDGPUBlockPingpong();
  mlir::registerTritonAMDGPUStreamPipeline();
  mlir::registerTritonAMDGPUCanonicalizePointers();
  mlir::registerTritonAMDGPUConvertToBufferOps();
  mlir::triton::registerTritonAMDGPUInsertInstructionSchedHints();
  mlir::triton::registerTritonAMDGPULowerInstructionSchedHints();

  // Register all external models.
  mlir::LLVM::registerInlinerInterface(registry);
  mlir::NVVM::registerNVVMTargetInterfaceExternalModels(registry);
  mlir::ROCDL::registerROCDLTargetInterfaceExternalModels(registry);
  mlir::affine::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::arith::registerBufferDeallocationOpInterfaceExternalModels(registry);
  mlir::arith::registerBufferViewFlowOpInterfaceExternalModels(registry);
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::arith::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  mlir::builtin::registerCastOpInterfaceExternalModels(registry);
  mlir::cf::registerBufferDeallocationOpInterfaceExternalModels(registry);
  mlir::cf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::gpu::registerBufferDeallocationOpInterfaceExternalModels(registry);
  mlir::linalg::registerAllDialectInterfaceImplementations(registry);
  mlir::linalg::registerRuntimeVerifiableOpInterfaceExternalModels(registry);
  mlir::memref::registerAllocationOpInterfaceExternalModels(registry);
  mlir::memref::registerBufferViewFlowOpInterfaceExternalModels(registry);
  mlir::memref::registerMemorySlotExternalModels(registry);
  mlir::memref::registerRuntimeVerifiableOpInterfaceExternalModels(registry);
  mlir::memref::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::ml_program::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferDeallocationOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::shape::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::sparse_tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::spirv::registerSPIRVTargetInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerFindPayloadReplacementOpInterfaceExternalModels(
      registry);
  mlir::tensor::registerInferTypeOpInterfaceExternalModels(registry);
  mlir::tensor::registerSubsetOpInterfaceExternalModels(registry);
  mlir::tensor::registerTilingInterfaceExternalModels(registry);
  mlir::tensor::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::tosa::registerShardingInterfaceExternalModels(registry);
  mlir::vector::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::vector::registerSubsetOpInterfaceExternalModels(registry);
  mlir::vector::registerValueBoundsOpInterfaceExternalModels(registry);
}

} // namespace

void tritonMlirRegisterTritonDialectsAndPasses(MlirDialectRegistry registry) {
  registerTritonDialectsAndPasses(*unwrap(registry));
}

MlirTypeID tritonMlirPointerTypeGetTypeID() {
  return wrap(mlir::triton::PointerType::getTypeID());
}

bool tritonMlirTypeIsAPointerType(MlirType type) {
  return llvm::isa<mlir::triton::PointerType>(unwrap(type));
}

bool tritonMlirTypeIsATensorOfPointer(MlirType type) {
  auto type_ = unwrap(type);
  return llvm::isa<mlir::RankedTensorType>(type_) &&
         llvm::isa<mlir::triton::PointerType>(
             llvm::cast<mlir::RankedTensorType>(type_).getElementType());
}

MlirType tritonMlirPointerTypeOfPointeeType(MlirType type, int addressSpace) {
  return wrap(mlir::triton::PointerType::get(unwrap(type), addressSpace));
}

MlirType tritonMlirPointerTypeGetPointeeType(MlirType type) {
  return wrap(
      llvm::cast<mlir::triton::PointerType>(unwrap(type)).getPointeeType());
}

void addControlConstant(llvm::Module *module, const char *name,
                        uint32_t bitwidth, uint32_t value) {
  using llvm::GlobalVariable;

  llvm::IntegerType *type =
      llvm::IntegerType::getIntNTy(module->getContext(), bitwidth);
  auto *initializer = llvm::ConstantInt::get(type, value, /*isSigned=*/false);
  auto *constant = new llvm::GlobalVariable(
      *module, type, /*isConstant=*/true,
      GlobalVariable::LinkageTypes::LinkOnceODRLinkage, initializer, name,
      /*before=*/nullptr, GlobalVariable::ThreadLocalMode::NotThreadLocal,
      /*addressSpace=*/4);
  constant->setAlignment(llvm::MaybeAlign(bitwidth / 8));
  constant->setUnnamedAddr(GlobalVariable::UnnamedAddr::Local);
  constant->setVisibility(GlobalVariable::VisibilityTypes::ProtectedVisibility);
}

std::unique_ptr<llvm::TargetMachine>
createTargetMachine(llvm::Module *module, std::string proc,
                    bool enable_fp_fusion, const std::string &features) {
  std::string error;
  auto target =
      llvm::TargetRegistry::lookupTarget(module->getTargetTriple(), error);
  llvm::TargetOptions opt;
  bool disableLLVMOpt = mlir::triton::tools::getBoolEnv("DISABLE_LLVM_OPT");
  if (enable_fp_fusion)
    opt.AllowFPOpFusion = llvm::FPOpFusion::Fast;
  opt.UnsafeFPMath = false;
  opt.NoInfsFPMath = false;
  opt.NoNaNsFPMath = true;
  opt.TrapUnreachable = true;
  opt.MCOptions.AsmVerbose = true;
  opt.MCOptions.PreserveAsmComments = true;
  std::unique_ptr<llvm::TargetMachine> machine{target->createTargetMachine(
      module->getTargetTriple(), proc, features, opt, llvm::Reloc::PIC_,
      std::nullopt,
      disableLLVMOpt ? llvm::CodeGenOptLevel::None
                     : llvm::CodeGenOptLevel::Aggressive)};
  return machine;
}

std::string translateLLVMIRToASM(llvm::Module &module,
                                 const std::string &triple,
                                 const std::string &proc,
                                 const std::string &features,
                                 const std::vector<std::string> &flags,
                                 bool enable_fp_fusion, bool isObject) {
  using namespace mlir;
  // options
  auto options = llvm::cl::getRegisteredOptions();
  for (std::string flag : flags) {
    auto *shortPtr = static_cast<llvm::cl::opt<bool> *>(options[flag]);
    assert(shortPtr);
    shortPtr->setValue(true);
  }
  if (triton::tools::getBoolEnv("LLVM_IR_ENABLE_DUMP")) {
    auto optIt = options.find("print-after-all");
    if (optIt != options.end()) {
      auto optPtr = static_cast<llvm::cl::opt<bool> *>(optIt->second);
      *optPtr = true;
    }
  }
  bool disableLLVMOpt = triton::tools::getBoolEnv("DISABLE_LLVM_OPT");
  if (!disableLLVMOpt) {
    // Check to see if we are passing a list of flags to disable optimizations.
    auto flagList = triton::tools::getStrEnv("DISABLE_LLVM_OPT");
    if (!flagList.empty()) {
      llvm::SmallVector<StringRef, 3> split;
      StringRef(flagList.c_str()).split(split, ',');
      for (auto flag : split) {
        auto optIt = options.find(flag);
        if (optIt != options.end()) {
          auto optPtr = static_cast<llvm::cl::opt<bool> *>(optIt->second);
          *optPtr = true;
        }
      }
    }
  }

  // inline everything
  for (llvm::Function &f : module.functions())
    if (!f.hasFnAttribute(llvm::Attribute::NoInline))
      f.addFnAttr(llvm::Attribute::AlwaysInline);
  // verify and store llvm
  llvm::legacy::PassManager pm;
  pm.add(llvm::createAlwaysInlinerLegacyPass());
  pm.add(llvm::createVerifierPass());

  const bool enabledTiming = triton::tools::getBoolEnv("LLVM_ENABLE_TIMING");
  if (enabledTiming) {
    llvm::TimePassesIsEnabled = true;
    llvm::TimePassesPerRun = true;
  }

  pm.run(module);

  SmallString<0> timePassesStr;
  llvm::raw_svector_ostream reportStream(timePassesStr);

  if (enabledTiming) {
    reportAndResetTimings(&reportStream);
    llvm::dbgs() << reportStream.str();
    timePassesStr.clear();
  }
  // module->print(llvm::outs(), nullptr);

  // create machine
  module.setTargetTriple(triple);
  auto machine = createTargetMachine(&module, proc, enable_fp_fusion, features);
  // set data layout
  module.setDataLayout(machine->createDataLayout());
  // emit machine code
  std::string result;
  {
    llvm::raw_string_ostream stream(result);
    llvm::buffer_ostream pstream(stream);
    llvm::legacy::PassManager pass;
    // emit
    auto fileType = isObject ? llvm::CodeGenFileType::ObjectFile
                             : llvm::CodeGenFileType::AssemblyFile;
    machine->addPassesToEmitFile(pass, pstream, nullptr, fileType);
    pass.run(module);

    if (enabledTiming) {
      reportAndResetTimings(&reportStream);
      llvm::dbgs() << reportStream.str();
      timePassesStr.clear();
    }
  }
  return result;
}

bool hasMatrixCoreFeature(MlirStringRef arch) {
  using mlir::triton::AMD::ISAFamily;
  switch (mlir::triton::AMD::deduceISAFamily(unwrap(arch).str())) {
  case ISAFamily::CDNA3:
  case ISAFamily::CDNA2:
  case ISAFamily::CDNA1:
  case ISAFamily::RDNA3:
    return true;
  default:
    return false;
  }
}

LLD_HAS_DRIVER(elf)

static bool lldInvoke(const char *inPath, const char *outPath) {
  std::vector<const char *> args{"ld.lld", "-shared", inPath, "-o", outPath};
  lld::Result s = lld::lldMain(args, llvm::outs(), llvm::errs(),
                               {{lld::Gnu, &lld::elf::link}});
  return !s.retCode && s.canRunAgain;
}

static const std::string amdTargetTriple = "amdgcn-amd-amdhsa";

void populateTritonExtension(nanobind::module_ &m) {

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

  llvm.def("init_targets", []() {
    static std::once_flag init_flag;
    std::call_once(init_flag, []() {
      llvm::InitializeAllTargetInfos();
      llvm::InitializeAllTargets();
      llvm::InitializeAllTargetMCs();
      llvm::InitializeAllAsmParsers();
      llvm::InitializeAllAsmPrinters();
    });
  });

  nb::class_<llvm::OptimizationLevel>(llvm, "optimization_level");
  llvm.attr("OPTIMIZE_O0") = llvm::OptimizationLevel::O0;
  llvm.attr("OPTIMIZE_O1") = llvm::OptimizationLevel::O1;
  llvm.attr("OPTIMIZE_O2") = llvm::OptimizationLevel::O2;
  llvm.attr("OPTIMIZE_O3") = llvm::OptimizationLevel::O3;
  llvm.attr("OPTIMIZE_Os") = llvm::OptimizationLevel::Os;
  llvm.attr("OPTIMIZE_Oz") = llvm::OptimizationLevel::Oz;

  auto llvmContext =
      nb::class_<llvm::LLVMContext>(llvm, "context", nb::sig("class context()"))
          .def(nb::init<>());

  auto sourceMgr =
      nb::class_<llvm::SourceMgr>(llvm, "source_mgr").def(nb::init<>());
  auto llvmModule =
      nb::class_<llvm::Module>(llvm, "module", nb::sig("class module()"))
          .def(
              "__str__",
              [](llvm::Module *self) {
                std::string str;
                llvm::raw_string_ostream os(str);
                os << *self;
                return os.str();
              },
              nb::rv_policy::take_ownership)
          .def(
              "get_functions",
              [](llvm::Module *mod) -> llvm::Module::FunctionListType & {
                // Note: Backends assume that we are compiling exactly one
                // kernel (i.e. one function that's that's called by the CPU)
                // and that it's the first function in this list.
                return mod->getFunctionList();
              },
              nb::rv_policy::reference_internal)
          .def("add_flag",
               [](llvm::Module *mod, llvm::Module::ModFlagBehavior behavior,
                  std::string &key, uint32_t value) {
                 return mod->addModuleFlag(behavior, key, value);
               });

  nb::class_<llvm::Function>(llvm, "function")
      .def_prop_ro("name",
                   [](llvm::Function *fn) { return fn->getName().str(); })
      .def("set_calling_conv", &llvm::Function::setCallingConv)
      .def("add_fn_attr", [](llvm::Function *fn, std::string &name,
                             std::string &val) { fn->addFnAttr(name, val); })
      .def("add_fn_asan_attr",
           [](llvm::Function *fn) {
             fn->addFnAttr(llvm::Attribute::SanitizeAddress);
           })
      .def("add_fn_target_feature",
           [](llvm::Function *fn, std::string &val) {
             fn->addFnAttr("target-features", val);
           })
      // Sets the nvvm.maxreg property on the given function.
      .def("set_nvvm_maxnreg",
           [](llvm::Function *fn, int maxnreg) {
             auto op = llvm::MDNode::get(
                 fn->getContext(),
                 {
                     llvm::ValueAsMetadata::get(fn),
                     llvm::MDString::get(fn->getContext(), "maxnreg"),
                     llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                         llvm::Type::getInt32Ty(fn->getContext()), maxnreg)),
                 });
             fn->getParent()
                 ->getOrInsertNamedMetadata("nvvm.annotations")
                 ->addOperand(op);
           })
      // External functions that are definitions (i.e. not declarations) are
      // kernel functions.
      .def("is_declaration", &llvm::Function::isDeclaration)
      .def("is_external_linkage", [](llvm::Function *fn) {
        return fn->getLinkage() == llvm::GlobalValue::ExternalLinkage;
      });

  nb::class_<llvm::Module::FunctionListType>(m, "function_list")
      .def(
          "__iter__",
          [](llvm::Module::FunctionListType &s) {
            return nb::make_iterator<nb::rv_policy::reference_internal>(
                nb::type<llvm::Module::FunctionListType>(),
                "function_list_iterator", s.begin(), s.end());
          },
          nb::keep_alive<0, 1>());

  llvm.def(
      "to_module",
      [](MlirOperation mod, llvm::LLVMContext &ctx) {
        return mlir::translateModuleToLLVMIR(unwrap(mod), ctx);
      },
      nb::keep_alive<0, 2>());

  llvm.def("attach_datalayout", [](llvm::Module *mod, const std::string &triple,
                                   const std::string &proc,
                                   const std::string &features) {
    std::string error;
    auto target = llvm::TargetRegistry::lookupTarget(triple, error);
    if (!target) {
      throw std::runtime_error("target lookup error: " + error);
    }
    llvm::TargetOptions opt;
    // Target machine is only used to create the data layout.
    std::unique_ptr<llvm::TargetMachine> machine{target->createTargetMachine(
        triple, proc, features, opt, llvm::Reloc::PIC_, std::nullopt,
        llvm::CodeGenOptLevel::None)};
    // set data layout
    mod->setDataLayout(machine->createDataLayout());
  });

  llvm.def(
      "optimize_module",
      [](llvm::Module *mod, const llvm::OptimizationLevel &opt,
         std::string arch, std::string features, std::vector<std::string> flags,
         bool enable_fp_fusion) {
        if (mlir::triton::tools::getBoolEnv("DISABLE_LLVM_OPT"))
          return;
        // Check to see if we are passing a list of flags to disable
        // optimizations.
        auto flagList = mlir::triton::tools::getStrEnv("DISABLE_LLVM_OPT");
        if (!flagList.empty()) {
          auto options = llvm::cl::getRegisteredOptions();
          llvm::SmallVector<llvm::StringRef, 3> split;
          llvm::StringRef(flagList.c_str()).split(split, ',');
          for (auto flag : split) {
            auto optIt = options.find(flag);
            if (optIt != options.end()) {
              auto optPtr = static_cast<llvm::cl::opt<bool> *>(optIt->second);
              *optPtr = true;
            }
          }
        }
        llvm::LoopAnalysisManager lam;
        llvm::FunctionAnalysisManager fam;
        llvm::CGSCCAnalysisManager cgam;
        llvm::ModuleAnalysisManager mam;

        llvm::PassInstrumentationCallbacks *instrCbPtr = nullptr;
        llvm::PassInstrumentationCallbacks passInstrCb;
        llvm::StandardInstrumentations standardInstr(mod->getContext(),
                                                     /*DebugLogging*/ true);
        if (mlir::triton::tools::getBoolEnv("LLVM_IR_ENABLE_DUMP")) {
          auto optMap = llvm::cl::getRegisteredOptions();
          auto optIt = optMap.find("print-after-all");
          if (optIt != optMap.end()) {
            auto optPtr = static_cast<llvm::cl::opt<bool> *>(optIt->second);
            *optPtr = true;
          }
          standardInstr.registerCallbacks(passInstrCb, &mam);
          instrCbPtr = &passInstrCb;
        }

        llvm::PipelineTuningOptions tuningOptions;
        tuningOptions.LoopUnrolling = true;
        tuningOptions.LoopInterleaving = true;
        tuningOptions.LoopVectorization = true;
        // TODO: currently we run SLP vectorizer with an empty target machine.
        // This cause the vectorizer to create larger vector which could be bad.
        // Disabling it would currently cause regressions as this pass also
        // applies some scheduling that helps performance in some cases. We
        // should work on using NVPTX target instead and address the performance
        // regressions with some scheduling solution.
        tuningOptions.SLPVectorization = true;

        std::string pluginFile =
            mlir::triton::tools::getStrEnv("LLVM_PASS_PLUGIN_PATH");

        // We don't pass the targetMachine to the LLVM-IR pass builder, unless
        // `arch` is specified.
        //
        // Don't set target machine in LLVM pass builder when using LLVM IR
        // level plugins. LLVM IR level plugin passes typically want to insert
        // calls to externally generated code (i.e. precompile a Cuda/Hip kernel
        // with Clang and then insert a call to it within an instrumentation
        // pass) setting the targetMachine value here can can cause a mismatch
        // in the target machine between the MLIR and Clang generated kernels
        // and break the lowering of some target specific intrinsics.
        std::unique_ptr<llvm::TargetMachine> targetMachine = nullptr;
        if (!arch.empty() && pluginFile.empty())
          targetMachine =
              createTargetMachine(mod, arch, enable_fp_fusion, features);
        llvm::PassBuilder pb(/*targetMachine=*/targetMachine.get(),
                             tuningOptions, std::nullopt, instrCbPtr);

        if (!pluginFile.empty()) {
          // TODO: Add some logging here that we inserted a pass into the LLVM
          // pass pipeline
          auto passPlugin = llvm::PassPlugin::Load(pluginFile);
          if (!passPlugin) {
            llvm::Error Err = passPlugin.takeError();
            std::string ErrMsg =
                "Pass Plugin Error: " + llvm::toString(std::move(Err));
            throw std::runtime_error(ErrMsg);
          }
          passPlugin->registerPassBuilderCallbacks(pb);
        }

        pb.registerModuleAnalyses(mam);
        pb.registerCGSCCAnalyses(cgam);
        pb.registerFunctionAnalyses(fam);
        pb.registerLoopAnalyses(lam);
        pb.crossRegisterProxies(lam, fam, cgam, mam);

        llvm::ModulePassManager mpm;
        pb.registerVectorizerStartEPCallback(
            [&](llvm::FunctionPassManager &fpm, llvm::OptimizationLevel level) {
              // Triton generates large structure of scalars which may pessimise
              // optimizations, we run a pass to break up phi of struct to make
              // sure all the struct are removed for the following passes.
              fpm.addPass(llvm::BreakStructPhiNodesPass());
              fpm.addPass(llvm::InstCombinePass());
            });
        bool enableAddressSanitizer =
            mlir::triton::tools::getBoolEnv("TRITON_ENABLE_ASAN");
        if (enableAddressSanitizer) {
          llvm::AddressSanitizerOptions Opts;
          mpm.addPass(llvm::AddressSanitizerPass(Opts));
        }
        mpm.addPass(pb.buildPerModuleDefaultPipeline(opt));
        mpm.run(*mod, mam);
      },
      // Mandatory parameters
      nb::arg("mod"), nb::arg("opt"),
      // If we want to specify the target machine, we require additional
      // (optional) parameters
      nb::arg("arch") = "", nb::arg("features") = "",
      nb::arg("flags") = std::vector<std::string>{},
      nb::arg("enable_fp_fusion") = false);

  llvm.def(
      "translate_to_asm",
      [](llvm::Module *module, std::string triple, std::string proc,
         std::string features, std::vector<std::string> flags,
         bool enable_fp_fusion, bool isObject) -> nb::object {
        std::string obj;
        {
          // when allow_threads goes out of scope, gil will be released
          nb::gil_scoped_release allow_threads;
          // create LLVM module from C++
          obj = translateLLVMIRToASM(*module, triple, proc, features, flags,
                                     enable_fp_fusion, isObject);
        }
        if (isObject)
          return nb::bytes(obj.c_str());
        return nb::str(obj.c_str());
      },
      nb::rv_policy::take_ownership);

  auto amd = m.def_submodule("amd");
  amd.attr("TARGET_TRIPLE") = amdTargetTriple;
  amd.attr("CALLING_CONV_AMDGPU_KERNEL") =
      static_cast<unsigned>(llvm::CallingConv::AMDGPU_KERNEL);

  amd.def("attach_target_triple", [](llvm::Module *module) {
    module->setTargetTriple(amdTargetTriple);
  });

  amd.def("set_isa_version", [](llvm::Module *module, const std::string &arch) {
    llvm::AMDGPU::IsaVersion version = llvm::AMDGPU::getIsaVersion(arch);
    addControlConstant(module, "__oclc_ISA_version", /*bitwidth=*/32,
                       version.Major * 1000 + version.Minor * 100 +
                           version.Stepping);
  });

  amd.def("set_abi_version", [](llvm::Module *module, int version) {
    // Inject the control constant into the LLVM module so that device libraries
    // linked against module can resolve their references to it.
    llvm::Type *i32Ty = llvm::Type::getInt32Ty(module->getContext());
    llvm::GlobalVariable *abi = new llvm::GlobalVariable(
        *module, i32Ty, /*isConstant=*/true,
        llvm::GlobalValue::LinkageTypes::LinkOnceODRLinkage,
        llvm::ConstantInt::get(i32Ty, version), "__oclc_ABI_version", nullptr,
        llvm::GlobalValue::ThreadLocalMode::NotThreadLocal, 4);
    abi->setVisibility(llvm::GlobalValue::VisibilityTypes::ProtectedVisibility);
    abi->setAlignment(llvm::MaybeAlign(4));
    abi->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Local);

    // Also attach the control attribute on the LLVM module. This is also needed
    // in addition to the above for various transformations to know what code
    // object version we are targeting at.
    module->addModuleFlag(llvm::Module::Error, "amdhsa_code_object_version",
                          version);
  });

  amd.def("set_bool_control_constant",
          [](llvm::Module *module, const std::string &name, bool enable) {
            addControlConstant(module, name.c_str(), /*bitwidth=*/8, enable);
          });

  amd.def("set_all_fn_arg_inreg", [](llvm::Function *fn) {
    for (llvm::Argument &arg : fn->args()) {
      // Check for incompatible attributes.
      if (arg.hasByRefAttr() || arg.hasNestAttr())
        continue;
      arg.addAttr(llvm::Attribute::InReg);
    }
  });

  amd.def("cleanup_bitcode_metadata", [](llvm::Module *module) {
    // We can have Clang version metadata from device libraries linked in. We
    // don't care about them so drop them.
    if (auto *ident = module->getNamedMetadata("llvm.ident"))
      module->eraseNamedMetadata(ident);
    // Also various OpenCL version details.
    if (auto *openclVersion = module->getNamedMetadata("opencl.ocl.version"))
      module->eraseNamedMetadata(openclVersion);
  });

  amd.def("disable_print_inline", [](llvm::Module *module) {
    // List of functions name prefixes we want to forbid inline.
    std::array<const char *, 2> prefixes = {"__ockl_fprintf", "__ockl_printf"};

    for (llvm::Function &f : module->functions()) {
      if (!f.hasName())
        continue;
      llvm::StringRef name = f.getName();

      auto isNamePrefixed = [&name](const char *prefix) {
        return name.starts_with(prefix);
      };

      if (llvm::any_of(prefixes, isNamePrefixed))
        f.addFnAttr(llvm::Attribute::NoInline);
    }
  });

  amd.def(
      "assemble_amdgcn",
      [](const std::string &assembly, const std::string &arch,
         const std::string &features) {
        std::string error;

        llvm::Triple triple(amdTargetTriple);
        const llvm::Target *target =
            llvm::TargetRegistry::lookupTarget(triple.normalize(), error);
        if (!target)
          throw std::runtime_error("target lookup error: " + error);

        llvm::SourceMgr srcMgr;
        srcMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(assembly),
                                  llvm::SMLoc());

        const llvm::MCTargetOptions mcOptions;
        std::unique_ptr<llvm::MCRegisterInfo> mri(
            target->createMCRegInfo(amdTargetTriple));
        std::unique_ptr<llvm::MCAsmInfo> mai(
            target->createMCAsmInfo(*mri, amdTargetTriple, mcOptions));
        std::unique_ptr<llvm::MCSubtargetInfo> sti(
            target->createMCSubtargetInfo(amdTargetTriple, arch, features));

        llvm::MCContext ctx(triple, mai.get(), mri.get(), sti.get(), &srcMgr,
                            &mcOptions);
        std::unique_ptr<llvm::MCObjectFileInfo> mofi(
            target->createMCObjectFileInfo(ctx, /*PIC=*/false,
                                           /*LargeCodeModel=*/false));
        ctx.setObjectFileInfo(mofi.get());

        llvm::SmallString<128> cwd;
        if (!llvm::sys::fs::current_path(cwd))
          ctx.setCompilationDir(cwd);

        llvm::SmallVector<char, 0> result;
        llvm::raw_svector_ostream svos(result);

        std::unique_ptr<llvm::MCStreamer> mcStreamer;
        std::unique_ptr<llvm::MCInstrInfo> mcii(target->createMCInstrInfo());

        std::unique_ptr<llvm::MCCodeEmitter> ce(
            target->createMCCodeEmitter(*mcii, ctx));
        std::unique_ptr<llvm::MCAsmBackend> mab(
            target->createMCAsmBackend(*sti, *mri, mcOptions));
        std::unique_ptr<llvm::MCObjectWriter> ow(mab->createObjectWriter(svos));
        mcStreamer.reset(target->createMCObjectStreamer(
            triple, ctx, std::move(mab), std::move(ow), std::move(ce), *sti));

        std::unique_ptr<llvm::MCAsmParser> parser(
            createMCAsmParser(srcMgr, ctx, *mcStreamer, *mai));
        std::unique_ptr<llvm::MCTargetAsmParser> tap(
            target->createMCAsmParser(*sti, *parser, *mcii, mcOptions));
        if (!tap)
          throw std::runtime_error("assembler initializtion error");

        parser->setTargetParser(*tap);
        parser->Run(/*NoInitialTextSection=*/false);

        auto rs = std::string(result.begin(), result.end());
        return nb::bytes(rs.data(), rs.size());
      },
      nb::rv_policy::take_ownership);

  amd.def("link_hsaco",
          [](const std::string &inPath, const std::string &outPath) {
            if (!lldInvoke(inPath.c_str(), outPath.c_str()))
              throw std::runtime_error("couldn't link");
          });
}

const char *HERE = "HERE";

#include <dlfcn.h>
static std::optional<std::string> getHere() {
  Dl_info info;
  if (dladdr(HERE, &info)) {
    if (info.dli_fname) {
      return {std::string(info.dli_fname)};
    }
  }
  return {};
}

NB_MODULE(_triton, m) {
  nb::set_leak_warnings(false);
  populateTritonExtension(m);
}

NB_MODULE(_site_initialize_0, m) {
  m.def("register_dialects", [](MlirDialectRegistry registry) {
    tritonMlirRegisterTritonDialectsAndPasses(registry);
  });
}
