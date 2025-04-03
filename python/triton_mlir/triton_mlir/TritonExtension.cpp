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
#include "circt/Conversion/Passes.h"
#include "circt/Dialect/SMT/SMTOps.h"
#include "circt/Dialect/Verif/VerifDialect.h"
#include "circt/Dialect/Verif/VerifPasses.h"
#include "lib/Target/LLVMIR/LLVMPasses.h"
#include "lld/Common/Driver.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "nvidia/include/NVGPUToLLVM/Passes.h"
#include "nvidia/include/TritonNVIDIAGPUToLLVM/Passes.h"
#include "proton/dialect/include/Dialect/Proton/IR/Dialect.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Membar.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
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
#include "llvm/Linker/Linker.h"
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
#include <nanobind/stl/optional.h>
#include <nanobind/stl/unique_ptr.h>
#include <numeric>
#include <unordered_set>

#include "passes.h"

namespace mlir::triton::AMD {
enum class ISAFamily;
void registerConvertArithToSMTPass();
void registerLowerContractsPassInFunctions();
} // namespace mlir::triton::AMD
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
              mlir::xegpu::XeGPUDialect,

              circt::smt::SMTDialect,
              circt::verif::VerifDialect
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
  mlir::triton::gpu::registerAllocateSharedMemoryPass();
  mlir::triton::registerConvertNVGPUToLLVMPass();
  mlir::triton::registerConvertTritonGPUToLLVMPass();
  mlir::triton::registerConvertTritonToTritonGPUPass();
  mlir::triton::gpu::registerTritonGPUGlobalScratchAllocationPass();

  // CIRCT passes
  mlir::triton::AMD::registerConvertArithToSMTPass();
  mlir::triton::AMD::registerLowerContractsPassInFunctions();
  circt::verif::registerStripContractsPass();
  circt::registerConvertVerifToSMT();

  // TritonAMDGPUToLLVM passes
  mlir::triton::registerConvertTritonAMDGPUToLLVM();
  mlir::triton::registerConvertBuiltinFuncToLLVM();
  // mlir::triton::registerDecomposeUnsupportedAMDConversions();
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

int tritonMlirPointerTypeGetAddressSpace(MlirType type) {
  return llvm::cast<mlir::triton::PointerType>(unwrap(type)).getAddressSpace();
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
  module.setTargetTriple(llvm::Triple(triple));
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

void populateTTDialect(nb::module_ &m) {
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
      .def_property_readonly("pointee_type",
                             [](MlirType self) {
                               return tritonMlirPointerTypeGetPointeeType(self);
                             })
      .def_property_readonly("address_space", [](MlirType self) {
        return tritonMlirPointerTypeGetAddressSpace(self);
      });
}

namespace {

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
  module.setTargetTriple(llvm::Triple(triple));
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

using ret = nb::rv_policy;

void init_triton_llvm(nb::module_ &m) {

  nb::class_<llvm::LLVMContext>(m, "context").def(nb::init<>());
  nb::class_<llvm::SourceMgr>(m, "source_mgr").def(nb::init<>());

  nb::class_<llvm::Module::FunctionListType>(m, "function_list")
      .def(
          "__iter__",
          [](llvm::Module::FunctionListType &s) {
            return nb::make_iterator<nb::rv_policy::reference_internal>(
                nb::type<llvm::Module::FunctionListType>(),
                "function_list_iterator", s.begin(), s.end());
          },
          nb::keep_alive<0, 1>());

  // Module Flag behavior. See
  // https://llvm.org/doxygen/classllvm_1_1Module.html#a0a5c55e12c97b80021330fe82b642293
  // for details.
  // nb::class_<llvm::Module::ModFlagBehavior>(m, "module_flag_behavior");
  // m.attr("MODULE_FLAG_BEHAVIOR_ERROR") = llvm::Module::Error;
  // m.attr("MODULE_FLAG_BEHAVIOR_WARNING") = llvm::Module::Warning;
  // m.attr("MODULE_FLAG_BEHAVIOR_REQUIRE") = llvm::Module::Require;
  // m.attr("MODULE_FLAG_BEHAVIOR_OVERRIDE") = llvm::Module::Override;
  // m.attr("MODULE_FLAG_BEHAVIOR_APPEND") = llvm::Module::Append;
  // m.attr("MODULE_FLAG_BEHAVIOR_APPEND_UNIQUE") = llvm::Module::AppendUnique;
  // m.attr("MODULE_FLAG_BEHAVIOR_MAX") = llvm::Module::Max;
  // m.attr("MODULE_FLAG_BEHAVIOR_MIN") = llvm::Module::Min;

  nb::class_<llvm::Module>(m, "module")
      .def(
          "__str__",
          [](llvm::Module *self) {
            std::string str;
            llvm::raw_string_ostream os(str);
            os << *self;
            return os.str();
          },
          ret::take_ownership)
      .def(
          "get_functions",
          [](llvm::Module *mod) -> llvm::Module::FunctionListType & {
            // Note: Backends assume that we are compiling exactly one kernel
            // (i.e. one function that's that's called by the CPU) and that it's
            // the first function in this list.
            return mod->getFunctionList();
          },
          ret::reference_internal)
      .def("add_flag",
           [](llvm::Module *mod, llvm::Module::ModFlagBehavior behavior,
              std::string &key, uint32_t value) {
             return mod->addModuleFlag(behavior, key, value);
           })
      .def("verify", [](llvm::Module *self) {
        std::string str;
        llvm::raw_string_ostream os(str);
        if (verifyModule(*self, &os)) {
          os.flush();
          throw std::runtime_error(str);
        }
        return true;
      });

  nb::class_<llvm::Function>(m, "function")
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

  // optimization levels
  nb::class_<llvm::OptimizationLevel>(m, "optimization_level");
  m.attr("OPTIMIZE_O0") = llvm::OptimizationLevel::O0;
  m.attr("OPTIMIZE_O1") = llvm::OptimizationLevel::O1;
  m.attr("OPTIMIZE_O2") = llvm::OptimizationLevel::O2;
  m.attr("OPTIMIZE_O3") = llvm::OptimizationLevel::O3;
  m.attr("OPTIMIZE_Os") = llvm::OptimizationLevel::Os;
  m.attr("OPTIMIZE_Oz") = llvm::OptimizationLevel::Oz;

  m.def(
      "to_module",
      [](mlir::ModuleOp &mod, llvm::LLVMContext &ctx) {
        return mlir::translateModuleToLLVMIR(mod, ctx);
      },
      nb::keep_alive<0, 2>());

  m.def("attach_datalayout", [](llvm::Module *mod, const std::string triple,
                                const std::string proc,
                                const std::string features) {
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

  m.def(
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
        using namespace llvm;
        LoopAnalysisManager lam;
        FunctionAnalysisManager fam;
        CGSCCAnalysisManager cgam;
        llvm::ModuleAnalysisManager mam;

        PassInstrumentationCallbacks *instrCbPtr = nullptr;
        PassInstrumentationCallbacks passInstrCb;
        StandardInstrumentations standardInstr(mod->getContext(),
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

        PipelineTuningOptions tuningOptions;
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
        std::unique_ptr<TargetMachine> targetMachine = nullptr;
        if (!arch.empty() && pluginFile.empty())
          targetMachine =
              createTargetMachine(mod, arch, enable_fp_fusion, features);
        PassBuilder pb(/*targetMachine=*/targetMachine.get(), tuningOptions,
                       std::nullopt, instrCbPtr);

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

        ModulePassManager mpm;
        pb.registerVectorizerStartEPCallback(
            [&](llvm::FunctionPassManager &fpm, llvm::OptimizationLevel level) {
              // Triton generates large structure of scalars which may pessimise
              // optimizations, we run a pass to break up phi of struct to make
              // sure all the struct are removed for the following passes.
              fpm.addPass(BreakStructPhiNodesPass());
              fpm.addPass(InstCombinePass());
            });
        bool enableAddressSanitizer =
            mlir::triton::tools::getBoolEnv("TRITON_ENABLE_ASAN");
        if (enableAddressSanitizer) {
          AddressSanitizerOptions Opts;
          mpm.addPass(AddressSanitizerPass(Opts));
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

  m.def(
      "translate_to_asm",
      [](std::string llvmIR, std::string triple, std::string proc,
         std::string features, std::vector<std::string> flags,
         bool enable_fp_fusion, bool isObject) -> nb::object {
        std::string obj;
        {
          // when allow_threads goes out of scope, gil will be released
          nb::gil_scoped_release allow_threads;
          // create LLVM module from C++
          llvm::LLVMContext context;
          std::unique_ptr<llvm::MemoryBuffer> buffer =
              llvm::MemoryBuffer::getMemBuffer(llvmIR.c_str());
          llvm::SMDiagnostic error;
          std::unique_ptr<llvm::Module> module =
              llvm::parseIR(buffer->getMemBufferRef(), error, context);
          if (!module) {
            llvm::report_fatal_error(
                "failed to parse IR: " + error.getMessage() +
                "lineno: " + std::to_string(error.getLineNo()));
          }
          obj = translateLLVMIRToASM(*module, triple, proc, features, flags,
                                     enable_fp_fusion, isObject);
        }
        if (isObject)
          return nb::bytes(obj.c_str());
        else
          return nb::str(obj.c_str());
      },
      ret::take_ownership);

  m.def("init_targets", []() {
    static std::once_flag init_flag;
    std::call_once(init_flag, []() {
      llvm::InitializeAllTargetInfos();
      llvm::InitializeAllTargets();
      llvm::InitializeAllTargetMCs();
      llvm::InitializeAllAsmParsers();
      llvm::InitializeAllAsmPrinters();
    });
  });

  m.def("link_extern_libs", [](llvm::Module *dstMod,
                               const std::vector<std::string> &paths) {
    if (paths.empty())
      return;

    llvm::LLVMContext &ctx = dstMod->getContext();
    llvm::Linker linker(*dstMod);
    for (const std::string &path : paths) {
      llvm::SMDiagnostic err;
      std::unique_ptr<llvm::Module> libMod = llvm::parseIRFile(path, err, ctx);
      if (!libMod) {
        std::string message = "Failed to parse library at " + path;
        throw std::invalid_argument(message);
      }
      libMod->setTargetTriple(dstMod->getTargetTriple());
      libMod->setDataLayout(dstMod->getDataLayout());

      std::unordered_set<std::string> externalFns;
      for (llvm::Function &fn : libMod->functions()) {
        if (!fn.isDeclaration())
          externalFns.insert(fn.getName().str());
      }

      if (linker.linkInModule(std::move(libMod),
                              llvm::Linker::Flags::LinkOnlyNeeded)) {
        std::string message = "Failed to link library at " + path;
        throw std::invalid_argument(message);
      }

      // Mark linked-in functions as internal because backends use external
      // linkage as a signifier of kernel functions.
      for (llvm::Function &fn : dstMod->functions()) {
        if (externalFns.count(fn.getName().str())) {
          fn.setLinkage(llvm::GlobalValue::InternalLinkage);
        }
      }
    }
  });
}

void init_triton_analysis(nb::module_ &&m) {
  nb::class_<mlir::ModuleAllocation>(m, "allocation")
      .def(nb::init<mlir::ModuleOp>());
  nb::class_<mlir::ModuleMembarAnalysis>(m, "membar")
      .def(nb::init<mlir::ModuleAllocation *>())
      .def("run", &mlir::ModuleMembarAnalysis::run);
}

void init_triton_passes_common(nb::module_ &&m) {
  using namespace mlir;
  ADD_PASS_WRAPPER_0("add_sccp", createSCCPPass);
  ADD_PASS_WRAPPER_0("add_symbol_dce", createSymbolDCEPass);
  ADD_PASS_WRAPPER_0("add_inliner", createInlinerPass);
  ADD_PASS_WRAPPER_0("add_canonicalizer", createCanonicalizerPass);
  ADD_PASS_WRAPPER_0("add_cse", createCSEPass);
  // ADD_PASS_WRAPPER_0("add_licm", createLoopInvariantCodeMotionPass);
  ADD_PASS_WRAPPER_0("print_ir", createPrintIRPass);
}

void init_triton_passes_ttir(nb::module_ &&m) {
  using namespace mlir::triton;
  ADD_PASS_WRAPPER_0("add_combine", createCombineOpsPass);
  ADD_PASS_WRAPPER_0("add_reorder_broadcast", createReorderBroadcastPass);
  ADD_PASS_WRAPPER_0("add_rewrite_tensor_pointer",
                     createRewriteTensorPointerPass);
  ADD_PASS_WRAPPER_0("add_loop_unroll", createLoopUnrollPass);
  ADD_PASS_WRAPPER_4("add_convert_to_ttgpuir",
                     createConvertTritonToTritonGPUPass, const std::string &,
                     int, int, int);
}

void init_triton_passes_ttgpuir(nb::module_ &&m) {
  using namespace mlir::triton::gpu;
  ADD_PASS_WRAPPER_0("add_coalesce", createTritonGPUCoalesce);
  ADD_PASS_WRAPPER_0("add_optimize_thread_locality",
                     createTritonGPUOptimizeThreadLocality);
  ADD_PASS_OPTION_WRAPPER_2("add_pipeline", createTritonGPUPipeline, int, bool);
  ADD_PASS_WRAPPER_0("add_prefetch", createTritonGPUPrefetch);
  ADD_PASS_WRAPPER_0("add_accelerate_matmul", createTritonGPUAccelerateMatmul);
  ADD_PASS_WRAPPER_0("add_reorder_instructions",
                     createTritonGPUReorderInstructions);
  ADD_PASS_WRAPPER_0("add_f32_dot_tc", createTritonGPUF32DotTC);
  ADD_PASS_OPTION_WRAPPER_1("add_optimize_dot_operands",
                            createTritonGPUOptimizeDotOperands, bool);
  ADD_PASS_WRAPPER_0("add_remove_layout_conversions",
                     createTritonGPURemoveLayoutConversions);
  ADD_PASS_WRAPPER_0("add_reduce_data_duplication",
                     createTritonGPUReduceDataDuplication);
  ADD_PASS_WRAPPER_0("add_allocate_warp_groups",
                     createTritonGPUAllocateWarpGroups);
  ADD_PASS_WRAPPER_0("add_allocate_shared_memory", createAllocateSharedMemory);
  ADD_PASS_WRAPPER_0("add_allocate_global_scratch_memory",
                     createTritonGPUGlobalScratchAllocationPass);
  ADD_PASS_WRAPPER_0("add_combine_tensor_select_and_if",
                     createTritonGPUCombineTensorSelectAndIf);
  ADD_PASS_WRAPPER_0("add_optimize_accumulator_init",
                     createTritonGPUOptimizeAccumulatorInit);
  ADD_PASS_WRAPPER_0("add_fuse_nested_loops", createTritonGPUFuseNestedLoops);
  ADD_PASS_WRAPPER_0("add_coalesce_async_copy",
                     createTritonGPUCoalesceAsyncCopy);
}

void init_triton_passes_convert(nb::module_ &&m) {
  using namespace mlir;
  ADD_PASS_WRAPPER_0("add_scf_to_cf", createSCFToControlFlowPass);
  ADD_PASS_WRAPPER_0("add_cf_to_llvmir", createConvertControlFlowToLLVMPass);
  ADD_PASS_WRAPPER_0("add_index_to_llvmir", createConvertIndexToLLVMPass);
  ADD_PASS_WRAPPER_0("add_arith_to_llvmir", createArithToLLVMConversionPass);
}

void init_triton_passes_llvmir(nb::module_ &&m) {
  using namespace mlir;
  ADD_PASS_WRAPPER_0("add_di_scope", createLLVMDIScopePass);
}

void init_triton_passes(nb::module_ &m) {
  init_triton_analysis(m.def_submodule("analysis"));
  init_triton_passes_common(m.def_submodule("common"));
  init_triton_passes_convert(m.def_submodule("convert"));
  init_triton_passes_ttir(m.def_submodule("ttir"));
  init_triton_passes_ttgpuir(m.def_submodule("ttgpuir"));
  init_triton_passes_llvmir(m.def_submodule("llvmir"));
}

const char *const amdTargetTriple = "amdgcn-amd-amdhsa";

void init_triton_amd_passes_ttgpuir(nb::module_ &&m) {
  using namespace mlir::triton;
  m.def("add_to_llvmir",
        [](mlir::PassManager &pm, const std::string &arch, bool ftz) {
          pm.addPass(createConvertTritonAMDGPUToLLVMPass(arch, ftz));
        });
  m.def("add_builtin_func_to_llvmir", [](mlir::PassManager &pm, bool ftz) {
    pm.addPass(createConvertBuiltinFuncToLLVMPass(ftz));
  });
  m.def("insert_instruction_sched_hints", [](mlir::PassManager &pm,
                                             const std::string &variant) {
    pm.addPass(createTritonAMDGPUInsertInstructionSchedHintsPass(variant));
  });
  m.def("lower_instruction_sched_hints",
        [](mlir::PassManager &pm, const std::string &arch, int32_t numStages) {
          pm.addPass(createTritonAMDGPULowerInstructionSchedHintsPass(
              arch, numStages));
        });
  m.def("add_decompose_unsupported_conversions",
        [](mlir::PassManager &pm, const std::string &arch) {
          // pm.addPass(
          //     mlir::triton::AMD::createDecomposeUnsupportedConversionsPass(arch));
        });
  ADD_PASS_WRAPPER_2("add_optimize_lds_usage",
                     mlir::triton::AMD::createOptimizeLDSUsagePass,
                     const std::string &, int32_t);
  ADD_PASS_WRAPPER_3("add_accelerate_matmul",
                     mlir::createTritonAMDGPUAccelerateMatmulPass,
                     const std::string, int, int);
  ADD_PASS_WRAPPER_0("add_optimize_epilogue",
                     mlir::createTritonAMDGPUOptimizeEpiloguePass);
  m.def("add_canonicalize_pointers", [](mlir::PassManager &pm) {
    pm.addNestedPass<mlir::triton::FuncOp>(
        mlir::createTritonAMDGPUCanonicalizePointersPass());
  });
  ADD_PASS_WRAPPER_1("add_convert_to_buffer_ops",
                     mlir::createTritonAMDGPUConvertToBufferOpsPass,
                     const std::string &);
  ADD_PASS_WRAPPER_0("add_reorder_instructions",
                     mlir::createTritonAMDGPUReorderInstructionsPass);
  ADD_PASS_WRAPPER_0("add_block_pingpong",
                     mlir::createTritonAMDGPUBlockPingpongPass);
  ADD_PASS_WRAPPER_3("add_stream_pipeline",
                     mlir::createTritonAMDGPUStreamPipelinePass, int, int, int);
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

void init_triton_amd(nb::module_ &m) {
  m.doc() = "Python bindings to the AMD Triton backend";

  auto passes = m.def_submodule("passes");
  init_triton_amd_passes_ttgpuir(passes.def_submodule("ttgpuir"));

  m.attr("TARGET_TRIPLE") = amdTargetTriple;
  m.attr("CALLING_CONV_AMDGPU_KERNEL") =
      (unsigned)llvm::CallingConv::AMDGPU_KERNEL;

  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::amdgpu::TritonAMDGPUDialect>();
    // registry.insert<mlir::ROCDL::ROCDLDialect>();
    mlir::registerROCDLDialectTranslation(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  m.def("attach_target_triple", [](llvm::Module *module) {
    module->setTargetTriple(llvm::Triple(amdTargetTriple));
  });

  // Set target architecture ISA version
  m.def("set_isa_version", [](llvm::Module *module, const std::string &arch) {
    llvm::AMDGPU::IsaVersion version = llvm::AMDGPU::getIsaVersion(arch);
    addControlConstant(module, "__oclc_ISA_version", /*bitwidth=*/32,
                       version.Major * 1000 + version.Minor * 100 +
                           version.Stepping);
  });

  // Set boolean control constant
  m.def("set_bool_control_constant",
        [](llvm::Module *module, const std::string &name, bool enable) {
          addControlConstant(module, name.c_str(), /*bitwidth=*/8, enable);
        });

  // Set code object ABI version
  m.def("set_abi_version", [](llvm::Module *module, int version) {
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

  m.def("cleanup_bitcode_metadata", [](llvm::Module *module) {
    // We can have Clang version metadata from device libraries linked in. We
    // don't care about them so drop them.
    if (auto *ident = module->getNamedMetadata("llvm.ident"))
      module->eraseNamedMetadata(ident);
    // Also various OpenCL version details.
    if (auto *openclVersion = module->getNamedMetadata("opencl.ocl.version"))
      module->eraseNamedMetadata(openclVersion);
  });

  m.def("disable_print_inline", [](llvm::Module *module) {
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

  m.def(
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

        return nb::bytes(result.data(), result.size());
      },
      nb::rv_policy::take_ownership);

  m.def("need_extern_lib", [](llvm::Module *module, const std::string &lib) {
    for (llvm::Function &f : module->functions()) {
      if (f.hasExternalLinkage() && f.hasName() && !f.hasExactDefinition()) {
        llvm::StringRef funcName = f.getName();
        // The rule for linking the extern lib:
        //    if the function name includes ocml or ockl, link
        //    ocml or ockl accordingly.
        if (funcName.contains(lib))
          return true;
        if (funcName.contains("__nv_")) {
          std::stringstream message;
          message << "Implicit conversion of CUDA " << funcName.str()
                  << " device function has been dropped; "
                  << "please, update your source program to use "
                     "triton.language.extra.<op> "
                  << "to replace triton.language.extra.cuda.<op>";
          throw std::runtime_error(message.str());
        }
      }
    }
    return false;
  });

  m.def("has_matrix_core_feature", [](const std::string &arch) {
    using mlir::triton::AMD::ISAFamily;
    switch (mlir::triton::AMD::deduceISAFamily(arch)) {
    case ISAFamily::CDNA4:
    case ISAFamily::CDNA3:
    case ISAFamily::CDNA2:
    case ISAFamily::CDNA1:
    case ISAFamily::RDNA3:
      return true;
    default:
      return false;
    }
  });

  m.def("set_all_fn_arg_inreg", [](llvm::Function *fn) {
    for (llvm::Argument &arg : fn->args()) {
      // Check for incompatible attributes.
      if (arg.hasByRefAttr() || arg.hasNestAttr())
        continue;
      arg.addAttr(llvm::Attribute::InReg);
    }
  });

  m.def("link_hsaco",
        [](const std::string &inPath, const std::string &outPath) {
          if (!lldInvoke(inPath.c_str(), outPath.c_str()))
            throw std::runtime_error("couldn't link");
        });
}

std::string locationToString(Location loc) {
  std::string str;
  llvm::raw_string_ostream os(str);
  loc.print(os);
  os.flush(); // Make sure all the content is dumped into the 'str' string
  return str;
}

void outputWarning(Location loc, const std::string &msg) {
  std::string locStr = locationToString(loc);

  PyErr_WarnEx(PyExc_UserWarning, (locStr + ": " + msg).c_str(),
               /*stack_level=*/2);
}

llvm::raw_fd_ostream &mlir_dumps() {
  std::error_code EC;
  static llvm::raw_fd_ostream S(::triton::tools::getStrEnv("MLIR_DUMP_PATH"),
                                EC, llvm::sys::fs::CD_CreateAlways);
  assert(!EC);
  return S;
}

llvm::raw_ostream &mlir_dumps_or_dbgs() {
  if (!::triton::tools::getStrEnv("MLIR_DUMP_PATH").empty()) {
    return mlir_dumps();
  } else {
    return llvm::dbgs();
  }
}

// Allow dump a reproducer in the console on crash.
struct ConsoleReproducerStream : public mlir::ReproducerStream {
  ~ConsoleReproducerStream() override {}

  StringRef description() override {
    return "std::errs, please share the reproducer above with Triton project.";
  }
  raw_ostream &os() override { return llvm::errs(); }
};

static ReproducerStreamFactory makeConsoleReproducer() {
  return [](std::string &error) -> std::unique_ptr<ReproducerStream> {
    return std::make_unique<ConsoleReproducerStream>();
  };
}

// Function to parse a comma-separated string into a vector of C-style strings
llvm::SmallVector<const char *, 3>
parseCommaSeparatedValues(const std::string &input,
                          llvm::SmallVector<std::string, 3> &storage) {
  llvm::SmallVector<StringRef, 3> split;
  llvm::SmallVector<const char *, 3> result;
  StringRef(input.c_str()).split(split, ',');
  llvm::transform(split, std::back_inserter(result), [&storage](StringRef str) {
    // StringRefs are not always null-terminated.
    // The purpose for this storage pattern is to
    // produce a collection of C-strings that are.
    storage.push_back(str.str());
    return storage.back().c_str();
  });
  return result;
}

// Run the pass manager under a source manager diagnostic handler, which
// enables emitted MLIR diagnostics to directly reference Python source
// code. This diagnostic handler supports filtering diagnostic info by
// severity levels.
struct TritonSourceMgrDiagnosticHandler : public SourceMgrDiagnosticHandler {
  TritonSourceMgrDiagnosticHandler(MLIRContext *ctx,
                                   DiagnosticSeverity minSeverity)
      : SourceMgrDiagnosticHandler(sourceMgr, ctx, llvm::errs()) {
    setHandler([this, minSeverity](Diagnostic &diag) {
      auto severity = diag.getSeverity();
      switch (severity) {
      case DiagnosticSeverity::Error:
        break;
      case DiagnosticSeverity::Warning:
        if (minSeverity == DiagnosticSeverity::Error)
          return success();
        break;
      case DiagnosticSeverity::Remark:
        if (minSeverity == DiagnosticSeverity::Error ||
            minSeverity == DiagnosticSeverity::Warning)
          return success();
        break;
      case DiagnosticSeverity::Note:
        // notes are handled somewhere else.
        return failure();
      default:
        llvm_unreachable("Unknown diagnostic severity");
      }
      emitDiagnostic(diag);
      return success();
    });
  }

  llvm::SourceMgr sourceMgr;
};

void init_triton_ir(nb::module_ &m) {
  using namespace mlir;
  using namespace triton;
  nb::enum_<PaddingOption>(m, "PADDING_OPTION")
      .value("PAD_ZERO", PaddingOption::PAD_ZERO)
      .value("PAD_NAN", PaddingOption::PAD_NAN)
      .export_values();

  nb::enum_<CacheModifier>(m, "CACHE_MODIFIER")
      .value("NONE", CacheModifier::NONE)
      .value("CA", CacheModifier::CA)
      .value("CG", CacheModifier::CG)
      .value("WB", CacheModifier::WB)
      .value("CS", CacheModifier::CS)
      .value("WT", CacheModifier::WT)
      .value("CV", CacheModifier::CV)
      .export_values();

  nb::enum_<MemSemantic>(m, "MEM_SEMANTIC")
      .value("ACQUIRE_RELEASE", MemSemantic::ACQUIRE_RELEASE)
      .value("ACQUIRE", MemSemantic::ACQUIRE)
      .value("RELEASE", MemSemantic::RELEASE)
      .value("RELAXED", MemSemantic::RELAXED)
      .export_values();

  nb::enum_<MemSyncScope>(m, "MEM_SYNC_SCOPE")
      .value("GPU", MemSyncScope::GPU)
      .value("CTA", MemSyncScope::CTA)
      .value("SYSTEM", MemSyncScope::SYSTEM)
      .export_values();

  nb::enum_<EvictionPolicy>(m, "EVICTION_POLICY")
      .value("NORMAL", EvictionPolicy::NORMAL)
      .value("EVICT_FIRST", EvictionPolicy::EVICT_FIRST)
      .value("EVICT_LAST", EvictionPolicy::EVICT_LAST)
      .export_values();

  nb::enum_<RMWOp>(m, "ATOMIC_OP")
      .value("ADD", RMWOp::ADD)
      .value("FADD", RMWOp::FADD)
      .value("AND", RMWOp::AND)
      .value("OR", RMWOp::OR)
      .value("XOR", RMWOp::XOR)
      .value("XCHG", RMWOp::XCHG)
      .value("MAX", RMWOp::MAX)
      .value("MIN", RMWOp::MIN)
      .value("UMIN", RMWOp::UMIN)
      .value("UMAX", RMWOp::UMAX);

  nb::enum_<RoundingMode>(m, "ROUNDING_MODE")
      .value("RTZ", RoundingMode::RTZ)
      .value("RTNE", RoundingMode::RTNE);

  nb::enum_<PropagateNan>(m, "PROPAGATE_NAN")
      .value("NONE", PropagateNan::NONE)
      .value("ALL", PropagateNan::ALL);

  nb::enum_<InputPrecision>(m, "INPUT_PRECISION")
      .value("TF32", InputPrecision::TF32)
      .value("TF32x3", InputPrecision::TF32x3)
      .value("IEEE", InputPrecision::IEEE)
      .export_values();

  nb::enum_<ScaleDotElemType>(m, "ScaleDotElemTypeTY")
      .value("E4M3", ScaleDotElemType::E4M3)
      .value("E5M2", ScaleDotElemType::E5M2)
      .value("E2M3", ScaleDotElemType::E2M3)
      .value("E3M2", ScaleDotElemType::E3M2)
      .value("E2M1", ScaleDotElemType::E2M1)
      .value("BF16", ScaleDotElemType::BF16)
      .value("FP16", ScaleDotElemType::FP16)
      .export_values();

  nb::class_<MLIRContext>(m, "context")
      .def(nb::init<>())
      .def("printOpOnDiagnostic",
           [](MLIRContext &self, bool v) { self.printOpOnDiagnostic(v); })
      .def("printStackTraceOnDiagnostic",
           [](MLIRContext &self, bool v) {
             self.printStackTraceOnDiagnostic(v);
           })
      .def("disable_multithreading",
           [](MLIRContext &self) { self.disableMultithreading(); });

  nb::class_<SourceMgrDiagnosticHandler>(m, "source_mgr_diag")
      .def(nb::init<llvm::SourceMgr &, MLIRContext *>());

  m.def("load_dialects", [](MLIRContext &context) {
    DialectRegistry registry;
    registry.insert<TritonDialect, ::mlir::triton::gpu::TritonGPUDialect,
                    math::MathDialect, arith::ArithDialect, scf::SCFDialect,
                    ::mlir::gpu::GPUDialect, cf::ControlFlowDialect,
                    ::mlir::triton::proton::ProtonDialect, LLVM::LLVMDialect,
                    mlir::ub::UBDialect>();
    mlir::LLVM::registerInlinerInterface(registry);
    registerBuiltinDialectTranslation(registry);
    registerLLVMDialectTranslation(registry);
    mlir::LLVM::registerInlinerInterface(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  nb::class_<Type>(m, "type")
      .def("is_integer",
           [](Type &self, unsigned width) { return self.isInteger(width); })
      .def("is_fp16", &Type::isF16)
      .def("__eq__",
           [](Type &self, nb::object &other) {
             Type *other_ty = nb::cast<Type *>(other);
             return (other_ty != nullptr) && (*other_ty == self);
           })
      .def("__ne__",
           [](Type &self, nb::object &other) {
             Type *other_ty = nb::cast<Type *>(other);
             return (other_ty == nullptr) || (*other_ty != self);
           })
      .def("__str__", [](Type &self) {
        std::string str;
        llvm::raw_string_ostream os(str);
        self.print(os);
        return os.str();
      });

  nb::class_<FunctionType>(m, "function_type")
      .def("param_types", [](FunctionType &self) {
        return std::vector<Type>(self.getInputs().begin(),
                                 self.getInputs().end());
      });

  nb::class_<Location>(m, "location").def("__str__", [](Location &self) {
    std::string str;
    llvm::raw_string_ostream os(str);
    self.print(os);
    return os.str();
  });

  nb::class_<Value>(m, "value")
      .def("set_attr",
           [](Value &self, std::string &name, Attribute &attr) -> void {
             if (Operation *definingOp = self.getDefiningOp())
               definingOp->setAttr(name, attr);
             else {
               auto arg = mlir::cast<BlockArgument>(self);
               int id = arg.getArgNumber();
               std::string attrName = name + "_arg" + std::to_string(id);
               Block *owner = arg.getOwner();
               if (owner->isEntryBlock() &&
                   !isa<FuncOp>(owner->getParentOp())) {
                 owner->getParentOp()->setAttr(attrName, attr);
               }
             }
           })
      .def("get_context", &Value::getContext)
      .def("replace_all_uses_with",
           [](Value &self, Value &newValue) {
             self.replaceAllUsesWith(newValue);
           })
      .def("get_type", &Value::getType)
      .def("id", [](Value &self) {
        // The Value is identified by and compared with
        // other Values via the underlying ValueImpl
        return (uint64_t)self.getImpl();
      });

  nb::class_<OpResult, Value>(m, "op_result");

  nb::class_<BlockArgument, Value>(m, "block_argument");

  nb::class_<Region>(m, "region")
      .def("get_parent_region", &Region::getParentRegion, ret::reference)
      .def("size", [](Region &self) { return self.getBlocks().size(); })
      .def("empty", &Region::empty)
      .def("id", [](Region &self) { return (uint64_t)&self; });

  nb::class_<Block>(m, "block")
      .def("arg",
           [](Block &self, int index) -> BlockArgument {
             if (index >= self.getNumArguments())
               throw nb::index_error("Block argument index out of range");
             return self.getArgument(index);
           })
      .def("add_argument",
           [](Block &self, Type ty) {
             auto loc = UnknownLoc::get(ty.getContext());
             self.addArgument(ty, loc);
           })
      .def("get_num_arguments", &Block::getNumArguments)
      .def("get_argument", &Block::getArgument)
      .def("dump", &Block::dump)
      .def("move_before",
           [](Block &self, Block &dst) { self.moveBefore(&dst); })
      .def("insert_before", &Block::insertBefore)
      .def("get_parent", &Block::getParent, ret::reference)
      .def("merge_block_before",
           [](Block &self, Block &dst) {
             // ref: RewriterBase::mergeBlocks()
             if (self.getNumArguments() != 0)
               throw std::runtime_error(
                   "This block has arguments, don't merge");
             dst.getOperations().splice(dst.begin(), self.getOperations());
             self.dropAllUses();
             self.erase();
           })
      .def("replace_use_in_block_with",
           [](Block &self, Value &v, Value &newVal) {
             v.replaceUsesWithIf(newVal, [&](OpOperand &operand) {
               Operation *user = operand.getOwner();
               Block *currentBlock = user->getBlock();
               while (currentBlock) {
                 if (currentBlock == &self)
                   return true;
                 // Move up one level
                 currentBlock =
                     currentBlock->getParent()->getParentOp()->getBlock();
               }
               return false;
             });
           })
      .def("__str__",
           [](Block &self) {
             std::string str;
             llvm::raw_string_ostream os(str);
             self.print(os);
             return str;
           })
      .def("has_terminator",
           [](Block &self) {
             return !self.empty() &&
                    self.back().hasTrait<OpTrait::IsTerminator>();
           })
      .def("has_return",
           [](Block &self) {
             return !self.empty() &&
                    self.back().hasTrait<OpTrait::ReturnLike>();
           })
      .def("erase", [](Block &self) { self.erase(); })
      .def("id", [](Block &self) { return (uint64_t)&self; });

  nb::class_<Attribute>(m, "attribute");
  nb::class_<IntegerAttr, Attribute>(m, "integer_attr");
  nb::class_<BoolAttr, Attribute>(m, "bool_attr");
  nb::class_<UnitAttr, Attribute>(m, "unit_attr");

  // Ops
  nb::class_<OpState>(m, "OpState")
      .def("set_attr",
           [](OpState &self, std::string &name, Attribute &attr) -> void {
             self->setAttr(name, attr);
           })
      .def("get_num_results",
           [](OpState &self) -> unsigned { return self->getNumResults(); })
      .def("get_result",
           [](OpState &self, unsigned idx) -> Value {
             if (idx >= self->getNumResults())
               throw nb::index_error("Op result index out of range");
             return self->getResult(idx);
           })
      .def(
          "get_region",
          [](OpState &self, unsigned idx) -> Region & {
            if (idx >= self->getNumRegions())
              throw nb::index_error("Op region index out of range");
            return self->getRegion(idx);
          },
          ret::reference)
      .def(
          "get_body",
          [](scf::ForOp &self, unsigned idx) -> Block * {
            if (idx >= self->getNumRegions())
              throw nb::index_error("Op region index out of range");
            return self.getBody(idx);
          },
          ret::reference)
      .def("dump", [](OpState &self) { self->dump(); })
      .def("__str__",
           [](OpState &self) -> std::string {
             std::string str;
             llvm::raw_string_ostream os(str);
             auto printingFlags = OpPrintingFlags();
             printingFlags.enableDebugInfo();
             self->print(os, printingFlags);
             return str;
           })
      .def("append_operand",
           [](OpState &self, Value &val) {
             self->insertOperands(self->getNumOperands(), val);
           })
      .def("verify", [](OpState &self) -> bool {
        return succeeded(verify(self.getOperation()));
      });
  // scf Ops
  nb::class_<scf::ForOp, OpState>(m, "ForOp")
      .def("get_induction_var", &scf::ForOp::getInductionVar);

  nb::class_<scf::IfOp, OpState>(m, "IfOp")
      .def("get_then_block", &scf::IfOp::thenBlock, ret::reference)
      .def("get_else_block", &scf::IfOp::elseBlock, ret::reference)
      .def("get_then_yield", &scf::IfOp::thenYield)
      .def("get_else_yield", &scf::IfOp::elseYield);
  nb::class_<scf::YieldOp, OpState>(m, "YieldOp");
  nb::class_<scf::WhileOp, OpState>(m, "WhileOp")
      .def("get_before", &scf::WhileOp::getBefore, ret::reference)
      .def("get_after", &scf::WhileOp::getAfter, ret::reference);
  nb::class_<scf::ConditionOp, OpState>(m, "ConditionOp");

  nb::class_<Operation>(m, "operation")
      .def("get_name",
           [](Operation &self) {
             llvm::StringRef opName = self.getName().getStringRef();
             return opName.str();
           })
      .def("get_num_operands", &Operation::getNumOperands)
      .def("get_operand", &Operation::getOperand)
      .def("get_num_results", &Operation::getNumResults)
      .def("get_result", &Operation::getResult)
      .def("get_num_regions", &Operation::getNumRegions)
      .def("get_region", &Operation::getRegion, ret::reference)
      .def("get_block", &Operation::getBlock, ret::reference)
      .def("get_str_attr",
           [](Operation &self, const std::string &name) -> nb::object {
             auto ret = self.getAttrOfType<StringAttr>(name);
             if (!ret)
               return nb::none();
             return nb::str(ret.getValue().str().c_str());
           })
      .def("get_bool_attr",
           [](Operation &self, const std::string &name) -> nb::object {
             auto ret = self.getAttrOfType<BoolAttr>(name);
             if (!ret)
               return nb::none();
             return nb::bool_(ret.getValue());
           })
      .def("get_flat_symbol_ref_attr",
           [](Operation &self, const std::string &name) -> nb::object {
             auto ret = self.getAttrOfType<FlatSymbolRefAttr>(name);
             if (!ret)
               return nb::none();
             return nb::str(ret.getValue().str().c_str());
           });

  nb::class_<ModuleOp, OpState>(m, "module", nb::dynamic_attr())
      // Triton uses a dynamic_attr to "transfer" context to the module???
      .def_prop_ro("context", &ModuleOp::getContext)
      .def("dump", &ModuleOp::dump)
      .def("str",
           [](ModuleOp &self) -> std::string {
             std::string str;
             llvm::raw_string_ostream os(str);
             auto printingFlags = OpPrintingFlags();
             printingFlags.enableDebugInfo();
             self.print(os, printingFlags);
             return str;
           })
      .def("push_back",
           [](ModuleOp &self, FuncOp &funcOp) -> void {
             self.push_back(funcOp);
           })
      .def("get_entry_func_name",
           [](ModuleOp &self) -> std::string {
             for (auto &op : self.getOps()) {
               if (auto func = dyn_cast<FuncOp>(op)) {
                 if (LLVM::isKernel(func))
                   return func.getName().str();
               }
             }
             return "";
           })
      .def("has_function",
           [](ModuleOp &self, std::string &funcName) -> bool {
             if (self.lookupSymbol(funcName))
               return true;
             return false;
           })
      .def("get_function",
           [](ModuleOp &self, std::string &funcName) -> FuncOp {
             return self.lookupSymbol<FuncOp>(funcName);
           })
      /*
       * def ty_to_cpp(ty) is the consumer of this function.
       * If the type is a ptr it expects ty[0] == '*', else the type itself.
       */

      .def("get_function_signature",
           [](ModuleOp &self, FuncOp &func) -> std::vector<std::string> {
             std::vector<std::string> strVec;

             auto type = func.getFunctionType();
             unsigned numArgs = type.getNumInputs();
             for (unsigned i = 0; i != numArgs; ++i) {
               std::string tempType;
               llvm::raw_string_ostream os(tempType);

               auto ty = type.getInput(i);
               if (auto attributes = func.getCallableArgAttrs()) {
                 Attribute attr = attributes[i];
                 // Check for tt.nv_tma_desc = 1
                 if (auto dAttr = dyn_cast<DictionaryAttr>(attr)) {
                   if (dAttr.contains("tt.nv_tma_desc")) {
                     strVec.push_back("nvTmaDesc");
                     continue;
                   }
                 }
               }
               if (auto ptrType = dyn_cast<PointerType>(ty)) {
                 auto pType = ptrType.getPointeeType();
                 os << "*";
                 pType.print(os);
               } else {
                 ty.print(os);
               }
               strVec.push_back(tempType);
             }
             return strVec;
           })
      .def("get_int_attr",
           [](ModuleOp &self, std::string name) -> nb::object {
             auto ret = self->getAttrOfType<IntegerAttr>(name);
             if (!ret)
               return nb::none();
             return nb::int_(ret.getInt());
           })
      .def("create_location_snapshot",
           [](ModuleOp &self, const std::string &fileName) -> void {
             generateLocationsFromIR(/*raw_ostream=*/llvm::nulls(),
                                     /*fileName=*/fileName,
                                     /*op=*/self, /*flags=*/{});
           })
      .def("walk",
           [](ModuleOp &self, const std::function<void(Operation *)> &fn) {
             self.walk(fn);
           });

  m.def("make_attr", [](const std::vector<int> &values, MLIRContext &context) {
    return mlir::cast<Attribute>(DenseIntElementsAttr::get(
        RankedTensorType::get({static_cast<int64_t>(values.size())},
                              IntegerType::get(&context, 32)),
        values));
  });

  m.def(
      "parse_mlir_module",
      [](const std::string &inputFilename, MLIRContext &context) {
        // parse module
        OwningOpRef<ModuleOp> module =
            parseSourceFile<ModuleOp>(inputFilename, &context);
        if (!module)
          throw std::runtime_error("Parse MLIR file failed.");
        return module->clone();
      },
      ret::take_ownership);

  nb::class_<FuncOp, OpState>(m, "function")
      // .def_prop_ro("attrs", &ir::function::attrs)
      // .def("add_attr", &ir::function::add_attr);
      .def("args",
           [](FuncOp &self, unsigned idx) -> BlockArgument {
             if (idx >= self.getNumArguments())
               throw nb::index_error("Function argument index out of range");
             return self.getArgument(idx);
           })
      .def("get_num_args", &FuncOp::getNumArguments)
      .def(
          "add_entry_block",
          [](FuncOp &self) -> Block * { return self.addEntryBlock(); },
          ret::reference)
      .def(
          "set_arg_attr",
          [](FuncOp &self, int arg_no, const std::string &name, int val) {
            if (arg_no >= self.getNumArguments())
              throw nb::index_error("Function argument index out of range");
            // set arg attributes "name" to value "val"
            auto attrTy = IntegerType::get(self.getContext(), 32);
            self.setArgAttr(arg_no, name, IntegerAttr::get(attrTy, val));
          },
          ret::reference)
      //  .def("has_attr", &::FuncOp::hasAttr)
      .def("finalize",
           [](FuncOp &self) -> void {
             // Check if the result of tl.advance is used
             self.walk([&](AdvanceOp op) {
               if (op->getResult(0).use_empty())
                 outputWarning(op->getLoc(), "The result of tl.advance is not "
                                             "being used. Note that tl.advance "
                                             "does not have any side effects. "
                                             "To move the block pointer, you "
                                             "need to assign the result of "
                                             "tl.advance to a variable.");
             });
           })
      .def_prop_ro("type", &FuncOp::getFunctionType)
      .def("reset_type", &FuncOp::setType);

  nb::class_<OpBuilder::InsertPoint>(m, "InsertPoint");

  nb::class_<PassManager>(m, "pass_manager")
      .def(nb::init<MLIRContext *>())
      .def("enable_debug",
           [](PassManager &self) {
             auto *context = self.getContext();
             bool haveDump = ::triton::tools::getBoolEnv("MLIR_ENABLE_DUMP");
             std::string funcToDump;
             if (!haveDump) {
               funcToDump = triton::tools::getStrEnv("MLIR_ENABLE_DUMP");
               if (!funcToDump.empty())
                 haveDump = true;
             }
             if (haveDump) {
               context->disableMultithreading();
               auto printingFlags = OpPrintingFlags();
               printingFlags.elideLargeElementsAttrs(16);
               printingFlags.enableDebugInfo();
               auto printAlways = [funcToDump](Pass *, Operation *op) -> bool {
                 if (funcToDump.empty())
                   return true;
                 if (auto mod = dyn_cast<mlir::ModuleOp>(op)) {
                   return mod.lookupSymbol(funcToDump);
                 }
                 if (auto func = dyn_cast<triton::FuncOp>(op)) {
                   return SymbolTable::getSymbolName(func).getValue() ==
                          funcToDump;
                 }

                 return false;
               };
               self.enableIRPrinting(
                   /*shouldPrintBeforePass=*/printAlways,
                   /*shouldPrintAfterPass=*/printAlways,
                   /*printModuleScope=*/true,
                   /*printAfterOnlyOnChange=*/false,
                   /*printAfterOnlyOnFailure*/ true, mlir_dumps_or_dbgs(),
                   printingFlags);
             }
           })
      .def("get_pipeline_str",
           [](PassManager &self) {
             std::string str;
             llvm::raw_string_ostream os(str);
             self.printAsTextualPipeline(os);
             return str;
           })
      .def("run", [](PassManager &self, ModuleOp &mod) {
        // TODO: maybe dump module to file and print error for better
        // diagnostics

        auto *context = mod.getContext();

        auto reproducerPath =
            triton::tools::getStrEnv("TRITON_REPRODUCER_PATH");
        if (!reproducerPath.empty()) {
          auto anchorName = self.getOpAnchorName();
          auto passes = self.getPasses();
          Operation *op = mod.getOperation();
          // Save a reproducer for the current pass manager invocation
          // immediately.
          makeReproducer(anchorName, passes, op, reproducerPath);
          // But if the pass manager crashes, attempt to generate a local
          // reproducer instead.
          context->disableMultithreading();
          self.enableCrashReproducerGeneration(reproducerPath,
                                               /*genLocalReproducer=*/true);
        } else {
          self.enableCrashReproducerGeneration(makeConsoleReproducer());
        }

        if (triton::tools::getBoolEnv("TRITON_ENABLE_LLVM_DEBUG")) {
          ::llvm::DebugFlag = true;
        }

        if (auto debugOnly = triton::tools::getStrEnv("TRITON_LLVM_DEBUG_ONLY");
            !debugOnly.empty()) {
          llvm::SmallVector<std::string, 3> storage;
          llvm::SmallVector<const char *, 3> debugTypes =
              parseCommaSeparatedValues(debugOnly, storage);
          ::llvm::DebugFlag = true;
          using namespace llvm;
          setCurrentDebugTypes(debugTypes.data(), debugTypes.size());
        }

        bool haveTiming = ::triton::tools::getBoolEnv("MLIR_ENABLE_TIMING");
        if (haveTiming) {
          self.enableTiming();
        }

        // setting up diagnostics
        bool showOperations = false, showStacktraces = false,
             showRemarks = false, showWarnings = false;

        if (auto enableDiagnostics =
                triton::tools::getStrEnv("MLIR_ENABLE_DIAGNOSTICS");
            !enableDiagnostics.empty()) {
          llvm::SmallVector<std::string, 3> storage;
          parseCommaSeparatedValues(enableDiagnostics, storage);
          for (auto &str : storage) {
            if (str == "warnings") {
              showWarnings = true;
            } else if (str == "remarks") {
              showRemarks = true;
            } else if (str == "stacktraces") {
              showStacktraces = true;
            } else if (str == "operations") {
              showOperations = true;
            }
            // we show errors by default, so no need to set it
          }
        }

        DiagnosticSeverity minSeverity = showWarnings
                                             ? DiagnosticSeverity::Warning
                                             : DiagnosticSeverity::Error;
        minSeverity = showRemarks ? DiagnosticSeverity::Remark : minSeverity;

        TritonSourceMgrDiagnosticHandler diagHandler(context, minSeverity);

        context->printOpOnDiagnostic(showOperations);
        context->printStackTraceOnDiagnostic(showStacktraces);
        if (showStacktraces) {
          context->disableMultithreading();
        }
        if (failed(self.run(mod.getOperation())))
          throw std::runtime_error("PassManager::run failed");
      });
}

} // namespace

namespace eudsl {
using namespace mlir;
using namespace triton::gpu;
using namespace triton::amdgpu;
#include "BlockedEncodingAttr_decls_defns.cpp.inc"
#include "SliceEncodingAttr_decls_defns.cpp.inc"
#include "SwizzledSharedEncodingAttr_decls_defns.h.inc"
#include "TritonAMDGPUAttrDefs_MlirAttribute_decls.h.inc"
#include "TritonAMDGPUAttrDefs_MlirAttribute_defns.cpp.inc"
#include "TritonGPUAttrDefs_MlirAttribute_decls.h.inc"
#include "TritonGPUAttrDefs_MlirAttribute_defns.cpp.inc"
#include "TritonGPUTypes_MlirType_decls.h.inc"
#include "TritonGPUTypes_MlirType_defns.cpp.inc"

MlirAttribute mlirSharedMemorySpaceAttributeGet(MlirContext mlirContext) {
  mlir::MLIRContext *context = unwrap(mlirContext);
  return wrap(SharedMemorySpaceAttr::get(context));
}

MlirTypeID mlirSharedMemorySpaceAttrGetTypeID() {
  return wrap(SharedMemorySpaceAttr::getTypeID());
}

bool isaMlirSharedMemorySpaceAttr(MlirAttribute thing) {
  return isa<SharedMemorySpaceAttr>(unwrap(thing));
}
} // namespace eudsl

void populateTTGDialect(nb::module_ &m) {
  using namespace eudsl;
  using namespace mlir::triton::gpu;
  using namespace triton::amdgpu;
#include "BlockedEncodingAttr_nbclasses.cpp.inc"

#include "SliceEncodingAttr_nbclasses.cpp.inc"

#include "SwizzledSharedEncodingAttr_nbclasses.cpp.inc"
#include "TritonGPUAttrDefs_MlirAttribute_nbclasses.cpp.inc"
#include "TritonGPUTypes_MlirType_nbclasses.cpp.inc"

  auto nbSharedMemorySpaceAttr = mlir_attribute_subclass(
      m, "SharedMemorySpaceAttr", isaMlirSharedMemorySpaceAttr,
      mlirSharedMemorySpaceAttrGetTypeID);
  nbSharedMemorySpaceAttr.def_staticmethod(
      "get",
      [](MlirContext context) {
        return mlirSharedMemorySpaceAttributeGet(context);
      },
      "context"_a = nb::none());
}

void populateAMDGPUDialect(nb::module_ &m) {
  using namespace eudsl;
  using namespace mlir::triton::gpu;
  using namespace triton::amdgpu;
#include "TritonAMDGPUAttrDefs_MlirAttribute_nbclasses.cpp.inc"
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
  // populateTritonExtension(m);
  auto ir = m.def_submodule("tritonir");
  init_triton_ir(ir);
  auto llvm = m.def_submodule("llvm");
  init_triton_llvm(llvm);
  auto amd = m.def_submodule("amd");
  init_triton_amd(amd);

  auto passes = m.def_submodule("passes");
  init_triton_passes(passes);

  auto ttDialect = m.def_submodule("tt");
  populateTTDialect(ttDialect);

  auto ttgDialect = m.def_submodule("ttg");
  populateTTGDialect(ttgDialect);

  auto amdgpuDialect = m.def_submodule("amdgpu");
  populateAMDGPUDialect(amdgpuDialect);

  m.def(
      "unwrap_c_context", [](MlirContext context) { return unwrap(context); },
      nb::rv_policy::reference);

  m.def("wrap_context", [](mlir::MLIRContext *context) {
    nb::object capsule =
        nb::steal<nb::object>(mlirPythonContextToCapsule(wrap(context)));
    return nb::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
        .attr("Context")
        .attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
        .release();
  });

  m.def("unwrap_c_type", [](MlirType type) { return unwrap(type); });

  m.def("wrap_type", [](mlir::Type type) {
    nb::object capsule =
        nb::steal<nb::object>(mlirPythonTypeToCapsule(wrap(type)));
    return nb::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
        .attr("Type")
        .attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
        .release();
  });

  m.def("unwrap_c_attribute",
        [](MlirAttribute attribute) { return unwrap(attribute); });

  m.def("wrap_attribute", [](mlir::Attribute attribute) {
    nb::object capsule =
        nb::steal<nb::object>(mlirPythonAttributeToCapsule(wrap(attribute)));
    return nb::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
        .attr("Attribute")
        .attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
        .release();
  });

  m.def("unwrap_c_module_op", [](MlirOperation mlirMaybeModuleOp) {
    Operation *maybeModOp = unwrap(mlirMaybeModuleOp);
    if (auto modOp = llvm::dyn_cast<ModuleOp>(maybeModOp))
      return modOp;
    throw std::runtime_error("operation isn't ModuleOp");
  });
  m.def("unwrap_c_module_op",
        [](MlirModule moduleOp) { return unwrap(moduleOp); });

  m.def("wrap_module_op", [](mlir::ModuleOp moduleOp) {
    nb::object capsule = nb::steal<nb::object>(
        mlirPythonOperationToCapsule(wrap(moduleOp.getOperation())));
    return nb::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
        .attr("Module")
        .attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
        .release();
  });

  m.def("unwrap_c_op", [](MlirOperation mlirMaybeModuleOp) {
    Operation *maybeModOp = unwrap(mlirMaybeModuleOp);
    if (auto modOp = llvm::dyn_cast<ModuleOp>(maybeModOp))
      return modOp;
    throw std::runtime_error("operation isn't ModuleOp");
  });

  m.def("wrap_op", [](mlir::ModuleOp moduleOp) {
    nb::object capsule = nb::steal<nb::object>(
        mlirPythonOperationToCapsule(wrap(moduleOp.getOperation())));
    return nb::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
        .attr("Operation")
        .attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
        .release();
  });
}

NB_MODULE(_site_initialize_0, m) {
  m.def("register_dialects", [](MlirDialectRegistry registry) {
    tritonMlirRegisterTritonDialectsAndPasses(registry);
  });
}