//===- Dialects.cpp - CAPI for dialects -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "triton-c/Dialects.h"

#include "amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "amd/include/TritonAMDGPUToLLVM/Passes.h"
#include "amd/include/TritonAMDGPUToLLVM/TargetUtils.h"
#include "amd/include/TritonAMDGPUTransforms/Passes.h"
#include "lib/Target/LLVMIR/LLVMPasses.h"
#include "mlir-c/Target/LLVMIR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
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
#include "llvm-c/Core.h"
#include "llvm-c/TargetMachine.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/TargetParser.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizer.h"

namespace mlir::test {
void registerTestAliasPass();
void registerTestAlignmentPass();
void registerTestAllocationPass();
void registerTestMembarPass();
} // namespace mlir::test

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

void addModuleFlag(LLVMModuleRef llvmModRef, MlirStringRef Key, uint32_t Val) {
  llvm::unwrap(llvmModRef)
      ->addModuleFlag(llvm::Module::Error, unwrap(Key), Val);
}

IsaVersion getIsaVersion(MlirStringRef arch) {
  auto isav = llvm::AMDGPU::getIsaVersion(unwrap(arch));
  return {isav.Major, isav.Minor, isav.Stepping};
}

unsigned getCallingConvAMDGPUKernel() {
  return llvm::CallingConv::AMDGPU_KERNEL;
}

static LLVMTargetMachineRef wrap(const llvm::TargetMachine *P) {
  return reinterpret_cast<LLVMTargetMachineRef>(
      const_cast<llvm::TargetMachine *>(P));
}

static llvm::TargetMachine *unwrap(LLVMTargetMachineRef P) {
  return reinterpret_cast<llvm::TargetMachine *>(P);
}

LLVMTargetMachineRef createTargetMachine(MlirStringRef targetTripleRef,
                                         MlirStringRef proc,
                                         bool enable_fp_fusion,
                                         MlirStringRef features,
                                         bool disableLLVMOpt) {
  std::string error;
  auto target =
      llvm::TargetRegistry::lookupTarget(unwrap(targetTripleRef), error);
  if (!error.empty())
    llvm::report_fatal_error(error.c_str());
  llvm::TargetOptions opt;
  if (enable_fp_fusion)
    opt.AllowFPOpFusion = llvm::FPOpFusion::Fast;
  opt.UnsafeFPMath = false;
  opt.NoInfsFPMath = false;
  opt.NoNaNsFPMath = true;
  opt.TrapUnreachable = true;
  opt.MCOptions.AsmVerbose = true;
  opt.MCOptions.PreserveAsmComments = true;
  llvm::TargetMachine *machine = target->createTargetMachine(
      unwrap(targetTripleRef), unwrap(proc), unwrap(features), opt,
      llvm::Reloc::PIC_, std::nullopt,
      disableLLVMOpt ? llvm::CodeGenOptLevel::None
                     : llvm::CodeGenOptLevel::Aggressive);
  return wrap(machine);
}

void initAllLLVMTargets() {
  static std::once_flag init_flag;
  std::call_once(init_flag, []() {
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllAsmPrinters();
  });
}

LLVMModuleRef optimizeModule(LLVMModuleRef llvmModRef, MlirStringRef flagList,
                             MlirStringRef arch, MlirStringRef features,
                             LLVMOptimizationLevel optLevel,
                             bool enableFPFusion, bool llvmIREnableDump,
                             bool enableAddressSanitizer, bool disableLLVMOpt) {
  if (disableLLVMOpt)
    return llvmModRef;
  // Check to see if we are passing a list of flags to disable
  // optimizations.
  if (flagList.data) {
    auto options = llvm::cl::getRegisteredOptions();
    llvm::SmallVector<llvm::StringRef, 3> split;
    llvm::StringRef(unwrap(flagList)).split(split, ',');
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
  llvm::Module *mod = llvm::unwrap(llvmModRef);
  llvm::StandardInstrumentations standardInstr(mod->getContext(),
                                               /*DebugLogging*/ true);
  if (llvmIREnableDump) {
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
  llvm::TargetMachine *targetMachine = nullptr;
  if (arch.data)
    targetMachine = unwrap(createTargetMachine(
        mlirStringRefCreateFromCString(mod->getTargetTriple().c_str()), arch,
        enableFPFusion, features, disableLLVMOpt));
  llvm::PassBuilder pb(/*targetMachine=*/targetMachine, tuningOptions,
                       std::nullopt, instrCbPtr);

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
  if (enableAddressSanitizer) {
    llvm::AddressSanitizerOptions Opts;
    mpm.addPass(llvm::AddressSanitizerPass(Opts));
  }
  llvm::OptimizationLevel OL;
  switch (optLevel) {
  case O0:
    OL = llvm::OptimizationLevel::O0;
    break;
  case O1:
    OL = llvm::OptimizationLevel::O1;
    break;
  case O2:
    OL = llvm::OptimizationLevel::O2;
    break;
  case O3:
    OL = llvm::OptimizationLevel::O3;
    break;
  case Os:
    OL = llvm::OptimizationLevel::Os;
    break;
  case Oz:
    OL = llvm::OptimizationLevel::Oz;
    break;
  }
  mpm.addPass(pb.buildPerModuleDefaultPipeline(OL));
  mpm.run(*mod, mam);
  return llvm::wrap(mod);
}

void cleanupBitcodeMetadata(LLVMModuleRef llvmModRef) {
  llvm::Module *module = llvm::unwrap(llvmModRef);
  // We can have Clang version metadata from device libraries linked in. We
  // don't care about them so drop them.
  if (auto *ident = module->getNamedMetadata("llvm.ident"))
    module->eraseNamedMetadata(ident);
  // Also various OpenCL version details.
  if (auto *openclVersion = module->getNamedMetadata("opencl.ocl.version"))
    module->eraseNamedMetadata(openclVersion);
}

LLVMModuleRef translateToLLVMIR(MlirModule module) {
  LLVMContextRef llvmContext = LLVMContextCreate();
  MlirOperation operation = mlirModuleGetOperation(module);
  auto mod = mlirTranslateModuleToLLVMIR(operation, llvmContext);
  return mod;
};