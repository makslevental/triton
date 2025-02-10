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
#include "amd/include/TritonAMDGPUTransforms/TritonGPUConversion.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/MLIRContext.h"
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