//===- Dialects.h - CAPI for dialects -----------------------------*- C -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_C_DIALECTS_H
#define TRITON_C_DIALECTS_H

#include "mlir-c/IR.h"
#include "llvm-c/TargetMachine.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED void
tritonMlirRegisterTritonDialectsAndPasses(MlirDialectRegistry registry);

MLIR_CAPI_EXPORTED MlirTypeID tritonMlirPointerTypeGetTypeID();

MLIR_CAPI_EXPORTED bool tritonMlirTypeIsAPointerType(MlirType type);

MLIR_CAPI_EXPORTED bool tritonMlirTypeIsATensorOfPointer(MlirType type);

MLIR_CAPI_EXPORTED MlirType
tritonMlirPointerTypeOfPointeeType(MlirType type, int addressSpace);

MLIR_CAPI_EXPORTED MlirType tritonMlirPointerTypeGetPointeeType(MlirType type);

MLIR_CAPI_EXPORTED void addModuleFlag(LLVMModuleRef llvmModRef,
                                      MlirStringRef Key, uint32_t Val);

MLIR_CAPI_EXPORTED bool hasMatrixCoreFeature(MlirStringRef arch);

struct IsaVersion {
  unsigned Major;
  unsigned Minor;
  unsigned Stepping;
};

MLIR_CAPI_EXPORTED IsaVersion getIsaVersion(MlirStringRef arch);

MLIR_CAPI_EXPORTED unsigned getCallingConvAMDGPUKernel();

enum LLVMOptimizationLevel { O0, O1, O2, O3, Os, Oz };

MLIR_CAPI_EXPORTED void initAllLLVMTargets();

MLIR_CAPI_EXPORTED LLVMTargetMachineRef createTargetMachine(
    MlirStringRef targetTripleRef, MlirStringRef proc, bool enable_fp_fusion,
    MlirStringRef features, bool disableLLVMOpt);

MLIR_CAPI_EXPORTED LLVMModuleRef optimizeModule(
    LLVMModuleRef llvmModRef, MlirStringRef flagList, MlirStringRef arch,
    MlirStringRef features, LLVMOptimizationLevel optLevel, bool enableFPFusion,
    bool llvmIREnableDump, bool enableAddressSanitizer, bool disableLLVMOpt);

MLIR_CAPI_EXPORTED void cleanupBitcodeMetadata(LLVMModuleRef llvmModRef);

MLIR_CAPI_EXPORTED LLVMModuleRef translateToLLVMIR(MlirModule module);

#ifdef __cplusplus
}
#endif

#endif // TRITON_C_DIALECTS_H
