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

MLIR_CAPI_EXPORTED bool hasMatrixCoreFeature(MlirStringRef arch);

#ifdef __cplusplus
}
#endif

#endif // TRITON_C_DIALECTS_H
