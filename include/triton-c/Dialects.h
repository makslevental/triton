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

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Triton, triton);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(TritonGPU, triton_gpu);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(TritonAMDGPU, triton_amd_gpu);

#ifdef __cplusplus
}
#endif

#endif // TRITON_C_DIALECTS_H
