//===- Dialects.cpp - CAPI for dialects -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "triton-c/Dialects.h"

#include "mlir/CAPI/Registration.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Triton, triton,
                                      mlir::triton::TritonDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(TritonGPU, triton_gpu,
                                      mlir::triton::gpu::TritonGPUDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(TritonAMDGPU, triton_amd_gpu,
                                      mlir::triton::amdgpu::TritonAMDGPUDialect)
