include(AddMLIRPython)

add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=triton_mlir.")


################################################################################
# Sources
################################################################################

declare_mlir_python_sources(TritonPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT TritonPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/triton_mlir"
  TD_FILE dialects/TritonOps.td
  SOURCES
    dialects/tt.py
  DIALECT_NAME tt)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT TritonPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/triton_mlir"
  TD_FILE dialects/TritonGPUOps.td
  SOURCES
    dialects/triton_gpu.py
  DIALECT_NAME triton_gpu)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT TritonPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/triton_mlir"
  TD_FILE dialects/TritonAMDGPUOps.td
  SOURCES
    dialects/amdgpu.py
  DIALECT_NAME amdgpu)

declare_mlir_python_extension(TritonPythonSources.Extension
  MODULE_NAME _site_initialize_0
  ADD_TO_PARENT TritonPythonSources
  SOURCES
    TritonExtension.cpp
  EMBED_CAPI_LINK_LIBS
    TritonCAPI
)

################################################################################
# Common CAPI
################################################################################

add_mlir_python_common_capi_library(TritonPythonCAPI
  INSTALL_COMPONENT TritonPythonModules
  INSTALL_DESTINATION python_packages/triton_mlir/_mlir_libs
  OUTPUT_DIRECTORY "${TRITON_BINARY_DIR}/python_packages/triton_mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../.."
  DECLARED_SOURCES
    TritonPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
)

################################################################################
# Instantiation of all Python modules
################################################################################

add_mlir_python_modules(TritonPythonModules
  ROOT_PREFIX "${TRITON_BINARY_DIR}/python_packages/triton_mlir"
  INSTALL_PREFIX "python_packages/triton_mlir"
  DECLARED_SOURCES
    TritonPythonSources
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
    MLIRPythonSources.Core.Python
    MLIRPythonSources.Core.Python.Extras
    MLIRPythonSources.Dialects.affine
    # MLIRPythonSources.Dialects.amdgpu
    MLIRPythonSources.Dialects.arith
    MLIRPythonSources.Dialects.async
    MLIRPythonSources.Dialects.bufferization
    MLIRPythonSources.Dialects.bufferization_transform
    MLIRPythonSources.Dialects.builtin
    MLIRPythonSources.Dialects.cf
    MLIRPythonSources.Dialects.complex
    MLIRPythonSources.Dialects.func
    MLIRPythonSources.Dialects.gpu
    MLIRPythonSources.Dialects.gpu_transform
    MLIRPythonSources.Dialects.index
    MLIRPythonSources.Dialects.linalg
    MLIRPythonSources.Dialects.llvm
    MLIRPythonSources.Dialects.loop_transform
    MLIRPythonSources.Dialects.math
    MLIRPythonSources.Dialects.memref
    MLIRPythonSources.Dialects.memref_transform
    MLIRPythonSources.Dialects.ml_program
    # MLIRPythonSources.Dialects.nvgpu
    # MLIRPythonSources.Dialects.nvgpu_transform
    MLIRPythonSources.Dialects.nvvm
    MLIRPythonSources.Dialects.omp
    MLIRPythonSources.Dialects.pdl
    MLIRPythonSources.Dialects.quant
    MLIRPythonSources.Dialects.rocdl
    MLIRPythonSources.Dialects.scf
    MLIRPythonSources.Dialects.shape
    MLIRPythonSources.Dialects.sparse_tensor
    MLIRPythonSources.Dialects.sparse_tensor_transform
    MLIRPythonSources.Dialects.spirv
    MLIRPythonSources.Dialects.structured_transform
    MLIRPythonSources.Dialects.tensor
    MLIRPythonSources.Dialects.tensor_transform
    MLIRPythonSources.Dialects.tosa
    MLIRPythonSources.Dialects.transform
    MLIRPythonSources.Dialects.transform.extras
    MLIRPythonSources.Dialects.transform.interpreter
    MLIRPythonSources.Dialects.transform_pdl_extension
    MLIRPythonSources.Dialects.vector
    MLIRPythonSources.Dialects.vector_transform
    MLIRPythonSources.ExecutionEngine
  COMMON_CAPI_LINK_LIBS
    TritonPythonCAPI
  )
