add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=triton_mlir.")

include(AddMLIRPython)

include_directories(${TRITON_SOURCE_DIR}/third_party)
include_directories(${TRITON_BINARY_DIR}/third_party)
include_directories(${TRITON_SOURCE_DIR}/third_party/amd/include)
include_directories(${TRITON_BINARY_DIR}/third_party/amd/include)

include_directories(${TRITON_SOURCE_DIR}/third_party/nvidia/include)
include_directories(${TRITON_BINARY_DIR}/third_party/nvidia/include)
# this is for tablegen
include_directories(${TRITON_SOURCE_DIR}/third_party/nvidia/include/Dialect/NVGPU/IR)

include_directories(${TRITON_SOURCE_DIR}/third_party/proton/include)
include_directories(${TRITON_BINARY_DIR}/third_party/proton/include)
# this is for tablegen
include_directories(${TRITON_SOURCE_DIR}/third_party/proton/dialect/include/Dialect/Proton/IR)

set(CMAKE_C_VISIBILITY_PRESET "hidden")
set(CMAKE_CXX_VISIBILITY_PRESET "hidden")
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
set(MLIR_BINDINGS_PYTHON_NB_DOMAIN "triton_mlir;NB_SHARED" CACHE STRING "" FORCE)
set(MLIR_DISABLE_CONFIGURE_PYTHON_DEV_PACKAGES ON CACHE BOOL "" FORCE)

################################################################################
# Sources
################################################################################

declare_mlir_python_sources(TritonPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT TritonPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/triton_mlir"
  TD_FILE dialects/TritonOps.td
  GEN_ENUM_BINDINGS_TD_FILE
    "../../../include/triton/Dialect/Triton/IR/TritonAttrDefs.td"
  DIALECT_NAME tt)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT TritonPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/triton_mlir"
  TD_FILE dialects/TritonGPUOps.td
  GEN_ENUM_BINDINGS_TD_FILE
    "../../../include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td"
  DIALECT_NAME ttg)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT TritonPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/triton_mlir"
  TD_FILE dialects/TritonAMDGPUOps.td
  GEN_ENUM_BINDINGS_TD_FILE
    "../../../third_party/amd/include/Dialect/TritonAMDGPU/IR/TritonAMDGPUAttrDefs.td"
  DIALECT_NAME amdgpu)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT TritonPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/triton_mlir"
  TD_FILE dialects/TritonNVGPUOps.td
  GEN_ENUM_BINDINGS_TD_FILE
    "../../../third_party/nvidia/include/Dialect/NVGPU/IR/NVGPUAttrDefs.td"
  DIALECT_NAME nvgpu)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT TritonPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/triton_mlir"
  TD_FILE dialects/TritonNVIDIAGPUOps.td
  GEN_ENUM_BINDINGS_TD_FILE
    "../../../include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUAttrDefs.td"
  DIALECT_NAME ttng)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT TritonPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/triton_mlir"
  TD_FILE dialects/TritonProtonOps.td
  GEN_ENUM_BINDINGS_TD_FILE
    "../../../third_party/amd/include/Dialect/TritonAMDGPU/IR/TritonAMDGPUAttrDefs.td"
  DIALECT_NAME proton)

set(embed_capi_link_libs
  MLIRCAPIIR
  # MLIRCAPIExecutionEngine
  MLIRCAPIDebug
  MLIRCAPIInterfaces
  MLIRCAPITransforms
  MLIRCAPITarget
  # MLIRCAPIConversion
  MLIRCAPIArith
  MLIRCAPIAsync
  MLIRCAPIControlFlow
  MLIRCAPIEmitC
  MLIRCAPIMath
  MLIRCAPIMemRef
  MLIRCAPIGPU
  MLIRCAPIIRDL
  MLIRCAPILLVM
  MLIRCAPILinalg
  MLIRCAPIMLProgram
  MLIRCAPINVVM
  MLIRCAPIROCDL
  MLIRCAPISCF
  MLIRCAPIShape
  MLIRCAPISparseTensor
  MLIRCAPIFunc
  MLIRCAPISPIRV
  MLIRCAPITensor
  MLIRCAPITransformDialect
  MLIRCAPITransformDialectTransforms
  MLIRCAPIQuant
  MLIRCAPIOpenMP
  MLIRCAPIPDL
  MLIRCAPIVector
  TritonCAPI
)

declare_mlir_python_extension(TritonPythonSources.site_initialize_0
  PYTHON_BINDINGS_LIBRARY nanobind
  MODULE_NAME _site_initialize_0
  ADD_TO_PARENT TritonPythonSources
  OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/triton_mlir/_mlir_libs"
  SOURCES
    triton_mlir/site_initialize_0.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    ${embed_capi_link_libs}
)

declare_mlir_python_extension(TritonPythonSources.Extension
  PYTHON_BINDINGS_LIBRARY nanobind
  MODULE_NAME _TritonExtension
  ADD_TO_PARENT TritonPythonSources
  OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/triton_mlir/_mlir_libs"
  SOURCES
    triton_mlir/TritonExtension.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    ${embed_capi_link_libs}
)

################################################################################
# Common CAPI
################################################################################

add_mlir_python_common_capi_library(TritonPythonCAPI
  INSTALL_COMPONENT TritonPythonModules
  INSTALL_DESTINATION python_packages/triton_mlir/_mlir_libs
  OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/triton_mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../.."
  DECLARED_SOURCES
    TritonPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    # MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
)
target_link_libraries(TritonPythonCAPI PRIVATE TritonCAPI)
if(APPLE)
  target_link_options(TritonPythonCAPI PRIVATE "LINKER:-u,_populateTritonExtension")
elseif(UNIX)
  target_link_options(TritonPythonCAPI PRIVATE "LINKER:-u,populateTritonExtension")
endif()

################################################################################
# Instantiation of all Python modules
################################################################################

add_mlir_python_modules(TritonPythonModules
  ROOT_PREFIX "${CMAKE_CURRENT_LIST_DIR}/triton_mlir"
  INSTALL_PREFIX "python_packages/triton_mlir"
  DECLARED_SOURCES
    TritonPythonSources
    # MLIRPythonExtension.RegisterEverything
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
    # MLIRPythonSources.ExecutionEngine
  COMMON_CAPI_LINK_LIBS
    TritonPythonCAPI
)

target_link_libraries(TritonCAPI PUBLIC nanobind-triton_mlir Python::Module)
target_link_options(nanobind-triton_mlir PRIVATE
  $<$<PLATFORM_ID:Linux>:-fexceptions -frtti>
  $<$<PLATFORM_ID:Darwin>:-fexceptions -frtti>
  $<$<PLATFORM_ID:Windows>:/EHsc /GR>)

macro(remove_flag_from_target _target _flag)
  get_target_property(_target_cxx_flags ${_target} COMPILE_OPTIONS)
  if(_target_cxx_flags)
    list(REMOVE_ITEM _target_cxx_flags ${_flag})
    set_target_properties(${_target} PROPERTIES COMPILE_OPTIONS "${_target_cxx_flags}")
  endif()
  get_target_property(_target_cxx_flags ${_target} COMPILE_FLAGS)
  if(_target_cxx_flags)
    list(REMOVE_ITEM _target_cxx_flags ${_flag})
    set_target_properties(${_target} PROPERTIES COMPILE_FLAGS "${_target_cxx_flags}")
  endif()
endmacro()

set(tgts
  _mlir
  _mlirAsyncPasses
  _mlirDialectsGPU
  _mlirDialectsLinalg
  _mlirDialectsLLVM
  _mlirDialectsPDL
  _mlirDialectsQuant
  _mlirDialectsSparseTensor
  _mlirDialectsTransform
  # _mlirExecutionEngine
  _mlirGPUPasses
  _mlirLinalgPasses
  _mlirSparseTensorPasses
  _mlirTransformInterpreter
  _site_initialize_0
)

foreach(tgt ${tgts})
  remove_flag_from_target("TritonPythonModules.extension.${tgt}.dso" "-fPIC")
endforeach()
remove_flag_from_target(TritonCAPI "-fPIC")
