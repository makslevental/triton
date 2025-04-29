set(MLIR_PYTHON_PACKAGE_PREFIX "triton_mlir")
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=${MLIR_PYTHON_PACKAGE_PREFIX}.")

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

if(DEFINED MLIR_INSTALL_PREFIX)
   find_package(LLD REQUIRED CONFIG PATHS ${MLIR_DIR})
   include_directories(${LLD_INCLUDE_DIRS})
endif()

set(CMAKE_C_VISIBILITY_PRESET "hidden")
set(CMAKE_CXX_VISIBILITY_PRESET "hidden")
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
set(MLIR_BINDINGS_PYTHON_NB_DOMAIN "triton_mlir" CACHE STRING "" FORCE)

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

install(
  FILES
    "triton_mlir/dialects/amdgpu.py"
    "triton_mlir/dialects/nvgpu.py"
    "triton_mlir/dialects/proton.py"
    "triton_mlir/dialects/tt.py"
    "triton_mlir/dialects/ttg.py"
    "triton_mlir/dialects/ttng.py"
    "triton_mlir/dialects/ttpp.py"
  DESTINATION "triton_mlir/dialects"
)

configure_file(
  "${CMAKE_CURRENT_LIST_DIR}/triton_mlir/__init__.py.in"
  "${CMAKE_CURRENT_LIST_DIR}/triton_mlir/__init__.py"
  @ONLY)

add_custom_command(OUTPUT "${CMAKE_CURRENT_LIST_DIR}/triton_mlir/__init__.py"
  COMMAND ${Python3_EXECUTABLE} -c "
    f = open('triton_mlir/__init__.py.in', 'r').read();
    f = f.replace('@PYTHON_MODULE_EXTENSION@', '${PYTHON_MODULE_EXTENSION}');
    open('triton_mlir/__init__.py', 'w').write(f)"
  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
  COMMENT "generating ${CMAKE_CURRENT_LIST_DIR}/triton_mlir/__init__.py"
  VERBATIM
)
add_custom_target (
  __init__py DEPENDS
  "${CMAKE_CURRENT_LIST_DIR}/triton_mlir/__init__.py"
)
install(
  FILES "triton_mlir/__init__.py"
  DESTINATION "triton_mlir"
)

get_property(_all_libs GLOBAL PROPERTY MLIR_ALL_LIBS)
get_property(_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(_translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
get_property(_conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(_extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(_triton_libs GLOBAL PROPERTY TRITON_LIBS)
get_property(_mlir_upstream_capi_libs GLOBAL PROPERTY MLIR_UPSTREAM_CAPI_LIBS)

set(_libs
  TritonIR
  TritonGPUIR
  TritonAMDGPUIR
  TritonAnalysis
  TritonGPUTransforms
  TritonAMDGPUToLLVM
  TritonAMDUtils
  TritonAMDGPUTransforms
  TritonAMDGPUDialectToLLVM
  TritonNVIDIAGPUToLLVM
  TritonAMDAnalysis
#  TritonTestAnalysis
#  NVGPUIR
#  NVGPUToLLVM

  ${_all_libs}
  ${_dialect_libs}
  ${_translation_libs}
  ${_conversion_libs}
  ${_extension_libs}
  ${_triton_libs}
  ${_mlir_upstream_capi_libs}

  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation

  # LLVM
  ${LLVM_AVAILABLE_LIBS}
  LLVMCodeGen
  LLVMAnalysis
  LLVMCore
  LLVMSupport
  LLVMOption
  LLVMCodeGen
  LLVMSupport
  LLVMTargetParser
  LLVMCore
  LLVMPasses
  LLVMMC
  LLVMMCParser
  LLVMMCDisassembler
  lldCommon
  lldELF

  # AMDGPU
  LLVMAMDGPUCodeGen
  LLVMAMDGPUAsmParser
  LLVMAMDGPUDisassembler
  LLVMAMDGPUTargetMCA
  LLVMAMDGPUDesc
  LLVMAMDGPUInfo
  LLVMAMDGPUUtils
  # NVPTX
  LLVMNVPTXCodeGen
  LLVMNVPTXDesc
  LLVMNVPTXInfo
)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  list(APPEND _libs
    LLVMX86Info
    LLVMX86CodeGen
    LLVMX86AsmParser
    LLVMX86Disassembler)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
  list(APPEND _libs
    LLVMAArch64Info
    LLVMAArch64Desc
    LLVMAArch64CodeGen
    LLVMAArch64AsmParser
    LLVMAArch64Disassembler)
endif()

#list(REMOVE_DUPLICATES _libs)
list(REMOVE_ITEM _libs
  mlir_c_runner_utils
  mlir_runner_utils
  MLIRSparseTensorRuntime
  mlir_float16_utils
  mlir_async_runtime
  mlir_arm_sme_abi_stubs
  mlir_arm_runner_utils
  mlir_spirv_cpu_runtime
)

# Dig out all of the upstream cpp source files associated with the C extension modules.
set(_python_dialect_sources
  async
  gpu
  linalg
  llvm
  pdl
  quant
  sparse_tensor
  transform
)
list(TRANSFORM _python_dialect_sources PREPEND "MLIRPythonSources.Dialects.")
_flatten_mlir_python_targets(mlir_python_sources MLIRPythonSources ${_python_dialect_sources})
set(_mlir_python_cpp_sources)
foreach(s ${mlir_python_sources})
  get_target_property(_sources ${s} INTERFACE_SOURCES)
  if(NOT DEFINED MLIR_INSTALL_PREFIX)
    string(REGEX MATCH "<BUILD_INTERFACE:(.+)>;" _sources "${_sources}")
    set(_sources ${CMAKE_MATCH_1})
  endif()
  list(FILTER _sources INCLUDE REGEX "\.cpp$")
  if(_sources)
    list(APPEND _mlir_python_cpp_sources ${_sources})
  endif()
endforeach()

set(tblgen_deps
  TritonTableGen
  TritonTransformsIncGen
  TritonCombineIncGen
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen
  TritonGPUTypeInterfacesIncGen
  TritonGPUTransformsIncGen
  TritonAMDGPUTransformsIncGen
  TritonNvidiaGPUTableGen
  TritonNvidiaGPUAttrDefsIncGen
  TritonNvidiaGPUTransformsIncGen
  TritonNvidiaGPUOpInterfacesIncGen
)

find_package(eudsl_tblgen CONFIG)
if(NOT eudsl_tblgen_FOUND)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m eudsl_tblgen --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE eudsl_tblgen_DIR
    # COMMAND_ECHO STDERR
  )
endif()
find_package(eudsl_tblgen CONFIG REQUIRED)
message(STATUS "Using eudsl_tblgen @ ${eudsl_tblgen_DIR}")

eudsl_tblgen(eudsl_TritonGPUAttrDefs
  TD_FILE
    "../../include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td"
  OUTPUT_DIRECTORY
    "${CMAKE_CURRENT_SOURCE_DIR}/triton_mlir"
  KIND
    MlirAttribute
  EXCLUDE
    BlockedEncodingAttr
    SliceEncodingAttr
    SwizzledSharedEncodingAttr
)

eudsl_tblgen(eudsl_TritonAMDGPUGPUAttrDefs
  TD_FILE
    "../../third_party/amd/include/Dialect/TritonAMDGPU/IR/TritonAMDGPUAttrDefs.td"
  OUTPUT_DIRECTORY
    "${CMAKE_CURRENT_SOURCE_DIR}/triton_mlir"
  KIND
    MlirAttribute
  EXCLUDE
    TritonAMDGPU_SchedHintVariantAttr
    TritonAMDGPU_InstCounter
)

eudsl_tblgen(eudsl_TritonGPUTypes
  TD_FILE
    "../../include/triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"
  OUTPUT_DIRECTORY
    "${CMAKE_CURRENT_SOURCE_DIR}/triton_mlir"
  KIND
    MlirType
)

declare_mlir_python_extension(TritonPythonSources.Extension
  PYTHON_BINDINGS_LIBRARY nanobind
  ROOT_DIR "/"
  MODULE_NAME _triton
  ADD_TO_PARENT TritonPythonSources
  OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/triton_mlir/_mlir_libs"
  SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/triton_mlir/TritonExtension.cpp
    "${_mlir_python_cpp_sources}"
  PRIVATE_LINK_LIBS
    ${_libs}
)

set(LLD_INCLUDE_DIR "${LLVM_EXTERNAL_LLD_SOURCE_DIR}/include")
target_include_directories(
  TritonPythonSources.Extension
  INTERFACE $<BUILD_INTERFACE:${LLD_INCLUDE_DIR}>
)
if(DEFINED MLIR_INSTALL_PREFIX)
  set(PYBINDINGS_SRC "${MLIR_INSTALL_PREFIX}/src/python/MLIRPythonExtension.Core")
  target_include_directories(
    TritonPythonSources.Extension
    INTERFACE $<BUILD_INTERFACE:${PYBINDINGS_SRC}>
  )
endif()

# Kill all of the dependencies (mlir_python_DEPENDS) on cpp sources in the dialect build targets
# so the C extension modules don't get built but the Python sources do get installed.
_flatten_mlir_python_targets(mlir_python_sources_deps MLIRPythonSources)
list(FILTER mlir_python_sources_deps INCLUDE REGEX "^MLIRPythonSources.Dialects")
list(FILTER mlir_python_sources_deps EXCLUDE REGEX "^MLIRPythonSources.Dialects.amdgpu")
list(FILTER mlir_python_sources_deps EXCLUDE REGEX "^MLIRPythonSources.Dialects.nvgpu")
foreach(t ${mlir_python_sources_deps})
  set_property(TARGET ${t} PROPERTY mlir_python_DEPENDS "")
endforeach()

add_mlir_python_modules(TritonPythonModules
  ROOT_PREFIX "${CMAKE_CURRENT_LIST_DIR}/triton_mlir"
  INSTALL_PREFIX "triton_mlir"
  DECLARED_SOURCES
    ${mlir_python_sources_deps}
    MLIRPythonSources.Core.Python
    TritonPythonSources
)

set(NB_STUBGEN_CMD "${Python_EXECUTABLE}" "-m" "nanobind.stubgen"
        --module "${MLIR_PYTHON_PACKAGE_PREFIX}._mlir_libs._triton"
        --recursive --include-private --output-dir triton_mlir/_mlir_libs/_triton)
set(NB_STUBGEN_OUTPUTS "_triton.pyi")
# note WORKING_DIRECTORY
add_custom_command(
        OUTPUT ${NB_STUBGEN_OUTPUTS}
        COMMAND ${NB_STUBGEN_CMD}
        WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
        DEPENDS TritonPythonModules.extension._triton.dso
)
add_custom_target(triton_ext_stub DEPENDS ${NB_STUBGEN_OUTPUTS})
install(
  DIRECTORY "triton_mlir/_mlir_libs/_triton"
  DESTINATION "triton_mlir/_mlir_libs"
  PATTERN "*.pyi"
)
install(
  DIRECTORY "triton_mlir/lib"
  DESTINATION "triton_mlir"
  PATTERN "*.pyi"
)

add_dependencies(TritonPythonModules
  ${tblgen_deps}
  __init__py
  triton_ext_stub
  eudsl_TritonGPUAttrDefs
  eudsl_TritonGPUTypes
)

set(_uninstalled
  NVGPUIR
  NVGPUToLLVM
  ProtonIR
  TritonAMDGPUDialectToLLVM
  TritonAMDGPUIR
  TritonAMDGPUToLLVM
  TritonAMDGPUTransforms
  TritonAMDUtils
  TritonAnalysis
  TritonGPUIR
  TritonGPUToLLVM
  TritonGPUTransforms
  TritonIR
  TritonLLVMIR
  TritonNVIDIAGPUToLLVM
  TritonNvidiaGPUIR
  TritonNvidiaGPUTransforms
  TritonProtonToLLVM
  TritonToTritonGPU
  TritonTools
  TritonTransforms
  f2reduce
)
install(TARGETS ${_uninstalled} EXPORT MLIRTargets)