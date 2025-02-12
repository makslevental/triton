add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=triton_mlir.")

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)

include_directories(${TRITON_SOURCE_DIR}/third_party)
include_directories(${TRITON_BINARY_DIR}/third_party)
include_directories(${TRITON_SOURCE_DIR}/third_party/amd/include)
include_directories(${TRITON_BINARY_DIR}/third_party/amd/include)
include_directories(${TRITON_SOURCE_DIR}/third_party/nvidia/include)
include_directories(${TRITON_BINARY_DIR}/third_party/nvidia/include)
include_directories(${TRITON_SOURCE_DIR}/third_party/proton/include)
include_directories(${TRITON_BINARY_DIR}/third_party/proton/include)

find_package(Python 3.9...<3.14 REQUIRED
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

set(libs
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
  TritonTestAnalysis
  TritonTestDialectTritonGPU
  NVGPUIR
  NVGPUToLLVM

  ${dialect_libs}
  ${translation_libs}
  ${conversion_libs}
  ${extension_libs}
  ${triton_libs}

  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation

  # MLIR C API
  MLIRCAPIIR
  MLIRCAPILLVM

  # LLVM
  LLVMTargetParser
  LLVMCore
  LLVMPasses
  LLVMMC
  LLVMMCParser
  LLVMMCDisassembler

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
  list(APPEND libs
    LLVMX86Info
    LLVMX86CodeGen
    LLVMX86AsmParser
    LLVMX86Disassembler)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
  list(APPEND libs
    LLVMAArch64Info
    LLVMAArch64Desc
    LLVMAArch64CodeGen
    LLVMAArch64AsmParser
    LLVMAArch64Disassembler)
endif()

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --include_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_include_dir)

add_library(TritonCAPI STATIC Dialects.cpp)
set(LLVM_REQUIRES_RTTI YES)
set(LLVM_REQUIRES_EH YES)
llvm_update_compile_flags(TritonCAPI)
target_link_libraries(TritonCAPI PUBLIC ${libs})
target_include_directories(TritonCAPI PRIVATE ${nanobind_include_dir} ${Python_INCLUDE_DIRS})
target_link_options(TritonCAPI PRIVATE
  $<$<PLATFORM_ID:Linux>:-fexceptions -frtti>
  $<$<PLATFORM_ID:Darwin>:-fexceptions -frtti>
  $<$<PLATFORM_ID:Windows>:/EHsc /GR>)
target_compile_definitions(TritonCAPI
  PRIVATE
  -DMLIR_CAPI_BUILDING_LIBRARY=1
)
add_dependencies(TritonCAPI
  TritonTableGen
  TritonTransformsIncGen
  TritonCombineIncGen
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen
  TritonGPUTypeInterfacesIncGen
  TritonGPUTransformsIncGen
  TritonNvidiaGPUTableGen
  TritonNvidiaGPUAttrDefsIncGen
  TritonNvidiaGPUTransformsIncGen
)
