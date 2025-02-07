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

add_triton_library(TritonCAPI
  Dialects.cpp
  LINK_LIBS PUBLIC
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

  DEPENDS
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

target_compile_definitions(TritonCAPI
  PRIVATE
  -DMLIR_CAPI_BUILDING_LIBRARY=1
)
