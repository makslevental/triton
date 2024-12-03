add_mlir_public_c_api_library(TritonCAPI
  Dialects.cpp
  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
  TritonAMDGPUIR
  TritonAnalysis
  TritonGPUTransforms
)
