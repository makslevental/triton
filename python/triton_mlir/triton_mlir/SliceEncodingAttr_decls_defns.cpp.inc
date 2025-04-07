MLIR_CAPI_EXPORTED MlirAttribute mlirSliceEncodingAttrAttributeGet(
    unsigned dim, MlirAttribute parent, MlirContext mlirContext);
MLIR_CAPI_EXPORTED unsigned
mlirSliceEncodingAttrGetdim(MlirAttribute mlirSliceEncodingAttr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirSliceEncodingAttrGetparent(MlirAttribute mlirSliceEncodingAttr);
MLIR_CAPI_EXPORTED MlirTypeID mlirSliceEncodingAttrGetTypeID(void);
MLIR_CAPI_EXPORTED bool isaMlirSliceEncodingAttr(MlirAttribute thing);

MlirAttribute mlirSliceEncodingAttrAttributeGet(unsigned dim,
                                                MlirAttribute parent,
                                                MlirContext mlirContext) {
  mlir::MLIRContext *context = unwrap(mlirContext);
  unsigned dim_ = dim;
  DistributedEncodingTrait parent_ =
      llvm::cast<DistributedEncodingTrait>(unwrap(parent));
  return wrap(SliceEncodingAttr::get(context, dim_, parent_));
}

unsigned mlirSliceEncodingAttrGetdim(MlirAttribute mlirSliceEncodingAttr) {
  return llvm::cast<SliceEncodingAttr>(unwrap(mlirSliceEncodingAttr)).getDim();
}

MlirAttribute
mlirSliceEncodingAttrGetparent(MlirAttribute mlirSliceEncodingAttr) {
  return wrap(
      llvm::cast<SliceEncodingAttr>(unwrap(mlirSliceEncodingAttr)).getParent());
}

MlirTypeID mlirSliceEncodingAttrGetTypeID() {
  return wrap(SliceEncodingAttr::getTypeID());
}

bool isaMlirSliceEncodingAttr(MlirAttribute thing) {
  return isa<SliceEncodingAttr>(unwrap(thing));
}
