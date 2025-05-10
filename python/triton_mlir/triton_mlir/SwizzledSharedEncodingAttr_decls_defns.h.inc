MLIR_CAPI_EXPORTED MlirAttribute mlirSwizzledSharedEncodingAttrAttributeGet(
    unsigned vec, unsigned perPhase, unsigned maxPhase, unsigned *order,
    unsigned norders, MlirAttribute CTALayout, MlirContext mlirContext);
MLIR_CAPI_EXPORTED unsigned mlirSwizzledSharedEncodingAttrGetvec(
    MlirAttribute mlirSwizzledSharedEncodingAttr);
MLIR_CAPI_EXPORTED unsigned mlirSwizzledSharedEncodingAttrGetperPhase(
    MlirAttribute mlirSwizzledSharedEncodingAttr);
MLIR_CAPI_EXPORTED unsigned mlirSwizzledSharedEncodingAttrGetmaxPhase(
    MlirAttribute mlirSwizzledSharedEncodingAttr);
MLIR_CAPI_EXPORTED void mlirSwizzledSharedEncodingAttrGetorder(
    MlirAttribute mlirSwizzledSharedEncodingAttr, unsigned **orderPtr,
    unsigned *norders);
MLIR_CAPI_EXPORTED MlirAttribute mlirSwizzledSharedEncodingAttrGetCTALayout(
    MlirAttribute mlirSwizzledSharedEncodingAttr);
MLIR_CAPI_EXPORTED MlirTypeID mlirSwizzledSharedEncodingAttrGetTypeID(void);
MLIR_CAPI_EXPORTED bool isaMlirSwizzledSharedEncodingAttr(MlirAttribute thing);

MlirAttribute mlirSwizzledSharedEncodingAttrAttributeGet(
    unsigned vec, unsigned perPhase, unsigned maxPhase, unsigned *order,
    unsigned norders, MlirAttribute CTALayout, MlirContext mlirContext) {
  mlir::MLIRContext *context = unwrap(mlirContext);
  unsigned vec_ = vec;
  unsigned perPhase_ = perPhase;
  unsigned maxPhase_ = maxPhase;
  ::llvm::ArrayRef<unsigned> order_ = {order, norders};
  CTALayoutAttr CTALayout_ = llvm::cast<CTALayoutAttr>(unwrap(CTALayout));
  return wrap(SwizzledSharedEncodingAttr::get(context, vec_, perPhase_,
                                              maxPhase_, order_, CTALayout_));
}

unsigned mlirSwizzledSharedEncodingAttrGetvec(
    MlirAttribute mlirSwizzledSharedEncodingAttr) {
  return llvm::cast<SwizzledSharedEncodingAttr>(
             unwrap(mlirSwizzledSharedEncodingAttr))
      .getVec();
}

unsigned mlirSwizzledSharedEncodingAttrGetperPhase(
    MlirAttribute mlirSwizzledSharedEncodingAttr) {
  return llvm::cast<SwizzledSharedEncodingAttr>(
             unwrap(mlirSwizzledSharedEncodingAttr))
      .getPerPhase();
}

unsigned mlirSwizzledSharedEncodingAttrGetmaxPhase(
    MlirAttribute mlirSwizzledSharedEncodingAttr) {
  return llvm::cast<SwizzledSharedEncodingAttr>(
             unwrap(mlirSwizzledSharedEncodingAttr))
      .getMaxPhase();
}

void mlirSwizzledSharedEncodingAttrGetorder(
    MlirAttribute mlirSwizzledSharedEncodingAttr, unsigned **orderPtr,
    unsigned *norders) {
  ::llvm::ArrayRef<unsigned> order = llvm::cast<SwizzledSharedEncodingAttr>(
                                         unwrap(mlirSwizzledSharedEncodingAttr))
                                         .getOrder();
  *norders = order.size();
  *orderPtr = const_cast<unsigned *>(order.data());
}

MlirAttribute mlirSwizzledSharedEncodingAttrGetCTALayout(
    MlirAttribute mlirSwizzledSharedEncodingAttr) {
  return wrap(llvm::cast<SwizzledSharedEncodingAttr>(
                  unwrap(mlirSwizzledSharedEncodingAttr))
                  .getCTALayout());
}

MlirTypeID mlirSwizzledSharedEncodingAttrGetTypeID() {
  return wrap(SwizzledSharedEncodingAttr::getTypeID());
}

bool isaMlirSwizzledSharedEncodingAttr(MlirAttribute thing) {
  return isa<SwizzledSharedEncodingAttr>(unwrap(thing));
}
