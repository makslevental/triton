MLIR_CAPI_EXPORTED MlirAttribute mlirBlockedEncodingAttrAttributeGet(
    unsigned *sizePerThread, unsigned nsizePerThreads,
    unsigned *threadsPerWarp__, unsigned nthreadsPerWarp__s,
    unsigned *warpsPerCTA__, unsigned nwarpsPerCTA__s, unsigned *order,
    unsigned norders, MlirAttribute CTALayout, MlirContext mlirContext);
MLIR_CAPI_EXPORTED void
mlirBlockedEncodingAttrGetsizePerThread(MlirAttribute mlirBlockedEncodingAttr,
                                        unsigned **sizePerThreadPtr,
                                        unsigned *nsizePerThreads);
MLIR_CAPI_EXPORTED void mlirBlockedEncodingAttrGetthreadsPerWarp__(
    MlirAttribute mlirBlockedEncodingAttr, unsigned **threadsPerWarp__Ptr,
    unsigned *nthreadsPerWarp__s);
MLIR_CAPI_EXPORTED void
mlirBlockedEncodingAttrGetwarpsPerCTA__(MlirAttribute mlirBlockedEncodingAttr,
                                        unsigned **warpsPerCTA__Ptr,
                                        unsigned *nwarpsPerCTA__s);
MLIR_CAPI_EXPORTED void
mlirBlockedEncodingAttrGetorder(MlirAttribute mlirBlockedEncodingAttr,
                                unsigned **orderPtr, unsigned *norders);
MLIR_CAPI_EXPORTED MlirAttribute
mlirBlockedEncodingAttrGetCTALayout(MlirAttribute mlirBlockedEncodingAttr);
MLIR_CAPI_EXPORTED MlirTypeID mlirBlockedEncodingAttrGetTypeID(void);
MLIR_CAPI_EXPORTED bool isaMlirBlockedEncodingAttr(MlirAttribute thing);

MlirAttribute mlirBlockedEncodingAttrAttributeGet(
    unsigned *sizePerThread, unsigned nsizePerThreads,
    unsigned *threadsPerWarp__, unsigned nthreadsPerWarp__s,
    unsigned *warpsPerCTA__, unsigned nwarpsPerCTA__s, unsigned *order,
    unsigned norders, MlirAttribute CTALayout, MlirContext mlirContext) {
  mlir::MLIRContext *context = unwrap(mlirContext);
  ::llvm::ArrayRef<unsigned> sizePerThread_ = {sizePerThread, nsizePerThreads};
  ::llvm::ArrayRef<unsigned> threadsPerWarp___ = {threadsPerWarp__,
                                                  nthreadsPerWarp__s};
  ::llvm::ArrayRef<unsigned> warpsPerCTA___ = {warpsPerCTA__, nwarpsPerCTA__s};
  ::llvm::ArrayRef<unsigned> order_ = {order, norders};
  CTALayoutAttr CTALayout_ = llvm::cast<CTALayoutAttr>(unwrap(CTALayout));
  return wrap(BlockedEncodingAttr::get(context, sizePerThread_,
                                       threadsPerWarp___, warpsPerCTA___,
                                       order_, CTALayout_));
}

void mlirBlockedEncodingAttrGetsizePerThread(
    MlirAttribute mlirBlockedEncodingAttr, unsigned **sizePerThreadPtr,
    unsigned *nsizePerThreads) {
  ::llvm::ArrayRef<unsigned> sizePerThread =
      llvm::cast<BlockedEncodingAttr>(unwrap(mlirBlockedEncodingAttr))
          .getSizePerThread();
  *nsizePerThreads = sizePerThread.size();
  *sizePerThreadPtr = const_cast<unsigned *>(sizePerThread.data());
}

void mlirBlockedEncodingAttrGetthreadsPerWarp__(
    MlirAttribute mlirBlockedEncodingAttr, unsigned **threadsPerWarp__Ptr,
    unsigned *nthreadsPerWarp__s) {
  ::llvm::ArrayRef<unsigned> threadsPerWarp__ =
      llvm::cast<BlockedEncodingAttr>(unwrap(mlirBlockedEncodingAttr))
          .getThreadsPerWarp__();
  *nthreadsPerWarp__s = threadsPerWarp__.size();
  *threadsPerWarp__Ptr = const_cast<unsigned *>(threadsPerWarp__.data());
}

void mlirBlockedEncodingAttrGetwarpsPerCTA__(
    MlirAttribute mlirBlockedEncodingAttr, unsigned **warpsPerCTA__Ptr,
    unsigned *nwarpsPerCTA__s) {
  ::llvm::ArrayRef<unsigned> warpsPerCTA__ =
      llvm::cast<BlockedEncodingAttr>(unwrap(mlirBlockedEncodingAttr))
          .getWarpsPerCTA__();
  *nwarpsPerCTA__s = warpsPerCTA__.size();
  *warpsPerCTA__Ptr = const_cast<unsigned *>(warpsPerCTA__.data());
}

void mlirBlockedEncodingAttrGetorder(MlirAttribute mlirBlockedEncodingAttr,
                                     unsigned **orderPtr, unsigned *norders) {
  ::llvm::ArrayRef<unsigned> order =
      llvm::cast<BlockedEncodingAttr>(unwrap(mlirBlockedEncodingAttr))
          .getOrder();
  *norders = order.size();
  *orderPtr = const_cast<unsigned *>(order.data());
}

MlirAttribute
mlirBlockedEncodingAttrGetCTALayout(MlirAttribute mlirBlockedEncodingAttr) {
  return wrap(llvm::cast<BlockedEncodingAttr>(unwrap(mlirBlockedEncodingAttr))
                  .getCTALayout());
}

MlirTypeID mlirBlockedEncodingAttrGetTypeID() {
  return wrap(BlockedEncodingAttr::getTypeID());
}

bool isaMlirBlockedEncodingAttr(MlirAttribute thing) {
  return isa<BlockedEncodingAttr>(unwrap(thing));
}

MlirAttribute mlirCTALayoutAttrAttributeGetDefault(unsigned rank,
                                                   MlirContext mlirContext) {
  return wrap(CTALayoutAttr::getDefault(unwrap(mlirContext), rank));
}
