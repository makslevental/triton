MLIR_CAPI_EXPORTED MlirAttribute mlirBlockedEncodingAttrAttributeGet(
    unsigned *sizePerThread, unsigned nsizePerThreads, unsigned *threadsPerWarp,
    unsigned nthreadsPerWarps, unsigned *warpsPerCTA, unsigned nwarpsPerCTAs,
    unsigned *order, unsigned norders, MlirAttribute CTALayout,
    MlirContext mlirContext);
MLIR_CAPI_EXPORTED void
mlirBlockedEncodingAttrGetsizePerThread(MlirAttribute mlirBlockedEncodingAttr,
                                        unsigned **sizePerThreadPtr,
                                        unsigned *nsizePerThreads);
MLIR_CAPI_EXPORTED void
mlirBlockedEncodingAttrGetthreadsPerWarp(MlirAttribute mlirBlockedEncodingAttr,
                                         unsigned **threadsPerWarpPtr,
                                         unsigned *nthreadsPerWarps);
MLIR_CAPI_EXPORTED void
mlirBlockedEncodingAttrGetwarpsPerCTA(MlirAttribute mlirBlockedEncodingAttr,
                                      unsigned **warpsPerCTAPtr,
                                      unsigned *nwarpsPerCTAs);
MLIR_CAPI_EXPORTED void
mlirBlockedEncodingAttrGetorder(MlirAttribute mlirBlockedEncodingAttr,
                                unsigned **orderPtr, unsigned *norders);
MLIR_CAPI_EXPORTED MlirAttribute
mlirBlockedEncodingAttrGetCTALayout(MlirAttribute mlirBlockedEncodingAttr);
MLIR_CAPI_EXPORTED MlirTypeID mlirBlockedEncodingAttrGetTypeID(void);
MLIR_CAPI_EXPORTED bool isaMlirBlockedEncodingAttr(MlirAttribute thing);

MlirAttribute mlirBlockedEncodingAttrAttributeGet(
    unsigned *sizePerThread, unsigned nsizePerThreads, unsigned *threadsPerWarp,
    unsigned nthreadsPerWarps, unsigned *warpsPerCTA, unsigned nwarpsPerCTAs,
    unsigned *order, unsigned norders, MlirAttribute CTALayout,
    MlirContext mlirContext) {
  mlir::MLIRContext *context = unwrap(mlirContext);
  ::llvm::ArrayRef<unsigned> sizePerThread_ = {sizePerThread, nsizePerThreads};
  ::llvm::ArrayRef<unsigned> threadsPerWarp_ = {threadsPerWarp,
                                                nthreadsPerWarps};
  ::llvm::ArrayRef<unsigned> warpsPerCTA_ = {warpsPerCTA, nwarpsPerCTAs};
  ::llvm::ArrayRef<unsigned> order_ = {order, norders};
  CTALayoutAttr CTALayout_ = llvm::cast<CTALayoutAttr>(unwrap(CTALayout));
  return wrap(BlockedEncodingAttr::get(context, sizePerThread_, threadsPerWarp_,
                                       warpsPerCTA_, order_, CTALayout_));
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

void mlirBlockedEncodingAttrGetthreadsPerWarp(
    MlirAttribute mlirBlockedEncodingAttr, unsigned **threadsPerWarpPtr,
    unsigned *nthreadsPerWarps) {
  ::llvm::ArrayRef<unsigned> threadsPerWarp =
      llvm::cast<BlockedEncodingAttr>(unwrap(mlirBlockedEncodingAttr))
          .getThreadsPerWarp();
  *nthreadsPerWarps = threadsPerWarp.size();
  *threadsPerWarpPtr = const_cast<unsigned *>(threadsPerWarp.data());
}

void mlirBlockedEncodingAttrGetwarpsPerCTA(
    MlirAttribute mlirBlockedEncodingAttr, unsigned **warpsPerCTAPtr,
    unsigned *nwarpsPerCTAs) {
  ::llvm::ArrayRef<unsigned> warpsPerCTA =
      llvm::cast<BlockedEncodingAttr>(unwrap(mlirBlockedEncodingAttr))
          .getWarpsPerCTA();
  *nwarpsPerCTAs = warpsPerCTA.size();
  *warpsPerCTAPtr = const_cast<unsigned *>(warpsPerCTA.data());
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
