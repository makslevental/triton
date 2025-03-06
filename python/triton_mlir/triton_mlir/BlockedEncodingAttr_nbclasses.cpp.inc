auto nbBlockedEncodingAttr = mlir_attribute_subclass(
    m, "BlockedEncodingAttr", isaMlirBlockedEncodingAttr,
    mlirBlockedEncodingAttrGetTypeID);

nbBlockedEncodingAttr.def_staticmethod(
    "get",
    [](std::vector<unsigned> &sizePerThread,
       std::vector<unsigned> &threadsPerWarp__,
       std::vector<unsigned> &warpsPerCTA__, std::vector<unsigned> &order,
       std::optional<MlirAttribute> CTALayout, MlirContext context) {
      if (!CTALayout) {
        auto rank = sizePerThread.size();
        CTALayout = mlirCTALayoutAttrAttributeGetDefault(rank, context);
      }
      return mlirBlockedEncodingAttrAttributeGet(
          sizePerThread.data(), sizePerThread.size(), threadsPerWarp__.data(),
          threadsPerWarp__.size(), warpsPerCTA__.data(), warpsPerCTA__.size(),
          order.data(), order.size(),
          CTALayout.has_value() ? *CTALayout : wrap(CTALayoutAttr()), context);
    },
    "size_per_thread"_a, "threads_per_warp__"_a, "warps_per_cta__"_a, "order"_a,
    "cta_layout"_a = nb::none(), "context"_a = nb::none());

nbBlockedEncodingAttr.def_property_readonly(
    "size_per_thread", [](MlirAttribute self) {
      unsigned nsizePerThreads;
      unsigned *sizePerThreadPtr;
      mlirBlockedEncodingAttrGetsizePerThread(self, &sizePerThreadPtr,
                                              &nsizePerThreads);
      return std::vector<unsigned>{sizePerThreadPtr,
                                   sizePerThreadPtr + nsizePerThreads};
    });

nbBlockedEncodingAttr.def_property_readonly(
    "threads_per_warp__", [](MlirAttribute self) {
      unsigned nthreadsPerWarp__s;
      unsigned *threadsPerWarp__Ptr;
      mlirBlockedEncodingAttrGetthreadsPerWarp__(self, &threadsPerWarp__Ptr,
                                                 &nthreadsPerWarp__s);
      return std::vector<unsigned>{threadsPerWarp__Ptr,
                                   threadsPerWarp__Ptr + nthreadsPerWarp__s};
    });

nbBlockedEncodingAttr.def_property_readonly(
    "warps_per_cta__", [](MlirAttribute self) {
      unsigned nwarpsPerCTA__s;
      unsigned *warpsPerCTA__Ptr;
      mlirBlockedEncodingAttrGetwarpsPerCTA__(self, &warpsPerCTA__Ptr,
                                              &nwarpsPerCTA__s);
      return std::vector<unsigned>{warpsPerCTA__Ptr,
                                   warpsPerCTA__Ptr + nwarpsPerCTA__s};
    });

nbBlockedEncodingAttr.def_property_readonly("order", [](MlirAttribute self) {
  unsigned norders;
  unsigned *orderPtr;
  mlirBlockedEncodingAttrGetorder(self, &orderPtr, &norders);
  return std::vector<unsigned>{orderPtr, orderPtr + norders};
});

nbBlockedEncodingAttr.def_property_readonly(
    "cta_layout", [](MlirAttribute self) {
      return mlirBlockedEncodingAttrGetCTALayout(self);
    });
