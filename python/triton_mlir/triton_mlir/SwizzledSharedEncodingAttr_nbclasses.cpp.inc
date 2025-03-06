
auto nbSwizzledSharedEncodingAttr = mlir_attribute_subclass(
    m, "SwizzledSharedEncodingAttr", isaMlirSwizzledSharedEncodingAttr,
    mlirSwizzledSharedEncodingAttrGetTypeID);
nbSwizzledSharedEncodingAttr.def_staticmethod(
    "get",
    [](unsigned vec, unsigned perPhase, unsigned maxPhase,
       std::vector<unsigned> &order, std::optional<MlirAttribute> CTALayout,
       MlirContext context) {
      if (!CTALayout) {
        auto rank = order.size();
        CTALayout = mlirCTALayoutAttrAttributeGetDefault(rank, context);
      }
      return mlirSwizzledSharedEncodingAttrAttributeGet(
          vec, perPhase, maxPhase, order.data(), order.size(), *CTALayout,
          context);
    },
    "vec"_a, "per_phase"_a, "max_phase"_a, "order"_a,
    "cta_layout"_a = nb::none(), "context"_a = nb::none());

nbSwizzledSharedEncodingAttr.def_property_readonly(
    "vec", [](MlirAttribute self) {
      return mlirSwizzledSharedEncodingAttrGetvec(self);
    });

nbSwizzledSharedEncodingAttr.def_property_readonly(
    "per_phase", [](MlirAttribute self) {
      return mlirSwizzledSharedEncodingAttrGetperPhase(self);
    });

nbSwizzledSharedEncodingAttr.def_property_readonly(
    "max_phase", [](MlirAttribute self) {
      return mlirSwizzledSharedEncodingAttrGetmaxPhase(self);
    });

nbSwizzledSharedEncodingAttr.def_property_readonly(
    "order", [](MlirAttribute self) {
      unsigned norders;
      unsigned *orderPtr;
      mlirSwizzledSharedEncodingAttrGetorder(self, &orderPtr, &norders);
      return std::vector<unsigned>{orderPtr, orderPtr + norders};
    });

nbSwizzledSharedEncodingAttr.def_property_readonly(
    "cta_layout", [](MlirAttribute self) {
      return mlirSwizzledSharedEncodingAttrGetCTALayout(self);
    });
