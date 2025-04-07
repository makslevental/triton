auto nbSliceEncodingAttr =
    mlir_attribute_subclass(m, "SliceEncodingAttr", isaMlirSliceEncodingAttr,
                            mlirSliceEncodingAttrGetTypeID);
nbSliceEncodingAttr.def_staticmethod(
    "get",
    [](unsigned dim, MlirAttribute parent, MlirContext context) {
      return mlirSliceEncodingAttrAttributeGet(dim, parent, context);
    },
    "dim"_a, "parent"_a, "context"_a = nb::none());

nbSliceEncodingAttr.def_property_readonly("dim", [](MlirAttribute self) {
  return mlirSliceEncodingAttrGetdim(self);
});

nbSliceEncodingAttr.def_property_readonly("parent", [](MlirAttribute self) {
  return mlirSliceEncodingAttrGetparent(self);
});
