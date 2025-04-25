from pathlib import Path

from triton_mlir.compiler import (
    HIPBackend,
    unwrap_c_module_op,
    tritonir,
    llvm,
    make_backend,
)

llvmir = open(Path(__file__).parent / "test_v_pk.ll").read()

metadata = {}
backend = make_backend("gfx942", 32)
options = backend.parse_options({})
llvm.init_targets()
asm = backend.make_amdgcn(llvmir, metadata, options)
print(asm)
