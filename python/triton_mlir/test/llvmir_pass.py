from pathlib import Path

from triton_mlir.compiler import (
    HIPBackend,
    unwrap_c_module_op,
    tritonir,
    llvm,
    make_backend,
)
from triton_mlir.extras.context import mlir_mod_ctx

ttgir = open(Path(__file__).parent / "attn_fwd.ttir").read()

metadata = {}
backend = make_backend("gfx950", 64)
options = backend.parse_options(
    {
        "hash": "626bb48a6c6013dc0da6e64dacbc32ccda9cc1848a89d16eb737495d89e868a0",
        "target": {"backend": "hip", "arch": "gfx950", "warp_size": 64},
        "num_warps": 8,
        "waves_per_eu": 2,
        "num_stages": 4,
        "num_ctas": 1,
        "extern_libs": [
            [
                "ocml",
                "/var/lib/jenkins/OAI-triton/python/triton/backends/amd/lib/ocml.bc",
            ],
            [
                "ockl",
                "/var/lib/jenkins/OAI-triton/python/triton/backends/amd/lib/ockl.bc",
            ],
        ],
        "cluster_dims": [1, 1, 1],
        "debug": False,
        "sanitize_overflow": True,
        "arch": "gfx950",
        "supported_fp8_dtypes": ["fp8e4nv", "fp8e5"],
        "deprecated_fp8_dtypes": [],
        "default_dot_input_precision": "ieee",
        "allowed_dot_input_precisions": ["ieee"],
        "enable_fp_fusion": True,
        "launch_cooperative_grid": False,
        "matrix_instr_nonkdim": 0,
        "kpack": 1,
        "allow_flush_denorm": False,
        "max_num_imprecise_acc_default": 0,
        "backend_name": "hip",
        "schedule_hint": "none",
        "warp_size": 64,
        "TRITON_HIP_USE_ASYNC_COPY": "true",
        "triton_version": "3.3.0",
        "shared": 65536,
        "name": "attn_fwd",
    }
)

with mlir_mod_ctx(allow_unregistered_dialects=True) as ctx:
    ttir_mod = ctx.module.parse(ttgir)
    assert ttir_mod.operation.verify()
    unwrapped_ttir_mod = unwrap_c_module_op(ttir_mod.operation)

    ttgir_mod = backend.make_ttgir(unwrapped_ttir_mod, options)

    metadata = {}
    llvm_mod = backend.make_llir(ttgir_mod, metadata, options)
    assert isinstance(llvm_mod, llvm.module)
    # assert metadata.get("shared") == 0
    assert llvm_mod.verify()
    print(llvm_mod)

    amdgcn = backend.make_amdgcn(str(llvm_mod), metadata, options)
    print(amdgcn)
