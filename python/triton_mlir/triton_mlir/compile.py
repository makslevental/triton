import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from ._mlir_libs._triton import llvm, amd
from .ir import Module
from .passes import TritonPipeline
from .passmanager import PassManager


@dataclass(frozen=True)
class HIPOptions:
    num_warps: int = 4
    warp_size: int = 32
    waves_per_eu: int = 1
    num_stages: int = 2
    num_ctas: int = 1
    extern_libs: dict = None
    cluster_dims: tuple = (1, 1, 1)
    debug: bool = False
    sanitize_overflow: bool = True
    arch: str = None
    supported_fp8_dtypes: Tuple[str] = ("fp8e5",)
    deprecated_fp8_dtypes: Tuple[str] = ()
    default_dot_input_precision: str = "ieee"
    allowed_dot_input_precisions: Tuple[str] = ("ieee",)
    enable_fp_fusion: bool = True
    launch_cooperative_grid: bool = False
    matrix_instr_nonkdim: int = 0
    kpack: int = 1
    allow_flush_denorm: bool = False
    max_num_imprecise_acc_default: int = 0
    backend_name: str = "hip"

    # The following option provides hints to the AMDGPU backend regarding instruction scheduling
    # for all `tt.dot` operations in a kernel. The "none" variant preserves the default
    # instruction scheduling of the AMDGPU backend which aims at maximizing occupancy.
    # The option is experimental and may change at any time regarding its semantics and/or may
    # be gone entirely anytime.
    #
    # Current experimental scheduling variants:
    #
    # llvm-iglp-0: injects `llvm.amdgcn.iglp_opt` intrinsic call with value `0` to the GEMM's
    #              k-loop; i.e., "interleave DS and MFMA instructions for small GEMM kernels".
    # llvm-iglp-1: injects `llvm.amdgcn.iglp_opt` intrinsic call with value `1` to the GEMM's
    #              k-loop; i.e., "interleave DS and MFMA instructions for single wave small
    #              GEMM kernels.".
    # local-prefetch: implements instruction scheduling similar to the one from the ROCm Composable
    #                 Kernel library. Note, this variant requires the use of buffer load/store ops
    #                 and a special software pipelining style - i.e., 1x LDS and 1x register
    #                 prefetch buffers for each GEMM tile.
    instruction_sched_variant: str = "none"

    def __post_init__(self):
        warp_size = (
            32
            if "gfx10" in self.arch or "gfx11" in self.arch or "gfx12" in self.arch
            else 64
        )
        object.__setattr__(self, "warp_size", warp_size)
        assert (
            self.num_warps > 0 and (self.num_warps & (self.num_warps - 1)) == 0
        ), "num_warps must be a power of 2"


def parse_options(target_arch, **opts) -> HIPOptions:
    args = {"arch": os.getenv("TRITON_OVERRIDE_ARCH", target_arch)}

    # Enable XF32 (TF32) for CDNA3 GPUs
    if target_arch in ("gfx940", "gfx941", "gfx942"):
        allowed_dot_input_precisions = set(HIPOptions.allowed_dot_input_precisions)
        allowed_dot_input_precisions.update({"tf32"})
        args["allowed_dot_input_precisions"] = tuple(
            sorted(allowed_dot_input_precisions)
        )

    if "supported_fp8_dtypes" not in opts:
        supported_fp8_dtypes = set(HIPOptions.supported_fp8_dtypes)
        if target_arch in ("gfx940", "gfx941", "gfx942"):
            supported_fp8_dtypes.update({"fp8e4nv", "fp8e4b8", "fp8e5b16"})
        args["supported_fp8_dtypes"] = tuple(sorted(supported_fp8_dtypes))

    if "enable_fp_fusion" not in opts:
        args["enable_fp_fusion"] = strtobool(
            os.getenv("TRITON_DEFAULT_FP_FUSION", "True")
        )
    args.update(
        {
            k: opts[k]
            for k in HIPOptions.__dataclass_fields__.keys()
            if k in opts and opts[k] is not None
        }
    )
    return HIPOptions(**args)


def make_ttir(mod: Module):
    p = (
        TritonPipeline()
        .inline()
        .triton_rewrite_tensor_pointer()
        .canonicalize()
        .triton_combine()
        .triton_reorder_broadcast()
        .cse()
        .loop_invariant_code_motion()
        .symbol_dce()
        .triton_loop_unroll()
    )
    pm = PassManager.parse(p.materialize())
    pm.run(mod.operation)
    return mod


def make_ttgir(mod: Module, options: HIPOptions):
    p = TritonPipeline().convert_triton_to_tritongpu(
        num_warps=options.num_warps,
        threads_per_warp=options.num_warps,
        num_ctas=options.num_ctas,
        target=f"hip:{options.arch}",
    )
    pm = PassManager.parse(p.materialize())
    pm.run(mod.operation)
    p = (
        TritonPipeline()
        .tritongpu_coalesce()
        .tritongpu_remove_layout_conversions()
        .tritongpu_optimize_thread_locality()
        .tritonamdgpu_accelerate_matmul(
            arch_generation_name=options.arch,
            matrix_instruction_size=options.matrix_instr_nonkdim,
            kPack=options.kpack,
        )
        .tritongpu_remove_layout_conversions()
        .tritonamdgpu_optimize_epilogue()
        .tritongpu_optimize_dot_operands(hoist_layout_conversion=True)
    )

    stream_prefetch = strtobool(os.getenv("TRITON_HIP_STREAM_PREFETCH", "False"))
    # The `local-prefetch` scheduling variant requires turning on buffer ops.
    if options.instruction_sched_variant == "local-prefetch":
        stream_prefetch = True

    if llvm.has_matrix_core_feature(options.arch):
        assert options.num_stages != 0, (
            "Triton AMD backend pipeliner has been updated. "
            "We used to trigger software pipelining with "
            "num_stages == 0. Now it will not happen anymore; "
            "please update to use num_stages == 2 for "
            "equivalent behavior in the past."
        )
        p = p.tritonamdgpu_stream_pipeline(
            num_stages=options.num_stages, prefetch=stream_prefetch
        ).canonicalize()
    if options.instruction_sched_variant.lower() != "none":
        p = p.triton_amdgpu_insert_instruction_sched_hints(
            variant=options.instruction_sched_variant
        )
    p = (
        p.tritongpu_optimize_dot_operands(hoist_layout_conversion=True)
        .tritongpu_remove_layout_conversions()
        .tritongpu_reduce_data_duplication()
    )
    if llvm.has_matrix_core_feature(options.arch):
        p = p.tritonamdgpu_reorder_instructions()
        use_block_pingpong = strtobool(
            os.getenv("TRITON_HIP_USE_BLOCK_PINGPONG", "False")
        )
        if use_block_pingpong and options.num_stages == 2:
            p = p.tritonamdgpu_block_pingpong()

    use_buffer_ops = strtobool(os.environ.get("AMDGCN_USE_BUFFER_OPS", "False"))
    if use_buffer_ops:
        p = (
            p.tritonamdgpu_canonicalize_pointers()
            .canonicalize()
            .tritonamdgpu_convert_buffer_ops(arch_generation_name=options.arch)
        )
    p = p.canonicalize().cse().symbol_dce()
    pm = PassManager.parse(p.materialize())
    pm.run(mod.operation)
    return mod


AMD_TRIPLE = "amdgcn-amd-amdhsa"


def strtobool(val):
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif val in ("n", "no", "f", "false", "off", "0"):
        return False
    else:
        raise ValueError("invalid truth value %r" % (val,))


def add_enum_attr_to_function_arg(fn, i, attr_name):
    llvm_context = llvm.get_value_context(fn)
    k = llvm.get_enum_attribute_kind_for_name(attr_name, len(attr_name))
    attr = llvm.create_enum_attribute(llvm_context, k, val=0)
    # Value must be zero for enum attributes
    llvm.add_attribute_at_index(fn, i, attr)


LLVMAttributeFunctionIndex = 0xFFFFFFFF


def add_enum_attr_to_function(fn, attr_name):
    llvm_context = llvm.get_value_context(fn)
    k = llvm.get_enum_attribute_kind_for_name(attr_name, len(attr_name))
    attr = llvm.create_enum_attribute(llvm_context, k, val=0)
    # Value must be zero for enum attributes
    llvm.add_attribute_at_index(
        fn,
        LLVMAttributeFunctionIndex,
        attr,
    )


def add_string_attr_to_function(fn, attr_name, attr_val):
    llvm_context = llvm.get_value_context(fn)
    attr = llvm.create_string_attribute(
        llvm_context,
        attr_name,
        len(attr_name),
        attr_val,
        len(attr_val),
    )
    llvm.add_attribute_at_index(fn, LLVMAttributeFunctionIndex, attr)


def add_control_constant(llvm_mod, name, bitwidth, value):
    ctx = llvm.get_module_context(llvm_mod)
    int_ty = llvm.int_type_in_context(ctx, bitwidth)
    initer = llvm.const_int(int_ty, value, sign_extend=False)
    constant = llvm.add_global_in_address_space(llvm_mod, int_ty, name, address_space=4)
    llvm.set_global_constant(constant, is_constant=True)
    llvm.set_linkage(constant, llvm.Linkage.link_once_odr_linkage)
    llvm.set_initializer(constant, initer)
    llvm.set_thread_local(constant, is_thread_local=False)

    llvm.set_alignment(constant, bitwidth // 8)
    llvm.set_unnamed_address(constant, llvm.UnnamedAddr.local_unnamed_addr)
    llvm.set_visibility(constant, llvm.Visibility.protected_visibility)


def make_llir(mod, options: HIPOptions):
    # # Get some metadata
    # metadata["shared"] = mod.get_int_attr("ttg.shared")

    # TritonGPU -> LLVM-IR (MLIR)
    p = TritonPipeline().decompose_unsupported_amd_conversions(options.arch)
    # custom_lds_size is an experimental parameter that defines amount of LDS available
    # for one thread block. Measured in bytes.
    #
    # If custom_lds_size = 0, pass will consider all LDS is available for one threads block,
    # LDS size is determined by provided arch name.
    custom_lds_size = 0
    p = (
        p.optimize_amd_lds_usage(options.arch, custom_lds_size)
        .convert_scf_to_cf()
        .convert_index_to_llvm()
        .allocate_shared_memory()
    )
    ## __HIP_FTZ is used to control the denorm flushing behavior of exp2 op as follows:
    ## 1. If __HIP_FTZ = 1, exp2 flushes denorms in input and output regardless
    ##    of the value of kernel arg `allow_flush_denorm`.
    ## 2. If __HIP_FTZ = 0, whether exp2 flushes denorms in input and output
    ##    depends on the value of kernel arg `allow_flush_denorm`.
    ## 3. __HIP_FTZ is default to 1 and not exposed as a kernel argument.
    ##    For now it is used as a controller for developers only.
    __HIP_FTZ = True
    p = (
        p.convert_triton_amdgpu_to_llvm(arch=options.arch, ftz=__HIP_FTZ)
        .canonicalize()
        .cse()
        .convert_cf_to_llvm()
        .convert_arith_to_llvm()
        .canonicalize()
        .cse()
        .symbol_dce()
    )

    if options.instruction_sched_variant.lower() != "none":
        p = p.triton_amdgpu_lower_insert_instruction_sched_hints(
            arch=options.arch, num_stages=options.num_stages
        )
    if not strtobool(os.environ.get("TRITON_DISABLE_LINE_INFO", "False")):
        p = p.ensure_debug_info_scope_on_llvm_func()
    p = p.convert_builtin_func_to_llvm(__HIP_FTZ)
    pm = PassManager.parse(p.materialize())
    pm.run(mod.operation)

    # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
    llvm.init_targets()
    context = llvm.context()
    llvm_mod = llvm.to_module(mod.operation, context)
    amd.attach_target_triple(llvm_mod)
    target_features = ""
    if strtobool(os.environ.get("TRITON_ENABLE_ASAN", "False")):
        target_features = "+xnack"
    llvm.attach_datalayout(llvm_mod, amd.TARGET_TRIPLE, options.arch, target_features)

    # Set various control constants on the LLVM module so that device
    # libraries can resolve references to them.
    amd.set_isa_version(llvm_mod, options.arch)
    amd.set_abi_version(llvm_mod, 500)
    amd.set_bool_control_constant(llvm_mod, "__oclc_finite_only_opt", False)
    amd.set_bool_control_constant(llvm_mod, "__oclc_correctly_rounded_sqrt32", True)
    amd.set_bool_control_constant(llvm_mod, "__oclc_unsafe_math_opt", False)
    amd.set_bool_control_constant(
        llvm_mod, "__oclc_wavefrontsize64", options.warp_size == 64
    )

    # Set kernel attributes first given this may affect later optimizations.
    fns = [fn for fn in llvm_mod.get_functions() if not fn.is_declaration()]
    # The public kernel should be kernel 0.
    fns[0].set_calling_conv(amd.CALLING_CONV_AMDGPU_KERNEL)
    fns[0].add_fn_attr(
        "amdgpu-flat-work-group-size", f"1,{options.num_warps * options.warp_size}"
    )
    # LLVM AMDGPU backend supports the attribute "amdgpu-waves-per-eu"="<min>[, <max>]".
    # This attribute may be attached to a kernel function definition and is an optimization hint.
    # <min> parameter specifies the requested minimum number of waves per EU, and optional <max> parameter
    # specifies the requested maximum number of waves per EU (must be greater than <min> if specified).
    # If <max> is omitted, then there is no restriction on the maximum number of waves per EU other than
    # the one dictated by the hardware for which the kernel is compiled. Passing 0, 0 as <min>, <max>
    # implies the default behavior (no limits).
    fns[0].add_fn_attr("amdgpu-waves-per-eu", f"{options.waves_per_eu}")
    denormal_mode = "preserve-sign" if options.allow_flush_denorm else "ieee"
    fns[0].add_fn_attr("denormal-fp-math-f32", denormal_mode)
    if strtobool(os.environ.get("TRITON_ENABLE_ASAN", "False")):
        fns[0].add_fn_target_feature("+xnack")
        fns[0].add_fn_asan_attr()

    # Hint the compiler that we'd like the firmware to set the kernel arguments
    # to user SGPRs so that the kernel does not need to s_load its arguments
    # from memory.
    amd.set_all_fn_arg_inreg(fns[0])

    if strtobool(os.environ.get("TRITON_ENABLE_ASAN", "False")):
        default_libdir = Path(__file__).parent / "lib"
        paths = [
            str(default_libdir / "asanrtl.bc"),
            str(default_libdir / "ocml.bc"),
            str(default_libdir / "ockl.bc"),
        ]
        llvm.link_extern_libs(llvm_mod, paths)
    elif options.extern_libs:
        paths = [
            path
            for (name, path) in options.extern_libs
            if amd.need_extern_lib(llvm_mod, name)
        ]
        llvm.link_extern_libs(llvm_mod, paths)

    llvm.optimize_module(
        llvm_mod, llvm.OPTIMIZE_O3, options.arch, "", [], options.enable_fp_fusion
    )

    amd.cleanup_bitcode_metadata(llvm_mod)
    # Disable inlining of print related functions,
    # because inlining of these function could slow down compilation significantly
    amd.disable_print_inline(llvm_mod)
    return llvm_mod


def make_amdgcn(llvm_mod, options):
    amdgcn = llvm.translate_to_asm(
        llvm_mod,
        amd.TARGET_TRIPLE,
        options.arch,
        "",
        [],
        options.enable_fp_fusion,
        False,
    )
    if strtobool(os.environ.get("AMDGCN_ENABLE_DUMP", "False")):
        print("// -----// AMDGCN Dump //----- //")
        print(amdgcn)
    return amdgcn


def make_hsaco(src, options):
    target_features = ""
    if strtobool(os.environ.get("TRITON_ENABLE_ASAN", "False")):
        target_features = "+xnack"
    hsaco = amd.assemble_amdgcn(src, options.arch, target_features)
    with tempfile.NamedTemporaryFile() as tmp_out:
        with tempfile.NamedTemporaryFile() as tmp_in:
            with open(tmp_in.name, "wb") as fd_in:
                fd_in.write(hsaco)
            amd.link_hsaco(tmp_in.name, tmp_out.name)
        with open(tmp_out.name, "rb") as fd_out:
            ret = fd_out.read()
    return ret
