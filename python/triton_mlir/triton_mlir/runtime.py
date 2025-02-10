from triton_mlir.extras.runtime.passes import Pipeline
from triton_mlir.passmanager import PassManager
from triton_mlir.ir import Module


class TritonPipeline(Pipeline):

    def Func(self, p: "Pipeline"):
        return self.Nested("tt.func", p)

    def allocate_shared_memory(self):
        """Add metadata for shared memory allocation

        This pass uses the `ModuleAllocation` analysis to:
          - Annotate modules with an attribute with the amount of shared/local
            memory used.
          - Annotate operations with an offset into the total shared/local memory.

        """
        self.add_pass("allocate-shared-memory")
        return self

    def convert_triton_to_tritongpu(
        self,
        num_warps: int = None,
        threads_per_warp: int = None,
        num_ctas: int = None,
        target: str = None,
    ):
        """Convert Triton to TritonGPU

          This pass converts the Triton Dialect into the TritonGPU Dialect.
          This is a partial conversion that also affects other dialects
          (namely `Arith`, `Math`, `SCF` and `CF`).
          For these dialects, and many Triton dialect operations the conversions
          mainly consists of enhancing the tensor type and the `tt.ptr<tensor<>>`
          type with an appropriate layout encoding (these encodings generally
          include information on `numWarps`, `threadsPerWarp` and `numCTAs`).

        Args:
            num-warps: number of warps
            threads-per-warp: number of threads per warp
            num-ctas: number of ctas in a cga
            target: the GPU target, e.g., cuda:80, hip:gfx942
        """
        self.add_pass(
            "convert-triton-to-tritongpu",
            num_warps=num_warps,
            threads_per_warp=threads_per_warp,
            num_ctas=num_ctas,
            target=target,
        )
        return self

    def enable_line_info(self):
        """Materialize LLVM line info

        This pass materializes line mapping information for LLVM IR dialect operations.

        """
        self.add_pass("enable-line-info")
        return self

    def triton_combine(self):
        """combine ops

        This pass aims to optimize the five following patterns:
        - `dot(a, b, 0) + c => dot(a, b, c)`

        - `addptr(addptr(ptr, idx0), idx1) => addptr(ptr, AddI(idx0, idx1))`

        - `select(cond, load(ptrs, broadcast(cond), ???), other) =>
             load(ptrs, broadcast(cond), other)`

        - `broadcast(constant) => reshaped_constant`
        - `torch.sum(x[:,:,None].expand(-1,-1,n) * y[None,:,:].expand(m,-1,-1),1)
           => dot(x,y,splat(0))`

        """
        self.add_pass("triton-combine")
        return self

    def triton_loop_unroll(self):
        """Loop unroller

        The pass unrolls a scf loop with tt.loop_unroll_factor attribute. The attribute specialises how many iterations
        the loop should be unrolled.

        """
        self.add_pass("triton-loop-unroll")
        return self

    def triton_nvidia_gpu_fence_insertion(self, compute_capability: int = None):
        """Insert fences across generic and async proxy

        This pass is to insert memory fences to ensure that memory operations are
        properly ordered across generic and async operations.

        Args:
            compute-capability: device compute capability
        """
        self.add_pass(
            "triton-nvidia-gpu-fence-insertion", compute_capability=compute_capability
        )
        return self

    def triton_nvidia_gpu_plan_cta(self):
        """plan CTA

        This pass computes and applies "optimized" CTA tilings to DotOp, ReduceOp
        and StoreLikeOps operations.

        """
        self.add_pass("triton-nvidia-gpu-plan-cta")
        return self

    def triton_nvidia_mma_lowering(self):
        """lower mma operations if needed

        Lower MMA ops to prepare for conversion to LLVM.

        """
        self.add_pass("triton-nvidia-mma-lowering")
        return self

    def triton_nvidia_tma_lowering(self):
        """lower to TMA load/store operations

        Lower Triton experimental descriptor load to TMA load/store operations in TritonNvidiaGPUDialect.

        """
        self.add_pass("triton-nvidia-tma-lowering")
        return self

    def triton_reorder_broadcast(self):
        """Moves broadcast and splat after elementwise operations

        The purpose of this pass is to transform:
          - `elementwise(broadcast(a)) => broadcast(elementwise(a))`
          - `elementwise(splat(a), splat(b), ...) => splat(elementwise(a, b, ...))`
        In the event of a match, the broadcast (or splat) operation is delayed
        and performed after the ElementWise operation.

        """
        self.add_pass("triton-reorder-broadcast")
        return self

    def triton_rewrite_tensor_pointer(self):
        """Rewrite load/stores with tensor pointers into legacy load/stores

        This pass rewrites all load/store semantics initiated by a `tt.make_tensor_ptr` and `tt.advance` into legacy
        semantics. After this pass, `tt.make_tensor_ptr` and `tt.advance` will disappear, and it generates logics to compute
        the pointer/mask/other for each load/store.

        """
        self.add_pass("triton-rewrite-tensor-pointer")
        return self

    def triton_tensor_memory_allocation(self):
        """Assign tensor memory allocation

        Decide on tensor memory allocation and assign attributes to each allocation.

        """
        self.add_pass("triton-tensor-memory-allocation")
        return self

    def tritongpu_F32DotTC(self):
        """3xTF32 trick

        Decompose fp32 `DotOp` instructions into 4 pointwise ops and 3 fp16 `DotOp`s
        to allow using TensorCores. See https://github.com/NVIDIA/cutlass/discussions/385

        """
        self.add_pass("tritongpu-F32DotTC")
        return self

    def tritongpu_accelerate_matmul(self):
        """accelerate matmul

        Optimize the input/output layout of `dot` instruction to make them compatible hardware accelerators
        (e.g., Nvidia tensor cores)

        """
        self.add_pass("tritongpu-accelerate-matmul")
        return self

    def tritongpu_coalesce(self):
        """coalesce

        The pass analyses loads/stores with type `tensor<tt.ptr<>>` or
        `tt.ptr<tensor<>>` and replaces the layouts of these operations with
        coalesced layouts, i.e. cache friendly access patterns.
        Layout conversions are inserted before and after the load/store op
        to maintain consistency with the rest of the program.

        """
        self.add_pass("tritongpu-coalesce")
        return self

    def tritongpu_coalesce_async_copy(self):
        """Improve coalescing for async global to local copies
        For AsyncCopyGlobalToLocal ops where the shared encoding's vec is less than the blocked encoding's sizePerThread, this pass improves coalescing by clipping the sizePerThread value
        """
        self.add_pass("tritongpu-coalesce-async-copy")
        return self

    def tritongpu_combine_tensor_select_and_if(self):
        """Combine tensor select and if
        For select instruction that uses the same condition as the if instruction in the same block this pass combines the select into the if instruction, making the select operands returned by the then/else yields.
        """
        self.add_pass("tritongpu-combine-tensor-select-and-if")
        return self

    def tritongpu_fuse_nested_loops(self):
        """fuse nested loops for pipelining

        The `tritongpu-fuse-nested-loops` pass will analyze loop nests in the module
        that need to be pipelined and fuse them into a single loop. This composes
        with the pipeliner to pipeline loop nests.

        """
        self.add_pass("tritongpu-fuse-nested-loops")
        return self

    def tritongpu_global_scratch_memory_allocation(self):
        """Assign global scratch memory allocation

        Decide on global scratch space memory allocation and assign attributes to each allocation.

        """
        self.add_pass("tritongpu-global-scratch-memory-allocation")
        return self

    def tritongpu_keep_acc_in_tmem(self):
        """Keep accumulator in Tensor Memory

        For Tensor Core Gen 05 Dot operations called in the loop, where the accumulator is reused
        in the next iteration, we want to keep the accumulator in Tensor Memory, so that we can
        avoid the cost of loading the accumulator from registers to Tensor Memory.

        """
        self.add_pass("tritongpu-keep-acc-in-tmem")
        return self

    def tritongpu_loop_scheduling(self, num_stages: int = None):
        """Generate loop scheduling for SWP
        This pass sets up stages and clustering for software pipelining.
            Args:
                num-stages: number of pipeline stages
        """
        self.add_pass("tritongpu-loop-scheduling", num_stages=num_stages)
        return self

    def tritongpu_optimize_accumulator_init(self):
        """Replace accumulator zero-initialization with the flag indicating first use of the accumulator
        For the dot operations that support accumulator-use flag this pass replaces the zero-initialization of the accumulator with the flag indicating the first use of the accumulator.
        """
        self.add_pass("tritongpu-optimize-accumulator-init")
        return self

    def tritongpu_optimize_dot_operands(self, hoist_layout_conversion: bool = None):
        """fuse transpositions

        Re-arranged layouts of tensors used as matrix multiplication operands so as to promote the use of
        hardware-accelerated transpositions.

        Args:
            hoist-layout-conversion: whether to move conver to dot operand earlier pass elementwise ops
        """
        self.add_pass(
            "tritongpu-optimize-dot-operands",
            hoist_layout_conversion=hoist_layout_conversion,
        )
        return self

    def tritongpu_optimize_thread_locality(self):
        """Reduce the cost of synchronization between threads in an SM

        The aim of this pass is to reduce cross-thread communication for certain
        operations, like reductions, reshapes, and gathers.

        For reduction operations, this pass attempts to adjust the reduction size
        (or layout) to avoid splitting the reduction operation between multiple
        threads. Currently, this pass only optimizes reduction yielded by loop to be
        thread-local until after the loop completes.

        For gathers, this pass will attempt to pick an optimized layout for gather
        operations in the module. This is determined based on the shapes of the
        gather operands as well as their existing layouts. The pass applies
        heuristics to determine when it is appropriate to assign specific layouts
        and trigger their respective codegen paths. For now, the pass only attempts
        to apply layouts that result in warp-synchronous gathers.

        """
        self.add_pass("tritongpu-optimize-thread-locality")
        return self

    def tritongpu_pipeline(self, num_stages: int = None):
        """pipeline

        Applies software pipelining to loops in the module based on number of stages.
        This may convert some load into asynchronous loads, and multi-buffer the data.

        Args:
            num-stages: number of pipeline stages
        """
        self.add_pass("tritongpu-pipeline", num_stages=num_stages)
        return self

    def tritongpu_prefetch(self):
        """prefetch

        This pass attempts to prefetch from shared memory the operands (A and B)
        of a `tt.dot`, when this operation is located in a loop.
        Decompose `DotOp` instructions in loops into several finer-grained `DotOp`
        that may have their operands constructed at the end of the previous
        iteration.
        Transformations are performed in five different places:
          1. The pass emits a prologue to the loop where the data for the first
             loop iteration are prefetched.
          2. The loop arguments are extended with the new prefetched values.
          3. The dotOp parameters is updated with the new args.
          4. The prefetch operations for the next iteration are added to the loop.
          5. The yieldOp is updated by adding the prefetched values for the next
             iteration.

        """
        self.add_pass("tritongpu-prefetch")
        return self

    def tritongpu_promote_lhs_to_tmem(self):
        """Promote LHS operand of MMAv5 op to Tensor Memory

        Promote LHS operand of MMAv5 op to Tensor Memory.

        """
        self.add_pass("tritongpu-promote-lhs-to-tmem")
        return self

    def tritongpu_reduce_data_duplication(self):
        """Reduce data duplication in register by decomposing convert[distributed -> dotOperand] into convert[distributed -> shared -> dotOperand]
        Decomposing conversions this way makes it possible to use CSE and reuse #shared tensors
        """
        self.add_pass("tritongpu-reduce-data-duplication")
        return self

    def tritongpu_remove_layout_conversions(self):
        """remove superfluous layout conversions

        The purpose of this pass is to rewrite the `ConvertLayoutOps` to reduce
        the number of operations and to prefer favorable layouts like
        `BlockedEncodingAttr` layout for "expensive" loads and stores
        (good for coalescing) and `NvidiaMmaEncodingAttr` otherwise
        (good for tensor ops).

        """
        self.add_pass("tritongpu-remove-layout-conversions")
        return self

    def tritongpu_reorder_instructions(self):
        """Reorder instructions
        This pass reorder instructions so as to (1) decrease register pressure (e.g., by moving conversions from shared memory before their first use) and (2) promote LLVM instruction order more friendly to `ptxas`.
        """
        self.add_pass("tritongpu-reorder-instructions")
        return self

    def tritongpu_tc05mma_pipeline(self, disable_expander: bool = None):
        """Test pass calling TC05MMA pipeline

        This pass is used to test the TC05MMA pipelining under LIT. Internally it calls
        `getTC05MMASchedule` to get the schedule and then applies the pipelining.

        Args:
            disable-expander: Run only loop pre-process
        """
        self.add_pass("tritongpu-tc05mma-pipeline", disable_expander=disable_expander)
        return self

    def tritongpu_test_pipeline_assign_latencies(self, num_stages: int = None):
        """test assigning latencies to interesting ops ahead of pipelining

        This is a test pass that tests `assignLatencies` method of `TritonGPULoopScheduling`.

        Args:
            num-stages: number of pipeline stages
        """
        self.add_pass("tritongpu-test-pipeline-assign-latencies", num_stages=num_stages)
        return self

    def tritongpu_test_pipeline_schedule_loop(self):
        """test scheduling a loop for software pipelining

        This is a test pass that tests `scheduleLoop` method of `TritonGPULoopScheduling`.

        """
        self.add_pass("tritongpu-test-pipeline-schedule-loop")
        return self

    def convert_builtin_func_to_llvm(self, ftz: bool = None):
        """Convert Builtin Func to LLVM
        Args:
            ftz: flush denorms for math functions
        """
        self.add_pass("convert-builtin-func-to-llvm", ftz=ftz)
        return self

    def convert_triton_amdgpu_to_llvm(self, arch: str = None, ftz: bool = None):
        """Convert TritonGPU to LLVM
        Args:
            arch: gfx target device architecture, e.g., gfx942
            ftz: flush denorms for math functions
        """
        self.add_pass("convert-triton-amdgpu-to-llvm", arch=arch, ftz=ftz)
        return self

    def decompose_unsupported_amd_conversions(self, arch: str = None):
        """Decompose conversions that are not supported by TritonGPU -> LLVM
        Args:
            arch: gfx target device architecture, e.g., gfx942
        """
        self.add_pass("decompose-unsupported-amd-conversions", arch=arch)
        return self

    def optimize_amd_lds_usage(self, target_arch: str = None, lds_limit: int = None):
        """Minimize LDS usage
        Args:
            target-arch: gfx target device architecture, e.g., gfx942
            lds-limit: custom limit of LDS consumption, if not provided, maximum LDS size is used
        """
        self.add_pass(
            "optimize-amd-lds-usage", target_arch=target_arch, lds_limit=lds_limit
        )
        return self

    def triton_amdgpu_insert_instruction_sched_hints(self, variant: str = None):
        """Insert instruction scheduling hints after the dot ops in the main loop
        Args:
            variant: instruction scheduling variant
        """
        self.add_pass("triton-amdgpu-insert-instruction-sched-hints", variant=variant)
        return self

    def triton_amdgpu_lower_insert_instruction_sched_hints(
        self, arch: str = None, num_stages: int = None
    ):
        """Lower instruction scheduling hints to LLVM intrinsics
        Args:
            arch: gfx target device architecture, e.g., gfx942
            num_stages: number of pipeline stages
        """
        self.add_pass(
            "triton-amdgpu-lower-insert-instruction-sched-hints",
            arch=arch,
            num_stages=num_stages,
        )
        return self

    def tritonamdgpu_accelerate_matmul(
        self,
        arch_generation_name: str = None,
        matrix_instruction_size: int = None,
        kPack: int = None,
    ):
        """accelerate matmul

        Optimize the input/output layout of `dot` instruction to make them compatible hardware accelerators
        (e.g., AMD matrix cores)

        Args:
            arch-generation-name: GFX generation name of target device.
            matrix-instruction-size: enforce matrix instruction MN size
            kPack: KWidth / kBase
        """
        self.add_pass(
            "tritonamdgpu-accelerate-matmul",
            arch_generation_name=arch_generation_name,
            matrix_instruction_size=matrix_instruction_size,
            kPack=kPack,
        )
        return self

    def tritonamdgpu_block_pingpong(self):
        """Interleaving instructions from two warps on the same SIMD to better utilize matrix core

        This pass reorder instructions to interleave instructions from two warps on the same SIMD unit.
        We call this a ping-pong scheduling pattern, where two warps run concurrently in the synchronized fashion
        This block ping-pong pattern could be beneficial under few conditions including
        occupancy and number of warps.

        """
        self.add_pass("tritonamdgpu-block-pingpong")
        return self

    def tritonamdgpu_canonicalize_pointers(self):
        """Canonicalize pointers: rewrite pointers passed to load/store operation as a `<basePtr, offset>` pair.

        This pass pushes all the constant pointer arithmetic on a scalar basePtr, while all the vector
        pointer arithmetic to a vector offset. I.e., if we consider the following IR:
        ```
          %v_ptr = tt.splat %s_ptr
          %c_offset = tt.splat %s_offset
          %v_offset0 = tt.make_range
          %v_offset1 = tt.make_range
          %v_ptr0 = tt.addptr %v_ptr, %c_offset
          %v_ptr1 = tt.addptr %v_ptr0, %v_offset0
          %v_ptr2 = tt.addptr %v_ptr0, %v_offset1
          %data = tt.load(%v_ptr2)
        ```
        We transform this into:
        ```
          %s_ptr0 = tt.addptr %s_ptr, %s_offset
          %v_offset = %zero
          %v_offset = arith.addi %v_offset, %v_offset0
          %v_offset = arith.addi %v_offset, %v_offset1
          %c_ptr = tt.splat %s_ptr0
          %v_ptr = tt.addptr %c_ptr, %v_offset
          %data = tt.load(%v_ptr)
        ```
        In the above IR:
        -  `v_` means "variable vector across the program"
        -  `c_` means "constant vector across the program"
        -  `s_` means "scalar"
        So we transform the IR such that the constant updates become scalar updates, and the variable updates happen on the offset. Note that
        when we have to load the data, we splat the scalar pointer, add the "variable" offset and then issue the load.

        """
        self.add_pass("tritonamdgpu-canonicalize-pointers")
        return self

    def tritonamdgpu_convert_buffer_ops(self, arch_generation_name: str = None):
        """Convert memory operations to buffer operations
        This pass converts memory and atomic operations (e.g., tt.load/tt.store/tt.atomic_rmw) to  amdgpu buffer operations, if possible
            Args:
                arch-generation-name: GFX generation name of target device.
        """
        self.add_pass(
            "tritonamdgpu-convert-buffer-ops", arch_generation_name=arch_generation_name
        )
        return self

    def tritonamdgpu_optimize_epilogue(self):
        """Optimize epilogue: (1) Store accumulators directly without going thorough SMEM in epilogue."""
        self.add_pass("tritonamdgpu-optimize-epilogue")
        return self

    def tritonamdgpu_reorder_instructions(self):
        """Reorder instructions
        This pass reorder instructions so as to (1) decrease register pressure (e.g., by moving conversions from shared memory before their first use) and (2) promote LLVM instruction order more friendly to `ptxas`.
        """
        self.add_pass("tritonamdgpu-reorder-instructions")
        return self

    def tritonamdgpu_stream_pipeline(
        self, num_stages: int = None, prefetch: int = None
    ):
        """pipeline

        Pipeline global loads through registers to shared memory while computing on previous
        tile

        Args:
            num_stages: Number of Pipeline stages
            prefetch: Enable prefetch from shared memory
        """
        self.add_pass(
            "tritonamdgpu-stream-pipeline", num_stages=num_stages, prefetch=prefetch
        )
        return self


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


def make_ttgir(mod: Module, options):
    pm.enable_debug()
    passes.ttir.add_convert_to_ttgpuir(
        pm,
        f"hip:{options.arch}",
        options.num_warps,
        options.warp_size,
        options.num_ctas,
    )
    pm.run(mod)
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    passes.ttgpuir.add_coalesce(pm)
    passes.ttgpuir.add_remove_layout_conversions(pm)
    passes.ttgpuir.add_optimize_thread_locality(pm)
    amd.passes.ttgpuir.add_accelerate_matmul(
        pm, options.arch, options.matrix_instr_nonkdim, options.kpack
    )
    passes.ttgpuir.add_remove_layout_conversions(pm)
    amd.passes.ttgpuir.add_optimize_epilogue(pm)
    passes.ttgpuir.add_optimize_dot_operands(pm, True)

    stream_prefetch = os.getenv("TRITON_HIP_STREAM_PREFETCH", "0") == "1"

    # The `local-prefetch` scheduling variant requires turning on buffer ops.
    if options.instruction_sched_variant == "local-prefetch":
        stream_prefetch = True

    if amd.has_matrix_core_feature(options.arch):
        assert options.num_stages != 0, (
            "Triton AMD backend pipeliner has been updated. "
            "We used to trigger software pipelining with "
            "num_stages == 0. Now it will not happen anymore; "
            "please update to use num_stages == 2 for "
            "equivalent behavior in the past."
        )
        amd.passes.ttgpuir.add_stream_pipeline(pm, options.num_stages, stream_prefetch)
        passes.common.add_canonicalizer(pm)
    if options.instruction_sched_variant.lower() != "none":
        amd.passes.ttgpuir.insert_instruction_sched_hints(
            pm, options.instruction_sched_variant
        )
    passes.ttgpuir.add_optimize_dot_operands(pm, True)
    passes.ttgpuir.add_remove_layout_conversions(pm)
    passes.ttgpuir.add_reduce_data_duplication(pm)
    if amd.has_matrix_core_feature(options.arch):
        amd.passes.ttgpuir.add_reorder_instructions(pm)
        use_block_pingpong = os.getenv("TRITON_HIP_USE_BLOCK_PINGPONG", "0") == "1"
        if use_block_pingpong and options.num_stages == 2:
            amd.passes.ttgpuir.add_block_pingpong(pm)

    use_buffer_ops = os.environ.get("AMDGCN_USE_BUFFER_OPS", "0") == "1"
    if use_buffer_ops:
        amd.passes.ttgpuir.add_canonicalize_pointers(pm)
        passes.common.add_canonicalizer(pm)
        amd.passes.ttgpuir.add_convert_to_buffer_ops(pm, options.arch)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    passes.common.add_symbol_dce(pm)
    pm.run(mod)
    return mod


def make_llir(src, metadata, options):
    mod = src
    # TritonGPU -> LLVM-IR (MLIR)
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    amd.passes.ttgpuir.add_decompose_unsupported_conversions(pm, options.arch)
    # custom_lds_size is an experimental parameter that defines amount of LDS available
    # for one thread block. Measured in bytes.
    #
    # If custom_lds_size = 0, pass will consider all LDS is available for one threads block,
    # LDS size is determined by provided arch name.
    custom_lds_size = 0
    amd.passes.ttgpuir.add_optimize_lds_usage(pm, options.arch, custom_lds_size)
    passes.convert.add_scf_to_cf(pm)
    passes.convert.add_index_to_llvmir(pm)

    passes.ttgpuir.add_allocate_shared_memory(pm)
    ## __HIP_FTZ is used to control the denorm flushing behavior of exp2 op as follows:
    ## 1. If __HIP_FTZ = 1, exp2 flushes denorms in input and output regardless
    ##    of the value of kernel arg `allow_flush_denorm`.
    ## 2. If __HIP_FTZ = 0, whether exp2 flushes denorms in input and output
    ##    depends on the value of kernel arg `allow_flush_denorm`.
    ## 3. __HIP_FTZ is default to 1 and not exposed as a kernel argument.
    ##    For now it is used as a controller for developers only.
    __HIP_FTZ = True
    amd.passes.ttgpuir.add_to_llvmir(pm, options.arch, __HIP_FTZ)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)

    passes.convert.add_cf_to_llvmir(pm)
    passes.convert.add_arith_to_llvmir(pm)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    passes.common.add_symbol_dce(pm)
    if options.instruction_sched_variant.lower() != "none":
        amd.passes.ttgpuir.lower_instruction_sched_hints(
            pm, options.arch, options.num_stages
        )
    if os.environ.get("TRITON_DISABLE_LINE_INFO", "0") == "0":
        passes.llvmir.add_di_scope(pm)
    amd.passes.ttgpuir.add_builtin_func_to_llvmir(pm, __HIP_FTZ)
    pm.run(mod)

    # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
    llvm.init_targets()
    context = llvm.context()
    llvm_mod = llvm.to_module(mod, context)
    amd.attach_target_triple(llvm_mod)
    target_features = ""
    if os.environ.get("TRITON_ENABLE_ASAN", "0") == "1":
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
        "amdgpu-flat-work-group-size", f"1,{options.num_warps*options.warp_size}"
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
    if os.environ.get("TRITON_ENABLE_ASAN", "0") == "1":
        fns[0].add_fn_target_feature("+xnack")
        fns[0].add_fn_asan_attr()

    # Hint the compiler that we'd like the firmware to set the kernel arguments
    # to user SGPRs so that the kernel does not need to s_load its arguments
    # from memory.
    amd.set_all_fn_arg_inreg(fns[0])

    if os.environ.get("TRITON_ENABLE_ASAN", "0") == "1":
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

    # Get some metadata
    metadata["shared"] = src.get_int_attr("ttg.shared")

    amd.cleanup_bitcode_metadata(llvm_mod)
    # Disable inlining of print related functions,
    # because inlining of these function could slow down compilation significantly
    amd.disable_print_inline(llvm_mod)
    return str(llvm_mod)


def make_amdgcn(src, metadata, options):
    # Find kernel names (there should only be one)
    # We get the name at the last possible step to accomodate `triton.compile`
    # on user-provided LLVM
    names = re.findall(r"define amdgpu_kernel void @([a-zA-Z_][a-zA-Z0-9_]*)", src)
    assert len(names) == 1
    metadata["name"] = names[0]
    # llvm -> hsaco
    amdgcn = llvm.translate_to_asm(
        src, amd.TARGET_TRIPLE, options.arch, "", [], options.enable_fp_fusion, False
    )
    if os.environ.get("AMDGCN_ENABLE_DUMP", "0") == "1":
        print("// -----// AMDGCN Dump //----- //")
        print(amdgcn)
    return amdgcn


def make_hsaco(src, metadata, options):
    target_features = ""
    if os.environ.get("TRITON_ENABLE_ASAN", "0") == "1":
        target_features = "+xnack"
    hsaco = amd.assemble_amdgcn(src, options.arch, target_features)

    rocm_path = HIPBackend.path_to_rocm_lld()
    with tempfile.NamedTemporaryFile() as tmp_out:
        with tempfile.NamedTemporaryFile() as tmp_in:
            with open(tmp_in.name, "wb") as fd_in:
                fd_in.write(hsaco)
            subprocess.check_call(
                [
                    rocm_path,
                    "-flavor",
                    "gnu",
                    "-shared",
                    tmp_in.name,
                    "-o",
                    tmp_out.name,
                ]
            )
        with open(tmp_out.name, "rb") as fd_out:
            ret = fd_out.read()
    return ret
