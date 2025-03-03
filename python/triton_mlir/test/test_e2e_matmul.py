import ctypes
import inspect
import io
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest
from triton_mlir.extras.context import mlir_mod_ctx
from triton_mlir.extras.util import mlir_type_to_np_dtype
from triton_mlir.ir import (
    Module,
    WalkResult,
    WalkOrder,
    IntegerAttr,
    DenseI32ArrayAttr,
    BoolAttr,
    FlatSymbolRefAttr,
    Attribute,
    ArrayAttr,
    DenseIntElementsAttr,
    DenseFPElementsAttr,
    FloatAttr,
    StringAttr,
    TypeAttr,
    Value,
    OpView,
    OpOperandList,
    IntegerType,
    FloatType,
    RankedTensorType,
    F32Type,
    F16Type,
    F64Type,
)

# this needs to be below the triton_mlir_bindings
from triton_mlir.extras.dialects.ext import arith
from triton_mlir.dialects.arith import IntegerOverflowFlags

# noinspection PyUnresolvedReferences
from triton_mlir.extras.testing import mlir_ctx as ctx, filecheck, MLIRContext

from triton_mlir.dialects import tt, scf, llvm, builtin
from triton_mlir.types import T
from triton_mlir.compiler import (
    HIPBackend,
    unwrap_c_module_op,
    tritonir,
    llvm,
    GPUTarget,
)

# noinspection PyUnresolvedReferences
from triton_mlir.dialects.tt import PointerType
from triton_mlir.dialects import scf

from util import hip_bindings_not_installed, hip_check, backend

pytest.mark.usefixtures("backend")
pytest.mark.usefixtures("ctx")


def normalize_ssa(ssa: str | Value):
    if isinstance(ssa, Value):
        ssa = ssa.get_name()
    if ssa[1].isnumeric():
        ssa = ssa.replace("%", "v")
    else:
        ssa = ssa.replace("%", "")
    ssa = ssa.replace("-", "_")
    ssa = ssa.replace("#", "_")
    return ssa


def normalize_op_name(name: str):
    name = name.replace("(", "_").replace(")", "_").replace(", ", "_").replace(",", "_")
    split_on_dots = name.split(".")
    if len(split_on_dots) > 2:
        dialect, op = split_on_dots[0], "_".join(split_on_dots[1:])
        split_on_dots = [dialect, op]
    return ".".join(split_on_dots)


def np_array_from_shape_type(shape, dtype, splat_value=None):
    if splat_value:
        return np.full(shape, splat_value, dtype)
    return np.empty(shape, dtype)


_dense_i32_array_attr_reg = re.compile(r"array<i32: (.*?)>")
_integer_overflow_flags_reg = re.compile(r"#arith.overflow<(.*?)>")


def map_attr(attr):
    attr = attr.maybe_downcast()
    if isinstance(attr, (IntegerAttr, BoolAttr, FloatAttr)):
        return attr.value
    if isinstance(attr, (FlatSymbolRefAttr, StringAttr)):
        return repr(attr.value)
    # TODO(max): add things upstream to query ArrayAttr elements via .value
    if isinstance(attr, DenseI32ArrayAttr):
        s = str(attr)
        if s == "array<i32>":
            return "[]"
        elements = _dense_i32_array_attr_reg.findall(s)
        assert len(elements) == 1
        return f"[{elements[0]}]"
    if isinstance(attr, (DenseIntElementsAttr, DenseFPElementsAttr)):
        if attr.is_splat:
            splat_v = map_attr(attr.get_splat_value())
            # arr = np_array_from_shape_type(attr.type.shape, mlir_type_to_np_dtype(attr.type.element_type), splat_v)
            return f"np.full({attr.type.shape}, {splat_v}, np.{mlir_type_to_np_dtype(attr.type.element_type).__name__})"
    if attr.__class__ in {Attribute}:
        if "#arith.overflow" in str(attr):
            flag = _integer_overflow_flags_reg.findall(str(attr))
            assert len(flag) == 1
            return f"IntegerOverflowFlags.{flag[0]}"
        return f"Attribute.parse('{attr}')"
    # TODO(max): add things upstream to query ArrayAttr elements via .value
    if attr.__class__ in {ArrayAttr}:
        return f"ArrayAttr.parse('{attr}')"
    if attr.__class__ in {TypeAttr}:
        return f"TypeAttr.parse('{attr}')"
    raise NotImplementedError(attr)


def map_type(type):
    type = type.maybe_downcast()
    if isinstance(type, (IntegerType, F16Type, F32Type, F64Type)):
        if type.width == 1:
            return f"T.bool()"
        return f"T.{type}()"
    if isinstance(type, RankedTensorType):
        return f"T.tensor({', '.join(map(str, type.shape))}, {map_type(type.element_type)})"
    if isinstance(type, PointerType):
        return f"ttpp.ptr({map_type(type.pointee_type)}, {type.address_space})"
    raise NotImplementedError(type)


indent = 0
OUTPUT_BUF = io.StringIO()


def get_init_args(opview):
    klass = opview.__class__
    while not klass.__base__ is OpView:
        klass = klass.__base__
    init_sig = inspect.getfullargspec(klass.__init__)
    init_args = init_sig.args[1:] + init_sig.kwonlyargs
    init_args.remove("loc")
    init_args.remove("ip")
    return init_args


def expects_result_first_arg(opview):
    klass = opview.__class__
    while not klass.__base__ is OpView:
        klass = klass.__base__
    init_sig = inspect.getfullargspec(klass.__init__)
    first_arg = init_sig.args[1]
    if first_arg in {"result"}:
        return first_arg


# stolen from inflection
def underscore(word: str) -> str:
    word = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", word)
    word = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", word)
    word = word.replace("-", "_")
    return word.lower()


def print_opview(opview, name=None):
    print("    " * indent, file=OUTPUT_BUF, end="")
    if len(opview.results):
        print(
            ", ".join([normalize_ssa(r) for r in opview.results]),
            end=" = ",
            file=OUTPUT_BUF,
        )
    if name is None:
        name = opview.name
    name = normalize_op_name(name)
    print(f"{name}(", end="", file=OUTPUT_BUF)
    init_args = get_init_args(opview)
    operands_attrs = {}

    if init_args[0] in {"result"}:
        result = map_type(getattr(opview, init_args[0]).type)
        if isinstance(opview, tt.CallOp):
            result = f"[{result}]"
        operands_attrs["result"] = result
        init_args = init_args[1:]

    # using this causes a reference to the value to remain (causing LLVM ERROR: operation destroyed but still has uses at the end of the script)
    # results_ = {r for r in opview.results}
    results = {r.get_name() for r in opview.results}
    for oan in init_args:
        oa = getattr(opview, oan)
        py_oan = underscore(oan)
        if oa is None:
            continue
        if isinstance(oa, Value):
            if oa.get_name() not in results:
                operands_attrs[py_oan] = normalize_ssa(oa)
            else:
                assert len(
                    results
                ), "only single output result type currently supported"
                operands_attrs[py_oan] = map_type(oa.type)
        elif isinstance(oa, OpOperandList):
            operands_attrs[py_oan] = f"[{', '.join(normalize_ssa(o) for o in oa)}]"
        elif isinstance(oa, Attribute):
            operands_attrs[py_oan] = map_attr(oa.maybe_downcast())
        else:
            raise NotImplementedError(oa)
    print(
        ", ".join([f"{k}={v}" for k, v in operands_attrs.items()]),
        file=OUTPUT_BUF,
        end="",
    )
    print(f")", file=OUTPUT_BUF)


def print_tt_func_op(func_op: tt.FuncOp):
    # op.print(print_generic_op_form=True)
    print("    " * indent, file=OUTPUT_BUF, end="")
    print("@ttpp.jit(", file=OUTPUT_BUF, end="")
    if len(func_op.attributes):
        attrs = []
        for i in range(len(func_op.attributes)):
            attr = func_op.attributes[i]
            if attr.name == "function_type":
                fun_type = attr.attr.value
                inputs = f"[{', '.join([map_type(t) for t in fun_type.inputs])}]"
                results = f"[{', '.join([map_type(t) for t in fun_type.results])}]"
                attrs.append(
                    f"{attr.name}=T.function(inputs={inputs}, results={results})"
                )
            else:
                attrs.append(f"{attr.name}={map_attr(attr.attr)}")
        print(", ".join(attrs), end="", file=OUTPUT_BUF)
    print(")", file=OUTPUT_BUF)
    args = list(func_op.body.blocks[0].arguments)
    args = list(map(normalize_ssa, args))
    print(
        f"def {normalize_op_name(func_op.sym_name.value)}({', '.join(args)}):",
        file=OUTPUT_BUF,
    )


def print_arith_constant(constop: arith.ConstantOp):
    print("    " * indent, file=OUTPUT_BUF, end="")
    print(
        f"{normalize_ssa(constop.result)} = arith.constant({map_attr(constop.value)}, {map_type(constop.result.type)})",
        file=OUTPUT_BUF,
    )


def print_scf_for(for_op: scf.ForOp):
    iv = normalize_ssa(for_op.induction_variable)
    iter_args = [normalize_ssa(a) for a in for_op.inner_iter_args]
    results = [normalize_ssa(r) for r in for_op.results]
    if len(iter_args) > 1:
        opers_str = f"{iv}, [{', '.join(iter_args)}], [{', '.join(results)}]"
    elif len(iter_args) == 1:
        opers_str = f"{iv}, {iter_args[0]}, {results[0]}"
    else:
        opers_str = f"{iv}"
    start, stop, step = map(
        normalize_ssa, [for_op.lowerBound, for_op.upperBound, for_op.step]
    )
    init_args = [normalize_ssa(a) for a in for_op.initArgs]
    print(
        ("    " * indent)
        + f"for {opers_str} in scf.for_({start}, {stop}, {step}, iter_args=[{', '.join(init_args)}]):",
        file=OUTPUT_BUF,
    )


def generic_print_walk_callback(op):
    opview = op.opview
    if isinstance(opview, builtin.ModuleOp):
        return WalkResult.ADVANCE

    if isinstance(opview, tt.FuncOp):
        print("", file=OUTPUT_BUF)
        print_tt_func_op(opview)
    elif isinstance(opview, scf.ForOp):
        print_scf_for(opview)
    elif isinstance(opview, arith.ConstantOp):
        print_arith_constant(opview)
    elif isinstance(opview, scf.YieldOp):
        print_opview(opview, name=f"scf.yield_")
    elif isinstance(opview, tt.ReturnOp):
        print_opview(opview, name=f"tt.return_")
    elif isinstance(opview, tt.PrintOp):
        print_opview(opview, name=f"tt.print_")
    else:
        print_opview(opview)

    if len(op.regions):
        global indent
        indent += 1
        for bodyop in op.regions[0].blocks[0].operations:
            bodyop.walk(generic_print_walk_callback, WalkOrder.PRE_ORDER)
        indent -= 1
        return WalkResult.SKIP

    return WalkResult.ADVANCE


PROLOG = """\
import numpy as np
from triton_mlir.extras.context import RAIIMLIRContextModule
from triton_mlir.dialects import tt as ttpp, scf, llvm, _tt_ops_gen as tt
from triton_mlir.dialects.arith import IntegerOverflowFlags
from triton_mlir.ir import ArrayAttr
from triton_mlir.extras.dialects.ext import arith
from triton_mlir.extras import types as T

ctx = RAIIMLIRContextModule()
"""


def print_prolog():
    print(PROLOG, file=OUTPUT_BUF)


EPILOG = """
# cdiv__i32___1__cconstexpr_32_.emit()
# cdiv__i32___1__cconstexpr_64_.emit()
# zeros_____0_0_cconstexpr_64___0_1_cconstexpr_64___1__cconstexpr_fp32_.emit()

matmul_kernel.emit()
ctx.module.operation.verify()

def mod_str():
    return str(ctx.module)
"""


def print_epilog():
    print(EPILOG, file=OUTPUT_BUF)


HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))


@pytest.mark.parametrize("arch", ["gfx1100"])
def test_smoke(ctx, backend, arch):
    with open(HERE / "matmul_kernel.mlir") as src:
        matmul_src = src.read()

    mod = ctx.module.parse(matmul_src)
    assert mod.operation.verify()

    triton_mod = unwrap_c_module_op(mod.operation)
    hsaco = backend.compile(triton_mod, {"arch": arch})
    assert len(hsaco)
    assert "matmul_kernel" in str(hsaco)


def test_round_trip(ctx):
    with open(HERE / "matmul_kernel.mlir") as src:
        matmul_src = src.read()

    mod = Module.parse(matmul_src)
    assert mod.operation.verify()

    global OUTPUT_BUF
    with open(HERE / "e2e_matmul.py", "w") as OUTPUT_BUF:
        print_prolog()
        mod.operation.walk(generic_print_walk_callback, WalkOrder.PRE_ORDER)
        print_epilog()

    # noinspection PyUnresolvedReferences
    import e2e_matmul

    assert str(e2e_matmul.ctx.module).strip() == matmul_src.strip()


@pytest.mark.parametrize("arch", ["gfx1100"])
def test_compile(ctx, backend, arch):
    with open(HERE / "matmul_kernel.mlir") as src:
        matmul_src = src.read()

    mod = Module.parse(matmul_src)
    assert mod.operation.verify()

    global OUTPUT_BUF
    with open(HERE / "e2e_matmul.py", "w") as OUTPUT_BUF:
        print_prolog()
        mod.operation.walk(generic_print_walk_callback, WalkOrder.PRE_ORDER)
        print_epilog()

    # noinspection PyUnresolvedReferences
    import e2e_matmul

    mod = ctx.module.parse(e2e_matmul.mod_str())
    assert mod.operation.verify()

    triton_mod = unwrap_c_module_op(mod.operation)
    hsaco = backend.compile(triton_mod, {"arch": arch})
    assert len(hsaco)
    assert "matmul_kernel" in str(hsaco)


def get_torch_pointer_hip(a):
    from hip import hip

    if a is None:
        return None
    attributes = hip.hipPointerAttribute_t()
    data_ptr = hip.hipDeviceptr_t(a.data_ptr())
    hip_check(hip.hipPointerGetAttributes(attributes, data_ptr))
    return hip.hipDeviceptr_t(attributes.devicePointer)


def get_torch_pointer_chip(a):
    from triton_mlir import chip

    if a is None:
        return None
    attributes = chip.hipPointerAttribute_t()
    data_ptr = chip.hipDeviceptr_t(a.data_ptr())
    chip_check(chip.hipPointerGetAttributes(ctypes.byref(attributes), data_ptr))
    return chip.hipDeviceptr_t(attributes.devicePointer)


def chip_check(status):
    from triton_mlir import chip

    if status != 0:
        raise RuntimeError(
            f"HIP Error {status}, {ctypes.string_at(chip.hipGetErrorString(status)).decode()}"
        )


try:
    import torch
except ImportError:

    class torch:
        class Tensor:
            pass


def launch(
    function,
    gridX,
    gridY,
    gridZ,
    stream,
    warp_size,
    num_warps,
    shared_memory,
    *args,
):
    from triton_mlir import chip

    from hip._util.types import DeviceArray

    params = [None] * len(args)
    addresses = [None] * len(args)
    for i, p in enumerate(args):
        if isinstance(p, torch.Tensor):
            params[i] = get_torch_pointer_chip(p)
            addresses[i] = ctypes.addressof(params[i])
        elif isinstance(p, DeviceArray):
            addresses[i] = params[i] = p.createRef().as_c_void_p()
        elif isinstance(p, int):
            params[i] = ctypes.c_int32(p)
            addresses[i] = ctypes.addressof(params[i])
        else:
            raise NotImplementedError(f"{p=} not supported with {p=}")

    global_scratch = chip.hipDeviceptr_t()
    addresses += [ctypes.addressof(global_scratch)]
    c_args = (ctypes.c_void_p * len(addresses))(*addresses)
    function = ctypes.cast(function, chip.hipFunction_t)
    stream = ctypes.cast(stream, chip.hipStream_t)
    chip_check(
        chip.hipModuleLaunchKernel(
            function,
            gridX,
            gridY,
            gridZ,
            warp_size * num_warps,
            1,
            1,
            shared_memory,
            stream,
            c_args,
            None,
        )
    )


@pytest.mark.skipif(hip_bindings_not_installed(), reason="hip not installed")
def test_run(ctx, backend):
    from hip import hip

    with open(HERE / "matmul_kernel.mlir") as src:
        matmul_src = src.read()

    mod = Module.parse(matmul_src)
    assert mod.operation.verify()

    global OUTPUT_BUF
    with open(HERE / "e2e_matmul.py", "w") as OUTPUT_BUF:
        print_prolog()
        mod.operation.walk(generic_print_walk_callback, WalkOrder.PRE_ORDER)
        print_epilog()

    # noinspection PyUnresolvedReferences
    import e2e_matmul

    props = hip.hipDeviceProp_t()
    hip_check(hip.hipGetDeviceProperties(props, 0))
    arch = props.gcnArchName.decode()
    options = backend.parse_options({"arch": arch, "waves_per_eu": 8})

    mod = ctx.module.parse(e2e_matmul.mod_str())
    assert mod.operation.verify()
    triton_mod = unwrap_c_module_op(mod.operation)
    hsaco, metadata = backend.compile(
        triton_mod,
        options=options,
        dump_ir=True,
        ir_dump_dir=Path(__file__).parent / "e2e_matmul_dump",
        dump_file_prefix="0",
    )

    module = hip_check(hip.hipModuleLoadData(hsaco))
    function = hip_check(
        hip.hipModuleGetFunction(module, metadata["name"].encode())
    ).as_c_void_p()

    # kernel launch

    M, K, N = 16, 16, 16
    BLOCK_SIZE_M = 8
    BLOCK_SIZE_N = 8
    BLOCK_SIZE_K = 4

    a_h = np.random.rand(M, K).astype(dtype=np.float32)
    b_h = np.random.rand(K, N).astype(dtype=np.float32)
    c_h = -3 * np.ones((M, N), dtype=np.float32)

    a_num_bytes = a_h.size * a_h.itemsize
    b_num_bytes = b_h.size * b_h.itemsize
    c_num_bytes = c_h.size * c_h.itemsize

    a_d = hip_check(hip.hipMalloc(a_num_bytes)).configure(
        typestr="float32", shape=(M, K)
    )
    b_d = hip_check(hip.hipMalloc(b_num_bytes)).configure(
        typestr="float32", shape=(K, N)
    )
    c_d = hip_check(hip.hipMalloc(c_num_bytes)).configure(
        typestr="float32", shape=(M, N)
    )

    hip_check(
        hip.hipMemcpy(a_d, a_h, a_num_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice)
    )
    hip_check(
        hip.hipMemcpy(b_d, b_h, b_num_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice)
    )
    hip_check(
        hip.hipMemcpy(c_d, c_h, c_num_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice)
    )

    stream = 0
    gridX = math.ceil(M / BLOCK_SIZE_M) * math.ceil(N / BLOCK_SIZE_N)
    gridY = 1
    gridZ = 1
    num_warps = options.num_warps
    shared_memory = metadata["shared"]

    launch(
        function,
        gridX,
        gridY,
        gridZ,
        stream,
        options.warp_size,
        num_warps,
        shared_memory,
        a_d,
        b_d,
        c_d,
        M,
        N,
        K,
        *(np.array(a_h.strides) // a_h.itemsize).tolist(),
        *(np.array(b_h.strides) // b_h.itemsize).tolist(),
        *(np.array(c_h.strides) // c_h.itemsize).tolist(),
    )

    hip_check(
        hip.hipMemcpy(c_h, c_d, c_num_bytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost)
    )

    assert np.allclose(c_h, a_h @ b_h)


@pytest.mark.skipif(hip_bindings_not_installed(), reason="hip not installed")
def test_inline_mod(ctx, backend):
    from hip import hip

    BLOCK_SIZE_M = 8
    BLOCK_SIZE_N = 8
    BLOCK_SIZE_K = 4
    GROUP_SIZE_M = 1

    @tt.jit(
        arg_attrs=ArrayAttr.parse(
            "[{tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, {}, {}, {}, {}, {}, {}, {}, {}, {}]"
        ),
        noinline=False,
        sym_name="matmul_kernel_2",
        sym_visibility="public",
    )
    def matmul_kernel_2(
        a_ptr: +T.f32,
        b_ptr: +T.f32,
        c_ptr: +T.f32,
        M: T.i32,
        N: T.i32,
        K: T.i32,
        stride_am: T.i32,
        stride_ak: T.i32,
        stride_bk: T.i32,
        stride_bn: T.i32,
        stride_cm: T.i32,
        stride_cn: T.i32,
    ):
        c0_i32 = arith.constant(0, T.i32)
        c1_i32 = arith.constant(1, T.i32)
        GROUP_SIZE_M_ = arith.constant(1, T.i32)
        BLOCK_SIZE_K_ = arith.constant(BLOCK_SIZE_K, T.i32)
        BLOCK_SIZE_M_ = arith.constant(BLOCK_SIZE_M, T.i32)
        BLOCK_SIZE_N_ = arith.constant(BLOCK_SIZE_N, T.i32)

        pid = tt.get_program_id(axis=0)

        num_pid_m = arith.ceildivsi(M, BLOCK_SIZE_M_)
        num_pid_n = arith.ceildivsi(N, BLOCK_SIZE_N_)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid / num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = arith.minsi(num_pid_m - first_pid_m, GROUP_SIZE_M_)
        pid_m = first_pid_m + ((pid % num_pid_n) % group_size_m)
        pid_n = (pid % num_pid_n) / group_size_m

        arr = tt.arange(start=0, end=BLOCK_SIZE_M)

        v15 = tt.splat(pid_m * BLOCK_SIZE_M, (BLOCK_SIZE_M,)) + arr
        offs_am = v15 % tt.splat(M, (BLOCK_SIZE_M,))

        v20 = tt.splat(pid_n * BLOCK_SIZE_N, (BLOCK_SIZE_N,)) + arr
        offs_bn = v20 % tt.splat(N, (BLOCK_SIZE_N,))

        offs_k = tt.arange(start=0, end=BLOCK_SIZE_K)

        v26 = tt.expand_dims(offs_am, axis=1) * tt.splat(stride_am, (BLOCK_SIZE_M, 1))
        v27 = tt.expand_dims(offs_k, axis=0)
        v29 = v27 * tt.splat(stride_ak, (1, BLOCK_SIZE_K))
        v30 = tt.broadcast(v26, (BLOCK_SIZE_M, BLOCK_SIZE_K))
        v31 = tt.broadcast(v29, (BLOCK_SIZE_M, BLOCK_SIZE_K))
        v32 = v30 + v31
        v33 = tt.splat(a_ptr, (BLOCK_SIZE_M, BLOCK_SIZE_K))
        a_ptrs = tt.addptr(v33, offset=v32)

        v35 = tt.expand_dims(offs_k, axis=1)
        v37 = v35 * tt.splat(stride_bk, (BLOCK_SIZE_K, 1))
        v40 = tt.expand_dims(offs_bn, axis=0) * tt.splat(stride_bn, (1, BLOCK_SIZE_N))
        v41 = tt.broadcast(v37, (BLOCK_SIZE_K, BLOCK_SIZE_N))
        v42 = tt.broadcast(v40, (BLOCK_SIZE_K, BLOCK_SIZE_N))
        v43 = v41 + v42
        v44 = tt.splat(b_ptr, (BLOCK_SIZE_K, BLOCK_SIZE_N))
        b_ptrs = tt.addptr(v44, offset=v43)

        a_ptr_incr = tt.splat(stride_ak * BLOCK_SIZE_K, (BLOCK_SIZE_M, BLOCK_SIZE_K))
        b_ptr_incr = tt.splat(stride_bk * BLOCK_SIZE_K, (BLOCK_SIZE_K, BLOCK_SIZE_N))

        accum = arith.constant(
            np.full([BLOCK_SIZE_M, BLOCK_SIZE_N], 0.0, np.float32),
            T.tensor(BLOCK_SIZE_M, BLOCK_SIZE_N, T.f32),
        )
        cst_0 = arith.constant(
            np.full([BLOCK_SIZE_K, BLOCK_SIZE_N], 0.0, np.float32),
            T.tensor(BLOCK_SIZE_K, BLOCK_SIZE_N, T.f32),
        )
        cst_1 = arith.constant(
            np.full([BLOCK_SIZE_M, BLOCK_SIZE_K], 0.0, np.float32),
            T.tensor(BLOCK_SIZE_M, BLOCK_SIZE_K, T.f32),
        )
        stop = arith.ceildivsi(K, BLOCK_SIZE_K_)
        for k, [accum, a_ptrs, b_ptrs], [c, _a_ptrs, _b_ptrs] in scf.for_(
            c0_i32, stop, c1_i32, iter_args=[accum, a_ptrs, b_ptrs]
        ):
            v72 = K - k * BLOCK_SIZE_K

            a_mask = tt.broadcast(
                v27 < tt.splat(v72, (1, BLOCK_SIZE_K)), (BLOCK_SIZE_M, BLOCK_SIZE_K)
            )
            a = tt.load(a_ptrs, mask=a_mask, other=cst_1)

            b_mask = tt.broadcast(
                v35 < tt.splat(v72, (BLOCK_SIZE_K, 1)), (BLOCK_SIZE_K, BLOCK_SIZE_N)
            )
            b = tt.load(b_ptrs, mask=b_mask, other=cst_0)

            accum = tt.dot(a, b, c=accum)
            a_ptrs = tt.addptr(a_ptrs, offset=a_ptr_incr)
            b_ptrs = tt.addptr(b_ptrs, offset=b_ptr_incr)

            scf.yield_(results_=[accum, a_ptrs, b_ptrs])

        offs_cm = tt.expand_dims(v15, axis=1)

        c_ptr = tt.splat(c_ptr, (BLOCK_SIZE_M, 1))
        c_ptr = tt.addptr(
            c_ptr, offset=tt.splat(stride_cm, (BLOCK_SIZE_M, 1)) * offs_cm
        )
        c_ptr = tt.broadcast(c_ptr, (BLOCK_SIZE_M, BLOCK_SIZE_N))

        offs_cn = tt.expand_dims(v20, axis=0)
        v60 = tt.splat(stride_cn, (1, BLOCK_SIZE_N)) * offs_cn
        c_ptrs = tt.addptr(
            c_ptr, offset=tt.broadcast(v60, (BLOCK_SIZE_M, BLOCK_SIZE_N))
        )

        v68 = tt.broadcast(
            offs_cm < tt.splat(M, (BLOCK_SIZE_M, 1)), (BLOCK_SIZE_M, BLOCK_SIZE_N)
        )
        v69 = tt.broadcast(
            offs_cn < tt.splat(N, (1, BLOCK_SIZE_N)), (BLOCK_SIZE_M, BLOCK_SIZE_N)
        )
        c_mask = v68 & v69

        tt.store(c_ptrs, value=c, mask=c_mask)

        tt.return_(srcs=[])

    matmul_kernel_2.emit()
    assert ctx.module.operation.verify()
    triton_mod = unwrap_c_module_op(ctx.module.operation)

    props = hip.hipDeviceProp_t()
    hip_check(hip.hipGetDeviceProperties(props, 0))
    arch = props.gcnArchName.decode()
    options = backend.parse_options({"arch": arch, "waves_per_eu": 8})

    hsaco, metadata = backend.compile(
        triton_mod,
        options=options,
        dump_ir=True,
        ir_dump_dir=Path(__file__).parent / "matmul_kernel_2",
    )

    module = hip_check(hip.hipModuleLoadData(hsaco))
    function = hip_check(
        hip.hipModuleGetFunction(module, metadata["name"].encode())
    ).as_c_void_p()

    # kernel launch

    M, K, N = 16, 16, 16
    a_h = np.random.rand(M, K).astype(dtype=np.float32)
    b_h = np.random.rand(K, N).astype(dtype=np.float32)
    c_h = -3 * np.ones((M, N), dtype=np.float32)

    a_num_bytes = a_h.size * a_h.itemsize
    b_num_bytes = b_h.size * b_h.itemsize
    c_num_bytes = c_h.size * c_h.itemsize

    a_d = hip_check(hip.hipMalloc(a_num_bytes)).configure(
        typestr="float32", shape=(M, K)
    )
    b_d = hip_check(hip.hipMalloc(b_num_bytes)).configure(
        typestr="float32", shape=(K, N)
    )
    c_d = hip_check(hip.hipMalloc(c_num_bytes)).configure(
        typestr="float32", shape=(M, N)
    )

    hip_check(
        hip.hipMemcpy(a_d, a_h, a_num_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice)
    )
    hip_check(
        hip.hipMemcpy(b_d, b_h, b_num_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice)
    )
    hip_check(
        hip.hipMemcpy(c_d, c_h, c_num_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice)
    )

    stream = 0
    gridX = math.ceil(M / BLOCK_SIZE_M) * math.ceil(N / BLOCK_SIZE_N)
    gridY = 1
    gridZ = 1
    num_warps = options.num_warps
    shared_memory = metadata["shared"]

    launch(
        function,
        gridX,
        gridY,
        gridZ,
        stream,
        options.warp_size,
        num_warps,
        shared_memory,
        a_d,
        b_d,
        c_d,
        M,
        N,
        K,
        *(np.array(a_h.strides) // a_h.itemsize).tolist(),
        *(np.array(b_h.strides) // b_h.itemsize).tolist(),
        *(np.array(c_h.strides) // c_h.itemsize).tolist(),
    )

    hip_check(
        hip.hipMemcpy(c_h, c_d, c_num_bytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost)
    )

    assert np.allclose(c_h, a_h @ b_h)


if __name__ == "__main__":
    with mlir_mod_ctx(allow_unregistered_dialects=True) as ctx:
        test_smoke(ctx)
    with mlir_mod_ctx(allow_unregistered_dialects=True) as ctx:
        test_round_trip(ctx)
    with mlir_mod_ctx(allow_unregistered_dialects=True) as ctx:
        test_compile(ctx)
    with mlir_mod_ctx(allow_unregistered_dialects=True) as ctx:
        test_run(ctx)
    with mlir_mod_ctx(allow_unregistered_dialects=True) as ctx:
        test_inline_mod(ctx)
