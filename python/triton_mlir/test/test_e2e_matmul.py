import ctypes
import inspect
import io
import math
import re
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from triton_mlir.compiler import unwrap_c_module_op
from triton_mlir.dialects import tt, scf, builtin, ttg, amdgpu
from triton_mlir.extras.ast.canonicalize import canonicalize

# noinspection PyUnresolvedReferences
from triton_mlir.dialects.tt import (
    PointerType,
    broadcast,
    splat,
    load,
    dot,
    addptr,
    expand_dims,
    store,
    get_program_id,
    make_range,
)
from triton_mlir.extras.context import mlir_mod_ctx

# this needs to be below the triton_mlir_bindings
from triton_mlir.extras.dialects.ext import arith
from triton_mlir.extras.dialects.ext import scf

# noinspection PyUnresolvedReferences
from triton_mlir.extras.testing import mlir_ctx as ctx, filecheck, MLIRContext
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
    RankedTensorType,
    F32Type,
    F16Type,
    F64Type,
)
from triton_mlir.types import T

# noinspection PyUnresolvedReferences
from util import hip_bindings_not_installed, hip_check, backend_, backend

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
    if isinstance(attr, ttg.BlockedEncodingAttr):
        return f"ttg.BlockedEncodingAttr.get(size_per_thread={attr.size_per_thread}, threads_per_warp__={attr.threads_per_warp__}, warps_per_cta__={attr.warps_per_cta__}, order={attr.order})"
    if isinstance(attr, ttg.SliceEncodingAttr):
        return (
            f"ttg.SliceEncodingAttr.get(dim={attr.dim}, parent={map_attr(attr.parent)})"
        )
    if isinstance(attr, ttg.SwizzledSharedEncodingAttr):
        return f"ttg.SwizzledSharedEncodingAttr.get(vec={attr.vec}, per_phase={attr.per_phase}, max_phase={attr.max_phase}, order={attr.order})"
    if isinstance(attr, ttg.DotOperandEncodingAttr):
        return f"ttg.DotOperandEncodingAttr.get(op_idx={attr.op_idx}, parent={map_attr(attr.parent)})"
    if isinstance(attr, ttg.SharedMemorySpaceAttr):
        return f"ttg.SharedMemorySpaceAttr.get()"
    if isinstance(attr, ttg.SharedMemorySpaceAttr):
        return f"ttg.SharedMemorySpaceAttr.get()"
    if isinstance(attr, amdgpu.OpIdxAttr):
        return f"amdgpu.OpIdxAttr.get({attr.value})"
    return f"Attribute.parse('{attr}')"


def map_type(type):
    type = type.maybe_downcast()
    if isinstance(type, (IntegerType, F16Type, F32Type, F64Type)):
        if type.width == 1:
            return f"T.bool()"
        return f"T.{type}()"
    if isinstance(type, RankedTensorType):
        encoding = "None"
        if type.encoding is not None:
            encoding = map_attr(type.encoding)
        return f"T.tensor({', '.join(map(str, type.shape))}, {map_type(type.element_type)}, encoding={encoding})"
    if isinstance(type, PointerType):
        return f"ttpp.ptr({map_type(type.pointee_type)}, {type.address_space})"
    if isinstance(type, ttg.MemDescType):
        return f"ttg.MemDescType.get(shape={type.shape}, element_type={map_type(type.element_type)}, encoding={map_attr(type.encoding)}, memory_space={map_attr(type.memory_space)}, mutable_memory={type.mutable_memory}, alloc_shape={type.alloc_shape})"
    return f"Type.parse('{type}')"


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


opidx_counter = 0


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

    attrs = {attr.name: attr.attr for attr in opview.attributes}
    op_idx_owner_name = None
    if "OpIdx" in attrs:
        global opidx_counter
        if len(opview.results):
            assert len(opview.results) == 1
            op_idx_owner_name = f"{normalize_ssa(opview.results[0])}"
        else:
            op_idx_owner_name = f"{name}_{opidx_counter}"
            print(op_idx_owner_name, end=" = ", file=OUTPUT_BUF)
        opidx_counter += 1

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

    if op_idx_owner_name is not None:
        if len(results):
            owner = f"{op_idx_owner_name}.owner"
        else:
            owner = f"{op_idx_owner_name}"
        print(
            "    " * indent
            + f"{owner}.attributes['OpIdx'] = amdgpu.OpIdxAttr.get({attrs['OpIdx'].value})",
            file=OUTPUT_BUF,
        )


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


def print_scf_if(if_op: scf.IfOp):
    assert len(if_op.results) == 1
    res = if_op.results[0]
    res_name = normalize_ssa(res)
    global indent

    def print_yield_as_return(yield_op: scf.YieldOp):
        opers = [normalize_ssa(a) for a in yield_op.operands]
        print(
            ("    " * indent) + f"return {', '.join(opers)}",
            file=OUTPUT_BUF,
        )

    print(
        textwrap.indent(
            textwrap.dedent(
                f"""\
                    @ext.scf.if_({normalize_ssa(if_op.condition)}, results=[{map_type(res.type)}])
                    def {res_name}():\
                """
            ),
            "    " * indent,
        ),
        file=OUTPUT_BUF,
    )
    indent += 1
    for bodyop in if_op.thenRegion.blocks[0].operations:
        if isinstance(bodyop, scf.YieldOp):
            print_yield_as_return(bodyop)
        else:
            bodyop.walk(generic_print_walk_callback, WalkOrder.PRE_ORDER)
    indent -= 1
    print(
        textwrap.indent(
            textwrap.dedent(
                f"""\
                    @ext.scf.else_({res_name})
                    def {res_name}_else():\
                """,
            ),
            "    " * indent,
        ),
        file=OUTPUT_BUF,
    )
    indent += 1
    for bodyop in if_op.elseRegion.blocks[0].operations:
        if isinstance(bodyop, scf.YieldOp):
            print_yield_as_return(bodyop)
        else:
            bodyop.walk(generic_print_walk_callback, WalkOrder.PRE_ORDER)
    indent -= 1


def generic_print_walk_callback(op):
    opview = op.opview
    if isinstance(opview, builtin.ModuleOp):
        for attr in opview.attributes:
            print(
                f"ctx.module.operation.attributes['{attr.name}'] = Attribute.parse('{(attr.attr)}')",
                file=OUTPUT_BUF,
            )
        return WalkResult.ADVANCE

    if isinstance(opview, tt.FuncOp):
        print("", file=OUTPUT_BUF)
        print_tt_func_op(opview)
    elif isinstance(opview, scf.ForOp):
        print_scf_for(opview)
    elif isinstance(opview, arith.ConstantOp):
        print_arith_constant(opview)
    elif isinstance(opview, scf.IfOp):
        print_scf_if(opview)
        return WalkResult.SKIP
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
from triton_mlir import types as _types
from triton_mlir.extras.context import RAIIMLIRContextModule
from triton_mlir.dialects import tt as ttpp, ttg, scf, llvm, _tt_ops_gen as tt, amdgpu
from triton_mlir.dialects.arith import IntegerOverflowFlags
from triton_mlir.ir import ArrayAttr, Type, Attribute
from triton_mlir.extras.dialects.ext import arith
from triton_mlir.extras.dialects import ext
import triton_mlir.extras.dialects.ext.scf
from triton_mlir.extras import types as T

ctx = RAIIMLIRContextModule()
"""


def print_prolog():
    print(PROLOG, file=OUTPUT_BUF)


EPILOG = """
matmul_kernel.emit()
ctx.module.operation.verify()

def mod_str():
    return str(ctx.module)
"""


def print_epilog():
    print(EPILOG, file=OUTPUT_BUF)


HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))


def test_smoke(ctx, backend):
    with open(HERE / "matmul_kernel.mlir") as src:
        matmul_src = src.read()

    mod = ctx.module.parse(matmul_src)
    assert mod.operation.verify()

    triton_mod = unwrap_c_module_op(mod.operation)
    hsaco = backend.compile(triton_mod, {"arch": backend.target.arch})
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

    import e2e_matmul

    assert str(e2e_matmul.mod_str()).strip() == matmul_src.strip()


def test_round_trip_ttg(ctx):
    with open(HERE / "matmul_kernel.ttgir") as src:
        matmul_src = src.read()

    mod = Module.parse(matmul_src)
    # mod.operation.print(use_local_scope=True)
    assert mod.operation.verify()

    global OUTPUT_BUF
    with open(HERE / "e2e_matmul_ttg.py", "w") as OUTPUT_BUF:
        print_prolog()
        mod.operation.walk(generic_print_walk_callback, WalkOrder.PRE_ORDER)
        print_epilog()

    import e2e_matmul_ttg

    assert str(e2e_matmul_ttg.mod_str()).strip() == matmul_src.strip()


def test_compile(ctx, backend):
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
    hsaco = backend.compile(triton_mod, {"arch": backend.target.arch})
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
def test_run_ttg(ctx, backend):
    from hip import hip

    with open(HERE / "matmul_kernel.ttgir") as src:
        matmul_src = src.read()

    mod = Module.parse(matmul_src)
    assert mod.operation.verify()

    global OUTPUT_BUF
    with open(HERE / "e2e_matmul_ttg.py", "w") as OUTPUT_BUF:
        print_prolog()
        mod.operation.walk(generic_print_walk_callback, WalkOrder.PRE_ORDER)
        print_epilog()

    # noinspection PyUnresolvedReferences
    import e2e_matmul_ttg

    props = hip.hipDeviceProp_t()
    hip_check(hip.hipGetDeviceProperties(props, 0))
    arch = props.gcnArchName.decode()
    options = backend.parse_options(
        {"arch": arch, "waves_per_eu": 8, "num_warps": 4, "num_stages": 2}
    )

    mod = ctx.module.parse(e2e_matmul_ttg.mod_str())
    mod.operation.attributes["ttg.target"] = StringAttr.get(f"hip:{arch}")
    assert mod.operation.verify()

    triton_mod = unwrap_c_module_op(mod.operation)
    hsaco, metadata = backend.compile(
        triton_mod,
        options=options,
        ttir=False,
        ttgir=False,
        dump_ir=True,
        ir_dump_dir=Path(__file__).parent / "e2e_matmul_ttg_dump",
        dump_file_prefix="0",
    )

    module = hip_check(hip.hipModuleLoadData(hsaco))
    function = hip_check(
        hip.hipModuleGetFunction(module, metadata["name"].encode())
    ).as_c_void_p()

    # kernel launch

    M, K, N = 128, 128, 128
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


autotune_configs = [
    {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 16,
        "GROUP_SIZE_M": 1,
        "WAVES_PER_EU": 2,
        "NUM_WARPS": 4,
    },
    {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 16,
        "GROUP_SIZE_M": 4,
        "WAVES_PER_EU": 2,
        "NUM_WARPS": 4,
    },
    {
        "BLOCK_SIZE_M": 256,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 16,
        "GROUP_SIZE_M": 4,
        "WAVES_PER_EU": 2,
        "NUM_WARPS": 8,
    },
    {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 1,
        "WAVES_PER_EU": 2,
        "NUM_WARPS": 8,
    },
    {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
        "WAVES_PER_EU": 3,
        "NUM_WARPS": 4,
    },
    {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 1,
        "WAVES_PER_EU": 8,
        "NUM_WARPS": 4,
    },
    {
        "BLOCK_SIZE_M": 8,
        "BLOCK_SIZE_N": 8,
        "BLOCK_SIZE_K": 4,
        "GROUP_SIZE_M": 1,
        "WAVES_PER_EU": 8,
        "NUM_WARPS": 4,
    },
]


@pytest.mark.skipif(hip_bindings_not_installed(), reason="hip not installed")
@pytest.mark.parametrize("autotune_config", autotune_configs)
def test_inline_mod(ctx, backend, autotune_config):
    from hip import hip

    M, K, N = 512, 512, 512
    BS_M = autotune_config["BLOCK_SIZE_M"]
    BS_N = autotune_config["BLOCK_SIZE_N"]
    BS_K = autotune_config["BLOCK_SIZE_K"]
    GS_M = autotune_config["GROUP_SIZE_M"]
    WAVES_PER_EU = autotune_config["WAVES_PER_EU"]
    NUM_WARPS = autotune_config["NUM_WARPS"]

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
        BS_M_ = arith.constant(BS_M, T.i32)
        BS_N_ = arith.constant(BS_N, T.i32)
        BS_K_ = arith.constant(BS_K, T.i32)
        GS_M_ = arith.constant(GS_M, T.i32)

        pid = get_program_id(axis=0)

        num_pid_m = arith.ceildivsi(M, BS_M_)
        num_pid_n = arith.ceildivsi(N, BS_N_)
        num_pid_in_group = GS_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GS_M
        group_size_m = arith.minsi(num_pid_m - first_pid_m, GS_M_)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        v15 = splat(pid_m * BS_M, (BS_M,)) + make_range(start=0, end=BS_M)
        offs_am = v15 % splat(M, (BS_M,))

        v20 = splat(pid_n * BS_N, (BS_N,)) + make_range(start=0, end=BS_N)
        offs_bn = v20 % splat(N, (BS_N,))

        offs_k = make_range(start=0, end=BS_K)

        v26 = expand_dims(offs_am, axis=1) * splat(stride_am, (BS_M, 1))
        v29 = expand_dims(offs_k, axis=0) * splat(stride_ak, (1, BS_K))
        v32 = broadcast(v26, (BS_M, BS_K)) + broadcast(v29, (BS_M, BS_K))
        v33 = splat(a_ptr, (BS_M, BS_K))
        a_ptrs = addptr(v33, offset=v32)

        v37 = expand_dims(offs_k, axis=1) * splat(stride_bk, (BS_K, 1))
        v40 = expand_dims(offs_bn, axis=0) * splat(stride_bn, (1, BS_N))
        v43 = broadcast(v37, (BS_K, BS_N)) + broadcast(v40, (BS_K, BS_N))
        v44 = splat(b_ptr, (BS_K, BS_N))
        b_ptrs = addptr(v44, offset=v43)

        a_ptr_incr = splat(stride_ak * BS_K, (BS_M, BS_K))
        b_ptr_incr = splat(stride_bk * BS_K, (BS_K, BS_N))

        accum = arith.constant(np.full([BS_M, BS_N], 0.0, np.float32))
        cst_0 = arith.constant(np.full([BS_K, BS_N], 0.0, np.float32))
        cst_1 = arith.constant(np.full([BS_M, BS_K], 0.0, np.float32))
        stop = arith.ceildivsi(K, BS_K_)
        for k, [accum, a_ptrs, b_ptrs], _ in scf.range_(
            0, stop, 1, iter_args=[accum, a_ptrs, b_ptrs]
        ):
            a_mask = broadcast(
                expand_dims(offs_k, axis=0) < splat(K - k * BS_K, (1, BS_K)),
                (BS_M, BS_K),
            )
            a = load(a_ptrs, mask=a_mask, other=cst_1)

            b_mask = broadcast(
                expand_dims(offs_k, axis=1) < splat(K - k * BS_K, (BS_K, 1)),
                (BS_K, BS_N),
            )
            b = load(b_ptrs, mask=b_mask, other=cst_0)

            accum = dot(a, b, c=accum)
            a_ptrs = addptr(a_ptrs, offset=a_ptr_incr)
            b_ptrs = addptr(b_ptrs, offset=b_ptr_incr)

            # these are the results
            c, _a_ptrs, _b_ptrs = scf.yield_(accum, a_ptrs, b_ptrs)

        offs_cm = expand_dims(v15, axis=1)

        c_ptr = splat(c_ptr, (BS_M, 1))
        c_ptr = addptr(c_ptr, offset=splat(stride_cm, (BS_M, 1)) * offs_cm)
        c_ptr = broadcast(c_ptr, (BS_M, BS_N))

        offs_cn = expand_dims(v20, axis=0)
        v60 = splat(stride_cn, (1, BS_N)) * offs_cn
        c_ptrs = addptr(c_ptr, offset=broadcast(v60, (BS_M, BS_N)))

        v68 = broadcast(offs_cm < splat(M, (BS_M, 1)), (BS_M, BS_N))
        v69 = broadcast(offs_cn < splat(N, (1, BS_N)), (BS_M, BS_N))
        c_mask = v68 & v69

        store(c_ptrs, value=c, mask=c_mask)

        tt.return_(srcs=[])

    matmul_kernel_2.emit()
    assert ctx.module.operation.verify()
    triton_mod = unwrap_c_module_op(ctx.module.operation)

    props = hip.hipDeviceProp_t()
    hip_check(hip.hipGetDeviceProperties(props, 0))
    arch = props.gcnArchName.decode()
    options = backend.parse_options(
        {"arch": arch, "waves_per_eu": WAVES_PER_EU, "num_warps": NUM_WARPS}
    )

    hsaco, metadata = backend.compile(
        triton_mod,
        options=options,
        dump_ir=True,
        ir_dump_dir=Path(__file__).parent / f"matmul_kernel_2_group_size_{GS_M}",
    )

    module = hip_check(hip.hipModuleLoadData(hsaco))
    function = hip_check(
        hip.hipModuleGetFunction(module, metadata["name"].encode())
    ).as_c_void_p()

    # kernel launch

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
    gridX = math.ceil(M / BS_M) * math.ceil(N / BS_N)
    gridY = 1
    gridZ = 1
    shared_memory = metadata["shared"]

    launch(
        function,
        gridX,
        gridY,
        gridZ,
        stream,
        options.warp_size,
        options.num_warps,
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

    correct = a_h @ b_h
    assert np.allclose(c_h, -3.0)
    assert not np.allclose(correct, c_h)
    hip_check(
        hip.hipMemcpy(c_h, c_d, c_num_bytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost)
    )
    assert np.allclose(c_h, correct)


@pytest.mark.skipif(hip_bindings_not_installed(), reason="hip not installed")
@pytest.mark.parametrize("autotune_config", autotune_configs[:1])
def test_inline_mod_ttpp(ctx, backend, autotune_config):
    from hip import hip

    M, K, N = 512, 512, 512
    BLOCK_SIZE_M = autotune_config["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = autotune_config["BLOCK_SIZE_N"]
    BLOCK_SIZE_K = autotune_config["BLOCK_SIZE_K"]
    GROUP_SIZE_M = autotune_config["GROUP_SIZE_M"]
    WAVES_PER_EU = autotune_config["WAVES_PER_EU"]
    NUM_WARPS = autotune_config["NUM_WARPS"]

    @tt.jit(
        arg_attrs=ArrayAttr.parse(
            "[{tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, {}, {}, {}, {}, {}, {}, {}, {}, {}]"
        ),
        noinline=False,
        sym_name="matmul_kernel_3",
        sym_visibility="public",
    )
    def matmul_kernel_3(
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
        BLOCK_SIZE_M_ = arith.constant(BLOCK_SIZE_M, T.i32)
        BLOCK_SIZE_N_ = arith.constant(BLOCK_SIZE_N, T.i32)
        BLOCK_SIZE_K_ = arith.constant(BLOCK_SIZE_K, T.i32)
        GROUP_SIZE_M_ = arith.constant(GROUP_SIZE_M, T.i32)

        pid = get_program_id(axis=0)

        num_pid_m = arith.ceildivsi(M, BLOCK_SIZE_M_)
        num_pid_n = arith.ceildivsi(N, BLOCK_SIZE_N_)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = arith.minsi(num_pid_m - first_pid_m, GROUP_SIZE_M_)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_am = (pid_m * BLOCK_SIZE_M + make_range(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + make_range(0, BLOCK_SIZE_N)) % N
        offs_k = make_range(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        accum = arith.constant(np.full([BLOCK_SIZE_M, BLOCK_SIZE_N], 0.0, np.float32))
        stop = arith.ceildivsi(K, BLOCK_SIZE_K_)
        for k, [accum, a_ptrs, b_ptrs], _ in scf.range_(
            0, stop, 1, iter_args=[accum, a_ptrs, b_ptrs]
        ):
            a = load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            accum += dot(a, b)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
            accum, *_ = scf.yield_(accum, a_ptrs, b_ptrs)

        c = accum

        offs_cm = pid_m * BLOCK_SIZE_M + make_range(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + make_range(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        store(c_ptrs, c, mask=c_mask)

        tt.return_(srcs=[])

    matmul_kernel_3.emit()
    assert ctx.module.operation.verify()
    triton_mod = unwrap_c_module_op(ctx.module.operation)

    props = hip.hipDeviceProp_t()
    hip_check(hip.hipGetDeviceProperties(props, 0))
    arch = props.gcnArchName.decode()
    options = backend.parse_options(
        {"arch": arch, "waves_per_eu": WAVES_PER_EU, "num_warps": NUM_WARPS}
    )

    hsaco, metadata = backend.compile(
        triton_mod,
        options=options,
        dump_ir=True,
        ir_dump_dir=Path(__file__).parent
        / f"matmul_kernel_3_group_size_{GROUP_SIZE_M}",
    )

    module = hip_check(hip.hipModuleLoadData(hsaco))
    function = hip_check(
        hip.hipModuleGetFunction(module, metadata["name"].encode())
    ).as_c_void_p()

    # kernel launch

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
    shared_memory = metadata["shared"]

    launch(
        function,
        gridX,
        gridY,
        gridZ,
        stream,
        options.warp_size,
        options.num_warps,
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

    correct = a_h @ b_h
    assert np.allclose(c_h, -3.0)
    assert not np.allclose(correct, c_h)
    hip_check(
        hip.hipMemcpy(c_h, c_d, c_num_bytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost)
    )
    assert np.allclose(c_h, correct)


@pytest.mark.skipif(hip_bindings_not_installed(), reason="hip not installed")
@pytest.mark.parametrize("autotune_config", [autotune_configs[-1]])
def test_inline_mod_ttg(ctx, backend, autotune_config):
    from hip import hip

    M, K, N = 16, 16, 16
    BS_M = autotune_config["BLOCK_SIZE_M"]
    BS_N = autotune_config["BLOCK_SIZE_N"]
    BS_K = autotune_config["BLOCK_SIZE_K"]
    GS_M = autotune_config["GROUP_SIZE_M"]
    WAVES_PER_EU = autotune_config["WAVES_PER_EU"]
    NUM_WARPS = autotune_config["NUM_WARPS"]

    ctx.module.operation.attributes["ttg.num-ctas"] = IntegerAttr.get(T.i32, 1)
    ctx.module.operation.attributes["ttg.num-warps"] = IntegerAttr.get(T.i32, 4)
    ctx.module.operation.attributes["ttg.threads-per-warp"] = IntegerAttr.get(T.i32, 32)

    blocked = ttg.BlockedEncodingAttr.get(
        size_per_thread=[1, 1],
        threads_per_warp__=[4, 8],
        warps_per_cta__=[4, 1],
        order=[1, 0],
    )
    blocked1 = ttg.BlockedEncodingAttr.get(
        size_per_thread=[1, 1],
        threads_per_warp__=[4, 8],
        warps_per_cta__=[1, 4],
        order=[0, 1],
    )
    blocked2 = ttg.BlockedEncodingAttr.get(
        size_per_thread=[1, 1],
        threads_per_warp__=[8, 4],
        warps_per_cta__=[1, 4],
        order=[0, 1],
    )
    sliced = ttg.SliceEncodingAttr.get(dim=1, parent=blocked2)
    sliced1 = ttg.SliceEncodingAttr.get(dim=0, parent=blocked2)
    sliced2 = ttg.SliceEncodingAttr.get(dim=1, parent=blocked1)
    sliced3 = ttg.SliceEncodingAttr.get(dim=0, parent=blocked1)
    memdesc = ttg.MemDescType.get(
        shape=[1, 8, 4],
        element_type=T.f32,
        encoding=ttg.SwizzledSharedEncodingAttr.get(
            vec=1, per_phase=1, max_phase=1, order=[0, 1]
        ),
        memory_space=ttg.SharedMemorySpaceAttr.get(),
        mutable_memory=True,
        alloc_shape=[1, 8, 4],
    )
    memdesc1 = ttg.MemDescType.get(
        shape=[1, 4, 8],
        element_type=T.f32,
        encoding=ttg.SwizzledSharedEncodingAttr.get(
            vec=1, per_phase=1, max_phase=1, order=[0, 1]
        ),
        memory_space=ttg.SharedMemorySpaceAttr.get(),
        mutable_memory=True,
        alloc_shape=[1, 4, 8],
    )
    memdesc2 = ttg.MemDescType.get(
        shape=[8, 4],
        element_type=T.f32,
        encoding=ttg.SwizzledSharedEncodingAttr.get(
            vec=1, per_phase=1, max_phase=1, order=[0, 1]
        ),
        memory_space=ttg.SharedMemorySpaceAttr.get(),
        mutable_memory=True,
        alloc_shape=[8, 4],
    )
    memdesc3 = ttg.MemDescType.get(
        shape=[4, 8],
        element_type=T.f32,
        encoding=ttg.SwizzledSharedEncodingAttr.get(
            vec=1, per_phase=1, max_phase=1, order=[0, 1]
        ),
        memory_space=ttg.SharedMemorySpaceAttr.get(),
        mutable_memory=True,
        alloc_shape=[4, 8],
    )

    @tt.jit(
        arg_attrs=ArrayAttr.parse(
            "[{tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {}, {}, {}, {}, {}, {}, {}, {}, {}]"
        ),
        function_type=T.function(
            inputs=[
                tt.ptr(T.f32),
                tt.ptr(T.f32),
                tt.ptr(T.f32),
                T.i32,
                T.i32,
                T.i32,
                T.i32,
                T.i32,
                T.i32,
                T.i32,
                T.i32,
                T.i32,
            ],
            results=[],
        ),
        noinline=False,
        sym_name="matmul_kernel",
        sym_visibility="public",
    )
    @canonicalize(using=scf.canonicalizer)
    def matmul_kernel_3(
        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11
    ):
        cst = arith.constant(
            np.full([8, 8], 0.0, np.float32), T.tensor(8, 8, T.f32, encoding=blocked)
        )
        cst_0 = arith.constant(
            np.full([4, 8], 0.0, np.float32), T.tensor(4, 8, T.f32, encoding=blocked1)
        )
        cst_1 = arith.constant(
            np.full([8, 4], 0.0, np.float32), T.tensor(8, 4, T.f32, encoding=blocked2)
        )
        c0_i32 = arith.constant(0, T.i32)
        c1_i32 = arith.constant(1, T.i32)
        c3_i32 = arith.constant(3, T.i32)
        c4_i32 = arith.constant(4, T.i32)
        c7_i32 = arith.constant(7, T.i32)
        c8_i32 = arith.constant(8, T.i32)
        v0 = tt.splat(result=T.tensor(8, 4, tt.ptr(T.f32), encoding=blocked2), src=arg0)
        v1 = tt.get_program_id(axis=0)
        v2 = arith.addi(arg4, c7_i32)
        v3 = arith.divsi(v2, c8_i32)
        v4 = arith.divsi(v1, v3)
        v5 = arith.remsi(v1, v3)
        v6 = arith.addi(arg3, c7_i32)
        v7 = arith.divsi(v6, c8_i32)
        v8 = arith.subi(v7, v4)
        v9 = arith.minsi(v8, c1_i32)
        v10 = arith.remsi(v5, v9)
        v11 = arith.addi(v4, v10)
        v12 = arith.muli(v11, c8_i32)
        v13 = tt.splat(result=T.tensor(8, T.i32, encoding=sliced), src=v12)
        v14 = tt.make_range(result=T.tensor(8, T.i32, encoding=sliced), start=0, end=8)
        v15 = arith.addi(v13, v14)
        v16 = tt.splat(result=T.tensor(8, T.i32, encoding=sliced), src=arg3)
        v17 = arith.remsi(v15, v16)
        v18 = tt.expand_dims(src=v17, axis=1)
        v19 = tt.splat(result=T.tensor(8, 1, T.i32, encoding=blocked2), src=arg6)
        v20 = arith.muli(v18, v19)
        v21 = tt.broadcast(result=T.tensor(8, 4, T.i32, encoding=blocked2), src=v20)
        v22 = tt.make_range(result=T.tensor(4, T.i32, encoding=sliced1), start=0, end=4)
        v23 = tt.expand_dims(src=v22, axis=0)
        v24 = tt.splat(result=T.tensor(1, 4, T.i32, encoding=blocked2), src=arg7)
        v25 = arith.muli(v23, v24)
        v26 = tt.broadcast(result=T.tensor(8, 4, T.i32, encoding=blocked2), src=v25)
        v27 = arith.addi(v21, v26)
        v28 = tt.addptr(
            result=T.tensor(8, 4, tt.ptr(T.f32), encoding=blocked2), ptr=v0, offset=v27
        )
        v29 = arith.addi(arg5, c3_i32)
        v30 = arith.divsi(v29, c4_i32)
        v31 = arith.cmpi("sgt", v30, c0_i32)
        v32 = tt.splat(result=T.tensor(8, 4, T.i1, encoding=blocked2), src=v31)
        v33 = tt.splat(result=T.tensor(1, 4, T.i32, encoding=blocked2), src=arg5)
        v34 = arith.cmpi("slt", v23, v33)
        v35 = tt.broadcast(result=T.tensor(8, 4, T.i1, encoding=blocked2), src=v34)
        v36 = arith.andi(v32, v35)
        v37 = tt.load(ptr=v28, mask=v36, other=cst_1)
        v37.owner.attributes["OpIdx"] = amdgpu.OpIdxAttr.get(0)
        v38 = tt.splat(
            result=T.tensor(4, 8, tt.ptr(T.f32), encoding=blocked1), src=arg1
        )
        v39 = tt.make_range(result=T.tensor(4, T.i32, encoding=sliced2), start=0, end=4)
        v40 = tt.expand_dims(src=v39, axis=1)
        v41 = tt.splat(result=T.tensor(4, 1, T.i32, encoding=blocked1), src=arg8)
        v42 = arith.muli(v40, v41)
        v43 = tt.broadcast(result=T.tensor(4, 8, T.i32, encoding=blocked1), src=v42)
        v44 = arith.divsi(v5, v9)
        v45 = arith.muli(v44, c8_i32)
        v46 = tt.splat(result=T.tensor(8, T.i32, encoding=sliced3), src=v45)
        v47 = tt.make_range(result=T.tensor(8, T.i32, encoding=sliced3), start=0, end=8)
        v48 = arith.addi(v46, v47)
        v49 = tt.splat(result=T.tensor(8, T.i32, encoding=sliced3), src=arg4)
        v50 = arith.remsi(v48, v49)
        v51 = tt.expand_dims(src=v50, axis=0)
        v52 = tt.splat(result=T.tensor(1, 8, T.i32, encoding=blocked1), src=arg9)
        v53 = arith.muli(v51, v52)
        v54 = tt.broadcast(result=T.tensor(4, 8, T.i32, encoding=blocked1), src=v53)
        v55 = arith.addi(v43, v54)
        v56 = tt.addptr(
            result=T.tensor(4, 8, tt.ptr(T.f32), encoding=blocked1), ptr=v38, offset=v55
        )
        v57 = tt.splat(result=T.tensor(4, 8, T.i1, encoding=blocked1), src=v31)
        v58 = tt.splat(result=T.tensor(4, 1, T.i32, encoding=blocked1), src=arg5)
        v59 = arith.cmpi("slt", v40, v58)
        v60 = tt.broadcast(result=T.tensor(4, 8, T.i1, encoding=blocked1), src=v59)
        v61 = arith.andi(v57, v60)
        v62 = tt.load(ptr=v56, mask=v61, other=cst_0)
        v62.owner.attributes["OpIdx"] = amdgpu.OpIdxAttr.get(1)
        v63 = tt.make_range(result=T.tensor(8, T.i32, encoding=sliced1), start=0, end=8)
        v64 = tt.splat(result=T.tensor(8, T.i32, encoding=sliced1), src=v45)
        v65 = arith.addi(v64, v63)
        v66 = arith.muli(arg7, c4_i32)
        v67 = tt.splat(result=T.tensor(8, 4, T.i32, encoding=blocked2), src=v66)
        v68 = arith.muli(arg8, c4_i32)
        v69 = tt.splat(result=T.tensor(4, 8, T.i32, encoding=blocked1), src=v68)
        v70 = ttg.local_alloc(result=memdesc)
        v71 = ttg.local_alloc(result=memdesc1)
        v72 = ttg.memdesc_subview(
            result=memdesc2, src=v70, offsets=[c0_i32, c0_i32, c0_i32]
        )
        ttg.local_store_2 = ttg.local_store(src=v37, dst=v72)
        ttg.local_store_2.attributes["OpIdx"] = amdgpu.OpIdxAttr.get(0)
        v73 = ttg.memdesc_subview(
            result=memdesc3, src=v71, offsets=[c0_i32, c0_i32, c0_i32]
        )
        ttg.local_store_3 = ttg.local_store(src=v62, dst=v73)
        ttg.local_store_3.attributes["OpIdx"] = amdgpu.OpIdxAttr.get(1)
        v74 = arith.subi(v30, c1_i32)
        for (
            arg12,
            [arg13, arg14, arg15, arg16, arg17, arg18],
            [v75_0, v75_1, v75_2, v75_3, v75_4, v75_5],
        ) in scf.range_(
            c0_i32, v74, c1_i32, iter_args=[cst, v28, v56, c0_i32, v72, v73]
        ):
            v100 = tt.addptr(
                result=T.tensor(8, 4, tt.ptr(T.f32), encoding=blocked2),
                ptr=arg14,
                offset=v67,
            )
            v101 = arith.addi(arg12, c1_i32)
            v102 = arith.muli(v101, c4_i32)
            v103 = arith.subi(arg5, v102)
            v104 = tt.splat(result=T.tensor(1, 4, T.i32, encoding=blocked2), src=v103)
            v105 = arith.cmpi("slt", v23, v104)
            v106 = tt.broadcast(
                result=T.tensor(8, 4, T.i1, encoding=blocked2), src=v105
            )
            v107 = tt.load(ptr=v100, mask=v106, other=cst_1)
            v107.owner.attributes["OpIdx"] = amdgpu.OpIdxAttr.get(0)
            v108 = tt.addptr(
                result=T.tensor(4, 8, tt.ptr(T.f32), encoding=blocked1),
                ptr=arg15,
                offset=v69,
            )
            v109 = tt.splat(result=T.tensor(4, 1, T.i32, encoding=blocked1), src=v103)
            v110 = arith.cmpi("slt", v40, v109)
            v111 = tt.broadcast(
                result=T.tensor(4, 8, T.i1, encoding=blocked1), src=v110
            )
            v112 = tt.load(ptr=v108, mask=v111, other=cst_0)
            v112.owner.attributes["OpIdx"] = amdgpu.OpIdxAttr.get(1)
            v113 = ttg.local_load(
                result=T.tensor(
                    8,
                    4,
                    T.f32,
                    encoding=ttg.DotOperandEncodingAttr.get(op_idx=0, parent=blocked),
                ),
                src=arg17,
            )
            v114 = ttg.local_load(
                result=T.tensor(
                    4,
                    8,
                    T.f32,
                    encoding=ttg.DotOperandEncodingAttr.get(op_idx=1, parent=blocked),
                ),
                src=arg18,
            )
            v115 = tt.dot(a=v113, b=v114, c=arg13)
            v116 = arith.addi(arg16, c1_i32)
            v117 = arith.cmpi("slt", v116, c1_i32)
            v118 = arith.select(condition=v117, true_value=v116, false_value=c0_i32)
            v119 = ttg.memdesc_subview(
                result=memdesc2, src=v70, offsets=[v118, c0_i32, c0_i32]
            )
            ttg.local_store_6 = ttg.local_store(src=v107, dst=v119)
            ttg.local_store_6.attributes["OpIdx"] = amdgpu.OpIdxAttr.get(0)
            v120 = ttg.memdesc_subview(
                result=memdesc3, src=v71, offsets=[v118, c0_i32, c0_i32]
            )
            ttg.local_store_7 = ttg.local_store(src=v112, dst=v120)
            ttg.local_store_7.attributes["OpIdx"] = amdgpu.OpIdxAttr.get(1)

            yield v115, v100, v108, v118, v119, v120

        v76 = arith.cmpi("sge", v30, c1_i32)
        v77 = ttg.local_load(
            result=T.tensor(
                8,
                4,
                T.f32,
                encoding=ttg.DotOperandEncodingAttr.get(op_idx=0, parent=blocked),
            ),
            src=v75_4,
        )
        v78 = ttg.local_load(
            result=T.tensor(
                4,
                8,
                T.f32,
                encoding=ttg.DotOperandEncodingAttr.get(op_idx=1, parent=blocked),
            ),
            src=v75_5,
        )

        if v76:
            v100 = tt.dot(a=v77, b=v78, c=v75_0)
            v79 = yield v100
        else:
            v79 = yield v75_0

        v80 = arith.select(condition=v76, true_value=v79, false_value=v75_0)
        ttg.local_dealloc(src=v70)
        ttg.local_dealloc(src=v71)
        v81 = tt.expand_dims(src=v15, axis=1)
        v82 = tt.splat(result=T.tensor(8, 1, T.i32, encoding=blocked2), src=arg10)
        v83 = arith.muli(v82, v81)
        v84 = tt.splat(
            result=T.tensor(8, 1, tt.ptr(T.f32), encoding=blocked2), src=arg2
        )
        v85 = tt.addptr(
            result=T.tensor(8, 1, tt.ptr(T.f32), encoding=blocked2), ptr=v84, offset=v83
        )
        v86 = tt.expand_dims(src=v65, axis=0)
        v87 = tt.splat(result=T.tensor(1, 8, T.i32, encoding=blocked2), src=arg11)
        v88 = arith.muli(v87, v86)
        v89 = tt.broadcast(
            result=T.tensor(8, 8, tt.ptr(T.f32), encoding=blocked2), src=v85
        )
        v90 = tt.broadcast(result=T.tensor(8, 8, T.i32, encoding=blocked2), src=v88)
        v91 = tt.addptr(
            result=T.tensor(8, 8, tt.ptr(T.f32), encoding=blocked2), ptr=v89, offset=v90
        )
        v92 = tt.splat(result=T.tensor(8, 1, T.i32, encoding=blocked2), src=arg3)
        v93 = arith.cmpi("slt", v81, v92)
        v94 = tt.splat(result=T.tensor(1, 8, T.i32, encoding=blocked2), src=arg4)
        v95 = arith.cmpi("slt", v86, v94)
        v96 = tt.broadcast(result=T.tensor(8, 8, T.i1, encoding=blocked2), src=v93)
        v97 = tt.broadcast(result=T.tensor(8, 8, T.i1, encoding=blocked2), src=v95)
        v98 = arith.andi(v96, v97)
        v99 = ttg.convert_layout(
            result=T.tensor(8, 8, T.f32, encoding=blocked2), src=v80
        )
        tt.store(ptr=v91, value=v99, mask=v98)

        tt.return_(srcs=[])

    matmul_kernel_3.emit()
    ctx.module.operation.verify()
    triton_mod = unwrap_c_module_op(ctx.module.operation)

    props = hip.hipDeviceProp_t()
    hip_check(hip.hipGetDeviceProperties(props, 0))
    arch = props.gcnArchName.decode()
    options = backend.parse_options(
        {"arch": arch, "waves_per_eu": WAVES_PER_EU, "num_warps": NUM_WARPS}
    )

    hsaco, metadata = backend.compile(
        triton_mod,
        options=options,
        dump_ir=True,
        ir_dump_dir=Path(__file__).parent / f"matmul_kernel_3_group_size_{GS_M}",
    )

    module = hip_check(hip.hipModuleLoadData(hsaco))
    function = hip_check(
        hip.hipModuleGetFunction(module, metadata["name"].encode())
    ).as_c_void_p()

    # kernel launch

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
    gridX = math.ceil(M / BS_M) * math.ceil(N / BS_N)
    gridY = 1
    gridZ = 1
    shared_memory = metadata["shared"]

    launch(
        function,
        gridX,
        gridY,
        gridZ,
        stream,
        options.warp_size,
        options.num_warps,
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

    correct = a_h @ b_h
    assert np.allclose(c_h, -3.0)
    assert not np.allclose(correct, c_h)
    hip_check(
        hip.hipMemcpy(c_h, c_d, c_num_bytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost)
    )
    assert np.allclose(c_h, correct)


if __name__ == "__main__":
    backend = backend_()
    with mlir_mod_ctx(allow_unregistered_dialects=True) as ctx:
        test_smoke(ctx, backend)
    with mlir_mod_ctx(allow_unregistered_dialects=True) as ctx:
        test_round_trip(ctx)
    with mlir_mod_ctx(allow_unregistered_dialects=True) as ctx:
        test_round_trip_ttg(ctx)
    with mlir_mod_ctx(allow_unregistered_dialects=True) as ctx:
        test_compile(ctx, backend)
    with mlir_mod_ctx(allow_unregistered_dialects=True) as ctx:
        test_run(ctx, backend)
    with mlir_mod_ctx(allow_unregistered_dialects=True) as ctx:
        test_run_ttg(ctx, backend)
    for ac in autotune_configs:
        with mlir_mod_ctx(allow_unregistered_dialects=True) as ctx:
            test_inline_mod(ctx, backend, ac)
        with mlir_mod_ctx(allow_unregistered_dialects=True) as ctx:
            test_inline_mod_ttpp(ctx, backend, ac)
        with mlir_mod_ctx(allow_unregistered_dialects=True) as ctx:
            test_inline_mod_ttg(ctx, backend, ac)
