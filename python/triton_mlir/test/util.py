import ctypes
import inspect
import io
import re
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from triton_mlir.compiler import HIPBackend, make_backend, ENV_OR_DEFAULT_ARCH
from triton_mlir.compiler import unwrap_c_module_op
from triton_mlir.dialects import tt, scf, builtin, ttg, amdgpu, ttpp

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


def hip_check(call_result):
    from hip import hip

    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    return result


def hip_bindings_not_installed():
    try:
        from hip import hip

        props = hip.hipDeviceProp_t()
        hip_check(hip.hipGetDeviceProperties(props, 0))

        # don't skip
        return False

    except:
        # skip
        return True


def chip_check(status):
    from triton_mlir import chip

    if status != 0:
        raise RuntimeError(
            f"HIP Error {status}, {ctypes.string_at(chip.hipGetErrorString(status)).decode()}"
        )


def hip_synchronize():
    from hip import hip

    hip.hipDeviceSynchronize()


def launch_kernel(
    function,
    blocks_per_grid_x,
    blocks_per_grid_y,
    blocks_per_grid_z,
    threads_per_block_x,
    threads_per_block_y,
    threads_per_block_z,
    shared_memory,
    stream,
    *args,
):
    from triton_mlir import chip

    import hip
    from hip._util.types import DeviceArray

    params = [None] * len(args)
    addresses = [None] * len(args)
    for i, p in enumerate(args):
        if isinstance(p, DeviceArray):
            addresses[i] = params[i] = p.createRef().as_c_void_p()
        elif isinstance(p, int):
            params[i] = ctypes.c_int32(p)
            addresses[i] = ctypes.addressof(params[i])
        else:
            raise NotImplementedError(f"{p=} not supported with {p=}")

    c_args = (ctypes.c_void_p * len(addresses))(*addresses)
    function = ctypes.cast(function, chip.hipFunction_t)
    stream = ctypes.cast(stream, chip.hipStream_t)

    tstart = hip_check(hip.hip.hipEventCreate())
    tstop = hip_check(hip.hip.hipEventCreate())
    hip_check(hip.hip.hipEventRecord(tstart, None))

    r = chip.hipModuleLaunchKernel(
        function,
        blocks_per_grid_x,
        blocks_per_grid_y,
        blocks_per_grid_z,
        threads_per_block_x,
        threads_per_block_y,
        threads_per_block_z,
        shared_memory,
        stream,
        c_args,
        None,
    )

    hip_check(hip.hip.hipEventRecord(tstop, None))
    hip_check(hip.hip.hipEventSynchronize(tstop))
    time_compute = hip_check(hip.hip.hipEventElapsedTime(tstart, tstop))

    chip_check(r)

    return time_compute


def backend_() -> HIPBackend:
    try:
        from hip import hip

        props = hip.hipDeviceProp_t()
        hip_check(hip.hipGetDeviceProperties(props, 0))
        arch = props.gcnArchName.decode()
    except:
        arch = "gfx1150"
    warp_size = 32 if "gfx10" in arch or "gfx11" in arch or "gfx12" in arch else 64
    return make_backend(arch, warp_size)


backend = pytest.fixture(backend_)


def normalize_ssa(ssa: str | Value):
    if isinstance(ssa, Value):
        ssa = ssa.get_name(use_name_loc_as_prefix=True)
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
    if attr in ATTR_ALIASES:
        return ATTR_ALIASES[attr]
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
        return f"ttg.DotOperandEncodingAttr.get(op_idx={attr.op_idx}, parent={map_attr(attr.parent)}, k_width={attr.k_width})"
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
ATTR_ALIASES = {}


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
        # print_opview(opview, name=f"tt.return_")
        pass
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
from triton_mlir.dialects import tt as ttpp, ttg, scf, llvm, _tt_ops_gen as tt, amdgpu, rocdl
from triton_mlir.dialects.arith import IntegerOverflowFlags
from triton_mlir.ir import ArrayAttr, Type, Attribute
from triton_mlir.extras.dialects.ext import arith
from triton_mlir.extras.dialects import ext
import triton_mlir.extras.dialects.ext.scf
from triton_mlir.extras import types as T

ctx = RAIIMLIRContextModule()
"""


def print_attr_alias(attr_line: str):
    print(attr_line)
    alias_name, attr_str = attr_line.split(" = ", maxsplit=1)
    assert alias_name.startswith("#")
    alias_name = alias_name[1:]
    attr = Attribute.parse(attr_str)
    print(f"{alias_name} = {map_attr(attr)}", file=OUTPUT_BUF)
    ATTR_ALIASES[attr] = alias_name


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
