from textwrap import dedent

import pytest
from triton_mlir._mlir_libs._mlir.passmanager import PassManager
from triton_mlir.dialects import tt, ttpp
from triton_mlir.types import T

# this needs to be below the triton_mlir_bindings
from triton_mlir.extras.dialects.ext import arith, tensor

# noinspection PyUnresolvedReferences
from triton_mlir.extras.dialects.ext.scf import range_, yield_

# noinspection PyUnresolvedReferences
from triton_mlir.extras.testing import mlir_ctx as ctx, filecheck, MLIRContext

pytest.mark.usefixtures("ctx")


def test_tensor_arithmetic(ctx: MLIRContext):
    # number of elements must be power of 2
    t_p_f32 = tensor.empty(16, 16, +T.float32)
    t_i32 = tensor.empty(16, 16, T.int32)
    res = t_p_f32 + t_i32

    ctx.module.operation.verify()
    filecheck(
        dedent(
            """\
    module {
      %0 = tensor.empty() : tensor<16x16x!tt.ptr<f32>>
      %1 = tensor.empty() : tensor<16x16xi32>
      %2 = tt.addptr %0, %1 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    }
    """
        ),
        ctx.module,
    )


def test_vadd(ctx: MLIRContext):
    BLOCK_SIZE = 64

    @tt.jit
    def kernel_0123(
        x_ptr: +T.float32,
        y_ptr: +T.float32,
        output_ptr: +T.float32,
        n_elements: T.int32,
    ):
        pid = tt.get_program_id(axis=tt.ProgramIDDim.X)
        block_size = arith.constant(BLOCK_SIZE, T.int32)
        block_start = pid * block_size
        offsets = block_start + tt.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tt.load(x_ptr + offsets, mask)
        y = tt.load(y_ptr + offsets, mask)

        output = x + y
        tt.store(output_ptr + offsets, output, mask)
        tt.return_(srcs=[])

    kernel_0123.emit()

    ctx.module.operation.verify()
    correct = dedent(
        """\
    module {
      tt.func @kernel_0123(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) {
        %0 = tt.get_program_id x : i32
        %c64_i32 = arith.constant 64 : i32
        %1 = arith.muli %0, %c64_i32 : i32
        %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
        %3 = tt.splat %1 : i32 -> tensor<64xi32>
        %4 = arith.addi %3, %2 : tensor<64xi32>
        %5 = tt.splat %arg3 : i32 -> tensor<64xi32>
        %6 = arith.cmpi slt, %4, %5 : tensor<64xi32>
        %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
        %8 = tt.addptr %7, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
        %9 = tt.load %8, %6 : tensor<64x!tt.ptr<f32>>
        %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
        %11 = tt.addptr %10, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
        %12 = tt.load %11, %6 : tensor<64x!tt.ptr<f32>>
        %13 = arith.addf %9, %12 : tensor<64xf32>
        %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
        %15 = tt.addptr %14, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
        tt.store %15, %13, %6 : tensor<64x!tt.ptr<f32>>
        tt.return
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_vadd_set_get(ctx: MLIRContext):
    BLOCK_SIZE = 64

    @tt.jit
    def kernel_0123(
        x_ptr: +T.float32,
        y_ptr: +T.float32,
        output_ptr: +T.float32,
        n_elements: T.int32,
    ):
        pid = tt.get_program_id(axis=tt.ProgramIDDim.X)
        block_size = arith.constant(BLOCK_SIZE, T.int32)
        block_start = pid * block_size
        offsets = block_start + tt.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x_ptr += offsets
        x = x_ptr[mask]
        y_ptr += offsets
        y = y_ptr[mask]

        output = x + y
        output_ptr += offsets
        output_ptr[mask] = output
        tt.return_(srcs=[])

    kernel_0123.emit()

    ctx.module.operation.verify()
    correct = dedent(
        """\
    module {
      tt.func @kernel_0123(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) {
        %0 = tt.get_program_id x : i32
        %c64_i32 = arith.constant 64 : i32
        %1 = arith.muli %0, %c64_i32 : i32
        %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
        %3 = tt.splat %1 : i32 -> tensor<64xi32>
        %4 = arith.addi %3, %2 : tensor<64xi32>
        %5 = tt.splat %arg3 : i32 -> tensor<64xi32>
        %6 = arith.cmpi slt, %4, %5 : tensor<64xi32>
        %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
        %8 = tt.addptr %7, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
        %9 = tt.load %8, %6 : tensor<64x!tt.ptr<f32>>
        %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
        %11 = tt.addptr %10, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
        %12 = tt.load %11, %6 : tensor<64x!tt.ptr<f32>>
        %13 = arith.addf %9, %12 : tensor<64xf32>
        %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
        %15 = tt.addptr %14, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
        tt.store %15, %13, %6 : tensor<64x!tt.ptr<f32>>
        tt.return
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_matmul(ctx: MLIRContext):
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    GROUP_SIZE_M = 2

    @tt.jit
    def matmul_kernel(
        a_ptr: +T.float32,
        b_ptr: +T.float32,
        c_ptr: +T.float32,
        M: T.int32,
        N: T.int32,
        K: T.int32,
        stride_am: T.int32,
        stride_ak: T.int32,
        stride_bk: T.int32,
        stride_bn: T.int32,
        stride_cm: T.int32,
        stride_cn: T.int32,
    ):
        pid = tt.get_program_id(axis=tt.ProgramIDDim.X)
        num_pid_m = tt.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tt.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        # offs_am = (pid_m * BLOCK_SIZE_M + tt.arange(0, BLOCK_SIZE_M)) % M
        # offs_bn = (pid_n * BLOCK_SIZE_N + tt.arange(0, BLOCK_SIZE_N)) % N
        offs_am = pid_m * BLOCK_SIZE_M + tt.arange(0, BLOCK_SIZE_M)
        offs_bn = pid_n * BLOCK_SIZE_N + tt.arange(0, BLOCK_SIZE_N)
        offs_k = tt.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        accumulator = tt.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=T.float32)
        acc = accumulator

        r = tt.cdiv(K, BLOCK_SIZE_K)
        zero = arith.constant(0)
        one = arith.constant(1)
        # r = 1
        for k, (acc, aptrs, bptrs), results in range_(
            zero, r, one, iter_args=[accumulator, a_ptrs, b_ptrs]
        ):
            mask = offs_k[None, :] < K - k * BLOCK_SIZE_K
            a = tt.load(a_ptrs, mask=mask, other=0.0)
            mask = offs_k[:, None] < K - k * BLOCK_SIZE_K
            b = tt.load(b_ptrs, mask=mask, other=0.0)
            # TODO(max): the problem here is the _update_frame_vars upstream
            acc += tt.dot(a, b)
            aptrs += BLOCK_SIZE_K * stride_ak
            bptrs += BLOCK_SIZE_K * stride_bk
            acc, *_ = yield_(acc, aptrs, bptrs)

        c = acc

        offs_cm = pid_m * BLOCK_SIZE_M + tt.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tt.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tt.store(c_ptrs, c, mask=c_mask)
        tt.return_(srcs=[])

    matmul_kernel.emit()

    ctx.module.operation.verify()
    pm = PassManager.parse("builtin.module(cse)")
    pm.run(ctx.module.operation)
    correct = dedent(
        """\
    module {
      tt.func @matmul_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) {
        %0 = tt.get_program_id x : i32
        %c16_i32 = arith.constant 16 : i32
        %1 = arith.ceildivsi %arg3, %c16_i32 : i32
        %2 = arith.ceildivsi %arg4, %c16_i32 : i32
        %c2_i32 = arith.constant 2 : i32
        %3 = arith.muli %2, %c2_i32 : i32
        %4 = arith.floordivsi %0, %3 : i32
        %5 = arith.muli %4, %c2_i32 : i32
        %6 = arith.subi %1, %5 : i32
        %7 = arith.remsi %0, %c2_i32 : i32
        %8 = arith.addi %5, %7 : i32
        %9 = arith.remsi %0, %3 : i32
        %10 = arith.floordivsi %9, %c2_i32 : i32
        %11 = arith.muli %8, %c16_i32 : i32
        %12 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
        %13 = tt.splat %11 : i32 -> tensor<16xi32>
        %14 = arith.addi %13, %12 : tensor<16xi32>
        %15 = arith.muli %10, %c16_i32 : i32
        %16 = tt.splat %15 : i32 -> tensor<16xi32>
        %17 = arith.addi %16, %12 : tensor<16xi32>
        %18 = tt.expand_dims %14 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
        %19 = tt.splat %arg6 : i32 -> tensor<16x1xi32>
        %20 = arith.muli %18, %19 : tensor<16x1xi32>
        %21 = tt.expand_dims %12 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
        %22 = tt.splat %arg7 : i32 -> tensor<1x16xi32>
        %23 = arith.muli %21, %22 : tensor<1x16xi32>
        %24 = tt.broadcast %20 : tensor<16x1xi32> -> tensor<16x16xi32>
        %25 = tt.broadcast %23 : tensor<1x16xi32> -> tensor<16x16xi32>
        %26 = arith.addi %24, %25 : tensor<16x16xi32>
        %27 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
        %28 = tt.addptr %27, %26 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
        %29 = tt.expand_dims %12 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
        %30 = tt.splat %arg8 : i32 -> tensor<16x1xi32>
        %31 = arith.muli %29, %30 : tensor<16x1xi32>
        %32 = tt.expand_dims %17 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
        %33 = tt.splat %arg9 : i32 -> tensor<1x16xi32>
        %34 = arith.muli %32, %33 : tensor<1x16xi32>
        %35 = tt.broadcast %31 : tensor<16x1xi32> -> tensor<16x16xi32>
        %36 = tt.broadcast %34 : tensor<1x16xi32> -> tensor<16x16xi32>
        %37 = arith.addi %35, %36 : tensor<16x16xi32>
        %38 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
        %39 = tt.addptr %38, %37 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
        %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
        %40 = arith.ceildivsi %arg5, %c16_i32 : i32
        %c0_i32 = arith.constant 0 : i32
        %c1_i32 = arith.constant 1 : i32
        %41:3 = scf.for %arg12 = %c0_i32 to %40 step %c1_i32 iter_args(%arg13 = %cst, %arg14 = %28, %arg15 = %39) -> (tensor<16x16xf32>, tensor<16x16x!tt.ptr<f32>>, tensor<16x16x!tt.ptr<f32>>)  : i32 {
          %58 = arith.muli %arg12, %c16_i32 : i32
          %59 = arith.subi %arg5, %58 : i32
          %60 = tt.splat %59 : i32 -> tensor<1x16xi32>
          %61 = arith.cmpi slt, %21, %60 : tensor<1x16xi32>
          %cst_0 = arith.constant 0.000000e+00 : f32
          %62 = tt.splat %cst_0 : f32 -> tensor<16x16xf32>
          %63 = tt.broadcast %61 : tensor<1x16xi1> -> tensor<16x16xi1>
          %64 = tt.load %28, %63, %62 : tensor<16x16x!tt.ptr<f32>>
          %65 = tt.splat %59 : i32 -> tensor<16x1xi32>
          %66 = arith.cmpi slt, %29, %65 : tensor<16x1xi32>
          %67 = tt.broadcast %66 : tensor<16x1xi1> -> tensor<16x16xi1>
          %68 = tt.load %39, %67, %62 : tensor<16x16x!tt.ptr<f32>>
          %69 = tt.dot %64, %68, %62 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
          %70 = arith.addf %arg13, %69 : tensor<16x16xf32>
          %71 = arith.muli %arg7, %c16_i32 : i32
          %72 = tt.splat %71 : i32 -> tensor<16x16xi32>
          %73 = tt.addptr %arg14, %72 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
          %74 = arith.muli %arg8, %c16_i32 : i32
          %75 = tt.splat %74 : i32 -> tensor<16x16xi32>
          %76 = tt.addptr %arg15, %75 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
          scf.yield %70, %73, %76 : tensor<16x16xf32>, tensor<16x16x!tt.ptr<f32>>, tensor<16x16x!tt.ptr<f32>>
        }
        %42 = tt.splat %arg10 : i32 -> tensor<16x1xi32>
        %43 = arith.muli %42, %18 : tensor<16x1xi32>
        %44 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>>
        %45 = tt.addptr %44, %43 : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi32>
        %46 = tt.splat %arg11 : i32 -> tensor<1x16xi32>
        %47 = arith.muli %46, %32 : tensor<1x16xi32>
        %48 = tt.broadcast %45 : tensor<16x1x!tt.ptr<f32>> -> tensor<16x16x!tt.ptr<f32>>
        %49 = tt.broadcast %47 : tensor<1x16xi32> -> tensor<16x16xi32>
        %50 = tt.addptr %48, %49 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
        %51 = tt.splat %arg3 : i32 -> tensor<16x1xi32>
        %52 = arith.cmpi slt, %18, %51 : tensor<16x1xi32>
        %53 = tt.splat %arg4 : i32 -> tensor<1x16xi32>
        %54 = arith.cmpi slt, %32, %53 : tensor<1x16xi32>
        %55 = tt.broadcast %52 : tensor<16x1xi1> -> tensor<16x16xi1>
        %56 = tt.broadcast %54 : tensor<1x16xi1> -> tensor<16x16xi1>
        %57 = arith.andi %55, %56 : tensor<16x16xi1>
        tt.store %50, %41#0, %57 : tensor<16x16x!tt.ptr<f32>>
        tt.return
      }
    }
    """
    )

    filecheck(correct, ctx.module)
