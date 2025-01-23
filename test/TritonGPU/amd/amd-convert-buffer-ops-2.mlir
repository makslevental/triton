// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --tritonamdgpu-convert-buffer-ops='arch-generation-name=gfx940' | FileCheck %s

// CHECK-LABEL:   tt.func @conversion1(
// CHECK-SAME:  %[[VAL_0:.*]]: !tt.ptr<f32>) -> tensor<1024xf32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant {output_range = [1024, 1024]} 1024 : i32
// CHECK:           %[[VAL_2:.*]] = tt.get_program_id x {output_range = [0, 2048]} : i32
// CHECK:           %[[VAL_3:.*]] = arith.muli %[[VAL_2]], %[[VAL_1]] {output_range = [0, 2097152]} : i32
// CHECK:           %[[VAL_4:.*]] = tt.addptr %[[VAL_0]], %[[VAL_3]] : !tt.ptr<f32>, i32
// CHECK:           %[[VAL_5:.*]] = tt.splat %[[VAL_4]] : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
// CHECK:           %[[VAL_6:.*]] = tt.load %[[VAL_5]] : tensor<1024x!tt.ptr<f32>>
// CHECK:           tt.return %[[VAL_6]] : tensor<1024xf32>
// CHECK:         }

module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @conversion1(%arg0: !tt.ptr<f32>) -> tensor<1024xf32> {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    %3 = tt.splat %2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %4 = tt.load %3 : tensor<1024x!tt.ptr<f32>>
    tt.return %4 : tensor<1024xf32>
  }
}

// -----

// CHECK-LABEL:   tt.func @conversion2(
// CHECK-SAME:  %[[VAL_0:.*]]: !tt.ptr<f32>) -> tensor<1024xf32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant {output_range = [1024, 1024]} 1024 : i32
// CHECK:           %[[VAL_2:.*]] = tt.get_program_id x {output_range = [0, 2048]} : i32
// CHECK:           %[[VAL_3:.*]] = arith.muli %[[VAL_2]], %[[VAL_1]] {output_range = [0, 2097152]} : i32
// CHECK:           %[[VAL_4:.*]] = tt.make_range {end = 1024 : i32, output_range = [0, 1024], start = 0 : i32} : tensor<1024xi32>
// CHECK:           %[[VAL_5:.*]] = tt.addptr %[[VAL_0]], %[[VAL_3]] : !tt.ptr<f32>, i32
// CHECK:           %[[VAL_6:.*]] = amdgpu.buffer_load %[[VAL_5]]{{\[}}%[[VAL_4]]] : tensor<1024xf32>
// CHECK:           tt.return %[[VAL_6]] : tensor<1024xf32>
// CHECK:         }

module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @conversion2(%arg0: !tt.ptr<f32>) -> tensor<1024xf32> {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    %4 = tt.splat %3 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %5 = tt.addptr %4, %2 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %6 = tt.load %5 : tensor<1024x!tt.ptr<f32>>
    tt.return %6 : tensor<1024xf32>
  }
}

// -----

// CHECK-LABEL:   tt.func @conversion3(
// CHECK-SAME:  %[[VAL_0:.*]]: !tt.ptr<f32>) -> tensor<1024xf32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant {output_range = [1024, 1024]} 1024 : i32
// CHECK:           %[[VAL_2:.*]] = tt.get_program_id x {output_range = [0, 2048]} : i32
// CHECK:           %[[VAL_3:.*]] = arith.muli %[[VAL_2]], %[[VAL_1]] {output_range = [0, 2097152]} : i32
// CHECK:           %[[VAL_4:.*]] = tt.make_range {end = 1024 : i32, output_range = [0, 1024], start = 0 : i32} : tensor<1024xi32>
// CHECK:           %[[VAL_5:.*]] = tt.addptr %[[VAL_0]], %[[VAL_3]] : !tt.ptr<f32>, i32
// CHECK:           %[[VAL_6:.*]] = arith.extsi %[[VAL_4]] {output_range = [0, 1024]} : tensor<1024xi32> to tensor<1024xi64>
// CHECK:           %[[VAL_7:.*]] = tt.addptr %[[VAL_5]], %[[VAL_3]] : !tt.ptr<f32>, i32
// CHECK:           %[[VAL_8:.*]] = arith.extsi %[[VAL_4]] {output_range = [0, 1024]} : tensor<1024xi32> to tensor<1024xi64>
// CHECK:           %[[VAL_9:.*]] = arith.addi %[[VAL_8]], %[[VAL_6]] {output_range = [0, 2048]} : tensor<1024xi64>
// CHECK:           %[[VAL_10:.*]] = tt.splat %[[VAL_7]] : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
// CHECK:           %[[VAL_11:.*]] = tt.addptr %[[VAL_10]], %[[VAL_9]] : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
// CHECK:           %[[VAL_12:.*]] = tt.load %[[VAL_11]] : tensor<1024x!tt.ptr<f32>>
// CHECK:           tt.return %[[VAL_12]] : tensor<1024xf32>
// CHECK:         }

module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @conversion3(%arg0: !tt.ptr<f32>) -> tensor<1024xf32> {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    %4 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    %5 = tt.addptr %3, %1 : !tt.ptr<f32>, i32
    %6 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    %7 = arith.addi %6, %4 : tensor<1024xi64>
    %8 = tt.splat %5 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %9 = tt.addptr %8, %7 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
    %10 = tt.load %9 : tensor<1024x!tt.ptr<f32>>
    tt.return %10 : tensor<1024xf32>
  }
}

// -----

// CHECK-LABEL:   tt.func @conversion4(
// CHECK-SAME:  %[[VAL_0:.*]]: !tt.ptr<f32> {tt.pointer_range = 32 : i32}) -> tensor<1024xf32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant {output_range = [1024, 1024]} 1024 : i32
// CHECK:           %[[VAL_2:.*]] = tt.get_program_id x {output_range = [0, 2048]} : i32
// CHECK:           %[[VAL_3:.*]] = arith.muli %[[VAL_2]], %[[VAL_1]] {output_range = [0, 2097152]} : i32
// CHECK:           %[[VAL_4:.*]] = tt.make_range {end = 1024 : i32, output_range = [0, 1024], start = 0 : i32} : tensor<1024xi32>
// CHECK:           %[[VAL_5:.*]] = tt.addptr %[[VAL_0]], %[[VAL_3]] : !tt.ptr<f32>, i32
// CHECK:           %[[VAL_6:.*]] = tt.addptr %[[VAL_5]], %[[VAL_3]] : !tt.ptr<f32>, i32
// CHECK:           %[[VAL_7:.*]] = arith.addi %[[VAL_4]], %[[VAL_4]] {output_range = [0, 2048]} : tensor<1024xi32>
// CHECK:           %[[VAL_8:.*]] = amdgpu.buffer_load %[[VAL_6]]{{\[}}%[[VAL_7]]] : tensor<1024xf32>
// CHECK:           tt.return %[[VAL_8]] : tensor<1024xf32>
// CHECK:         }

module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @conversion4(%arg0: !tt.ptr<f32> {tt.pointer_range = 32 : i32}) -> tensor<1024xf32> {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    %4 = tt.addptr %3, %1 : !tt.ptr<f32>, i32
    %5 = arith.addi %2, %2 : tensor<1024xi32>
    %6 = tt.splat %4 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %7 = tt.addptr %6, %5 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %8 = tt.load %7 : tensor<1024x!tt.ptr<f32>>
    tt.return %8 : tensor<1024xf32>
  }
}

// -----

// CHECK-LABEL:   tt.func @forOp(
// CHECK-SAME:  %[[VAL_0:.*]]: !tt.ptr<f32>, %[[VAL_1:.*]]: tensor<1024xf32>) -> tensor<1024xf32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant {output_range = [1024, 1024]} 1024 : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant {output_range = [0, 0]} 0 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant {output_range = [128, 128]} 128 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant {output_range = [1, 1]} 1 : index
// CHECK:           %[[VAL_6:.*]] = tt.get_program_id x {output_range = [0, 2048]} : i32
// CHECK:           %[[VAL_7:.*]] = arith.muli %[[VAL_6]], %[[VAL_2]] {output_range = [0, 2097152]} : i32
// CHECK:           %[[VAL_8:.*]] = tt.make_range {end = 1024 : i32, output_range = [0, 1024], start = 0 : i32} : tensor<1024xi32>
// CHECK:           %[[VAL_9:.*]] = tt.addptr %[[VAL_0]], %[[VAL_7]] : !tt.ptr<f32>, i32
// CHECK:           %[[VAL_10:.*]] = arith.extsi %[[VAL_8]] {output_range = [0, 1024]} : tensor<1024xi32> to tensor<1024xi64>
// CHECK:           %[[VAL_11:.*]]:3 = scf.for %[[VAL_12:.*]] = %[[VAL_3]] to %[[VAL_4]] step %[[VAL_5]] iter_args(%[[VAL_13:.*]] = %[[VAL_9]], %[[VAL_14:.*]] = %[[VAL_10]], %[[VAL_15:.*]] = %[[VAL_1]]) -> (!tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>) {
// CHECK:             %[[VAL_16:.*]] = tt.addptr %[[VAL_13]], %[[VAL_7]] : !tt.ptr<f32>, i32
// CHECK:             %[[VAL_17:.*]] = arith.extsi %[[VAL_8]] {output_range = [0, 1024]} : tensor<1024xi32> to tensor<1024xi64>
// CHECK:             %[[VAL_18:.*]] = arith.addi %[[VAL_17]], %[[VAL_14]] {output_range = [0, 130048]} : tensor<1024xi64>
// CHECK:             %[[VAL_19:.*]] = tt.splat %[[VAL_16]] : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
// CHECK:             %[[VAL_20:.*]] = tt.addptr %[[VAL_19]], %[[VAL_18]] : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
// CHECK:             %[[VAL_21:.*]] = tt.load %[[VAL_20]] : tensor<1024x!tt.ptr<f32>>
// CHECK:             %[[VAL_22:.*]] = arith.addf %[[VAL_21]], %[[VAL_15]] : tensor<1024xf32>
// CHECK:             scf.yield %[[VAL_16]], %[[VAL_18]], %[[VAL_22]] : !tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>
// CHECK:           }
// CHECK:           %[[VAL_23:.*]] = tt.addptr %[[VAL_24:.*]]#0, %[[VAL_7]] : !tt.ptr<f32>, i32
// CHECK:           %[[VAL_25:.*]] = arith.extsi %[[VAL_8]] {output_range = [0, 1024]} : tensor<1024xi32> to tensor<1024xi64>
// CHECK:           %[[VAL_26:.*]] = arith.addi %[[VAL_25]], %[[VAL_24]]#1 {output_range = [0, 131072]} : tensor<1024xi64>
// CHECK:           %[[VAL_27:.*]] = tt.splat %[[VAL_23]] : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
// CHECK:           %[[VAL_28:.*]] = tt.addptr %[[VAL_27]], %[[VAL_26]] : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
// CHECK:           %[[VAL_29:.*]] = tt.load %[[VAL_28]] : tensor<1024x!tt.ptr<f32>>
// CHECK:           tt.return %[[VAL_29]] : tensor<1024xf32>
// CHECK:         }

module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @forOp(%arg0: !tt.ptr<f32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %c1024_i32 = arith.constant 1024 : i32
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    %4 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    %5:3 = scf.for %arg2 = %c0 to %c128 step %c1 iter_args(%arg3 = %3, %arg4 = %4, %arg5 = %arg1) -> (!tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>) {
      %12 = tt.addptr %arg3, %1 : !tt.ptr<f32>, i32
      %13 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
      %14 = arith.addi %13, %arg4 : tensor<1024xi64>
      %15 = tt.splat %12 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      %16 = tt.addptr %15, %14 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
      %17 = tt.load %16 : tensor<1024x!tt.ptr<f32>>
      %18 = arith.addf %17, %arg5 : tensor<1024xf32>
      scf.yield %12, %14, %18 : !tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>
    }
    %6 = tt.addptr %5#0, %1 : !tt.ptr<f32>, i32
    %7 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    %8 = arith.addi %7, %5#1 : tensor<1024xi64>
    %9 = tt.splat %6 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %10 = tt.addptr %9, %8 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
    %11 = tt.load %10 : tensor<1024x!tt.ptr<f32>>
    tt.return %11 : tensor<1024xf32>
  }
}

// -----

// CHECK-LABEL:   tt.func @forOp2(
// CHECK-SAME:  %[[VAL_0:.*]]: !tt.ptr<f32>, %[[VAL_1:.*]]: tensor<1024xf32>) -> tensor<1024xf32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant {output_range = [0, 0]} dense<0> : tensor<1024xi64>
// CHECK:           %[[VAL_3:.*]] = arith.constant {output_range = [1024, 1024]} 1024 : i32
// CHECK:           %[[VAL_4:.*]] = arith.constant {output_range = [0, 0]} 0 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant {output_range = [128, 128]} 128 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant {output_range = [1, 1]} 1 : index
// CHECK:           %[[VAL_7:.*]] = tt.get_program_id x {output_range = [0, 2048]} : i32
// CHECK:           %[[VAL_8:.*]] = arith.muli %[[VAL_7]], %[[VAL_3]] {output_range = [0, 2097152]} : i32
// CHECK:           %[[VAL_9:.*]] = tt.make_range {end = 1024 : i32, output_range = [0, 1024], start = 0 : i32} : tensor<1024xi32>
// CHECK:           %[[VAL_10:.*]]:3 = scf.for %[[VAL_11:.*]] = %[[VAL_4]] to %[[VAL_5]] step %[[VAL_6]] iter_args(%[[VAL_12:.*]] = %[[VAL_0]], %[[VAL_13:.*]] = %[[VAL_2]], %[[VAL_14:.*]] = %[[VAL_1]]) -> (!tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>) {
// CHECK:             %[[VAL_15:.*]] = tt.addptr %[[VAL_12]], %[[VAL_8]] : !tt.ptr<f32>, i32
// CHECK:             %[[VAL_16:.*]] = arith.extsi %[[VAL_9]] {output_range = [0, 1024]} : tensor<1024xi32> to tensor<1024xi64>
// CHECK:             %[[VAL_17:.*]] = arith.addi %[[VAL_16]], %[[VAL_13]] {output_range = [0, 129024]} : tensor<1024xi64>
// CHECK:             %[[VAL_18:.*]] = tt.splat %[[VAL_15]] : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
// CHECK:             %[[VAL_19:.*]] = tt.addptr %[[VAL_18]], %[[VAL_17]] : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
// CHECK:             %[[VAL_20:.*]] = tt.load %[[VAL_19]] : tensor<1024x!tt.ptr<f32>>
// CHECK:             %[[VAL_21:.*]] = arith.addf %[[VAL_20]], %[[VAL_14]] : tensor<1024xf32>
// CHECK:             scf.yield %[[VAL_15]], %[[VAL_17]], %[[VAL_21]] : !tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>
// CHECK:           }
// CHECK:           %[[VAL_22:.*]] = tt.addptr %[[VAL_23:.*]]#0, %[[VAL_8]] : !tt.ptr<f32>, i32
// CHECK:           %[[VAL_24:.*]] = arith.extsi %[[VAL_9]] {output_range = [0, 1024]} : tensor<1024xi32> to tensor<1024xi64>
// CHECK:           %[[VAL_25:.*]] = arith.addi %[[VAL_24]], %[[VAL_23]]#1 {output_range = [0, 130048]} : tensor<1024xi64>
// CHECK:           %[[VAL_26:.*]] = tt.splat %[[VAL_22]] : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
// CHECK:           %[[VAL_27:.*]] = tt.addptr %[[VAL_26]], %[[VAL_25]] : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
// CHECK:           %[[VAL_28:.*]] = tt.load %[[VAL_27]] : tensor<1024x!tt.ptr<f32>>
// CHECK:           tt.return %[[VAL_28]] : tensor<1024xf32>
// CHECK:         }

module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @forOp2(%arg0: !tt.ptr<f32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = arith.constant dense<0> : tensor<1024xi64>
    %c1024_i32 = arith.constant 1024 : i32
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3:3 = scf.for %arg2 = %c0 to %c128 step %c1 iter_args(%arg3 = %arg0, %arg4 = %cst, %arg5 = %arg1) -> (!tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>) {
      %10 = tt.addptr %arg3, %1 : !tt.ptr<f32>, i32
      %11 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
      %12 = arith.addi %11, %arg4 : tensor<1024xi64>
      %13 = tt.splat %10 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      %14 = tt.addptr %13, %12 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
      %15 = tt.load %14 : tensor<1024x!tt.ptr<f32>>
      %16 = arith.addf %15, %arg5 : tensor<1024xf32>
      scf.yield %10, %12, %16 : !tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>
    }
    %4 = tt.addptr %3#0, %1 : !tt.ptr<f32>, i32
    %5 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    %6 = arith.addi %5, %3#1 : tensor<1024xi64>
    %7 = tt.splat %4 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %8 = tt.addptr %7, %6 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
    %9 = tt.load %8 : tensor<1024x!tt.ptr<f32>>
    tt.return %9 : tensor<1024xf32>
  }
}

// -----

// CHECK-LABEL:   tt.func @forNested(
// CHECK-SAME:  %[[VAL_0:.*]]: !tt.ptr<f32>, %[[VAL_1:.*]]: tensor<1024xf32>) -> tensor<1024xf32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant {output_range = [0, 0]} dense<0> : tensor<1024xi64>
// CHECK:           %[[VAL_3:.*]] = arith.constant {output_range = [1024, 1024]} 1024 : i32
// CHECK:           %[[VAL_4:.*]] = arith.constant {output_range = [0, 0]} 0 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant {output_range = [128, 128]} 128 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant {output_range = [1, 1]} 1 : index
// CHECK:           %[[VAL_7:.*]] = tt.get_program_id x {output_range = [0, 2048]} : i32
// CHECK:           %[[VAL_8:.*]] = arith.muli %[[VAL_7]], %[[VAL_3]] {output_range = [0, 2097152]} : i32
// CHECK:           %[[VAL_9:.*]] = tt.make_range {end = 1024 : i32, output_range = [0, 1024], start = 0 : i32} : tensor<1024xi32>
// CHECK:           %[[VAL_10:.*]]:3 = scf.for %[[VAL_11:.*]] = %[[VAL_4]] to %[[VAL_5]] step %[[VAL_6]] iter_args(%[[VAL_12:.*]] = %[[VAL_0]], %[[VAL_13:.*]] = %[[VAL_2]], %[[VAL_14:.*]] = %[[VAL_1]]) -> (!tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>) {
// CHECK:             %[[VAL_15:.*]]:3 = scf.for %[[VAL_16:.*]] = %[[VAL_4]] to %[[VAL_5]] step %[[VAL_6]] iter_args(%[[VAL_17:.*]] = %[[VAL_12]], %[[VAL_18:.*]] = %[[VAL_13]], %[[VAL_19:.*]] = %[[VAL_14]]) -> (!tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>) {
// CHECK:               %[[VAL_20:.*]] = tt.addptr %[[VAL_17]], %[[VAL_8]] : !tt.ptr<f32>, i32
// CHECK:               %[[VAL_21:.*]] = arith.extsi %[[VAL_9]] {output_range = [0, 1024]} : tensor<1024xi32> to tensor<1024xi64>
// CHECK:               %[[VAL_22:.*]] = arith.addi %[[VAL_21]], %[[VAL_18]] {output_range = [0, 44032]} : tensor<1024xi64>
// CHECK:               %[[VAL_23:.*]] = tt.splat %[[VAL_20]] : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
// CHECK:               %[[VAL_24:.*]] = tt.addptr %[[VAL_23]], %[[VAL_22]] : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
// CHECK:               %[[VAL_25:.*]] = tt.load %[[VAL_24]] : tensor<1024x!tt.ptr<f32>>
// CHECK:               %[[VAL_26:.*]] = arith.addf %[[VAL_25]], %[[VAL_19]] : tensor<1024xf32>
// CHECK:               scf.yield %[[VAL_20]], %[[VAL_22]], %[[VAL_26]] : !tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>
// CHECK:             }
// CHECK:             scf.yield %[[VAL_27:.*]]#0, %[[VAL_27]]#1, %[[VAL_27]]#2 : !tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>
// CHECK:           }
// CHECK:           %[[VAL_28:.*]] = tt.addptr %[[VAL_29:.*]]#0, %[[VAL_8]] : !tt.ptr<f32>, i32
// CHECK:           %[[VAL_30:.*]] = arith.extsi %[[VAL_9]] {output_range = [0, 1024]} : tensor<1024xi32> to tensor<1024xi64>
// CHECK:           %[[VAL_31:.*]] = arith.addi %[[VAL_30]], %[[VAL_29]]#1 {output_range = [0, 43008]} : tensor<1024xi64>
// CHECK:           %[[VAL_32:.*]] = tt.splat %[[VAL_28]] : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
// CHECK:           %[[VAL_33:.*]] = tt.addptr %[[VAL_32]], %[[VAL_31]] : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
// CHECK:           %[[VAL_34:.*]] = tt.load %[[VAL_33]] : tensor<1024x!tt.ptr<f32>>
// CHECK:           tt.return %[[VAL_34]] : tensor<1024xf32>
// CHECK:         }

module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @forNested(%arg0: !tt.ptr<f32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = arith.constant dense<0> : tensor<1024xi64>
    %c1024_i32 = arith.constant 1024 : i32
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3:3 = scf.for %arg2 = %c0 to %c128 step %c1 iter_args(%arg3 = %arg0, %arg4 = %cst, %arg5 = %arg1) -> (!tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>) {
      %10:3 = scf.for %arg6 = %c0 to %c128 step %c1 iter_args(%arg7 = %arg3, %arg8 = %arg4, %arg9 = %arg5) -> (!tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>) {
        %11 = tt.addptr %arg7, %1 : !tt.ptr<f32>, i32
        %12 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
        %13 = arith.addi %12, %arg8 : tensor<1024xi64>
        %14 = tt.splat %11 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
        %15 = tt.addptr %14, %13 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
        %16 = tt.load %15 : tensor<1024x!tt.ptr<f32>>
        %17 = arith.addf %16, %arg9 : tensor<1024xf32>
        scf.yield %11, %13, %17 : !tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>
      }
      scf.yield %10#0, %10#1, %10#2 : !tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>
    }
    %4 = tt.addptr %3#0, %1 : !tt.ptr<f32>, i32
    %5 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    %6 = arith.addi %5, %3#1 : tensor<1024xi64>
    %7 = tt.splat %4 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %8 = tt.addptr %7, %6 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
    %9 = tt.load %8 : tensor<1024x!tt.ptr<f32>>
    tt.return %9 : tensor<1024xf32>
  }
}

// -----

// CHECK-LABEL:   tt.func @ifOp(
// CHECK-SAME:  %[[VAL_0:.*]]: !tt.ptr<f32>, %[[VAL_1:.*]]: tensor<1024xf32>, %[[VAL_2:.*]]: i1) -> tensor<1024xf32> {
// CHECK:           %[[VAL_3:.*]] = arith.constant {output_range = [0, 0]} dense<0> : tensor<1024xi64>
// CHECK:           %[[VAL_4:.*]] = arith.constant {output_range = [1024, 1024]} 1024 : i32
// CHECK:           %[[VAL_5:.*]] = tt.get_program_id x {output_range = [0, 2048]} : i32
// CHECK:           %[[VAL_6:.*]] = arith.muli %[[VAL_5]], %[[VAL_4]] {output_range = [0, 2097152]} : i32
// CHECK:           %[[VAL_7:.*]] = tt.make_range {end = 1024 : i32, output_range = [0, 1024], start = 0 : i32} : tensor<1024xi32>
// CHECK:           %[[VAL_8:.*]]:2 = scf.if %[[VAL_2]] -> (!tt.ptr<f32>, tensor<1024xi64>) {
// CHECK:             %[[VAL_9:.*]] = tt.addptr %[[VAL_0]], %[[VAL_6]] : !tt.ptr<f32>, i32
// CHECK:             %[[VAL_10:.*]] = arith.extsi %[[VAL_7]] {output_range = [0, 1024]} : tensor<1024xi32> to tensor<1024xi64>
// CHECK:             scf.yield %[[VAL_9]], %[[VAL_10]] : !tt.ptr<f32>, tensor<1024xi64>
// CHECK:           } else {
// CHECK:             %[[VAL_11:.*]] = tt.addptr %[[VAL_0]], %[[VAL_6]] : !tt.ptr<f32>, i32
// CHECK:             scf.yield %[[VAL_11]], %[[VAL_3]] : !tt.ptr<f32>, tensor<1024xi64>
// CHECK:           }
// CHECK:           %[[VAL_12:.*]] = arith.trunci %[[VAL_13:.*]]#1 {output_range = [0, 1024]} : tensor<1024xi64> to tensor<1024xi32>
// CHECK:           %[[VAL_14:.*]] = amdgpu.buffer_load %[[VAL_13]]#0{{\[}}%[[VAL_12]]] : tensor<1024xf32>
// CHECK:           tt.return %[[VAL_14]] : tensor<1024xf32>
// CHECK:         }

module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @ifOp(%arg0: !tt.ptr<f32>, %arg1: tensor<1024xf32>, %arg2: i1) -> tensor<1024xf32> {
    %cst = arith.constant dense<0> : tensor<1024xi64>
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3:2 = scf.if %arg2 -> (!tt.ptr<f32>, tensor<1024xi64>) {
      %8 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
      %9 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
      scf.yield %8, %9 : !tt.ptr<f32>, tensor<1024xi64>
    } else {
      %8 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
      scf.yield %8, %cst : !tt.ptr<f32>, tensor<1024xi64>
    }
    %4 = arith.trunci %3#1 : tensor<1024xi64> to tensor<1024xi32>
    %5 = tt.splat %3#0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %6 = tt.addptr %5, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %7 = tt.load %6 : tensor<1024x!tt.ptr<f32>>
    tt.return %7 : tensor<1024xf32>
  }
}

// -----

// CHECK-LABEL:   tt.func @condBranch(
// CHECK-SAME:  %[[VAL_0:.*]]: !tt.ptr<f32>, %[[VAL_1:.*]]: i1) -> tensor<1024xf32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant {output_range = [0, 0]} dense<0> : tensor<1024xi64>
// CHECK:           %[[VAL_3:.*]] = arith.constant {output_range = [1024, 1024]} 1024 : i32
// CHECK:           %[[VAL_4:.*]] = tt.get_program_id x {output_range = [0, 2048]} : i32
// CHECK:           %[[VAL_5:.*]] = arith.muli %[[VAL_4]], %[[VAL_3]] {output_range = [0, 2097152]} : i32
// CHECK:           %[[VAL_6:.*]] = tt.make_range {end = 1024 : i32, output_range = [0, 1024], start = 0 : i32} : tensor<1024xi32>
// CHECK:           %[[VAL_7:.*]] = tt.addptr %[[VAL_0]], %[[VAL_5]] : !tt.ptr<f32>, i32
// CHECK:           %[[VAL_8:.*]] = arith.extsi %[[VAL_6]] {output_range = [0, 1024]} : tensor<1024xi32> to tensor<1024xi64>
// CHECK:           cf.cond_br %[[VAL_1]], ^bb1(%[[VAL_0]], %[[VAL_2]] : !tt.ptr<f32>, tensor<1024xi64>), ^bb1(%[[VAL_7]], %[[VAL_8]] : !tt.ptr<f32>, tensor<1024xi64>)
// CHECK:         ^bb1(%[[VAL_9:.*]]: !tt.ptr<f32>, %[[VAL_10:.*]]: tensor<1024xi64>):
// CHECK:           %[[VAL_11:.*]] = arith.trunci %[[VAL_10]] {output_range = [0, 1024]} : tensor<1024xi64> to tensor<1024xi32>
// CHECK:           %[[VAL_12:.*]] = tt.splat %[[VAL_9]] : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
// CHECK:           %[[VAL_13:.*]] = tt.addptr %[[VAL_12]], %[[VAL_11]] : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
// CHECK:           %[[VAL_14:.*]] = tt.load %[[VAL_13]] : tensor<1024x!tt.ptr<f32>>
// CHECK:           tt.return %[[VAL_14]] : tensor<1024xf32>
// CHECK:         }

module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @condBranch(%arg0: !tt.ptr<f32>, %arg1: i1) -> tensor<1024xf32> {
    %cst = arith.constant dense<0> : tensor<1024xi64>
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    %4 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    cf.cond_br %arg1, ^bb1(%arg0, %cst : !tt.ptr<f32>, tensor<1024xi64>), ^bb2(%3, %4 : !tt.ptr<f32>, tensor<1024xi64>)
  ^bb1(%5: !tt.ptr<f32>, %6: tensor<1024xi64>):  // pred: ^bb0
    %7 = arith.trunci %6 : tensor<1024xi64> to tensor<1024xi32>
    %8 = tt.splat %5 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %9 = tt.addptr %8, %7 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %10 = tt.load %9 : tensor<1024x!tt.ptr<f32>>
    tt.return %10 : tensor<1024xf32>
  ^bb2(%11: !tt.ptr<f32>, %12: tensor<1024xi64>):  // pred: ^bb0
    %13 = arith.trunci %12 : tensor<1024xi64> to tensor<1024xi32>
    %14 = tt.splat %11 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %15 = tt.addptr %14, %13 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %16 = tt.load %15 : tensor<1024x!tt.ptr<f32>>
    tt.return %16 : tensor<1024xf32>
  }
}

// -----

// CHECK-LABEL:   tt.func @branch(
// CHECK-SAME:  %[[VAL_0:.*]]: !tt.ptr<f32>, %[[VAL_1:.*]]: i1) -> tensor<1024xf32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant {output_range = [1024, 1024]} 1024 : i32
// CHECK:           %[[VAL_3:.*]] = tt.get_program_id x {output_range = [0, 2048]} : i32
// CHECK:           %[[VAL_4:.*]] = arith.muli %[[VAL_3]], %[[VAL_2]] {output_range = [0, 2097152]} : i32
// CHECK:           %[[VAL_5:.*]] = tt.make_range {end = 1024 : i32, output_range = [0, 1024], start = 0 : i32} : tensor<1024xi32>
// CHECK:           %[[VAL_6:.*]] = tt.addptr %[[VAL_0]], %[[VAL_4]] : !tt.ptr<f32>, i32
// CHECK:           %[[VAL_7:.*]] = amdgpu.buffer_load %[[VAL_6]]{{\[}}%[[VAL_5]]] : tensor<1024xf32>
// CHECK:           tt.return %[[VAL_7]] : tensor<1024xf32>
// CHECK:         }
// CHECK: #[[$ATTR_0:.+]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @branch(%arg0: !tt.ptr<f32>, %arg1: i1) -> tensor<1024xf32> {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    %4 = tt.splat %3 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %5 = tt.addptr %4, %2 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %6 = tt.load %5 : tensor<1024x!tt.ptr<f32>>
    tt.return %6 : tensor<1024xf32>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK-LABEL:   tt.func @tile_offset(
// CHECK-SAME:  %[[VAL_0:.*]]: !tt.ptr<f16>, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32) -> tensor<16x256xf16, #[[$ATTR_0]]> {
// CHECK:           %[[VAL_3:.*]] = arith.constant {output_range = [256, 256]} 256 : i32
// CHECK:           %[[VAL_4:.*]] = tt.get_program_id x {output_range = [0, 2048]} : i32
// CHECK:           %[[VAL_5:.*]] = arith.muli %[[VAL_4]], %[[VAL_3]] {output_range = [0, 524288]} : i32
// CHECK:           %[[VAL_6:.*]] = tt.make_range {end = 256 : i32, output_range = [0, 256], start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #[[$ATTR_0]]}>>
// CHECK:           %[[VAL_7:.*]] = tt.make_range {end = 16 : i32, output_range = [0, 16], start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #[[$ATTR_0]]}>>
// CHECK:           %[[VAL_8:.*]] = tt.expand_dims %[[VAL_7]] {axis = 1 : i32, output_range = [0, 16]} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #[[$ATTR_0]]}>> -> tensor<16x1xi32, #[[$ATTR_0]]>
// CHECK:           %[[VAL_9:.*]] = tt.splat %[[VAL_2]] {output_range = [-2147483648, 2147483647]} : i32 -> tensor<16x1xi32, #[[$ATTR_0]]>
// CHECK:           %[[VAL_10:.*]] = arith.muli %[[VAL_8]], %[[VAL_9]] {output_range = [-2147483648, 2147483647]} : tensor<16x1xi32, #[[$ATTR_0]]>
// CHECK:           %[[VAL_11:.*]] = tt.broadcast %[[VAL_10]] {output_range = [-2147483648, 2147483647]} : tensor<16x1xi32, #[[$ATTR_0]]> -> tensor<16x256xi32, #[[$ATTR_0]]>
// CHECK:           %[[VAL_12:.*]] = tt.expand_dims %[[VAL_6]] {axis = 0 : i32, output_range = [0, 256]} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #[[$ATTR_0]]}>> -> tensor<1x256xi32, #[[$ATTR_0]]>
// CHECK:           %[[VAL_13:.*]] = tt.broadcast %[[VAL_12]] {output_range = [-2147483648, 2147483647]} : tensor<1x256xi32, #[[$ATTR_0]]> -> tensor<16x256xi32, #[[$ATTR_0]]>
// CHECK:           %[[VAL_14:.*]] = arith.addi %[[VAL_11]], %[[VAL_13]] {output_range = [-2147483648, 2147483647]} : tensor<16x256xi32, #[[$ATTR_0]]>
// CHECK:           %[[VAL_15:.*]] = tt.addptr %[[VAL_0]], %[[VAL_5]] : !tt.ptr<f16>, i32
// CHECK:           %[[VAL_16:.*]] = tt.splat %[[VAL_15]] : !tt.ptr<f16> -> tensor<16x256x!tt.ptr<f16>, #[[$ATTR_0]]>
// CHECK:           %[[VAL_17:.*]] = tt.addptr %[[VAL_16]], %[[VAL_14]] : tensor<16x256x!tt.ptr<f16>, #[[$ATTR_0]]>, tensor<16x256xi32, #[[$ATTR_0]]>
// CHECK:           %[[VAL_18:.*]] = tt.load %[[VAL_17]] : tensor<16x256x!tt.ptr<f16>, #[[$ATTR_0]]>
// CHECK:           tt.return %[[VAL_18]] : tensor<16x256xf16, #[[$ATTR_0]]>
// CHECK:         }
// CHECK: #[[$ATTR_1:.+]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @tile_offset(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32) -> tensor<16x256xf16, #blocked> {
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %3 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %4 = tt.expand_dims %3 {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi32, #blocked>
    %5 = tt.splat %arg2 : i32 -> tensor<16x1xi32, #blocked>
    %6 = arith.muli %4, %5 : tensor<16x1xi32, #blocked>
    %7 = tt.broadcast %6 : tensor<16x1xi32, #blocked> -> tensor<16x256xi32, #blocked>
    %8 = tt.expand_dims %2 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
    %9 = tt.broadcast %8 : tensor<1x256xi32, #blocked> -> tensor<16x256xi32, #blocked>
    %10 = arith.addi %7, %9 : tensor<16x256xi32, #blocked>
    %11 = tt.addptr %arg0, %1 : !tt.ptr<f16>, i32
    %12 = tt.splat %11 : !tt.ptr<f16> -> tensor<16x256x!tt.ptr<f16>, #blocked>
    %13 = tt.addptr %12, %10 : tensor<16x256x!tt.ptr<f16>, #blocked>, tensor<16x256xi32, #blocked>
    %14 = tt.load %13 : tensor<16x256x!tt.ptr<f16>, #blocked>
    tt.return %14 : tensor<16x256xf16, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK-LABEL:   tt.func public @matmul_kernel(
// CHECK-SAME:  %[[VAL_0:.*]]: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %[[VAL_1:.*]]: i32 {tt.divisibility = 16 : i32}) -> tensor<128x16xf16, #[[$ATTR_1]]> {
// CHECK:           %[[VAL_2:.*]] = arith.constant {output_range = [128, 128]} 128 : i32
// CHECK:           %[[VAL_3:.*]] = tt.get_program_id x {output_range = [0, 2048]} : i32
// CHECK:           %[[VAL_4:.*]] = arith.muli %[[VAL_3]], %[[VAL_2]] {output_range = [0, 262144]} : i32
// CHECK:           %[[VAL_5:.*]] = tt.make_range {end = 128 : i32, output_range = [0, 128], start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #[[$ATTR_1]]}>>
// CHECK:           %[[VAL_6:.*]] = tt.make_range {end = 16 : i32, output_range = [0, 16], start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #[[$ATTR_1]]}>>
// CHECK:           %[[VAL_7:.*]] = tt.expand_dims %[[VAL_5]] {axis = 1 : i32, output_range = [0, 128]} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #[[$ATTR_1]]}>> -> tensor<128x1xi32, #[[$ATTR_1]]>
// CHECK:           %[[VAL_8:.*]] = arith.muli %[[VAL_4]], %[[VAL_1]] {output_range = [-2147483648, 2147483647]} : i32
// CHECK:           %[[VAL_9:.*]] = tt.splat %[[VAL_1]] {output_range = [-2147483648, 2147483647]} : i32 -> tensor<128x1xi32, #[[$ATTR_1]]>
// CHECK:           %[[VAL_10:.*]] = arith.muli %[[VAL_7]], %[[VAL_9]] {output_range = [-2147483648, 2147483647]} : tensor<128x1xi32, #[[$ATTR_1]]>
// CHECK:           %[[VAL_11:.*]] = tt.broadcast %[[VAL_10]] {output_range = [-2147483648, 2147483647]} : tensor<128x1xi32, #[[$ATTR_1]]> -> tensor<128x16xi32, #[[$ATTR_1]]>
// CHECK:           %[[VAL_12:.*]] = tt.expand_dims %[[VAL_6]] {axis = 0 : i32, output_range = [0, 16]} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #[[$ATTR_1]]}>> -> tensor<1x16xi32, #[[$ATTR_1]]>
// CHECK:           %[[VAL_13:.*]] = tt.broadcast %[[VAL_12]] {output_range = [-2147483648, 2147483647]} : tensor<1x16xi32, #[[$ATTR_1]]> -> tensor<128x16xi32, #[[$ATTR_1]]>
// CHECK:           %[[VAL_14:.*]] = arith.addi %[[VAL_11]], %[[VAL_13]] {output_range = [-2147483648, 2147483647]} : tensor<128x16xi32, #[[$ATTR_1]]>
// CHECK:           %[[VAL_15:.*]] = tt.addptr %[[VAL_0]], %[[VAL_8]] : !tt.ptr<f16>, i32
// CHECK:           %[[VAL_16:.*]] = tt.splat %[[VAL_15]] : !tt.ptr<f16> -> tensor<128x16x!tt.ptr<f16>, #[[$ATTR_1]]>
// CHECK:           %[[VAL_17:.*]] = tt.addptr %[[VAL_16]], %[[VAL_14]] : tensor<128x16x!tt.ptr<f16>, #[[$ATTR_1]]>, tensor<128x16xi32, #[[$ATTR_1]]>
// CHECK:           %[[VAL_18:.*]] = tt.load %[[VAL_17]] : tensor<128x16x!tt.ptr<f16>, #[[$ATTR_1]]>
// CHECK:           tt.return %[[VAL_18]] : tensor<128x16xf16, #[[$ATTR_1]]>
// CHECK:         }

module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}) -> tensor<128x16xf16, #blocked> {
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c128_i32 : i32
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %3 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %4 = tt.expand_dims %2 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %5 = arith.muli %1, %arg1 : i32
    %6 = tt.splat %arg1 : i32 -> tensor<128x1xi32, #blocked>
    %7 = arith.muli %4, %6 : tensor<128x1xi32, #blocked>
    %8 = tt.broadcast %7 : tensor<128x1xi32, #blocked> -> tensor<128x16xi32, #blocked>
    %9 = tt.expand_dims %3 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x16xi32, #blocked>
    %10 = tt.broadcast %9 : tensor<1x16xi32, #blocked> -> tensor<128x16xi32, #blocked>
    %11 = arith.addi %8, %10 : tensor<128x16xi32, #blocked>
    %12 = tt.addptr %arg0, %5 : !tt.ptr<f16>, i32
    %13 = tt.splat %12 : !tt.ptr<f16> -> tensor<128x16x!tt.ptr<f16>, #blocked>
    %14 = tt.addptr %13, %11 : tensor<128x16x!tt.ptr<f16>, #blocked>, tensor<128x16xi32, #blocked>
    %15 = tt.load %14 : tensor<128x16x!tt.ptr<f16>, #blocked>
    tt.return %15 : tensor<128x16xf16, #blocked>
  }
}

// -----

// CHECK-LABEL:   tt.func @select(
// CHECK-SAME:  %[[VAL_0:.*]]: !tt.ptr<f32>, %[[VAL_1:.*]]: i1) -> tensor<1024xf32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant {output_range = [0, 0]} dense<0> : tensor<1024xi64>
// CHECK:           %[[VAL_3:.*]] = arith.constant {output_range = [1024, 1024]} 1024 : i32
// CHECK:           %[[VAL_4:.*]] = tt.get_program_id x {output_range = [0, 2048]} : i32
// CHECK:           %[[VAL_5:.*]] = arith.muli %[[VAL_4]], %[[VAL_3]] {output_range = [0, 2097152]} : i32
// CHECK:           %[[VAL_6:.*]] = tt.make_range {end = 1024 : i32, output_range = [0, 1024], start = 0 : i32} : tensor<1024xi32>
// CHECK:           %[[VAL_7:.*]] = tt.addptr %[[VAL_0]], %[[VAL_5]] : !tt.ptr<f32>, i32
// CHECK:           %[[VAL_8:.*]] = arith.extsi %[[VAL_6]] {output_range = [0, 1024]} : tensor<1024xi32> to tensor<1024xi64>
// CHECK:           %[[VAL_9:.*]] = arith.select %[[VAL_1]], %[[VAL_0]], %[[VAL_7]] : !tt.ptr<f32>
// CHECK:           %[[VAL_10:.*]] = arith.select %[[VAL_1]], %[[VAL_2]], %[[VAL_8]] {output_range = [0, 1024]} : tensor<1024xi64>
// CHECK:           %[[VAL_11:.*]] = arith.trunci %[[VAL_10]] {output_range = [0, 1024]} : tensor<1024xi64> to tensor<1024xi32>
// CHECK:           %[[VAL_12:.*]] = tt.splat %[[VAL_9]] : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
// CHECK:           %[[VAL_13:.*]] = tt.addptr %[[VAL_12]], %[[VAL_11]] : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
// CHECK:           %[[VAL_14:.*]] = tt.load %[[VAL_13]] : tensor<1024x!tt.ptr<f32>>
// CHECK:           tt.return %[[VAL_14]] : tensor<1024xf32>
// CHECK:         }

module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @select(%arg0: !tt.ptr<f32>, %arg1: i1) -> tensor<1024xf32> {
    %cst = arith.constant dense<0> : tensor<1024xi64>
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    %4 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    %5 = arith.select %arg1, %arg0, %3 : !tt.ptr<f32>
    %6 = arith.select %arg1, %cst, %4 : tensor<1024xi64>
    %7 = arith.trunci %6 : tensor<1024xi64> to tensor<1024xi32>
    %8 = tt.splat %5 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %9 = tt.addptr %8, %7 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %10 = tt.load %9 : tensor<1024x!tt.ptr<f32>>
    tt.return %10 : tensor<1024xf32>
  }
}

// -----

// CHECK-LABEL:   tt.func @where_kernel(
// CHECK-SAME:  %[[VAL_0:.*]]: !tt.ptr<i64>, %[[VAL_1:.*]]: !tt.ptr<i64>, %[[VAL_2:.*]]: i8) -> tensor<1024xi64> {
// CHECK:           %[[VAL_3:.*]] = arith.constant {output_range = [0, 0]} 0 : i8
// CHECK:           %[[VAL_4:.*]] = arith.constant {output_range = [1024, 1024]} 1024 : i32
// CHECK:           %[[VAL_5:.*]] = tt.get_program_id x {output_range = [0, 2048]} : i32
// CHECK:           %[[VAL_6:.*]] = arith.muli %[[VAL_5]], %[[VAL_4]] {output_range = [0, 2097152]} : i32
// CHECK:           %[[VAL_7:.*]] = tt.make_range {end = 1024 : i32, output_range = [0, 1024], start = 0 : i32} : tensor<1024xi32>
// CHECK:           %[[VAL_8:.*]] = arith.cmpi ne, %[[VAL_2]], %[[VAL_3]] {output_range = [-1, 0]} : i8
// CHECK:           %[[VAL_9:.*]] = arith.select %[[VAL_8]], %[[VAL_0]], %[[VAL_1]] : !tt.ptr<i64>
// CHECK:           %[[VAL_10:.*]] = tt.addptr %[[VAL_9]], %[[VAL_6]] : !tt.ptr<i64>, i32
// CHECK:           %[[VAL_11:.*]] = amdgpu.buffer_load %[[VAL_10]]{{\[}}%[[VAL_7]]] {output_range = [-9223372036854775808, 9223372036854775807]} : tensor<1024xi64>
// CHECK:           tt.return %[[VAL_11]] : tensor<1024xi64>
// CHECK:         }

module attributes {"ttg.num-ctas" = 1 : i32} {
  tt.func @where_kernel(%arg0: !tt.ptr<i64>, %arg1: !tt.ptr<i64>, %arg2: i8) -> tensor<1024xi64> {
    %c0_i8 = arith.constant 0 : i8
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = arith.cmpi ne, %arg2, %c0_i8 : i8
    %4 = arith.select %3, %arg0, %arg1 : !tt.ptr<i64>
    %5 = tt.addptr %4, %1 : !tt.ptr<i64>, i32
    %6 = tt.splat %5 : !tt.ptr<i64> -> tensor<1024x!tt.ptr<i64>>
    %7 = tt.addptr %6, %2 : tensor<1024x!tt.ptr<i64>>, tensor<1024xi32>
    %8 = tt.load %7 : tensor<1024x!tt.ptr<i64>>
    tt.return %8 : tensor<1024xi64>
  }
}

// -----

// CHECK-LABEL:   tt.func @forOpWithHints(
// CHECK-SAME:  %[[VAL_0:.*]]: !tt.ptr<f32>, %[[VAL_1:.*]]: tensor<1024xf32>) -> tensor<1024xf32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant {output_range = [0, 0]} 0 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant {output_range = [1, 1]} 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant {output_range = [128, 128]} 128 : index
// CHECK:           %[[VAL_5:.*]] = tt.get_program_id x {output_range = [0, 2048]} : i32
// CHECK:           %[[VAL_6:.*]] = tt.make_range {end = 1024 : i32, output_range = [0, 1024], start = 0 : i32} : tensor<1024xi32>
// CHECK:           %[[VAL_7:.*]] = tt.addptr %[[VAL_0]], %[[VAL_5]] : !tt.ptr<f32>, i32
// CHECK:           %[[VAL_8:.*]] = arith.extsi %[[VAL_6]] {output_range = [0, 1024]} : tensor<1024xi32> to tensor<1024xi64>
// CHECK:           %[[VAL_9:.*]]:3 = scf.for %[[VAL_10:.*]] = %[[VAL_2]] to %[[VAL_4]] step %[[VAL_3]] iter_args(%[[VAL_11:.*]] = %[[VAL_7]], %[[VAL_12:.*]] = %[[VAL_8]], %[[VAL_13:.*]] = %[[VAL_1]]) -> (!tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>) {
// CHECK:             %[[VAL_14:.*]] = arith.trunci %[[VAL_12]] {output_range = [0, 129024]} : tensor<1024xi64> to tensor<1024xi32>
// CHECK:             %[[VAL_15:.*]] = tt.splat %[[VAL_11]] : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
// CHECK:             %[[VAL_16:.*]] = tt.addptr %[[VAL_15]], %[[VAL_14]] : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
// CHECK:             %[[VAL_17:.*]] = tt.load %[[VAL_16]] : tensor<1024x!tt.ptr<f32>>
// CHECK:             %[[VAL_18:.*]] = tt.addptr %[[VAL_11]], %[[VAL_5]] : !tt.ptr<f32>, i32
// CHECK:             %[[VAL_19:.*]] = arith.extsi %[[VAL_6]] {output_range = [0, 1024]} : tensor<1024xi32> to tensor<1024xi64>
// CHECK:             %[[VAL_20:.*]] = arith.addi %[[VAL_19]], %[[VAL_12]] {output_range = [0, 130048]} : tensor<1024xi64>
// CHECK:             %[[VAL_21:.*]] = tt.addptr %[[VAL_18]], %[[VAL_5]] : !tt.ptr<f32>, i32
// CHECK:             %[[VAL_22:.*]] = arith.addf %[[VAL_17]], %[[VAL_13]] : tensor<1024xf32>
// CHECK:             scf.yield %[[VAL_21]], %[[VAL_20]], %[[VAL_22]] : !tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>
// CHECK:           } {tt.divisibility_arg1 = dense<16> : tensor<1xi32>, tt.divisibility_arg2 = dense<16> : tensor<1xi32>}
// CHECK:           %[[VAL_23:.*]] = tt.addptr %[[VAL_24:.*]]#0, %[[VAL_5]] : !tt.ptr<f32>, i32
// CHECK:           %[[VAL_25:.*]] = arith.extsi %[[VAL_6]] {output_range = [0, 1024]} : tensor<1024xi32> to tensor<1024xi64>
// CHECK:           %[[VAL_26:.*]] = arith.addi %[[VAL_25]], %[[VAL_24]]#1 {output_range = [0, 131072]} : tensor<1024xi64>
// CHECK:           %[[VAL_27:.*]] = tt.splat %[[VAL_23]] : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
// CHECK:           %[[VAL_28:.*]] = tt.addptr %[[VAL_27]], %[[VAL_26]] : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
// CHECK:           %[[VAL_29:.*]] = tt.load %[[VAL_28]] : tensor<1024x!tt.ptr<f32>>
// CHECK:           tt.return %[[VAL_29]] : tensor<1024xf32>
// CHECK:         }

module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @forOpWithHints(%arg0: !tt.ptr<f32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %2 = tt.addptr %arg0, %0 : !tt.ptr<f32>, i32
    %3 = arith.extsi %1 : tensor<1024xi32> to tensor<1024xi64>
    %4:3 = scf.for %arg2 = %c0 to %c128 step %c1 iter_args(%arg3 = %2, %arg4 = %3, %arg5 = %arg1) -> (!tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>) {
      %11 = arith.trunci %arg4 : tensor<1024xi64> to tensor<1024xi32>
      %12 = tt.splat %arg3 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      %13 = tt.addptr %12, %11 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
      %14 = tt.load %13 : tensor<1024x!tt.ptr<f32>>
      %15 = tt.addptr %arg3, %0 : !tt.ptr<f32>, i32
      %16 = arith.extsi %1 : tensor<1024xi32> to tensor<1024xi64>
      %17 = arith.addi %16, %arg4 : tensor<1024xi64>
      %18 = tt.addptr %15, %0 : !tt.ptr<f32>, i32
      %19 = arith.addf %14, %arg5 : tensor<1024xf32>
      scf.yield %18, %17, %19 : !tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>
    } {tt.divisibility_arg1 = dense<16> : tensor<1xi32>, tt.divisibility_arg2 = dense<16> : tensor<1xi32>}
    %5 = tt.addptr %4#0, %0 : !tt.ptr<f32>, i32
    %6 = arith.extsi %1 : tensor<1024xi32> to tensor<1024xi64>
    %7 = arith.addi %6, %4#1 : tensor<1024xi64>
    %8 = tt.splat %5 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %9 = tt.addptr %8, %7 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
    %10 = tt.load %9 : tensor<1024x!tt.ptr<f32>>
    tt.return %10 : tensor<1024xf32>
  }
}

// -----

// CHECK-LABEL:   tt.func public @scalar_pointers(
// CHECK-SAME:  %[[VAL_0:.*]]: !tt.ptr<i64> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
// CHECK:           %[[VAL_1:.*]] = arith.constant {output_range = [0, 0]} 0 : i64
// CHECK:           %[[VAL_2:.*]] = arith.constant {output_range = [1, 1]} 1 : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant {output_range = [100, 100]} 100 : i32
// CHECK:           %[[VAL_4:.*]] = tt.addptr %[[VAL_0]], %[[VAL_2]] : !tt.ptr<i64>, i32
// CHECK:           %[[VAL_5:.*]] = scf.for %[[VAL_6:.*]] = %[[VAL_2]] to %[[VAL_3]] step %[[VAL_2]] iter_args(%[[VAL_7:.*]] = %[[VAL_4]]) -> (!tt.ptr<i64>)  : i32 {
// CHECK:             tt.store %[[VAL_7]], %[[VAL_1]] : !tt.ptr<i64>
// CHECK:             %[[VAL_8:.*]] = tt.addptr %[[VAL_7]], %[[VAL_2]] : !tt.ptr<i64>, i32
// CHECK:             scf.yield %[[VAL_8]] : !tt.ptr<i64>
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }

module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func public @scalar_pointers(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c0_i64 = arith.constant 0 : i64
    %c1_i32 = arith.constant 1 : i32
    %c100_i32 = arith.constant 100 : i32
    %0 = tt.addptr %arg0, %c1_i32 : !tt.ptr<i64>, i32
    %1 = scf.for %arg1 = %c1_i32 to %c100_i32 step %c1_i32 iter_args(%arg2 = %0) -> (!tt.ptr<i64>)  : i32 {
      tt.store %arg2, %c0_i64 : !tt.ptr<i64>
      %2 = tt.addptr %arg2, %c1_i32 : !tt.ptr<i64>, i32
      scf.yield %2 : !tt.ptr<i64>
    }
    tt.return
  }
}

// -----

// CHECK-LABEL:   tt.func @scalar_if(
// CHECK-SAME:  %[[VAL_0:.*]]: !tt.ptr<f32>, %[[VAL_1:.*]]: tensor<1024xf32>, %[[VAL_2:.*]]: i1) -> f32 {
// CHECK:           %[[VAL_3:.*]] = arith.constant {output_range = [1, 1]} 1 : i32
// CHECK:           %[[VAL_4:.*]] = arith.constant {output_range = [100, 100]} 100 : i32
// CHECK:           %[[VAL_5:.*]] = tt.addptr %[[VAL_0]], %[[VAL_3]] : !tt.ptr<f32>, i32
// CHECK:           %[[VAL_6:.*]] = scf.if %[[VAL_2]] -> (!tt.ptr<f32>) {
// CHECK:             %[[VAL_7:.*]] = tt.addptr %[[VAL_5]], %[[VAL_3]] : !tt.ptr<f32>, i32
// CHECK:             scf.yield %[[VAL_7]] : !tt.ptr<f32>
// CHECK:           } else {
// CHECK:             %[[VAL_8:.*]] = tt.addptr %[[VAL_5]], %[[VAL_4]] : !tt.ptr<f32>, i32
// CHECK:             scf.yield %[[VAL_8]] : !tt.ptr<f32>
// CHECK:           }
// CHECK:           %[[VAL_9:.*]] = tt.load %[[VAL_6]] : !tt.ptr<f32>
// CHECK:           tt.return %[[VAL_9]] : f32
// CHECK:         }

module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @scalar_if(%arg0: !tt.ptr<f32>, %arg1: tensor<1024xf32>, %arg2: i1) -> f32 {
    %c1_i32 = arith.constant 1 : i32
    %c100_i32 = arith.constant 100 : i32
    %0 = tt.addptr %arg0, %c1_i32 : !tt.ptr<f32>, i32
    %1 = scf.if %arg2 -> (!tt.ptr<f32>) {
      %3 = tt.addptr %0, %c1_i32 : !tt.ptr<f32>, i32
      scf.yield %3 : !tt.ptr<f32>
    } else {
      %3 = tt.addptr %0, %c100_i32 : !tt.ptr<f32>, i32
      scf.yield %3 : !tt.ptr<f32>
    }
    %2 = tt.load %1 : !tt.ptr<f32>
    tt.return %2 : f32
  }
}

// -----

// CHECK-LABEL:   tt.func @scalar_cond_branch(
// CHECK-SAME:  %[[VAL_0:.*]]: !tt.ptr<f32>, %[[VAL_1:.*]]: !tt.ptr<f32>, %[[VAL_2:.*]]: i1) -> f32 {
// CHECK:           cf.cond_br %[[VAL_2]], ^bb1(%[[VAL_0]] : !tt.ptr<f32>), ^bb1(%[[VAL_1]] : !tt.ptr<f32>)
// CHECK:         ^bb1(%[[VAL_3:.*]]: !tt.ptr<f32>):
// CHECK:           %[[VAL_4:.*]] = tt.load %[[VAL_3]] : !tt.ptr<f32>
// CHECK:           tt.return %[[VAL_4]] : f32
// CHECK:         }

module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @scalar_cond_branch(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i1) -> f32 {
    cf.cond_br %arg2, ^bb1(%arg0 : !tt.ptr<f32>), ^bb2(%arg1 : !tt.ptr<f32>)
  ^bb1(%0: !tt.ptr<f32>):  // pred: ^bb0
    %1 = tt.load %0 : !tt.ptr<f32>
    tt.return %1 : f32
  ^bb2(%2: !tt.ptr<f32>):  // pred: ^bb0
    %3 = tt.load %2 : !tt.ptr<f32>
    tt.return %3 : f32
  }
}

// -----

// CHECK-LABEL:   tt.func @flipFlopForOpSimple(
// CHECK-SAME:  %[[VAL_0:.*]]: !tt.ptr<f32>, %[[VAL_1:.*]]: tensor<1024xf32>) -> tensor<1024xf32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant {output_range = [1024, 1024]} 1024 : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant {output_range = [0, 0]} 0 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant {output_range = [128, 128]} 128 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant {output_range = [1, 1]} 1 : index
// CHECK:           %[[VAL_6:.*]] = tt.get_program_id x {output_range = [0, 2048]} : i32
// CHECK:           %[[VAL_7:.*]] = arith.muli %[[VAL_6]], %[[VAL_2]] {output_range = [0, 2097152]} : i32
// CHECK:           %[[VAL_8:.*]] = tt.make_range {end = 1024 : i32, output_range = [0, 1024], start = 0 : i32} : tensor<1024xi32>
// CHECK:           %[[VAL_9:.*]] = tt.addptr %[[VAL_0]], %[[VAL_7]] : !tt.ptr<f32>, i32
// CHECK:           %[[VAL_10:.*]] = arith.extsi %[[VAL_8]] {output_range = [0, 1024]} : tensor<1024xi32> to tensor<1024xi64>
// CHECK:           %[[VAL_11:.*]] = tt.addptr %[[VAL_0]], %[[VAL_7]] : !tt.ptr<f32>, i32
// CHECK:           %[[VAL_12:.*]] = arith.extsi %[[VAL_8]] {output_range = [0, 1024]} : tensor<1024xi32> to tensor<1024xi64>
// CHECK:           %[[VAL_13:.*]]:5 = scf.for %[[VAL_14:.*]] = %[[VAL_3]] to %[[VAL_4]] step %[[VAL_5]] iter_args(%[[VAL_15:.*]] = %[[VAL_11]], %[[VAL_16:.*]] = %[[VAL_12]], %[[VAL_17:.*]] = %[[VAL_9]], %[[VAL_18:.*]] = %[[VAL_10]], %[[VAL_19:.*]] = %[[VAL_1]]) -> (!tt.ptr<f32>, tensor<1024xi64>, !tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>) {
// CHECK:             %[[VAL_20:.*]] = tt.addptr %[[VAL_17]], %[[VAL_7]] : !tt.ptr<f32>, i32
// CHECK:             %[[VAL_21:.*]] = arith.extsi %[[VAL_8]] {output_range = [0, 1024]} : tensor<1024xi32> to tensor<1024xi64>
// CHECK:             %[[VAL_22:.*]] = arith.addi %[[VAL_21]], %[[VAL_18]] {output_range = [0, 66560]} : tensor<1024xi64>
// CHECK:             %[[VAL_23:.*]] = tt.splat %[[VAL_20]] : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
// CHECK:             %[[VAL_24:.*]] = tt.addptr %[[VAL_23]], %[[VAL_22]] : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
// CHECK:             %[[VAL_25:.*]] = tt.load %[[VAL_24]] : tensor<1024x!tt.ptr<f32>>
// CHECK:             %[[VAL_26:.*]] = arith.addf %[[VAL_25]], %[[VAL_19]] : tensor<1024xf32>
// CHECK:             scf.yield %[[VAL_20]], %[[VAL_22]], %[[VAL_15]], %[[VAL_16]], %[[VAL_26]] : !tt.ptr<f32>, tensor<1024xi64>, !tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>
// CHECK:           }
// CHECK:           %[[VAL_27:.*]] = tt.addptr %[[VAL_28:.*]]#0, %[[VAL_7]] : !tt.ptr<f32>, i32
// CHECK:           %[[VAL_29:.*]] = arith.extsi %[[VAL_8]] {output_range = [0, 1024]} : tensor<1024xi32> to tensor<1024xi64>
// CHECK:           %[[VAL_30:.*]] = arith.addi %[[VAL_29]], %[[VAL_28]]#1 {output_range = [0, 66560]} : tensor<1024xi64>
// CHECK:           %[[VAL_31:.*]] = tt.splat %[[VAL_27]] : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
// CHECK:           %[[VAL_32:.*]] = tt.addptr %[[VAL_31]], %[[VAL_30]] : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
// CHECK:           %[[VAL_33:.*]] = tt.load %[[VAL_32]] : tensor<1024x!tt.ptr<f32>>
// CHECK:           tt.return %[[VAL_33]] : tensor<1024xf32>
// CHECK:         }

module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @flipFlopForOpSimple(%arg0: !tt.ptr<f32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %c1024_i32 = arith.constant 1024 : i32
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    %4 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    %5 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    %6 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    %7:5 = scf.for %arg2 = %c0 to %c128 step %c1 iter_args(%arg3 = %5, %arg4 = %6, %arg5 = %3, %arg6 = %4, %arg7 = %arg1) -> (!tt.ptr<f32>, tensor<1024xi64>, !tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>) {
      %14 = tt.addptr %arg5, %1 : !tt.ptr<f32>, i32
      %15 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
      %16 = arith.addi %15, %arg6 : tensor<1024xi64>
      %17 = tt.splat %14 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      %18 = tt.addptr %17, %16 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
      %19 = tt.load %18 : tensor<1024x!tt.ptr<f32>>
      %20 = arith.addf %19, %arg7 : tensor<1024xf32>
      scf.yield %14, %16, %arg3, %arg4, %20 : !tt.ptr<f32>, tensor<1024xi64>, !tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>
    }
    %8 = tt.addptr %7#0, %1 : !tt.ptr<f32>, i32
    %9 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    %10 = arith.addi %9, %7#1 : tensor<1024xi64>
    %11 = tt.splat %8 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %12 = tt.addptr %11, %10 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
    %13 = tt.load %12 : tensor<1024x!tt.ptr<f32>>
    tt.return %13 : tensor<1024xf32>
  }
}

// -----

// CHECK-LABEL:   tt.func @flipFlopForOpComplex(
// CHECK-SAME:  %[[VAL_0:.*]]: !tt.ptr<f32>, %[[VAL_1:.*]]: !tt.ptr<f32>, %[[VAL_2:.*]]: tensor<1024xf32>) -> (tensor<1024xf32>, tensor<1024xf32>) {
// CHECK:           %[[VAL_3:.*]] = arith.constant {output_range = [1024, 1024]} 1024 : i32
// CHECK:           %[[VAL_4:.*]] = arith.constant {output_range = [0, 0]} 0 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant {output_range = [128, 128]} 128 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant {output_range = [1, 1]} 1 : index
// CHECK:           %[[VAL_7:.*]] = tt.get_program_id x {output_range = [0, 2048]} : i32
// CHECK:           %[[VAL_8:.*]] = arith.muli %[[VAL_7]], %[[VAL_3]] {output_range = [0, 2097152]} : i32
// CHECK:           %[[VAL_9:.*]] = tt.make_range {end = 1024 : i32, output_range = [0, 1024], start = 0 : i32} : tensor<1024xi32>
// CHECK:           %[[VAL_10:.*]] = tt.addptr %[[VAL_0]], %[[VAL_8]] : !tt.ptr<f32>, i32
// CHECK:           %[[VAL_11:.*]] = arith.extsi %[[VAL_9]] {output_range = [0, 1024]} : tensor<1024xi32> to tensor<1024xi64>
// CHECK:           %[[VAL_12:.*]] = tt.addptr %[[VAL_1]], %[[VAL_8]] : !tt.ptr<f32>, i32
// CHECK:           %[[VAL_13:.*]] = arith.extsi %[[VAL_9]] {output_range = [0, 1024]} : tensor<1024xi32> to tensor<1024xi64>
// CHECK:           %[[VAL_14:.*]]:6 = scf.for %[[VAL_15:.*]] = %[[VAL_4]] to %[[VAL_5]] step %[[VAL_6]] iter_args(%[[VAL_16:.*]] = %[[VAL_10]], %[[VAL_17:.*]] = %[[VAL_11]], %[[VAL_18:.*]] = %[[VAL_2]], %[[VAL_19:.*]] = %[[VAL_12]], %[[VAL_20:.*]] = %[[VAL_13]], %[[VAL_21:.*]] = %[[VAL_2]]) -> (!tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>, !tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>) {
// CHECK:             %[[VAL_22:.*]] = tt.addptr %[[VAL_16]], %[[VAL_8]] : !tt.ptr<f32>, i32
// CHECK:             %[[VAL_23:.*]] = arith.extsi %[[VAL_9]] {output_range = [0, 1024]} : tensor<1024xi32> to tensor<1024xi64>
// CHECK:             %[[VAL_24:.*]] = arith.addi %[[VAL_23]], %[[VAL_17]] {output_range = [0, 66560]} : tensor<1024xi64>
// CHECK:             %[[VAL_25:.*]] = tt.splat %[[VAL_22]] : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
// CHECK:             %[[VAL_26:.*]] = tt.addptr %[[VAL_25]], %[[VAL_24]] : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
// CHECK:             %[[VAL_27:.*]] = tt.load %[[VAL_26]] : tensor<1024x!tt.ptr<f32>>
// CHECK:             %[[VAL_28:.*]] = arith.addf %[[VAL_27]], %[[VAL_18]] : tensor<1024xf32>
// CHECK:             %[[VAL_29:.*]] = tt.addptr %[[VAL_19]], %[[VAL_8]] : !tt.ptr<f32>, i32
// CHECK:             %[[VAL_30:.*]] = arith.extsi %[[VAL_9]] {output_range = [0, 1024]} : tensor<1024xi32> to tensor<1024xi64>
// CHECK:             %[[VAL_31:.*]] = arith.addi %[[VAL_30]], %[[VAL_20]] {output_range = [0, 66560]} : tensor<1024xi64>
// CHECK:             %[[VAL_32:.*]] = tt.splat %[[VAL_29]] : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
// CHECK:             %[[VAL_33:.*]] = tt.addptr %[[VAL_32]], %[[VAL_31]] : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
// CHECK:             %[[VAL_34:.*]] = tt.load %[[VAL_33]] : tensor<1024x!tt.ptr<f32>>
// CHECK:             %[[VAL_35:.*]] = arith.addf %[[VAL_34]], %[[VAL_21]] : tensor<1024xf32>
// CHECK:             scf.yield %[[VAL_29]], %[[VAL_31]], %[[VAL_35]], %[[VAL_22]], %[[VAL_24]], %[[VAL_28]] : !tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>, !tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>
// CHECK:           }
// CHECK:           %[[VAL_36:.*]] = tt.addptr %[[VAL_37:.*]]#0, %[[VAL_8]] : !tt.ptr<f32>, i32
// CHECK:           %[[VAL_38:.*]] = arith.extsi %[[VAL_9]] {output_range = [0, 1024]} : tensor<1024xi32> to tensor<1024xi64>
// CHECK:           %[[VAL_39:.*]] = arith.addi %[[VAL_38]], %[[VAL_37]]#1 {output_range = [0, 67584]} : tensor<1024xi64>
// CHECK:           %[[VAL_40:.*]] = tt.splat %[[VAL_36]] : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
// CHECK:           %[[VAL_41:.*]] = tt.addptr %[[VAL_40]], %[[VAL_39]] : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
// CHECK:           %[[VAL_42:.*]] = tt.load %[[VAL_41]] : tensor<1024x!tt.ptr<f32>>
// CHECK:           %[[VAL_43:.*]] = tt.addptr %[[VAL_37]]#3, %[[VAL_8]] : !tt.ptr<f32>, i32
// CHECK:           %[[VAL_44:.*]] = arith.extsi %[[VAL_9]] {output_range = [0, 1024]} : tensor<1024xi32> to tensor<1024xi64>
// CHECK:           %[[VAL_45:.*]] = arith.addi %[[VAL_44]], %[[VAL_37]]#4 {output_range = [0, 67584]} : tensor<1024xi64>
// CHECK:           %[[VAL_46:.*]] = tt.splat %[[VAL_43]] : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
// CHECK:           %[[VAL_47:.*]] = tt.addptr %[[VAL_46]], %[[VAL_45]] : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
// CHECK:           %[[VAL_48:.*]] = tt.load %[[VAL_47]] : tensor<1024x!tt.ptr<f32>>
// CHECK:           tt.return %[[VAL_42]], %[[VAL_48]] : tensor<1024xf32>, tensor<1024xf32>
// CHECK:         }

module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @flipFlopForOpComplex(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: tensor<1024xf32>) -> (tensor<1024xf32>, tensor<1024xf32>) {
    %c1024_i32 = arith.constant 1024 : i32
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    %4 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    %5 = tt.addptr %arg1, %1 : !tt.ptr<f32>, i32
    %6 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    %7:6 = scf.for %arg3 = %c0 to %c128 step %c1 iter_args(%arg4 = %3, %arg5 = %4, %arg6 = %arg2, %arg7 = %5, %arg8 = %6, %arg9 = %arg2) -> (!tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>, !tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>) {
      %20 = tt.addptr %arg4, %1 : !tt.ptr<f32>, i32
      %21 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
      %22 = arith.addi %21, %arg5 : tensor<1024xi64>
      %23 = tt.splat %20 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      %24 = tt.addptr %23, %22 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
      %25 = tt.load %24 : tensor<1024x!tt.ptr<f32>>
      %26 = arith.addf %25, %arg6 : tensor<1024xf32>
      %27 = tt.addptr %arg7, %1 : !tt.ptr<f32>, i32
      %28 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
      %29 = arith.addi %28, %arg8 : tensor<1024xi64>
      %30 = tt.splat %27 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      %31 = tt.addptr %30, %29 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
      %32 = tt.load %31 : tensor<1024x!tt.ptr<f32>>
      %33 = arith.addf %32, %arg9 : tensor<1024xf32>
      scf.yield %27, %29, %33, %20, %22, %26 : !tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>, !tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>
    }
    %8 = tt.addptr %7#0, %1 : !tt.ptr<f32>, i32
    %9 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    %10 = arith.addi %9, %7#1 : tensor<1024xi64>
    %11 = tt.splat %8 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %12 = tt.addptr %11, %10 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
    %13 = tt.load %12 : tensor<1024x!tt.ptr<f32>>
    %14 = tt.addptr %7#3, %1 : !tt.ptr<f32>, i32
    %15 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    %16 = arith.addi %15, %7#4 : tensor<1024xi64>
    %17 = tt.splat %14 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %18 = tt.addptr %17, %16 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
    %19 = tt.load %18 : tensor<1024x!tt.ptr<f32>>
    tt.return %13, %19 : tensor<1024xf32>, tensor<1024xf32>
  }
}
