module {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {noinline = false} {
    %c3_i32 = arith.constant 3 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<8x8xf32>
    %c7_i32 = arith.constant 7 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<4x8xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<8x4xf32>
    %c4_i32 = arith.constant 4 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg3, %c7_i32 : i32
    %2 = arith.divsi %1, %c8_i32 : i32
    %3 = arith.addi %arg4, %c7_i32 : i32
    %4 = arith.divsi %3, %c8_i32 : i32
    %5 = arith.divsi %0, %4 : i32
    %6 = arith.subi %2, %5 : i32
    %7 = arith.minsi %6, %c1_i32 : i32
    %8 = arith.remsi %0, %4 : i32
    %9 = arith.remsi %8, %7 : i32
    %10 = arith.addi %5, %9 : i32
    %11 = arith.divsi %8, %7 : i32
    %12 = arith.muli %10, %c8_i32 : i32
    %13 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %14 = tt.splat %12 : i32 -> tensor<8xi32>
    %15 = arith.addi %14, %13 : tensor<8xi32>
    %16 = tt.splat %arg3 : i32 -> tensor<8xi32>
    %17 = arith.remsi %15, %16 : tensor<8xi32>
    %18 = arith.muli %11, %c8_i32 : i32
    %19 = tt.splat %18 : i32 -> tensor<8xi32>
    %20 = arith.addi %19, %13 : tensor<8xi32>
    %21 = tt.splat %arg4 : i32 -> tensor<8xi32>
    %22 = arith.remsi %20, %21 : tensor<8xi32>
    %23 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %24 = tt.expand_dims %17 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %25 = tt.splat %arg6 : i32 -> tensor<8x1xi32>
    %26 = arith.muli %24, %25 : tensor<8x1xi32>
    %27 = tt.expand_dims %23 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %28 = tt.splat %arg7 : i32 -> tensor<1x4xi32>
    %29 = arith.muli %27, %28 : tensor<1x4xi32>
    %30 = tt.broadcast %26 : tensor<8x1xi32> -> tensor<8x4xi32>
    %31 = tt.broadcast %29 : tensor<1x4xi32> -> tensor<8x4xi32>
    %32 = arith.addi %30, %31 : tensor<8x4xi32>
    %33 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<8x4x!tt.ptr<f32>>
    %34 = tt.addptr %33, %32 : tensor<8x4x!tt.ptr<f32>>, tensor<8x4xi32>
    %35 = tt.expand_dims %23 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %36 = tt.splat %arg8 : i32 -> tensor<4x1xi32>
    %37 = arith.muli %35, %36 : tensor<4x1xi32>
    %38 = tt.expand_dims %22 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %39 = tt.splat %arg9 : i32 -> tensor<1x8xi32>
    %40 = arith.muli %38, %39 : tensor<1x8xi32>
    %41 = tt.broadcast %37 : tensor<4x1xi32> -> tensor<4x8xi32>
    %42 = tt.broadcast %40 : tensor<1x8xi32> -> tensor<4x8xi32>
    %43 = arith.addi %41, %42 : tensor<4x8xi32>
    %44 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x8x!tt.ptr<f32>>
    %45 = tt.addptr %44, %43 : tensor<4x8x!tt.ptr<f32>>, tensor<4x8xi32>
    %46 = arith.addi %arg5, %c3_i32 : i32
    %47 = arith.divsi %46, %c4_i32 : i32
    %48 = arith.muli %arg7, %c4_i32 : i32
    %49 = tt.splat %48 : i32 -> tensor<8x4xi32>
    %50 = arith.muli %arg8, %c4_i32 : i32
    %51 = tt.splat %50 : i32 -> tensor<4x8xi32>
    %52:3 = scf.for %arg12 = %c0_i32 to %47 step %c1_i32 iter_args(%arg13 = %cst, %arg14 = %34, %arg15 = %45) -> (tensor<8x8xf32>, tensor<8x4x!tt.ptr<f32>>, tensor<4x8x!tt.ptr<f32>>)  : i32 {
      %71 = arith.muli %arg12, %c4_i32 : i32
      %72 = arith.subi %arg5, %71 : i32
      %73 = tt.splat %72 : i32 -> tensor<1x4xi32>
      %74 = arith.cmpi slt, %27, %73 : tensor<1x4xi32>
      %75 = tt.broadcast %74 : tensor<1x4xi1> -> tensor<8x4xi1>
      %76 = tt.load %arg14, %75, %cst_1 : tensor<8x4x!tt.ptr<f32>>
      %77 = tt.splat %72 : i32 -> tensor<4x1xi32>
      %78 = arith.cmpi slt, %35, %77 : tensor<4x1xi32>
      %79 = tt.broadcast %78 : tensor<4x1xi1> -> tensor<4x8xi1>
      %80 = tt.load %arg15, %79, %cst_0 : tensor<4x8x!tt.ptr<f32>>
      %81 = tt.dot %76, %80, %arg13 : tensor<8x4xf32> * tensor<4x8xf32> -> tensor<8x8xf32>
      %82 = tt.addptr %arg14, %49 : tensor<8x4x!tt.ptr<f32>>, tensor<8x4xi32>
      %83 = tt.addptr %arg15, %51 : tensor<4x8x!tt.ptr<f32>>, tensor<4x8xi32>
      scf.yield %81, %82, %83 : tensor<8x8xf32>, tensor<8x4x!tt.ptr<f32>>, tensor<4x8x!tt.ptr<f32>>
    }
    %53 = tt.expand_dims %15 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %54 = tt.splat %arg10 : i32 -> tensor<8x1xi32>
    %55 = arith.muli %54, %53 : tensor<8x1xi32>
    %56 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>>
    %57 = tt.addptr %56, %55 : tensor<8x1x!tt.ptr<f32>>, tensor<8x1xi32>
    %58 = tt.expand_dims %20 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %59 = tt.splat %arg11 : i32 -> tensor<1x8xi32>
    %60 = arith.muli %59, %58 : tensor<1x8xi32>
    %61 = tt.broadcast %57 : tensor<8x1x!tt.ptr<f32>> -> tensor<8x8x!tt.ptr<f32>>
    %62 = tt.broadcast %60 : tensor<1x8xi32> -> tensor<8x8xi32>
    %63 = tt.addptr %61, %62 : tensor<8x8x!tt.ptr<f32>>, tensor<8x8xi32>
    %64 = tt.splat %arg3 : i32 -> tensor<8x1xi32>
    %65 = arith.cmpi slt, %53, %64 : tensor<8x1xi32>
    %66 = tt.splat %arg4 : i32 -> tensor<1x8xi32>
    %67 = arith.cmpi slt, %58, %66 : tensor<1x8xi32>
    %68 = tt.broadcast %65 : tensor<8x1xi1> -> tensor<8x8xi1>
    %69 = tt.broadcast %67 : tensor<1x8xi1> -> tensor<8x8xi1>
    %70 = arith.andi %68, %69 : tensor<8x8xi1>
    tt.print " pid: " {hex = false, isSigned = array<i32: 1>} : %0 : i32
    tt.store %63, %52#0, %70 : tensor<8x8x!tt.ptr<f32>>
    tt.return
  }
}