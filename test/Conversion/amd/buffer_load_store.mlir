// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=arch=gfx942 | FileCheck %s

#blocked0 = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
    // CHECK-LABEL: buffer_load
    tt.func @buffer_load(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset : tensor<128xi32, #blocked0>{tt.divisibility=16:i32}) {
        // CHECK: %[[c_mask:.*]] = llvm.mlir.constant(true) : i1
        // CHECK: %[[offset:.*]] = llvm.select %[[c_mask]]
        // CHECK: rocdl.raw.ptr.buffer.load {{.*}}, %[[offset]]
        %ret = amdgpu.buffer_load %arg0[%offset] : tensor<128xf32, #blocked0>
        tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
    // CHECK-LABEL: buffer_load_mask
    tt.func @buffer_load_mask(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset : tensor<128xi32, #blocked0> {tt.divisibility=16:i32}, %N : i32 {tt.divisibility = 16 : i32}) {
        %c256_i32 = arith.constant 256 : i32
        %0 = tt.get_program_id x : i32
        %1 = arith.muli %0, %c256_i32 : i32
        %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked0>
        %3 = tt.splat %1 : i32 -> tensor<128xi32, #blocked0>
        %4 = arith.addi %3, %2 : tensor<128xi32, #blocked0>
        %5 = tt.splat %N: i32 -> tensor<128xi32, #blocked0>
        %7 = arith.cmpi slt, %4, %5: tensor<128xi32, #blocked0>
        // CHECK: %[[mask:.*]] = llvm.extractvalue %{{.*}} : !llvm.struct<(i1, i1, i1, i1)>
        // CHECK: %[[offset:.*]] = llvm.select %[[mask]]
        // CHECK: rocdl.raw.ptr.buffer.load {{.*}}, %[[offset]]
        %ret = amdgpu.buffer_load %arg0[%offset], %7: tensor<128xf32, #blocked0>
        tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
    // CHECK-LABEL: buffer_load_mask_other
    tt.func @buffer_load_mask_other(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset : tensor<128xi32, #blocked0> {tt.divisibility=16:i32}, %N : i32 {tt.divisibility = 16 : i32}) {
        %c256_i32 = arith.constant 256 : i32
        %0 = tt.get_program_id x : i32
        %1 = arith.muli %0, %c256_i32 : i32
        %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked0>
        %3 = tt.splat %1 : i32 -> tensor<128xi32, #blocked0>
        %4 = arith.addi %3, %2 : tensor<128xi32, #blocked0>
        %5 = tt.splat %N: i32 -> tensor<128xi32, #blocked0>
        %7 = arith.cmpi slt, %4, %5: tensor<128xi32, #blocked0>
        %other = arith.constant dense<0.00e+00> : tensor<128xf32, #blocked0>
        // CHECK: %[[mask:.*]] = llvm.extractvalue %{{.*}} : !llvm.struct<(i1, i1, i1, i1)>
        // CHECK: %[[offset:.*]] = llvm.select %[[mask]]
        // CHECK: rocdl.raw.ptr.buffer.load {{.*}}, %[[offset]]
        // CHECK: llvm.select
        %ret = amdgpu.buffer_load %arg0[%offset], %7, %other: tensor<128xf32, #blocked0>
        tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
    // CHECK-LABEL: buffer_store
    tt.func @buffer_store(%value : tensor<128xf32, #blocked0>, %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset : tensor<128xi32, #blocked0>{tt.divisibility=16:i32}) {
        // CHECK: %[[c_mask:.*]] = llvm.mlir.constant(true) : i1
        // CHECK: %[[offset:.*]] = llvm.select %[[c_mask]]
        // CHECK: rocdl.raw.ptr.buffer.store {{.*}}, {{.*}}, %[[offset]]
        amdgpu.buffer_store %value, %arg0[%offset] : tensor<128xf32, #blocked0>
        tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
    // CHECK-LABEL: buffer_store_mask
    tt.func @buffer_store_mask(%value : tensor<128xf32, #blocked0>, %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset : tensor<128xi32, #blocked0> {tt.divisibility=16:i32}, %N : i32 {tt.divisibility = 16 : i32}) {
        %c256_i32 = arith.constant 256 : i32
        %0 = tt.get_program_id x : i32
        %1 = arith.muli %0, %c256_i32 : i32
        %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked0>
        %3 = tt.splat %1 : i32 -> tensor<128xi32, #blocked0>
        %4 = arith.addi %3, %2 : tensor<128xi32, #blocked0>
        %5 = tt.splat %N: i32 -> tensor<128xi32, #blocked0>
        %7 = arith.cmpi slt, %4, %5: tensor<128xi32, #blocked0>
        // CHECK: %[[mask0:.*]] = llvm.extractvalue %{{.*}} : !llvm.struct<(i1, i1, i1, i1)>
        // CHECK: %[[mask1:.*]] = llvm.and %[[mask0]], {{.*}}
        // CHECK: %[[offset:.*]] = llvm.select %[[mask1]]
        // CHECK: rocdl.raw.ptr.buffer.store {{.*}}, {{.*}}, %[[offset]]
        amdgpu.buffer_store %value, %arg0[%offset], %7: tensor<128xf32, #blocked0>
        tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: buffer_load_store_vec4
    tt.func @buffer_load_store_vec4(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32) {
        %c256_i32 = arith.constant 256 : i32
        %0 = tt.get_program_id x : i32
        %1 = arith.muli %0, %c256_i32 : i32
        %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
        %3 = tt.splat %1 : i32 -> tensor<256xi32, #blocked0>
        %4 = arith.addi %3, %2 : tensor<256xi32, #blocked0>
        // Load 8 elements from A with two vectorized load instructions
        // CHECK-COUNT-2: rocdl.raw.ptr.buffer.load {{.*}} : vector<4xf32>
        %9 = amdgpu.buffer_load %arg0[%4] : tensor<256xf32, #blocked0>
        // Load 8 elements from B with two vectorized load instructions
        // CHECK-COUNT-2: rocdl.raw.ptr.buffer.load {{.*}} : vector<4xf32>
        %10 = amdgpu.buffer_load %arg1[%4] : tensor<256xf32, #blocked0>
        %11 = arith.addf %9, %10 : tensor<256xf32, #blocked0>
        // Store 8 elements into C with two vectorized store instructions
        // CHECK-COUNT-2: rocdl.raw.ptr.buffer.store {{.*}} : vector<4xf32>
        amdgpu.buffer_store %11, %arg2[%4]: tensor<256xf32, #blocked0>
        tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: buffer_load_store_vec1
    tt.func @buffer_load_store_vec1(%arg0: !tt.ptr<f32> , %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) {
        %c256_i32 = arith.constant 256 : i32
        %0 = tt.get_program_id x : i32
        %1 = arith.muli %0, %c256_i32 : i32
        %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
        %3 = tt.splat %1 : i32 -> tensor<256xi32, #blocked0>
        %4 = arith.addi %3, %2 : tensor<256xi32, #blocked0>
        %5 = tt.splat %arg3 : i32 -> tensor<256xi32, #blocked0>
        %7 = arith.cmpi slt, %4, %5: tensor<256xi32, #blocked0>
        // Load 8 elements from A with eight scalar load instructions
        // CHECK-COUNT-8: rocdl.raw.ptr.buffer.load {{.*}} : f32
        %9 = amdgpu.buffer_load %arg0[%4], %7 : tensor<256xf32, #blocked0>
        // Load 8 elements from B with two scalar load instructions
        // CHECK-COUNT-8: rocdl.raw.ptr.buffer.load {{.*}} : f32
        %10 = amdgpu.buffer_load %arg1[%4], %7 : tensor<256xf32, #blocked0>
        %11 = arith.addf %9, %10 : tensor<256xf32, #blocked0>
        // Store 8 elements into C with two scalar store instructions
        // CHECK-COUNT-8: rocdl.raw.ptr.buffer.store {{.*}} : f32
        amdgpu.buffer_store %11, %arg2[%4], %7 : tensor<256xf32, #blocked0>
        tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
    // CHECK-LABEL: buffer_load_store_vec2
    tt.func @buffer_load_store_vec2(%arg0: !tt.ptr<f16> {tt.divisibility = 4 : i32}, %arg1: !tt.ptr<f16>{tt.divisibility = 4 : i32}, %arg2: !tt.ptr<f16>{tt.divisibility = 4: i32}, %arg3: i32{tt.divisibility = 4: i32}) {
        %c256_i32 = arith.constant 256 : i32
        %0 = tt.get_program_id x : i32
        %1 = arith.muli %0, %c256_i32 : i32
        %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
        %3 = tt.splat %1 : i32 -> tensor<256xi32, #blocked0>
        %4 = arith.addi %3, %2 : tensor<256xi32, #blocked0>
        %5 = tt.splat %arg3 : i32 -> tensor<256xi32, #blocked0>
        %7 = arith.cmpi slt, %4, %5: tensor<256xi32, #blocked0>
        // Load 8 fp16 elements from A with four i32 scalar load instructions
        // CHECK-COUNT-4: rocdl.raw.ptr.buffer.load {{.*}} : i32
        %9 = amdgpu.buffer_load %arg0[%4], %7 : tensor<256xf16, #blocked0>
        // Load 8 fp16 elements from B with four i32 scalar load instructions
        // CHECK-COUNT-4: rocdl.raw.ptr.buffer.load {{.*}} : i32
        %10 = amdgpu.buffer_load %arg1[%4], %7 : tensor<256xf16, #blocked0>
        %11 = arith.addf %9, %10 : tensor<256xf16, #blocked0>
        // Store 8 fp16 elements into C with four i32 scalar store instructionss
        // CHECK-COUNT-4: rocdl.raw.ptr.buffer.store {{.*}} : i32
        amdgpu.buffer_store %11, %arg2[%4], %7 : tensor<256xf16, #blocked0>
        tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
    // LABEL: buffer_atomic_fadd_f16
    tt.func @buffer_atomic_fadd_f16(%value : tensor<64xf16, #blocked0>, %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %offset : tensor<64xi32, #blocked0>{tt.divisibility = 16 : i32}) {
        // CHECK: %5 = rocdl.make.buffer.rsrc %arg1, %2, %4, %3 : <1> to <8>
        // CHECK: %25 = llvm.bitcast %24 : vector<2xf16> to f32
        // CHECK: %29 = llvm.select %6, %28, %26 : i1, i32
        // CHECK: %30 = llvm.mlir.constant(0 : i32) : i32
        // CHECK: %31 = llvm.mlir.constant(0 : i32) : i32
        // CHECK: rocdl.raw.ptr.buffer.atomic.fadd %25, %5, %29, %30, %31 : f32
        amdgpu.buffer_atomic_fadd %value, %arg0[%offset] : tensor<64xf16, #blocked0>
        tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
    // LABEL: buffer_atomic_fadd_f32
    tt.func @buffer_atomic_fadd_32(%value : tensor<32xf32, #blocked0>, %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset : tensor<32xi32, #blocked0>{tt.divisibility = 16 : i32}) {
        // CHECK: %4 = rocdl.make.buffer.rsrc %arg1, %1, %3, %2 : <1> to <8>
        // CHECK: %21 = llvm.bitcast %20 : vector<1xf32> to f32
        // CHECK: %25 = llvm.select %5, %24, %22 : i1, i32
        // CHECK: %26 = llvm.mlir.constant(0 : i32) : i32
        // CHECK: %27 = llvm.mlir.constant(0 : i32) : i32
        // CHECK: rocdl.raw.ptr.buffer.atomic.fadd %21, %4, %25, %26, %27 : f32
        amdgpu.buffer_atomic_fadd %value, %arg0[%offset] : tensor<32xf32, #blocked0>
        tt.return
  }
}