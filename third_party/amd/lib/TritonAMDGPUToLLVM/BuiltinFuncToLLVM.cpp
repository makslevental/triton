#include "TritonAMDGPUToLLVM/Passes.h"

#include "Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace mlir::triton {
#define GEN_PASS_DEF_CONVERTBUILTINFUNCTOLLVM
#include "TritonAMDGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;

namespace {

class CallOpConversion : public OpRewritePattern<LLVM::CallOp> {
public:
  CallOpConversion(mlir::MLIRContext *context, bool ftz)
      : OpRewritePattern<LLVM::CallOp>(context, 1), ftz(ftz) {}

  LogicalResult
  matchAndRewrite(LLVM::CallOp callOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (isPredicatedLoad(callOp)) {
      return convertPredicatedLoad(callOp, rewriter);
    } else if (isPredicatedStore(callOp)) {
      return convertPredicatedStore(callOp, rewriter);
    } else if (isWrappedLLVMIntrinsic(callOp)) {
      return convertToLLVMIntrinsic(callOp, rewriter);
    } else {
      return failure();
    }
  }

private:
  bool isPredicatedLoad(LLVM::CallOp callOp) const {
    return callOp.getCallee().value().contains(mlir::LLVM::AMD::predicatedLoad);
  }

  bool isPredicatedStore(LLVM::CallOp callOp) const {
    return callOp.getCallee().value().contains(
        mlir::LLVM::AMD::predicatedStore);
  }

  bool isWrappedLLVMIntrinsic(LLVM::CallOp callOp) const {
    if (std::optional<StringRef> callee = callOp.getCallee()) {
      if (callee.value().starts_with("__triton_hip_")) {
        return true;
      }
    }
    return false;
  }

  LogicalResult convertPredicatedStore(LLVM::CallOp callOp,
                                       mlir::PatternRewriter &rewriter) const {
    auto operands = callOp.getOperands();

    auto loc = callOp.getLoc();
    auto ptr = operands[0];
    auto val = operands[1];
    auto pred = operands[2];

    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterStore =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    Block *trueBlock = rewriter.createBlock(afterStore);
    rewriter.setInsertionPointToEnd(currentBlock);
    rewriter.create<LLVM::CondBrOp>(loc, pred, trueBlock, afterStore);
    rewriter.setInsertionPointToStart(trueBlock);
    //               | vialatile | non-tmp | gcn instr gfx94
    // LLVM::StoreOp | 0         | 0       | (cg) global store
    //               | 0         | 1       | (cs) global store nt
    //               | 1         | 0/1     | (wt) global store sc0 sc1
    auto [volatileFlag, nonTmpFlag] =
        mlir::LLVM::AMD::getCacheModifierFlagsForPredicatedCall(callOp);
    auto storeOp = rewriter.create<LLVM::StoreOp>(
        loc, val, ptr, /*alignment=*/0, volatileFlag, nonTmpFlag);
    rewriter.create<LLVM::BrOp>(loc, afterStore);
    rewriter.setInsertionPointToStart(afterStore);
    rewriter.eraseOp(callOp);
    return mlir::success();
  }

  LogicalResult convertPredicatedLoad(LLVM::CallOp callOp,
                                      mlir::PatternRewriter &rewriter) const {
    auto operands = callOp.getOperands();
    auto result = callOp.getResult();

    auto loc = callOp.getLoc();
    auto elemTy = result.getType();
    auto ptr = operands[0];
    auto pred = operands[1];
    auto falseVal = operands[2];

    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterLoad =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    afterLoad->addArgument({elemTy}, {loc});
    Block *trueBlock = rewriter.createBlock(afterLoad);
    Block *falseBlock =
        rewriter.splitBlock(trueBlock, rewriter.getInsertionPoint());
    rewriter.setInsertionPointToEnd(currentBlock);
    rewriter.create<LLVM::CondBrOp>(loc, pred, trueBlock, falseBlock);
    rewriter.setInsertionPointToStart(trueBlock);
    //              | vialatile | non-tmp | gcn instr gfx94
    // LLVM::LoadOp | 0         | 0       | (ca) global load
    //              | 0/1       | 1       | (cg) global load nt
    //              | 1         | 0       | (cv) flat load sc0 sc1
    auto [volatileFlag, nonTmpFlag] =
        mlir::LLVM::AMD::getCacheModifierFlagsForPredicatedCall(callOp);
    auto loadOp = rewriter.create<LLVM::LoadOp>(
        loc, elemTy, ptr, /*alignment=*/0, volatileFlag, nonTmpFlag);
    rewriter.create<LLVM::BrOp>(loc, loadOp->getResult(0), afterLoad);
    rewriter.setInsertionPointToStart(falseBlock);
    rewriter.create<LLVM::BrOp>(loc, falseVal, afterLoad);
    rewriter.setInsertionPointToStart(afterLoad);
    Value loadVal = afterLoad->getArgument(0);
    rewriter.replaceOp(callOp, loadVal);
    return mlir::success();
  }

  LogicalResult convertToLLVMIntrinsic(LLVM::CallOp callOp,
                                       mlir::PatternRewriter &rewriter) const {
    StringRef calleeName = callOp.getCallee().value();

    auto operands = callOp.getOperands();
    auto result = callOp.getResult();

    LLVM::LLVMFunctionType calleeType = callOp.getCalleeFunctionType();
    Type returnType = calleeType.getReturnType();

    auto loc = callOp.getLoc();

    // clang-format off
    // define internal noundef i64 @load_acquire_workgroup(unsigned long AS1*)(ptr addrspace(1) noundef readonly captures(none) %input) #0 !dbg !12 {
    // entry:
    //   %0 = load atomic i64, ptr addrspace(1) %input syncscope("workgroup") acquire, align 8, !dbg !16
    //   ret i64 %0, !dbg !17
    // }}
    // clang-format on
    auto buildAtomicLoad =
        [&rewriter, &loc](Value inputPtr, LLVM::AtomicOrdering ordering,
                          std::optional<StringRef> syncGroup = std::nullopt) {
          return rewriter.create<LLVM::LoadOp>(
              loc, i64_ty, inputPtr, /*alignment=*/8,
              /*isVolatile =*/false, /*isNonTemporal*/ false,
              /*isInvariant =*/false, /*isInvariantGroup=*/false, ordering,
              syncGroup.value_or(StringRef()));
        };

    // clang-format off
    // define internal noundef i64 @store_release_workgroup(unsigned long AS1*)(ptr addrspace(1) noundef writeonly captures(none) %input) #0 !dbg !33 {
    // entry:
    //   store atomic i64 1, ptr addrspace(1) %input syncscope("workgroup") release, align 8, !dbg !34
    //   ret i64 1, !dbg !35
    // }
    // clang-format on
    auto buildAtomicStore =
        [&rewriter, &loc](Value value, Value inputPtr,
                          LLVM::AtomicOrdering ordering,
                          std::optional<StringRef> syncGroup = std::nullopt) {
          return rewriter.create<LLVM::StoreOp>(
              loc, value, inputPtr, /*alignment=*/8,
              /*isVolatile =*/false, /*isNonTemporal*/ false,
              /*isInvariantGroup=*/false, ordering,
              syncGroup.value_or(StringRef()));
        };

    // clang-format off
    // define internal noundef i32 @red_add_release_agent(int AS1*, int)(
    //  ptr addrspace(1) noundef captures(none) %atomic_address,
    //  i32 noundef %value
    // ) #0 !dbg !63 {
    // entry:
    //   %0 = atomicrmw add ptr addrspace(1) %atomic_address, i32 %value syncscope("agent") release, align 4, !dbg !64
    //   ret i32 %0, !dbg !65
    // }
    // clang-format on
    auto buildAtomicFetchAdd =
        [&rewriter, &loc](Value atomicAddress, Value value,
                          LLVM::AtomicOrdering ordering,
                          std::optional<StringRef> syncGroup = std::nullopt) {
          return rewriter.create<LLVM::AtomicRMWOp>(
              loc, LLVM::AtomicBinOp::add, atomicAddress, value, ordering,
              syncGroup.value_or(StringRef()), /*alignment*/ 4);
        };

    // clang-format off
    // define internal noundef range(i64 0, 2) i64 @atom_cas_acquire_relaxed_agent(int AS1*, int AS1*, int)(
    //  ptr addrspace(1) noundef captures(none) %atomic_address,
    //  ptr addrspace(1) noundef captures(none) %compare,
    //  i32 noundef %value
    // ) #0 !dbg !87 {
    // entry:
    //   %0 = load i32, ptr addrspace(1) %compare, align 4, !dbg !88
    //   %1 = cmpxchg ptr addrspace(1) %atomic_address, i32 %0, i32 %value syncscope("agent") acquire monotonic, align 4, !dbg !88
    //   %2 = extractvalue { i32, i1 } %1, 1, !dbg !88
    //   br i1 %2, label %cmpxchg.continue, label %cmpxchg.store_expected, !dbg !88

    // cmpxchg.store_expected:
    //   %3 = extractvalue { i32, i1 } %1, 0
    //   store i32 %3, ptr addrspace(1) %compare, align 4, !dbg !88
    //   br label %cmpxchg.continue, !dbg !88

    // cmpxchg.continue:
    //   %conv = zext i1 %2 to i64, !dbg !88
    //   ret i64 %conv, !dbg !89
    // }
    // clang-format on
    auto buildAtomicCompareExchangeStrong =
        [&rewriter, &loc](Value atomicAddress, Value compare, Value value,
                          LLVM::AtomicOrdering successOrdering,
                          LLVM::AtomicOrdering failureOrdering,
                          std::optional<StringRef> syncGroup = std::nullopt) {
          auto cmp = rewriter.create<LLVM::LoadOp>(loc, i32_ty, compare,
                                                   /*alignment=*/4);
          auto cmpxchg = rewriter.create<LLVM::AtomicCmpXchgOp>(
              loc, atomicAddress, cmp, value, successOrdering, failureOrdering,
              syncGroup.value_or(StringRef()),
              /*alignment*/ 4);
          auto extractOne = rewriter.create<LLVM::ExtractValueOp>(
              loc, cmpxchg, SmallVector<int64_t>{1});

          Block *currentBlock = rewriter.getInsertionBlock();
          Block *afterStore =
              rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
          Block *trueBlock = rewriter.createBlock(afterStore);
          rewriter.setInsertionPointToEnd(currentBlock);
          rewriter.create<LLVM::CondBrOp>(loc, extractOne, trueBlock,
                                          afterStore);
          rewriter.setInsertionPointToStart(trueBlock);
          // cmpxchg.store_expected:
          auto extractZero = rewriter.create<LLVM::ExtractValueOp>(
              loc, cmpxchg, SmallVector<int64_t>{0});
          (void)rewriter.create<LLVM::StoreOp>(loc, extractZero, compare,
                                               /*alignment=*/4);
          rewriter.create<LLVM::BrOp>(loc, afterStore);
          rewriter.setInsertionPointToStart(afterStore);
          // cmpxchg.continue:
          return rewriter.create<LLVM::ZExtOp>(loc, i64_ty, extractOne,
                                               /*nonNeg*/ true);
        };

    Operation *replacementOp = nullptr;
    if (calleeName == "__triton_hip_iabs") {
      assert(operands.size() == 1);
      replacementOp = rewriter.create<LLVM::AbsOp>(loc, returnType, operands[0],
                                                   /*is_int_min_poison=*/false);
    } else if (calleeName == "__triton_hip_fabs") {
      assert(operands.size() == 1);
      replacementOp =
          rewriter.create<LLVM::FAbsOp>(loc, returnType, operands[0]);
    } else if (calleeName == "__triton_hip_llrint") {
      assert(operands.size() == 1);
      // Note, LrintOp and LlrintOp result in a code-gen error
      Operation *op = rewriter.create<LLVM::RintOp>(loc, operands[0].getType(),
                                                    operands[0]);
      replacementOp =
          rewriter.create<LLVM::FPToSIOp>(loc, returnType, op->getResult(0));
    } else if (calleeName == "__triton_hip_fast_fdividef") {
      assert(operands.size() == 2);
      const char *intrinsic = "llvm.amdgcn.rcp.f32";
      auto rcpOp = LLVM::createLLVMIntrinsicCallOp(rewriter, loc, intrinsic,
                                                   returnType, operands[1]);

      LLVM::FastmathFlagsAttr defaultFlags{};
      replacementOp = rewriter.create<LLVM::FMulOp>(
          loc, returnType, operands[0], rcpOp->getResult(0), defaultFlags);
    } else if (calleeName == "__triton_hip_fast_expf") {
      assert(operands.size() == 1);
      assert(operands[0].getType().getIntOrFloatBitWidth() == 32);
      const double log2e = 1.4426950408889634;
      LLVM::FastmathFlagsAttr defaultFlags{};
      auto mulOp = rewriter.create<LLVM::FMulOp>(
          loc, rewriter.getF32Type(), operands[0],
          LLVM::createConstantF32(loc, rewriter, log2e), defaultFlags);
      const char *intrinsic = ftz ? "llvm.amdgcn.exp2.f32" : "llvm.exp2.f32";

      replacementOp = LLVM::createLLVMIntrinsicCallOp(
          rewriter, loc, intrinsic, returnType, mulOp->getResult(0));
    } else if (calleeName == "__triton_hip_load_acquire_workgroup") {
      assert(operands.size() == 1);
      replacementOp = buildAtomicLoad(
          operands[0], LLVM::AtomicOrdering::acquire, "workgroup");
    } else if (calleeName == "__triton_hip_load_relaxed_workgroup") {
      assert(operands.size() == 1);
      replacementOp = buildAtomicLoad(
          operands[0], LLVM::AtomicOrdering::monotonic, "workgroup");
    }

    else if (calleeName == "__triton_hip_load_acquire_agent") {
      assert(operands.size() == 1);
      replacementOp =
          buildAtomicLoad(operands[0], LLVM::AtomicOrdering::acquire, "agent");
    } else if (calleeName == "__triton_hip_load_relaxed_agent") {
      assert(operands.size() == 1);
      replacementOp = buildAtomicLoad(operands[0],
                                      LLVM::AtomicOrdering::monotonic, "agent");
    } else if (calleeName == "__triton_hip_load_acquire_system") {
      assert(operands.size() == 1);
      replacementOp =
          buildAtomicLoad(operands[0], LLVM::AtomicOrdering::acquire);
    } else if (calleeName == "__triton_hip_load_relaxed_system") {
      assert(operands.size() == 1);
      replacementOp =
          buildAtomicLoad(operands[0], LLVM::AtomicOrdering::monotonic);
    }

    else if (calleeName == "__triton_hip_store_release_workgroup") {
      assert(operands.size() == 1);
      Value one = rewriter.create<LLVM::ConstantOp>(
          loc, i64_ty, IntegerAttr::get(i64_ty, 1));
      (void)buildAtomicStore(one, operands[0], LLVM::AtomicOrdering::release,
                             "workgroup");
      replacementOp = one.getDefiningOp();
    } else if (calleeName == "__triton_hip_store_relaxed_workgroup") {
      assert(operands.size() == 1);
      Value one = rewriter.create<LLVM::ConstantOp>(
          loc, i64_ty, IntegerAttr::get(i64_ty, 1));
      (void)buildAtomicStore(one, operands[0], LLVM::AtomicOrdering::monotonic,
                             "workgroup");
      replacementOp = one.getDefiningOp();
    }

    else if (calleeName == "__triton_hip_store_release_agent") {
      assert(operands.size() == 1);
      Value one = rewriter.create<LLVM::ConstantOp>(
          loc, i64_ty, IntegerAttr::get(i64_ty, 1));
      (void)buildAtomicStore(one, operands[0], LLVM::AtomicOrdering::release,
                             "agent");
      replacementOp = one.getDefiningOp();
    }

    else if (calleeName == "__triton_hip_store_relaxed_agent") {
      assert(operands.size() == 1);
      Value one = rewriter.create<LLVM::ConstantOp>(
          loc, i64_ty, IntegerAttr::get(i64_ty, 1));
      (void)buildAtomicStore(one, operands[0], LLVM::AtomicOrdering::monotonic,
                             "agent");
      replacementOp = one.getDefiningOp();
    } else if (calleeName == "__triton_hip_store_release_system") {
      assert(operands.size() == 1);
      Value one = rewriter.create<LLVM::ConstantOp>(
          loc, i64_ty, IntegerAttr::get(i64_ty, 1));
      (void)buildAtomicStore(one, operands[0], LLVM::AtomicOrdering::release);
      replacementOp = one.getDefiningOp();
    } else if (calleeName == "__triton_hip_store_relaxed_system") {
      assert(operands.size() == 1);
      Value one = rewriter.create<LLVM::ConstantOp>(
          loc, i64_ty, IntegerAttr::get(i64_ty, 1));
      (void)buildAtomicStore(one, operands[0], LLVM::AtomicOrdering::monotonic);
      replacementOp = one.getDefiningOp();
    }

    // define internal noundef i64 @syncthreads()() #1 !dbg !51 {
    // entry:
    //   fence syncscope("workgroup") release, !dbg !52
    //   tail call void @llvm.amdgcn.s.barrier(), !dbg !60
    //   fence syncscope("workgroup") acquire, !dbg !61
    //   ret i64 0, !dbg !62
    // }
    else if (calleeName == "__triton_hip_syncthreads") {
      assert(operands.size() == 0);
      (void)rewriter.create<LLVM::FenceOp>(loc, LLVM::AtomicOrdering::release,
                                           "workgroup");
      (void)rewriter.create<ROCDL::SBarrierOp>(loc);
      (void)rewriter.create<LLVM::FenceOp>(loc, LLVM::AtomicOrdering::acquire,
                                           "workgroup");
      Value zero = rewriter.create<LLVM::ConstantOp>(
          loc, i64_ty, IntegerAttr::get(i64_ty, 0));
      replacementOp = zero.getDefiningOp();
    }

    else if (calleeName == "__triton_hip_red_add_release_agent") {
      assert(operands.size() == 2);
      replacementOp = buildAtomicFetchAdd(
          operands[0], operands[1], LLVM::AtomicOrdering::release, "agent");
    } else if (calleeName == "__triton_hip_red_add_release_system") {
      assert(operands.size() == 2);
      replacementOp = buildAtomicFetchAdd(operands[0], operands[1],
                                          LLVM::AtomicOrdering::release);
    }

    else if (calleeName == "__triton_hip_atom_add_acquire_agent") {
      assert(operands.size() == 2);
      replacementOp = buildAtomicFetchAdd(
          operands[0], operands[1], LLVM::AtomicOrdering::acquire, "agent");
    } else if (calleeName == "__triton_hip_atom_add_relaxed_agent") {
      assert(operands.size() == 2);
      replacementOp = buildAtomicFetchAdd(
          operands[0], operands[1], LLVM::AtomicOrdering::monotonic, "agent");
    } else if (calleeName == "__triton_hip_atom_add_acqrel_agent") {
      assert(operands.size() == 2);
      replacementOp = buildAtomicFetchAdd(
          operands[0], operands[1], LLVM::AtomicOrdering::acq_rel, "agent");
    }

    else if (calleeName == "__triton_hip_atom_add_acquire_system") {
      assert(operands.size() == 2);
      replacementOp = buildAtomicFetchAdd(operands[0], operands[1],
                                          LLVM::AtomicOrdering::acquire);
    } else if (calleeName == "__triton_hip_atom_add_relaxed_system") {
      assert(operands.size() == 2);
      replacementOp = buildAtomicFetchAdd(operands[0], operands[1],
                                          LLVM::AtomicOrdering::monotonic);
    } else if (calleeName == "__triton_hip_atom_add_acqrel_system") {
      assert(operands.size() == 2);
      replacementOp = buildAtomicFetchAdd(operands[0], operands[1],
                                          LLVM::AtomicOrdering::acq_rel);
    }

    else if (calleeName == "__triton_hip_atom_cas_acquire_relaxed_agent") {
      assert(operands.size() == 3);
      replacementOp = buildAtomicCompareExchangeStrong(
          operands[0], operands[1], operands[2], LLVM::AtomicOrdering::acquire,
          LLVM::AtomicOrdering::monotonic, "agent");
    }

    else if (calleeName == "__triton_hip_atom_cas_release_relaxed_agent") {
      assert(operands.size() == 3);
      replacementOp = buildAtomicCompareExchangeStrong(
          operands[0], operands[1], operands[2], LLVM::AtomicOrdering::release,
          LLVM::AtomicOrdering::monotonic, "agent");
    } else if (calleeName == "__triton_hip_atom_cas_relaxed_relaxed_agent") {
      assert(operands.size() == 3);
      replacementOp = buildAtomicCompareExchangeStrong(
          operands[0], operands[1], operands[2],
          LLVM::AtomicOrdering::monotonic, LLVM::AtomicOrdering::monotonic,
          "agent");
    }

    else if (calleeName == "__triton_hip_atom_cas_acqrel_relaxed_agent") {
      assert(operands.size() == 3);
      replacementOp = buildAtomicCompareExchangeStrong(
          operands[0], operands[1], operands[2], LLVM::AtomicOrdering::acq_rel,
          LLVM::AtomicOrdering::monotonic, "agent");
    }

    else if (calleeName == "__triton_hip_atom_cas_acquire_relaxed_system") {
      assert(operands.size() == 3);
      replacementOp = buildAtomicCompareExchangeStrong(
          operands[0], operands[1], operands[2], LLVM::AtomicOrdering::acquire,
          LLVM::AtomicOrdering::monotonic);
    }

    else if (calleeName == "__triton_hip_atom_cas_release_relaxed_system") {
      assert(operands.size() == 3);
      replacementOp = buildAtomicCompareExchangeStrong(
          operands[0], operands[1], operands[2], LLVM::AtomicOrdering::release,
          LLVM::AtomicOrdering::monotonic);
    }

    else if (calleeName == "__triton_hip_atom_cas_relaxed_relaxed_system") {
      assert(operands.size() == 3);
      replacementOp = buildAtomicCompareExchangeStrong(
          operands[0], operands[1], operands[2],
          LLVM::AtomicOrdering::monotonic, LLVM::AtomicOrdering::monotonic);
    }

    else if (calleeName == "__triton_hip_atom_cas_acqrel_relaxed_system") {
      assert(operands.size() == 3);
      replacementOp = buildAtomicCompareExchangeStrong(
          operands[0], operands[1], operands[2], LLVM::AtomicOrdering::acq_rel,
          LLVM::AtomicOrdering::monotonic);
    }

    if (replacementOp) {
      rewriter.replaceOp(callOp, replacementOp);
      return mlir::success();
    }

    return mlir::failure();
  }

private:
  bool ftz;
};

struct ConvertBuiltinFuncToLLVM
    : public triton::impl::ConvertBuiltinFuncToLLVMBase<
          ConvertBuiltinFuncToLLVM> {
  explicit ConvertBuiltinFuncToLLVM(bool ftz) { this->ftz = ftz; }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    GreedyRewriteConfig config;
    config.enableRegionSimplification = GreedySimplifyRegionLevel::Aggressive;

    RewritePatternSet patterns(context);
    patterns.add<CallOpConversion>(context, this->ftz);

    if (mlir::applyPatternsGreedily(mod, std::move(patterns), config)
            .failed()) {
      mod.emitError("failed to convert builtins/externs to llvm");
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir::triton {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertBuiltinFuncToLLVMPass(bool ftz) {
  return std::make_unique<ConvertBuiltinFuncToLLVM>(ftz);
}

} // namespace mlir::triton
