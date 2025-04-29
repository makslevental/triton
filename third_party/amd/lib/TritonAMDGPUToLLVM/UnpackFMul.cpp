#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"

#define DEBUG_TYPE "unpack-fmul"
#include "llvm/Transforms/Utils/InstructionWorklist.h"

using namespace llvm;
using namespace llvm::PatternMatch;

namespace {

// originates from VectorCombine::foldPermuteOfBinops
struct MaxsLLVMIRPass : FunctionPass {

  MaxsLLVMIRPass(Function &F)
      : F(F), Builder(F.getContext()), FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  bool unpackFMul(Instruction *inst) {
    Value *lhs, *rhs;
    if (!match(inst, m_BinOp(m_Value(lhs), m_Value(rhs))))
      return false;
    auto *VecLhs = dyn_cast<VectorType>(lhs->getType());
    if (!VecLhs)
      return false;
    assert(!VecLhs->isScalableTy() && "expected fixed-len vector");
    Builder.SetInsertPoint(inst);
    Value *UndefVec = llvm::UndefValue::get(VecLhs);
    for (int i = 0; i < VecLhs->getElementCount().getFixedValue(); ++i) {
      Value *mul = Builder.CreateFMul(Builder.CreateExtractElement(lhs, i),
                                      Builder.CreateExtractElement(rhs, i));
      UndefVec = Builder.CreateInsertElement(UndefVec, mul, i);
    }
    replaceValue(*inst, *UndefVec);
    // eraseInstruction(*inst);
    return true;
  }

  bool isMFMA(Instruction &inst) {
    if (llvm::isa<CallInst>(inst)) {
      auto &Call = llvm::cast<CallInst>(inst);
      if (Intrinsic::ID ID = Call.getIntrinsicID()) {
        switch (ID) {
        case Intrinsic::amdgcn_mfma_f32_16x16x16bf16_1k:
        case Intrinsic::amdgcn_mfma_f32_16x16x16f16:
        case Intrinsic::amdgcn_mfma_f32_16x16x1f32:
        case Intrinsic::amdgcn_mfma_f32_16x16x2bf16:
        case Intrinsic::amdgcn_mfma_f32_16x16x32_bf16:
        case Intrinsic::amdgcn_mfma_f32_16x16x32_bf8_bf8:
        case Intrinsic::amdgcn_mfma_f32_16x16x32_bf8_fp8:
        case Intrinsic::amdgcn_mfma_f32_16x16x32_f16:
        case Intrinsic::amdgcn_mfma_f32_16x16x32_fp8_bf8:
        case Intrinsic::amdgcn_mfma_f32_16x16x32_fp8_fp8:
        case Intrinsic::amdgcn_mfma_f32_16x16x4bf16_1k:
        case Intrinsic::amdgcn_mfma_f32_16x16x4f16:
        case Intrinsic::amdgcn_mfma_f32_16x16x4f32:
        case Intrinsic::amdgcn_mfma_f32_16x16x8_xf32:
        case Intrinsic::amdgcn_mfma_f32_16x16x8bf16:
        case Intrinsic::amdgcn_mfma_f32_32x32x16_bf8_bf8:
        case Intrinsic::amdgcn_mfma_f32_32x32x16_bf8_fp8:
        case Intrinsic::amdgcn_mfma_f32_32x32x16_f16:
        case Intrinsic::amdgcn_mfma_f32_32x32x16_fp8_bf8:
        case Intrinsic::amdgcn_mfma_f32_32x32x16_fp8_fp8:
        case Intrinsic::amdgcn_mfma_f32_32x32x1f32:
        case Intrinsic::amdgcn_mfma_f32_32x32x2bf16:
        case Intrinsic::amdgcn_mfma_f32_32x32x2f32:
        case Intrinsic::amdgcn_mfma_f32_32x32x4_xf32:
        case Intrinsic::amdgcn_mfma_f32_32x32x4bf16:
        case Intrinsic::amdgcn_mfma_f32_32x32x4bf16_1k:
        case Intrinsic::amdgcn_mfma_f32_32x32x4f16:
        case Intrinsic::amdgcn_mfma_f32_32x32x8bf16_1k:
        case Intrinsic::amdgcn_mfma_f32_32x32x8f16:
        case Intrinsic::amdgcn_mfma_f32_4x4x1f32:
        case Intrinsic::amdgcn_mfma_f32_4x4x2bf16:
        case Intrinsic::amdgcn_mfma_f32_4x4x4bf16_1k:
        case Intrinsic::amdgcn_mfma_f32_4x4x4f16:
        case Intrinsic::amdgcn_mfma_f64_16x16x4f64:
        case Intrinsic::amdgcn_mfma_f64_4x4x4f64:
        case Intrinsic::amdgcn_mfma_i32_16x16x16i8:
        case Intrinsic::amdgcn_mfma_i32_16x16x32_i8:
        case Intrinsic::amdgcn_mfma_i32_16x16x4i8:
        case Intrinsic::amdgcn_mfma_i32_16x16x64_i8:
        case Intrinsic::amdgcn_mfma_i32_32x32x16_i8:
        case Intrinsic::amdgcn_mfma_i32_32x32x32_i8:
        case Intrinsic::amdgcn_mfma_i32_32x32x4i8:
        case Intrinsic::amdgcn_mfma_i32_32x32x8i8:
        case Intrinsic::amdgcn_mfma_i32_4x4x4i8:
          return true;
        }
      }
    }
    return false;
  }

  bool containsMFMA(BasicBlock &BB) {
    for (Instruction &inst : BB) {
      if (isMFMA(inst))
        return true;
    }
    return false;
  }

  bool runOnFunction(Function &F) override {
    llvm::dbgs() << "maxs_llvmir_pass:\n";
    bool Changed = false;
    SmallVector<Instruction *> Ops(F.size());
    for (BasicBlock &BB : F) {
      for (Instruction &inst : BB) {
        auto c = containsMFMA(BB);
        if (inst.getOpcode() == Instruction::FMul && c) {
          if (unpackFMul(&inst)) {
            Ops.push_back(&inst);
            Changed = true;
          }
        }
      }
    }
    for (Instruction *inst : Ops) {
      if (inst)
        eraseInstruction(*inst);
    }
    return Changed;
  }

  void replaceValue(Value &Old, Value &New) {
    Old.replaceAllUsesWith(&New);
    if (auto *NewI = dyn_cast<Instruction>(&New)) {
      New.takeName(&Old);
      Worklist.pushUsersToWorkList(*NewI);
      Worklist.pushValue(NewI);
    }
    Worklist.pushValue(&Old);
  }

  void eraseInstruction(Instruction &I) {
    SmallVector<Value *> Ops(I.operands());
    Worklist.remove(&I);
    I.eraseFromParent();

    // Push remaining users of the operands and then the operand itself - allows
    // further folds that were hindered by OneUse limits.
    for (Value *Op : Ops)
      if (auto *OpI = dyn_cast<Instruction>(Op)) {
        Worklist.pushUsersToWorkList(*OpI);
        Worklist.pushValue(OpI);
      }
  }

  static char ID;
  Function &F;
  IRBuilder<> Builder;
  InstructionWorklist Worklist;
};

} // end anonymous namespace

char MaxsLLVMIRPass::ID = 0;

void runMaxsLLVMIRPassOnFunction(Function &F) {
  ModuleAnalysisManager MAM;
  FunctionAnalysisManager FAM;
  auto ttr = TargetIRAnalysis();
  MaxsLLVMIRPass pass(F);
  pass.runOnFunction(F);
  llvm::verifyFunction(F);
}