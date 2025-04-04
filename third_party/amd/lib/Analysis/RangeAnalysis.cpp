#include "third_party/amd/include/Analysis/RangeAnalysis.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/Utils/InferIntRangeCommon.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include <numeric>
#include <optional>

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonamdgpu-range-analysis"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;

namespace tt = mlir::triton;

namespace {

constexpr int64_t kDefaultMaxTripCount = 1024;
constexpr int64_t kDefaultMaxPrograms = 2 << 15; // 65536

void getEnclosingLoops(Operation &op, SmallVector<LoopLikeOpInterface> &ops) {
  Operation *currOp = op.getParentOp();
  while (currOp) {
    if (isa<LoopLikeOpInterface>(currOp))
      ops.push_back(llvm::cast<LoopLikeOpInterface>(currOp));
    currOp = currOp->getParentOp();
  }
}

void inferResultRangesPID(Operation *op, uint64_t max,
                          SetIntRangeFn setResultRange) {
  assert(op->getNumResults() == 1 && "expected op to have one result");
  auto result = op->getResult(0);
  assert(llvm::isa<IntegerType>(result.getType()) &&
         "expected result type to be int");
  IntegerType resTy = llvm::cast<IntegerType>(result.getType());
  auto bitWidth = mlir::ConstantIntRanges::getStorageBitwidth(resTy);
  setResultRange(result, ConstantIntRanges::range(
                             /*min*/ {/*numBits*/ bitWidth, /*val*/ 0,
                                      /*isSigned*/ resTy.isSigned()},
                             /*max*/
                             {/*numBits*/ bitWidth, /*val*/ max,
                              /*isSigned*/ resTy.isSigned()},
                             /*isSigned*/ resTy.isSigned()));
}

void inferResultRanges(tt::MakeRangeOp *op, SetIntRangeFn setResultRange) {
  auto result = op->getResult();
  RankedTensorType resTy = result.getType();
  assert(llvm::isa<IntegerType>(resTy.getElementType()) && "expected int type");
  IntegerType elTy = llvm::cast<IntegerType>(resTy.getElementType());
  auto bitWidth = mlir::ConstantIntRanges::getStorageBitwidth(elTy);
  setResultRange(result,
                 ConstantIntRanges::range(
                     /*min*/ {/*numBits*/ bitWidth, /*val*/ op->getStart(),
                              /*isSigned*/ elTy.isSigned()},
                     /*max*/
                     {/*numBits*/ bitWidth, /*val*/ op->getEnd(),
                      /*isSigned*/ elTy.isSigned()},
                     /*isSigned*/ elTy.isSigned()));
}

void inferResultRanges(tt::GatherOp *op, ArrayRef<ConstantIntRanges> argRanges,
                       SetIntRangeFn setResultRange) {
  assert(argRanges.size() == 2 && "expected two arg ranges");
  setResultRange(op->getResult(), argRanges[0]);
}

void inferResultRangesUnaryOpForwardArgRange(
    Operation *op, ArrayRef<ConstantIntRanges> argRanges,
    SetIntRangeFn setResultRange) {
  for (const auto &result : op->getResults())
    setResultRange(result, argRanges[0]);
}

void inferResultRangesBinaryOpUnionArgRanges(
    Operation *op, ArrayRef<ConstantIntRanges> argRanges,
    SetIntRangeFn setResultRange) {
  assert(op->getNumOperands() == 2 && "expected op to have two operands");
  assert(argRanges.size() == 2 && "expected two arg ranges");
  for (const auto &result : op->getResults())
    setResultRange(result, argRanges[0].rangeUnion(argRanges[1]));
}

void inferResultRangesMaxNonNegSigned(Operation *op,
                                      SetIntRangeFn setResultRange) {
  for (auto result : op->getResults()) {
    auto bitWidth =
        mlir::ConstantIntRanges::getStorageBitwidth(result.getType());
    setResultRange(result, ConstantIntRanges::fromSigned(
                               APInt::getZero(bitWidth).sext(bitWidth),
                               APInt::getMaxValue(bitWidth).sext(bitWidth)));
  }
}

std::optional<ConstantIntRanges> maybeGetAssumedRange(Operation *assumption,
                                                      Value anchor) {
  arith::CmpIOp cmpOp = llvm::dyn_cast<arith::CmpIOp>(assumption);
  if (!cmpOp) {
    emitRemark(assumption->getLoc(), "unsupported assumption operation");
    return {};
  }

  bool isSigned = true;
  switch (cmpOp.getPredicate()) {
  case arith::CmpIPredicate::uge:
  case arith::CmpIPredicate::ugt:
  case arith::CmpIPredicate::ule:
  case arith::CmpIPredicate::ult:
    isSigned = false;
  default:
    break;
  }

  bool anchorIsLhs = cmpOp.getLhs() == anchor;
  auto maybeConstantIntValue = getConstantIntValue(
      getAsOpFoldResult(anchorIsLhs ? cmpOp.getRhs() : cmpOp.getLhs()));
  if (auto constValue = maybeConstantIntValue) {
    unsigned bitWidth = ConstantIntRanges::getStorageBitwidth(anchor.getType());
    assert(bitWidth > 0 && "expected non-zero bitwdith");
    APInt apVal = {bitWidth, static_cast<uint64_t>(*constValue), isSigned};
    APInt min, max;
    if (isSigned) {
      min = APInt::getSignedMinValue(bitWidth);
      max = APInt::getSignedMaxValue(bitWidth);
    } else {
      min = APInt::getMinValue(bitWidth);
      max = APInt::getMaxValue(bitWidth);
    }

    switch (cmpOp.getPredicate()) {
    case arith::CmpIPredicate::eq:
      return mlir::ConstantIntRanges::constant(apVal);
    case arith::CmpIPredicate::uge:
    case arith::CmpIPredicate::sge: {
      // K >= apVal implies K ∈ [apVal, max]
      if (anchorIsLhs)
        return mlir::ConstantIntRanges::range(apVal, max, isSigned);
      // apVal >= K implies K ∈ [min, apVal]
      return mlir::ConstantIntRanges::range(min, apVal, isSigned);
    }
    case arith::CmpIPredicate::ugt:
    case arith::CmpIPredicate::sgt: {
      // K > apVal implies K >= apVal + 1 implies K ∈ [apVal + 1, max]
      if (anchorIsLhs)
        return mlir::ConstantIntRanges::range(apVal + 1, max, isSigned);
      // apVal > K implies apVal - 1 >= K implies K ∈ [min, apVal - 1]
      return mlir::ConstantIntRanges::range(min, apVal - 1, isSigned);
    }
    case arith::CmpIPredicate::ule:
    case arith::CmpIPredicate::sle: {
      // K <= apVal implies K ∈ [min, apVal]
      if (anchorIsLhs)
        return mlir::ConstantIntRanges::range(min, apVal, isSigned);
      // apVal <= K implies K ∈ [apVal, max]
      return mlir::ConstantIntRanges::range(apVal, max, isSigned);
    }
    case arith::CmpIPredicate::ult:
    case arith::CmpIPredicate::slt: {
      // K < apVal implies K <= apVal -1 implies K ∈ [min, apVal - 1]
      if (anchorIsLhs)
        return mlir::ConstantIntRanges::range(min, apVal - 1, isSigned);
      // apVal < K implies apVal + 1 <= K implies K ∈ [apVal + 1, max]
      return mlir::ConstantIntRanges::range(apVal + 1, max, isSigned);
    }
    default:
      emitRemark(cmpOp.getLoc(), "unsupported cmp predicate for assumption");
      return {};
    }
  }
  return {};
}

} // namespace

namespace mlir::triton::AMD {

bool isEmptyInitializedRange(ConstantIntRanges rv) {
  if (!rv.umin().getBitWidth() || !rv.umax().getBitWidth() ||
      !rv.smin().getBitWidth() || !rv.smax().getBitWidth())
    return true;
  return false;
}

std::optional<SmallVector<std::optional<ConstantIntRanges>>>
collectRanges(const DataFlowSolver &solver, ValueRange values) {
  SmallVector<std::optional<ConstantIntRanges>> ranges;
  for (Value val : values) {
    auto *maybeInferredRange =
        solver.lookupState<IntegerValueRangeWithMeetLattice>(val);
    if (!maybeInferredRange ||
        maybeInferredRange->getValue().isUninitialized()) {
      ranges.push_back(std::nullopt);
      continue;
    }
    const ConstantIntRanges &inferredRange =
        maybeInferredRange->getValue().getValue();
    if (isEmptyInitializedRange(inferredRange)) {
      ranges.push_back(std::nullopt);
      continue;
    }
    ranges.push_back(inferredRange);
  }
  return ranges;
}

bool cmpIIsStaticallyTrue(const DataFlowSolver &solver, arith::CmpIOp cmpOp) {
  if (auto inputRanges =
          collectRanges(solver, ValueRange{cmpOp.getOperands()})) {
    intrange::CmpPredicate pred =
        static_cast<intrange::CmpPredicate>(cmpOp.getPredicate());
    if (!(*inputRanges)[0] || !(*inputRanges)[1])
      return false;
    return intrange::evaluatePred(pred, *(*inputRanges)[0], *(*inputRanges)[1])
        .value_or(false);
  }
  return false;
}

std::optional<ConstantIntRanges>
TritonIntegerRangeAnalysis::maybeGetAssumedRange(Value anchor) const {
  auto matchingAssumptions = this->assumptions.lookup(anchor);
  if (matchingAssumptions.empty())
    return {};

  unsigned bitWidth = ConstantIntRanges::getStorageBitwidth(anchor.getType());
  assert(bitWidth > 0 && "expected non-zero bitwidth");
  ConstantIntRanges constIntRange = ConstantIntRanges::maxRange(bitWidth);
  for (auto assumption : matchingAssumptions) {
    if (auto constIntRange_ = ::maybeGetAssumedRange(assumption, anchor))
      constIntRange = constIntRange.intersection(*constIntRange_);
  }
  return constIntRange;
}

IntegerValueRangeWithMeet
IntegerValueRangeWithMeet::meet(const IntegerValueRangeWithMeet &lhs,
                                const IntegerValueRangeWithMeet &rhs) {
  if (lhs.isUninitialized() && rhs.isUninitialized())
    llvm::report_fatal_error(
        "expected at least one of lhs/rhs to be initialized");
  if (lhs.isUninitialized())
    return rhs;
  if (rhs.isUninitialized())
    return lhs;
  return {lhs.getValue().intersection(rhs.getValue())};
}

void IntegerValueRangeWithMeetLattice::onUpdate(DataFlowSolver *solver) const {
  Lattice::onUpdate(solver);

  // If the integer range can be narrowed to a constant, update the constant
  // value of the SSA value.
  std::optional<APInt> constant = getValue().getValue().getConstantValue();
  auto value = cast<Value>(anchor);
  auto *cv = solver->getOrCreateState<Lattice<dataflow::ConstantValue>>(value);
  if (!constant)
    return solver->propagateIfChanged(
        cv, cv->join(dataflow::ConstantValue::getUnknownConstant()));

  Dialect *dialect;
  if (auto *parent = value.getDefiningOp())
    dialect = parent->getDialect();
  else
    dialect = value.getParentBlock()->getParentOp()->getDialect();

  Type type = getElementTypeOrSelf(value);
  solver->propagateIfChanged(
      cv, cv->join(dataflow::ConstantValue(IntegerAttr::get(type, *constant),
                                           dialect)));
}

void TritonIntegerRangeAnalysis::setToEntryState(
    IntegerValueRangeWithMeetLattice *lattice) {
  auto anchor = lattice->getAnchor();
  IntegerValueRangeWithMeet range = IntegerValueRange::getMaxRange(anchor);
  if (auto maybeRange = maybeGetAssumedRange(anchor))
    range = *maybeRange;
  auto changed = lattice->join(range);
  LLVM_DEBUG({
    if (changed == ChangeResult::Change) {
      DBGS() << "Set range of ";
      anchor.printAsOperand(llvm::dbgs(), {});
      llvm::dbgs() << " to " << range << "\n";
    }
  });
  propagateIfChanged(lattice, changed);
}

LogicalResult TritonIntegerRangeAnalysis::visitOperation(
    Operation *op, ArrayRef<const IntegerValueRangeWithMeetLattice *> operands,
    ArrayRef<IntegerValueRangeWithMeetLattice *> resultsLattices) {
  LDBG("Inferring ranges for " << *op);
  // This callback is almost exactly like the callback in
  // IntegerRangeAnalysis::visitOperation except we do not "short-circuit" the
  // analysis by inferring a maximum range for loop results (instead we
  // perform a check based on visit counts in visitRegionSuccessors).
  auto joinCallback = [&op, &resultsLattices,
                       this](Value v,
                             const IntegerValueRangeWithMeet &incomingRange) {
    auto result = dyn_cast<OpResult>(v);
    if (!result)
      return;
    assert(llvm::is_contained(op->getResults(), result));

    IntegerValueRangeWithMeetLattice *lattice =
        resultsLattices[result.getResultNumber()];
    IntegerValueRangeWithMeet incomingRange_ = incomingRange;
    if (auto maybeRange = maybeGetAssumedRange(v)) {
      incomingRange_ = IntegerValueRangeWithMeet(
          incomingRange.getValue().intersection(*maybeRange));
    }
    ChangeResult changed = lattice->join(incomingRange_);
    LLVM_DEBUG({
      if (changed == ChangeResult::Change) {
        DBGS() << "Inferred range for ";
        v.printAsOperand(llvm::dbgs(), {});
        llvm::dbgs() << " to " << incomingRange << "\n";
      }
    });
    propagateIfChanged(lattice, changed);
  };

  // Initialize lattices with assumptions.
  for (const auto &resultLattice : resultsLattices) {
    if (!resultLattice->getValue().isUninitialized())
      continue;
    auto anchor = resultLattice->getAnchor();
    if (auto assumptions = this->assumptions.lookup(anchor);
        !assumptions.empty()) {
      setToEntryState(resultLattice);
      return success();
    }
  }

  // Ops with fixed/constant ranges.
  if (llvm::isa<GetProgramIdOp, MakeRangeOp, HistogramOp, GetNumProgramsOp>(
          op)) {
    llvm::TypeSwitch<Operation *>(op)
        .Case<GetProgramIdOp>([&](auto getPIDOp) {
          inferResultRangesPID(getPIDOp, kDefaultMaxPrograms - 1, joinCallback);
        })
        .Case<GetNumProgramsOp>([&](auto getPIDOp) {
          inferResultRangesPID(getPIDOp, kDefaultMaxPrograms, joinCallback);
        })
        .Case<MakeRangeOp>([&](MakeRangeOp makeROp) {
          inferResultRanges(&makeROp, joinCallback);
        })
        .Case<HistogramOp>([&](HistogramOp histOp) {
          return inferResultRangesMaxNonNegSigned(histOp, joinCallback);
        })
        .Default([&](auto) { llvm::report_fatal_error("unsupported op"); });
    return success();
  }

  SmallVector<IntegerValueRangeWithMeet> argIntValueRanges =
      llvm::map_to_vector(
          operands, [](const IntegerValueRangeWithMeetLattice *lattice) {
            return IntegerValueRangeWithMeet(lattice->getValue());
          });

  // Ops with actually changing/variable input/output ranges.
  if (llvm::isa<TransOp, SplitOp, BroadcastOp, ReshapeOp, gpu::ConvertLayoutOp,
                SplatOp, ExpandDimsOp, JoinOp, CatOp, GatherOp>(op)) {
    SmallVector<ConstantIntRanges> argConstIntRanges;
    for (const auto &r : argIntValueRanges) {
      if (r.isUninitialized()) {
        setAllToEntryStates(resultsLattices);
        return success();
      }
      argConstIntRanges.push_back(r.getValue());
    }
    llvm::TypeSwitch<Operation *>(op)
        .Case<TransOp, SplitOp, BroadcastOp, ExpandDimsOp, SplatOp, ReshapeOp,
              gpu::ConvertLayoutOp>([&](auto) {
          return inferResultRangesUnaryOpForwardArgRange(op, argConstIntRanges,
                                                         joinCallback);
        })
        .Case<JoinOp, CatOp>([&](auto joinOp) {
          return inferResultRangesBinaryOpUnionArgRanges(
              joinOp, argConstIntRanges, joinCallback);
        })
        .Case<GatherOp>([&](GatherOp gatherOp) {
          return inferResultRanges(&gatherOp, argConstIntRanges, joinCallback);
        })
        .Default([&](auto) { llvm::report_fatal_error("unsupported op"); });
    return success();
  }

  if (auto inferrable = dyn_cast<InferIntRangeInterface>(op)) {
    SmallVector<IntegerValueRange> argIntValueRanges = llvm::map_to_vector(
        operands, [](const IntegerValueRangeWithMeetLattice *lattice) {
          return static_cast<IntegerValueRange>(lattice->getValue());
        });
    inferrable.inferResultRangesFromOptional(argIntValueRanges, joinCallback);
    return success();
  }

  setAllToEntryStates(resultsLattices);
  return success();
}

void TritonIntegerRangeAnalysis::initializeFuncOp(tt::FuncOp *op) {
  for (BlockArgument argument : op->getArguments()) {
    if (auto assumptions = this->assumptions.lookup(argument);
        !assumptions.empty()) {
      IntegerValueRangeWithMeetLattice *argLattice =
          getLatticeElement(argument);
      auto anchor = argLattice->getAnchor();
      IntegerValueRangeWithMeet range =
          IntegerValueRangeWithMeet::getMaxRange(anchor);
      if (auto maybeRange = maybeGetAssumedRange(anchor))
        range = *maybeRange;
      (void)argLattice->join(range);
    }
  }
}

// template <typename PtrType> struct Storage {
//   inline static PtrType ptr;
// };
//
// template <auto V> struct PtrTaker {
//   struct Transferer {
//     Transferer() { Storage<decltype(V)>::ptr = V; }
//   };
//   inline static Transferer tr;
// };
//
// template struct PtrTaker<&TritonIntegerRangeAnalysis::solver>;

std::optional<int64_t>
TritonIntegerRangeAnalysis::maybeGetTripCount(LoopLikeOpInterface loop) {
  std::optional<OpFoldResult> lowerBound = loop.getSingleLowerBound();
  std::optional<OpFoldResult> upperBound = loop.getSingleUpperBound();
  std::optional<OpFoldResult> step = loop.getSingleStep();

  auto getLoopBoundFromFold = [&](std::optional<OpFoldResult> loopBound,
                                  Type boundType, Block *block,
                                  std::optional<bool> getUpper,
                                  std::optional<uint64_t> default_ =
                                      std::nullopt) {
    unsigned int width = ConstantIntRanges::getStorageBitwidth(boundType);
    if (loopBound.has_value()) {
      if (auto attr = dyn_cast<Attribute>(*loopBound)) {
        if (auto bound = dyn_cast_or_null<IntegerAttr>(attr))
          return bound.getValue();
      } else if (auto value = llvm::dyn_cast_if_present<Value>(*loopBound)) {
        const IntegerValueRangeWithMeetLattice *lattice =
            getLatticeElementFor(getProgramPointBefore(block), value);
        if (lattice != nullptr && !lattice->getValue().isUninitialized())
          return getUpper ? lattice->getValue().getValue().smax()
                          : lattice->getValue().getValue().smin();
      }
    }
    if (default_)
      return APInt(width, *default_, true);
    // Given the results of getConstant{Lower,Upper}Bound()
    // or getConstantStep() on a LoopLikeInterface return the lower/upper
    // bound
    return getUpper ? APInt::getSignedMaxValue(width)
                    : APInt::getSignedMinValue(width);
  };

  std::optional<Value> iv = loop.getSingleInductionVar();
  Block *block = iv->getParentBlock();
  APInt min = getLoopBoundFromFold(lowerBound, iv->getType(), block,
                                   /*getUpper=*/false);
  APInt max = getLoopBoundFromFold(upperBound, iv->getType(), block,
                                   /*getUpper=*/true);
  // Assume positivity for uniscoverable steps by way of getUpper = true.
  APInt stepVal =
      getLoopBoundFromFold(step, iv->getType(), block, /*getUpper=*/{}, 1);

  if (stepVal.isNegative()) {
    std::swap(min, max);
  } else {
    // Correct the upper bound by subtracting 1 so that it becomes a <=
    // bound, because loops do not generally include their upper bound.
    max -= 1;
  }

  // If we infer the lower bound to be larger than the upper bound, the
  // resulting range is meaningless and should not be used in further
  // inferences.
  if (max.sge(min)) {
    // auto &solver_ =
    //     this->*Storage<DataFlowSolver TritonIntegerRangeAnalysis::*>::ptr;
    // solver_.eraseState(*iv);
    IntegerValueRangeWithMeetLattice *ivEntry = getLatticeElement(*iv);
    auto ivRange = ConstantIntRanges::fromSigned(min, max);
    llvm::dbgs() << "ivRange: " << ivRange << "\n";
    auto changed = ivEntry->meet(IntegerValueRangeWithMeet{ivRange});
    if (changed == ChangeResult::Change) {
      llvm::dbgs() << (changed == ChangeResult::Change ? "changed"
                                                       : "not changed")
                   << "\n";
    }
    propagateIfChanged(ivEntry, changed);
    return llvm::divideCeilSigned(max.getSExtValue() - min.getSExtValue(),
                                  stepVal.getSExtValue());
  }
  return {};
}

void TritonIntegerRangeAnalysis::visitRegionSuccessors(
    ProgramPoint *point, RegionBranchOpInterface branch,
    RegionBranchPoint successor,
    ArrayRef<dataflow::AbstractSparseLattice *> abstractLattices) {
  LLVM_DEBUG({
    DBGS() << "Inferring ranges for ";
    OpPrintingFlags flags;
    flags.skipRegions(true);
    branch.print(llvm::dbgs(), flags);
    llvm::dbgs() << "\n";
  });
  SmallVector<IntegerValueRangeWithMeetLattice *> lattices;
  for (auto abstractLat : abstractLattices) {
    lattices.push_back(
        static_cast<IntegerValueRangeWithMeetLattice *>(abstractLat));
  }
  // Initialize loop trip counts
  LoopLikeOpInterface loop =
      llvm::dyn_cast<LoopLikeOpInterface>(branch.getOperation());

  if (loop) {
    // check if a new/smaller trip count has been inferred/computed
    SmallVector loops{loop};
    getEnclosingLoops(*loop, loops);
    int64_t loopTripCount =
        std::accumulate(loops.begin(), loops.end(), (int64_t)1,
                        [this](int64_t accum, LoopLikeOpInterface loop) {
                          return accum * maybeGetTripCount(loop).value_or(
                                             kDefaultMaxTripCount + 1);
                        });
    if (!loopTripCounts.contains(loop) ||
        loopTripCount < loopTripCounts[loop]) {
      loopTripCounts[loop] = loopTripCount;
      OpPrintingFlags flags;
      flags.skipRegions(true);
      loop.print(llvm::dbgs(), flags);
      llvm::dbgs() << "\n";
      llvm::dbgs() << "trip count: " << loopTripCount << '\n';
      for (auto argLat : lattices)
        loopVisits[{loop, argLat}] = 0;
    }
  }

  const auto *predecessors =
      getOrCreateFor<dataflow::PredecessorState>(point, point);
  assert(predecessors->allPredecessorsKnown() &&
         "unexpected unresolved region successors");

  for (Operation *op : predecessors->getKnownPredecessors()) {
    std::optional<OperandRange> operands;
    if (op == branch) {
      operands = branch.getEntrySuccessorOperands(successor);
    } else if (auto regionTerminator =
                   dyn_cast<RegionBranchTerminatorOpInterface>(op)) {
      operands = regionTerminator.getSuccessorOperands(successor);
    }
    if (!operands)
      return setAllToEntryStates(lattices);

    ValueRange inputs = predecessors->getSuccessorInputs(op);
    assert(inputs.size() == operands->size() &&
           "expected the same number of successor inputs as operands");

    unsigned firstIndex = 0;
    if (inputs.size() != lattices.size()) {
      if (!point->isBlockStart()) {
        if (!inputs.empty()) {
          firstIndex = cast<OpResult>(inputs.front()).getResultNumber();
        }
        visitNonControlFlowArguments(branch,
                                     RegionSuccessor(branch->getResults().slice(
                                         firstIndex, inputs.size())),
                                     lattices, firstIndex);
      } else {
        if (!inputs.empty()) {
          firstIndex = cast<BlockArgument>(inputs.front()).getArgNumber();
        }
        Region *region = point->getBlock()->getParent();
        visitNonControlFlowArguments(
            branch,
            RegionSuccessor(region, region->getArguments().slice(
                                        firstIndex, inputs.size())),
            lattices, firstIndex);
      }
    }

    for (auto [oper, argLat] :
         llvm::zip(*operands, ArrayRef(lattices).drop_front(firstIndex))) {
      std::pair loopArgLat = {loop, argLat};
      // If we've "run the loop" #tripcount times, stop propagating.
      if (loop && loopVisits[loopArgLat] >= loopTripCounts[loop])
        continue;
      ChangeResult changed;
      if (loop && loopTripCounts[loop] > kDefaultMaxTripCount) {
        // If the loop's tripcount is too large, infer the maximum range for
        // the arg lattices. This will have the effect that all users will
        // also be inferred to have maximum range and end the analysis will
        // end (the maximum range is the "top" of the lattice and thus no
        // further changes/updates are possible).

        // changed = argLat->join(IntegerValueRangeWithMeet::getMaxRange(oper));
      } else {
        // Else, propagate pred operands.
        auto operLat = *getLatticeElementFor(point, oper);
        changed = argLat->join(operLat);
        LLVM_DEBUG({
          DBGS() << "Operand lattice for ";
          OpPrintingFlags flags;
          flags.skipRegions(true);
          oper.print(llvm::dbgs(), flags);
          llvm::dbgs() << " ---> " << operLat << " ---> changed: "
                       << (changed == ChangeResult::Change ? "true" : "false")
                       << "\n";
        });
      }
      propagateIfChanged(argLat, changed);
      // Only increase the loop visitation count if have actually update the
      // lattice because otherwise we will over count the number of visits
      // (since not all iter_arg lattices are updated/propagated on each
      // visit).
      if (loop && changed == ChangeResult::Change) {
        ++loopVisits[loopArgLat];
        llvm::dbgs() << "loopVisits: " << loopVisits[loopArgLat] << "\n";
      }
    }
  }
}

void TritonIntegerRangeAnalysis::visitNonControlFlowArguments(
    Operation *op, const RegionSuccessor &successor,
    ArrayRef<IntegerValueRangeWithMeetLattice *> argLattices,
    unsigned firstIndex) {
  if (auto inferrable = dyn_cast<InferIntRangeInterface>(op)) {
    LLVM_DEBUG(llvm::dbgs() << "Inferring ranges for " << *op << "\n");

    auto joinCallback = [&](Value v, const IntegerValueRange &attrs) {
      auto arg = dyn_cast<BlockArgument>(v);
      if (!arg)
        return;
      if (!llvm::is_contained(successor.getSuccessor()->getArguments(), arg))
        return;

      LLVM_DEBUG(llvm::dbgs() << "Inferred range " << attrs << "\n");
      IntegerValueRangeWithMeetLattice *lattice =
          argLattices[arg.getArgNumber()];
      IntegerValueRange oldRange = lattice->getValue();

      ChangeResult changed = lattice->join(attrs);

      // Catch loop results with loop variant bounds and conservatively make
      // them [-inf, inf] so we don't circle around infinitely often (because
      // the dataflow analysis in MLIR doesn't attempt to work out trip counts
      // and often can't).
      bool isYieldedValue = llvm::any_of(v.getUsers(), [](Operation *op) {
        return op->hasTrait<OpTrait::IsTerminator>();
      });
      if (isYieldedValue && !oldRange.isUninitialized() &&
          !(lattice->getValue() == oldRange)) {
        LLVM_DEBUG(llvm::dbgs() << "Loop variant loop result detected\n");
        changed |= lattice->join(IntegerValueRange::getMaxRange(v));
      }
      propagateIfChanged(lattice, changed);
    };

    auto argRanges = llvm::map_to_vector(op->getOperands(), [&](Value value) {
      return static_cast<IntegerValueRange>(
          getLatticeElementFor(getProgramPointAfter(op), value)->getValue());
    });

    inferrable.inferResultRangesFromOptional(argRanges, joinCallback);
    return;
  }

  /// Given the results of getConstant{Lower,Upper}Bound() or getConstantStep()
  /// on a LoopLikeInterface return the lower/upper bound for that result if
  /// possible.
  auto getLoopBoundFromFold = [&](std::optional<OpFoldResult> loopBound,
                                  Type boundType, Block *block, bool getUpper) {
    unsigned int width = ConstantIntRanges::getStorageBitwidth(boundType);
    if (loopBound.has_value()) {
      if (auto attr = dyn_cast<Attribute>(*loopBound)) {
        if (auto bound = dyn_cast_or_null<IntegerAttr>(attr))
          return bound.getValue();
      } else if (auto value = llvm::dyn_cast_if_present<Value>(*loopBound)) {
        const IntegerValueRangeWithMeetLattice *lattice =
            getLatticeElementFor(getProgramPointBefore(block), value);
        if (lattice != nullptr && !lattice->getValue().isUninitialized())
          return getUpper ? lattice->getValue().getValue().smax()
                          : lattice->getValue().getValue().smin();
      }
    }
    // Given the results of getConstant{Lower,Upper}Bound()
    // or getConstantStep() on a LoopLikeInterface return the lower/upper
    // bound
    return getUpper ? APInt::getSignedMaxValue(width)
                    : APInt::getSignedMinValue(width);
  };

  // Infer bounds for loop arguments that have static bounds
  if (auto loop = dyn_cast<LoopLikeOpInterface>(op)) {
    std::optional<Value> iv = loop.getSingleInductionVar();
    if (!iv) {
      return SparseForwardDataFlowAnalysis ::visitNonControlFlowArguments(
          op, successor, argLattices, firstIndex);
    }
    Block *block = iv->getParentBlock();
    std::optional<OpFoldResult> lowerBound = loop.getSingleLowerBound();
    std::optional<OpFoldResult> upperBound = loop.getSingleUpperBound();
    std::optional<OpFoldResult> step = loop.getSingleStep();
    APInt min = getLoopBoundFromFold(lowerBound, iv->getType(), block,
                                     /*getUpper=*/false);
    APInt max = getLoopBoundFromFold(upperBound, iv->getType(), block,
                                     /*getUpper=*/true);
    // Assume positivity for uniscoverable steps by way of getUpper = true.
    APInt stepVal =
        getLoopBoundFromFold(step, iv->getType(), block, /*getUpper=*/true);

    if (stepVal.isNegative()) {
      std::swap(min, max);
    } else {
      // Correct the upper bound by subtracting 1 so that it becomes a <=
      // bound, because loops do not generally include their upper bound.
      max -= 1;
    }

    // If we infer the lower bound to be larger than the upper bound, the
    // resulting range is meaningless and should not be used in further
    // inferences.
    if (max.sge(min)) {
      IntegerValueRangeWithMeetLattice *ivEntry = getLatticeElement(*iv);
      auto ivRange = ConstantIntRanges::fromSigned(min, max);
      llvm::dbgs() << "ivRange non control flow: " << ivRange << "\n";
      propagateIfChanged(ivEntry, ivEntry->join(IntegerValueRange{ivRange}));
    }
    return;
  }

  return SparseForwardDataFlowAnalysis::visitNonControlFlowArguments(
      op, successor, argLattices, firstIndex);
}

DenseMap<Value, SetVector<Operation *>>
TritonIntegerRangeAnalysis::collectAssumptions(Operation *rootOp,
                                               bool filterConstants) {
  DenseMap<Value, SetVector<Operation *>> assumptions;
  rootOp->walk([&](LLVM::AssumeOp op) {
    auto assump = op.getCond().getDefiningOp();
    for (auto operand : assump->getOperands()) {
      if (filterConstants && getConstantIntValue(operand))
        continue;
      assumptions[operand].insert(assump);
    }
  });
  return assumptions;
}

struct FoldTrueCmpIOp : OpRewritePattern<arith::CmpIOp> {
  using OpRewritePattern::OpRewritePattern;

  FoldTrueCmpIOp(MLIRContext *context, DataFlowSolver *solver)
      : OpRewritePattern(context), solver(solver) {};

  LogicalResult matchAndRewrite(arith::CmpIOp cmpOp,
                                PatternRewriter &rewriter) const override {
    if (cmpIIsStaticallyTrue(*solver, cmpOp)) {
      if (failed(mlir::dataflow::maybeReplaceWithConstant(*solver, rewriter,
                                                          cmpOp.getResult()))) {
        LDBG("failed to replace with constant op: " << cmpOp);
        return failure();
      }
    } else {
      return failure();
    }
    return success();
  }

  DataFlowSolver *solver;
};

void populateFoldTrueCmpIOpPatterns(RewritePatternSet &patterns,
                                    DataFlowSolver *solver) {
  patterns.add<FoldTrueCmpIOp>(patterns.getContext(), solver);
}

void initializeFuncOps(Operation *op,
                       AMD::TritonIntegerRangeAnalysis *rangeAnalysis) {
  op->walk<WalkOrder::PreOrder>([&rangeAnalysis](FuncOp funcOp) {
    rangeAnalysis->initializeFuncOp(&funcOp);
  });
}

} // namespace mlir::triton::AMD
