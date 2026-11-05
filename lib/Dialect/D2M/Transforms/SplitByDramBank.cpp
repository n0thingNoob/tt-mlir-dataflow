// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MSPLITBYDRAMBANK
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

constexpr int64_t kTile = ttcore::TileType::getDefaultShape()[0];

// The function argument `value` is, if any.
BlockArgument getFuncArg(Value value) {
  auto arg = mlir::dyn_cast<BlockArgument>(value);
  return arg && mlir::isa<func::FuncOp>(arg.getOwner()->getParentOp())
             ? arg
             : nullptr;
}

bool isRank2(Value value) {
  return mlir::cast<RankedTensorType>(value.getType()).getRank() == 2;
}

// A matmul that can start a chain: no transpose_a, B a function argument,
// rank 2. Its result is [M, F] with F its N.
bool isSeed(ttir::MatmulOp matmul) {
  return !matmul.getTransposeA() && getFuncArg(matmul.getB()) &&
         isRank2(matmul.getA()) && isRank2(matmul.getB());
}

// The dim of a matmul's B that holds N (seed) or K (end).
int64_t nDimOfB(ttir::MatmulOp matmul) {
  return matmul.getTransposeB() ? 0 : 1;
}
int64_t kDimOfB(ttir::MatmulOp matmul) { return 1 - nDimOfB(matmul); }

bool isElementwise(Operation *op) {
  return mlir::isa<ttir::ElementwiseUnary, ttir::ElementwiseBinary>(op) &&
         op->getNumResults() == 1;
}

// A seed, the elementwise ops that follow it and the matmul that ends them.
struct Chain {
  // Seeds and elementwise ops, in block order.
  SmallVector<Operation *> ops;
  ttir::MatmulOp end;
  int64_t f = 0;
};

// Collect the chain that starts at `seed`, or say why there is none. `members`
// gets the seeds and elementwise ops visited either way.
FailureOr<Chain> collectChain(ttir::MatmulOp seed,
                              llvm::SetVector<Operation *> &members,
                              std::string &why) {
  auto fail = [&](const Twine &reason) {
    why = reason.str();
    return failure();
  };
  auto chainType = mlir::cast<RankedTensorType>(seed.getType());
  Chain chain;
  chain.f = chainType.getDimSize(1);

  members.insert(seed);
  SmallVector<Operation *> worklist = {seed};
  auto addSeed = [&](Operation *op) {
    auto matmul = mlir::dyn_cast_if_present<ttir::MatmulOp>(op);
    if (!matmul || !isSeed(matmul) || matmul.getType() != chainType ||
        matmul->getBlock() != seed->getBlock() ||
        members.contains(matmul.getA().getDefiningOp())) {
      return false;
    }
    if (members.insert(matmul)) {
      worklist.push_back(matmul);
    }
    return true;
  };

  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    for (Operation *user : op->getResult(0).getUsers()) {
      if (members.contains(user) || user == chain.end) {
        continue;
      }
      if (user->getBlock() != seed->getBlock()) {
        return fail("its result is used in another block");
      }
      auto matmul = mlir::dyn_cast<ttir::MatmulOp>(user);
      if (matmul && !matmul.getTransposeA() &&
          members.contains(matmul.getA().getDefiningOp()) &&
          !members.contains(matmul.getB().getDefiningOp()) &&
          getFuncArg(matmul.getB()) && isRank2(matmul.getB()) &&
          mlir::cast<RankedTensorType>(matmul.getB().getType())
                  .getDimSize(kDimOfB(matmul)) == chain.f) {
        if (chain.end) {
          return fail("two matmuls contract the chain");
        }
        chain.end = matmul;
        continue;
      }
      if (!isElementwise(user) || user->getResult(0).getType() != chainType) {
        return fail(Twine("its result reaches ") +
                    user->getName().getStringRef() +
                    ", which is not an elementwise op of the chain or a "
                    "matmul that contracts it over F with an argument");
      }
      members.insert(user);
      worklist.push_back(user);
      for (Value operand : user->getOperands()) {
        Operation *def = operand.getDefiningOp();
        if (!members.contains(def) && !addSeed(def)) {
          return fail(Twine("an operand of ") + user->getName().getStringRef() +
                      " is not a value of the chain");
        }
      }
    }
  }
  if (!chain.end) {
    return fail("no matmul contracts the chain over F with an argument");
  }
  if (members.contains(chain.end)) {
    return fail("the matmul that contracts the chain is also in it");
  }
  chain.ops = llvm::to_vector(members);
  llvm::sort(chain.ops,
             [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
  return chain;
}

// Slice [offset, offset + size) of `dim` of argument `weight`, in `bank`.
Value sliceWeight(IRRewriter &rewriter, Location loc, Value weight, int64_t dim,
                  int64_t offset, int64_t size, int64_t bank) {
  auto type = mlir::cast<RankedTensorType>(weight.getType());
  SmallVector<int32_t> begins = {0, 0};
  SmallVector<int32_t> ends = {static_cast<int32_t>(type.getDimSize(0)),
                               static_cast<int32_t>(type.getDimSize(1))};
  begins[dim] = static_cast<int32_t>(offset);
  ends[dim] = static_cast<int32_t>(offset + size);
  SmallVector<int64_t> shape = llvm::to_vector(type.getShape());
  shape[dim] = size;
  auto slice = rewriter.create<ttir::SliceStaticOp>(
      loc, RankedTensorType::get(shape, type.getElementType()), weight,
      rewriter.getI32ArrayAttr(begins), rewriter.getI32ArrayAttr(ends),
      rewriter.getI32ArrayAttr({1, 1}));
  slice->setAttr(utils::kDramBankAttr, rewriter.getI64IntegerAttr(bank));
  return slice.getResult();
}

// Replace `chain` by one copy per bank and a hub of adds, right before the
// end. Bank j gets the tiles [offset_j, offset_j + size_j) of F.
void splitChain(IRRewriter &rewriter, const Chain &chain,
                ArrayRef<int64_t> sizes) {
  rewriter.setInsertionPoint(chain.end);
  SmallVector<Value> partials;
  int64_t offset = 0;
  for (auto [bank, size] : llvm::enumerate(sizes)) {
    IRMapping mapping;
    for (Operation *op : chain.ops) {
      auto type = mlir::cast<RankedTensorType>(op->getResult(0).getType());
      auto partType = RankedTensorType::get({type.getDimSize(0), size},
                                            type.getElementType());
      if (auto seed = mlir::dyn_cast<ttir::MatmulOp>(op)) {
        Value slice = sliceWeight(rewriter, seed.getLoc(), seed.getB(),
                                  nDimOfB(seed), offset, size, bank);
        mapping.map(seed.getB(), slice);
      }
      Operation *part = rewriter.clone(*op, mapping);
      part->getResult(0).setType(partType);
    }
    ttir::MatmulOp end = chain.end;
    Value slice = sliceWeight(rewriter, end.getLoc(), end.getB(), kDimOfB(end),
                              offset, size, bank);
    mapping.map(end.getB(), slice);
    partials.push_back(rewriter.clone(*end, mapping)->getResult(0));
    offset += size;
  }

  ttir::MatmulOp end = chain.end;
  Value sum = partials.front();
  for (Value partial : llvm::drop_begin(partials)) {
    sum =
        rewriter.create<ttir::AddOp>(end.getLoc(), end.getType(), sum, partial);
  }
  rewriter.replaceOp(end, sum);
  for (Operation *op : llvm::reverse(chain.ops)) {
    rewriter.eraseOp(op);
  }
}

class D2MSplitByDramBank
    : public impl::D2MSplitByDramBankBase<D2MSplitByDramBank> {
public:
  using impl::D2MSplitByDramBankBase<
      D2MSplitByDramBank>::D2MSplitByDramBankBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    ttcore::SystemDescAttr systemDesc =
        ttcore::getCurrentScopeSystemDesc(module);
    if (!systemDesc) {
      return;
    }
    int64_t banks = systemDesc.getChipDescs().front().getNumDramChannels();
    if (!dramArgsAndResults) {
      if (remarks) {
        module.emitRemark("bank split: nothing split, the arguments are not "
                          "in DRAM (dram-args-and-results is not set)");
      }
      return;
    }

    IRRewriter rewriter(&getContext());
    module.walk([&](func::FuncOp func) {
      bool bankedArgs = llvm::any_of(
          llvm::seq<unsigned>(0, func.getNumArguments()), [&](unsigned i) {
            return static_cast<bool>(func.getArgAttr(i, utils::kDramBankAttr));
          });
      if (bankedArgs) {
        if (remarks) {
          func.emitRemark("bank split: nothing split, arguments already carry ")
              << utils::kDramBankAttr;
        }
        return;
      }

      SmallVector<ttir::MatmulOp> seeds;
      func.walk([&](ttir::MatmulOp matmul) {
        if (isSeed(matmul)) {
          seeds.push_back(matmul);
        }
      });
      llvm::SmallPtrSet<Operation *, 16> done;
      for (ttir::MatmulOp seed : seeds) {
        if (done.contains(seed)) {
          continue;
        }
        std::string why;
        llvm::SetVector<Operation *> members;
        FailureOr<Chain> chain = collectChain(seed, members, why);
        if (failed(chain)) {
          // One remark per chain: the other seeds and the matmuls that
          // consume it are not tried again.
          for (Operation *member : members) {
            done.insert(member);
            for (Operation *user : member->getResult(0).getUsers()) {
              if (mlir::isa<ttir::MatmulOp>(user)) {
                done.insert(user);
              }
            }
          }
          if (remarks) {
            seed.emitRemark("bank split: matmul not split: ") << why;
          }
          continue;
        }
        done.insert(chain->ops.begin(), chain->ops.end());
        done.insert(chain->end);

        int64_t f = chain->f;
        if (f % kTile != 0 || f / kTile < banks) {
          if (remarks) {
            InFlightDiagnostic remark =
                seed.emitRemark("bank split: matmul not split: F = ") << f;
            if (f % kTile != 0) {
              remark << " is not a multiple of " << kTile;
            } else {
              remark << " is " << f / kTile << " tiles, fewer than the "
                     << banks << " banks";
            }
          }
          continue;
        }
        int64_t tiles = f / kTile;
        SmallVector<int64_t> sizes;
        for (int64_t j = 0; j < banks; ++j) {
          sizes.push_back(kTile *
                          (tiles / banks + (j < tiles % banks ? 1 : 0)));
        }
        if (remarks) {
          int64_t rest = tiles % banks;
          InFlightDiagnostic remark = seed.emitRemark("bank split: ");
          remark << chain->ops.size()
                 << (chain->ops.size() == 1 ? " op" : " ops")
                 << " and the matmul that ends them split over " << banks
                 << " DRAM banks, F = " << tiles << " tiles: ";
          if (rest == 0) {
            remark << tiles / banks << " per bank";
          } else {
            auto range = [&](int64_t first, int64_t last) {
              return first == last ? "bank " + std::to_string(first)
                                   : "banks " + std::to_string(first) + "-" +
                                         std::to_string(last);
            };
            remark << tiles / banks + 1 << " on " << range(0, rest - 1) << ", "
                   << tiles / banks << " on " << range(rest, banks - 1);
          }
        }
        splitChain(rewriter, *chain, sizes);
      }
    });
  }
};

} // namespace
} // namespace mlir::tt::d2m
