// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Dialect/D2M/Analysis/GridAnalysis.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/Utils/SpatialGroupUtils.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <memory>
#include <optional>
#include <string>

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MSPATIALPLANNING
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

struct GenericDAGNode {
  GenericOp generic;
  SmallVector<unsigned> predecessors;
};

// A projection of block-local tensor SSA dependencies onto generic operations.
// This is not a memory-dependence or cross-control-flow scheduling analysis.
using GenericDAG = SmallVector<GenericDAGNode>;

static GenericDAG buildGenericDAG(Block &block) {
  GenericDAG dag;
  llvm::DenseMap<Operation *, unsigned> nodeIndices;
  for (GenericOp generic : block.getOps<GenericOp>()) {
    if (!generic.hasTensorSemantics()) {
      continue;
    }
    nodeIndices[generic.getOperation()] = dag.size();
    dag.push_back({generic, {}});
  }

  for (GenericDAGNode &node : dag) {
    llvm::SetVector<Value> capturedValues;
    getUsedValuesDefinedAbove(node.generic->getRegions(), capturedValues);
    SmallVector<Value> worklist(node.generic->getOperands());
    llvm::append_range(worklist, capturedValues);
    llvm::SmallPtrSet<Operation *, 16> visited;
    llvm::SetVector<unsigned> predecessors;
    while (!worklist.empty()) {
      Operation *def = worklist.pop_back_val().getDefiningOp();
      if (!def || def->getBlock() != &block || !visited.insert(def).second) {
        continue;
      }
      auto producer = nodeIndices.find(def);
      if (producer != nodeIndices.end()) {
        predecessors.insert(producer->second);
        continue;
      }
      // Region-bearing operations are opaque scheduling boundaries. Ordinary
      // SSA operations, including layout views, preserve dependency paths.
      if (def->getNumRegions() == 0) {
        llvm::append_range(worklist, def->getOperands());
      }
    }
    node.predecessors.assign(predecessors.begin(), predecessors.end());
    llvm::sort(node.predecessors);
  }
  return dag;
}

// What one generic costs inside a spatial, estimated before GridSelection.
struct NodeCost {
  // Region shape: the grid GridSelection picks for the generic's output when
  // the whole device is the target. With exactly this range as the target,
  // GridSelection picks the same grid, so every edge keeps the layout the
  // default flow uses between these two generics.
  SmallVector<int64_t, 2> grid;
  // One CB per input and output operand (measured on the S1 programs: the
  // lowered program of every compute generic has exactly that many).
  int64_t cbs = 0;
  // Local semaphores the generic uses by itself: a matmul multicasts its
  // operands with up to two semaphores per operand. Edges add their own.
  int64_t semaphores = 0;
};

// One edge candidate: result `result` of member `producer` read by member
// `consumer` (indices into a group's member list).
struct Edge {
  unsigned producer;
  unsigned result;
  unsigned consumer;
};

// A group under construction with a placed core range per member.
struct Group {
  SmallVector<unsigned> nodes;
  SmallVector<ttcore::CoreRangeAttr> ranges;
  // Region shape of each member: its output grid when GridAnalysis targets
  // targets[target] (0 = the whole device, NodeCost::grid). All members use
  // one target, so two members connected by an edge split the dimensions
  // they share the same way, as in the default flow, and the edge keeps one
  // layout at both ends. Shrinking a single member breaks that.
  SmallVector<SmallVector<int64_t, 2>> grids;
  unsigned target = 0;
  // Per member: whether it reads a DRAM tensor from outside the group (its
  // range is placed near DRAM, see place()), and the range first fit would
  // have given it with the number of DRAM-near cores that covers.
  SmallVector<bool> readsDram;
  SmallVector<std::pair<ttcore::CoreRangeAttr, int64_t>> firstFit;
  SmallVector<Edge> edges;
  // occupied[y][x]
  SmallVector<SmallVector<bool>> occupied;
};

// Largest number of local semaphores on one core of the group: a member's
// own semaphores, plus one per output that feeds an edge (shared by all its
// consumers) and one per producer output it reads through an edge.
static int64_t maxSemaphoresPerCore(ArrayRef<unsigned> nodes,
                                    ArrayRef<Edge> edges,
                                    ArrayRef<NodeCost> costs) {
  int64_t most = 0;
  for (unsigned i = 0; i < nodes.size(); ++i) {
    llvm::SmallSetVector<std::pair<unsigned, unsigned>, 4> outs, ins;
    for (const Edge &e : edges) {
      if (e.producer == i) {
        outs.insert({e.producer, e.result});
      }
      if (e.consumer == i) {
        ins.insert({e.producer, e.result});
      }
    }
    most = std::max<int64_t>(most, costs[nodes[i]].semaphores + outs.size() +
                                       ins.size());
  }
  return most;
}

class D2MSpatialPlanningPass
    : public impl::D2MSpatialPlanningBase<D2MSpatialPlanningPass> {
public:
  using impl::D2MSpatialPlanningBase<
      D2MSpatialPlanningPass>::D2MSpatialPlanningBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    bool greedy = policy == "greedy" && !ttnnMode;
    if (greedy) {
      deviceGrid = llvm::to_vector(
          ttcore::lookupDevice(module).getWorkerGrid().getShape());
      int64_t h = deviceGrid[0], w = deviceGrid[1];
      targets.assign(1, {h, w});
      for (int64_t a = 1; a <= h; ++a) {
        for (int64_t b = 1; b <= w; ++b) {
          if (h % a == 0 && w % b == 0 && (a != h || b != w)) {
            targets.push_back({a, b});
          }
        }
      }
      // Most cores first, then the aspect ratio closest to the device's (the
      // ratio of a * w to b * h nearest to 1), then fewer rows.
      auto skew = [&](ArrayRef<int64_t> t) {
        int64_t p = t[0] * w, q = t[1] * h;
        return std::make_pair(std::max(p, q), std::min(p, q));
      };
      std::stable_sort(targets.begin() + 1, targets.end(),
                       [&](ArrayRef<int64_t> l, ArrayRef<int64_t> r) {
                         if (l[0] * l[1] != r[0] * r[1]) {
                           return l[0] * l[1] > r[0] * r[1];
                         }
                         auto [ln, ld] = skew(l);
                         auto [rn, rd] = skew(r);
                         if (ln * rd != rn * ld) {
                           return ln * rd < rn * ld;
                         }
                         return l[0] < r[0];
                       });
      analyses.clear();
      analyses.resize(targets.size());
      analyses[0] =
          std::make_shared<GridAnalysis>(module, deviceGrid, ttnnMode);
      dramCores.clear();
      if (ttcore::SystemDescAttr systemDesc =
              ttcore::getCurrentScopeSystemDesc(module)) {
        for (ttcore::CoreCoordAttr core :
             systemDesc.getChipDescs()
                 .front()
                 .getDramBankToLogicalWorkerNoc0()) {
          dramCores.push_back({core.getY(), core.getX()});
        }
      }
    }
    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      for (Block &block : func.getBody()) {
        GenericDAG dag = buildGenericDAG(block);
        numGenericNodes += dag.size();
        for (const GenericDAGNode &node : dag) {
          numDependencyEdges += node.predecessors.size();
        }
        if (greedy) {
          planBlock(dag, block, *analyses[0]);
        }
        nextGenericIndex += dag.size();
      }
    }
    if (!greedy) {
      markAllAnalysesPreserved();
    }
  }

private:
  std::optional<NodeCost>
  costOf(GenericOp generic, const GridAnalysis &gridAnalysis, std::string &why);
  std::string tryAdd(Group &group, unsigned node, const GenericDAG &dag,
                     Block &block, ArrayRef<NodeCost> costs);
  bool place(Group &group, ArrayRef<int64_t> grid, bool nearDram);
  int64_t dramCoresIn(ttcore::CoreRangeAttr range) const;
  std::optional<SmallVector<int64_t, 2>> gridAt(unsigned node, unsigned target,
                                                const GenericDAG &dag,
                                                ArrayRef<NodeCost> costs);
  bool fit(Group &group, unsigned node, bool readsDram, const GenericDAG &dag,
           ArrayRef<NodeCost> costs, std::string &tried);
  void planBlock(const GenericDAG &dag, Block &block,
                 const GridAnalysis &gridAnalysis);
  Group emptyGroup();

  SmallVector<int64_t> deviceGrid;
  // GridAnalysis targets: the device grid, then its factor shapes (a divides
  // the device height, b its width) in the order fit() tries them. A grid
  // GridAnalysis picks for a smaller target is again picked exactly by
  // GridSelection with that grid as the region.
  SmallVector<SmallVector<int64_t, 2>> targets;
  // GridAnalysis per target, built on first use (shared: passes are copied
  // when cloned; runOnOperation rebuilds them).
  SmallVector<std::shared_ptr<GridAnalysis>> analyses;
  // Logical worker cores next to the DRAM banks (y, x), from the system desc.
  SmallVector<std::pair<int64_t, int64_t>> dramCores;
  int64_t nextGenericIndex = 0;
  int64_t nextGroupId = 0;
};

std::optional<NodeCost> D2MSpatialPlanningPass::costOf(
    GenericOp generic, const GridAnalysis &gridAnalysis, std::string &why) {
  if (generic->hasAttr(kSpatialGroupAttr)) {
    why = "it already has a group";
    return std::nullopt;
  }
  if (generic.getInputs().empty()) {
    why = "it has no inputs (a fill stays outside, as an input)";
    return std::nullopt;
  }
  if (generic.getOutputs().size() != 1 || generic.getNumResults() != 1) {
    why = "it does not have exactly one output";
    return std::nullopt;
  }
  const GenericGridAnalysisResult *result = gridAnalysis.lookup(generic);
  unsigned out = generic.getInputs().size();
  if (!result || result->normalizedOperandGrids.size() <= out) {
    why = "GridAnalysis has no grid for it";
    return std::nullopt;
  }
  ArrayRef<int64_t> grid = result->normalizedOperandGrids[out];
  if (grid.size() != 2 || grid[0] > deviceGrid[0] || grid[1] > deviceGrid[1]) {
    why = "its grid is not a 2D grid inside the device";
    return std::nullopt;
  }
  NodeCost cost;
  cost.grid.assign(grid.begin(), grid.end());
  cost.cbs = generic->getNumOperands();
  bool matmul = false;
  generic->walk([&](Operation *op) {
    matmul |= mlir::isa<TileMatmulOp, TileMatmulBlockOp>(op);
  });
  cost.semaphores = matmul ? 4 : 0;
  return cost;
}

Group D2MSpatialPlanningPass::emptyGroup() {
  Group group;
  group.occupied.assign(deviceGrid[0], SmallVector<bool>(deviceGrid[1], false));
  for (size_t i = 0; i + 1 < excludeCores.size(); i += 2) {
    int64_t y = excludeCores[i], x = excludeCores[i + 1];
    if (y >= 0 && y < deviceGrid[0] && x >= 0 && x < deviceGrid[1]) {
      group.occupied[y][x] = true;
    }
  }
  return group;
}

int64_t D2MSpatialPlanningPass::dramCoresIn(ttcore::CoreRangeAttr range) const {
  return llvm::count_if(dramCores, [&](std::pair<int64_t, int64_t> core) {
    return core.first >= range.getStartCoord().getY() &&
           core.first <= range.getEndCoord().getY() &&
           core.second >= range.getStartCoord().getX() &&
           core.second <= range.getEndCoord().getX();
  });
}

// Places a grid-shaped rectangle into the free cores: first fit, row-major,
// or for a generic that reads DRAM (`nearDram`) the free range covering the
// most DRAM-near cores, the first fit among equals.
bool D2MSpatialPlanningPass::place(Group &group, ArrayRef<int64_t> grid,
                                   bool nearDram) {
  int64_t h = grid[0], w = grid[1];
  MLIRContext *ctx = &getContext();
  auto rangeAt = [&](int64_t y, int64_t x) {
    return ttcore::CoreRangeAttr::get(
        ctx, ttcore::CoreCoordAttr::get(ctx, y, x),
        ttcore::CoreCoordAttr::get(ctx, y + h - 1, x + w - 1));
  };
  std::optional<std::pair<ttcore::CoreRangeAttr, int64_t>> first, best;
  for (int64_t y = 0; y + h <= deviceGrid[0] && (nearDram || !first); ++y) {
    for (int64_t x = 0; x + w <= deviceGrid[1] && (nearDram || !first); ++x) {
      bool free = true;
      for (int64_t dy = 0; dy < h && free; ++dy) {
        for (int64_t dx = 0; dx < w && free; ++dx) {
          free = !group.occupied[y + dy][x + dx];
        }
      }
      if (!free) {
        continue;
      }
      ttcore::CoreRangeAttr range = rangeAt(y, x);
      int64_t covered = dramCoresIn(range);
      if (!first) {
        first.emplace(range, covered);
      }
      if (!best || covered > best->second) {
        best.emplace(range, covered);
      }
    }
  }
  if (!first) {
    return false;
  }
  ttcore::CoreRangeAttr range = nearDram ? best->first : first->first;
  for (int64_t y = range.getStartCoord().getY();
       y <= range.getEndCoord().getY(); ++y) {
    for (int64_t x = range.getStartCoord().getX();
         x <= range.getEndCoord().getX(); ++x) {
      group.occupied[y][x] = true;
    }
  }
  group.ranges.push_back(range);
  group.firstFit.push_back(*first);
  return true;
}

// Region shape of `node` when GridAnalysis targets targets[target].
std::optional<SmallVector<int64_t, 2>>
D2MSpatialPlanningPass::gridAt(unsigned node, unsigned target,
                               const GenericDAG &dag,
                               ArrayRef<NodeCost> costs) {
  if (target == 0) {
    return costs[node].grid;
  }
  if (!analyses[target]) {
    analyses[target] = std::make_shared<GridAnalysis>(
        getOperation(), targets[target], ttnnMode);
  }
  GenericOp generic = dag[node].generic;
  const GenericGridAnalysisResult *result = analyses[target]->lookup(generic);
  unsigned out = generic.getInputs().size();
  if (!result || result->normalizedOperandGrids.size() <= out ||
      result->normalizedOperandGrids[out].size() != 2) {
    return std::nullopt;
  }
  // A grid larger than the target (seen for a reduction reading a DRAM
  // operand) is not what GridSelection picks in a range of that shape.
  ArrayRef<int64_t> grid = result->normalizedOperandGrids[out];
  if (grid[0] > targets[target][0] || grid[1] > targets[target][1]) {
    return std::nullopt;
  }
  return llvm::to_vector<2>(grid);
}

// Places `node` in `group` (it is not in group.nodes yet) at the group's
// target. If no range is free, places the whole group again, members first,
// at the first later target where everything fits. If no target works,
// leaves the group unchanged and lists the targets tried in `tried`.
bool D2MSpatialPlanningPass::fit(Group &group, unsigned node, bool readsDram,
                                 const GenericDAG &dag,
                                 ArrayRef<NodeCost> costs, std::string &tried) {
  auto grid = gridAt(node, group.target, dag, costs);
  if (grid && place(group, *grid, readsDram)) {
    group.grids.push_back(*grid);
    group.readsDram.push_back(readsDram);
    return true;
  }
  SmallVector<unsigned> nodes(group.nodes);
  nodes.push_back(node);
  SmallVector<bool> dram(group.readsDram);
  dram.push_back(readsDram);
  // Places the whole group again at target `t`, near DRAM where `nearDram`.
  auto placeAll = [&](unsigned t, bool nearDram) {
    Group trial = emptyGroup();
    trial.target = t;
    bool fits = llvm::all_of(llvm::zip(nodes, dram), [&](auto nd) {
      auto [n, d] = nd;
      auto g = gridAt(n, t, dag, costs);
      if (!g || !place(trial, *g, nearDram && d)) {
        return false;
      }
      trial.grids.push_back(*g);
      return true;
    });
    if (fits) {
      group.ranges = std::move(trial.ranges);
      group.grids = std::move(trial.grids);
      group.firstFit = std::move(trial.firstFit);
      group.occupied = std::move(trial.occupied);
      group.readsDram = dram;
      group.target = t;
    }
    return fits;
  };
  // Placing near DRAM can leave no room where first fit would: before
  // shrinking, try the current target with first fit only.
  if (llvm::is_contained(dram, true) && placeAll(group.target, false)) {
    return true;
  }
  for (unsigned t = group.target + 1; t < targets.size(); ++t) {
    if (placeAll(t, true) ||
        (llvm::is_contained(dram, true) && placeAll(t, false))) {
      return true;
    }
    tried += (tried.empty() ? "" : ", ") + std::to_string(targets[t][0]) + "x" +
             std::to_string(targets[t][1]);
  }
  return false;
}

// Suffix of a remark about a range that does not fit.
static std::string triedTargets(const std::string &tried) {
  return tried.empty() ? "" : " (nor with the group at targets " + tried + ")";
}

// Adds `node` to `group` if the group stays convex, every connection to the
// group can be an edge and the budgets hold; otherwise returns why not and
// leaves the group unchanged.
std::string D2MSpatialPlanningPass::tryAdd(Group &group, unsigned node,
                                           const GenericDAG &dag, Block &block,
                                           ArrayRef<NodeCost> costs) {
  GenericOp generic = dag[node].generic;
  llvm::DenseMap<Operation *, unsigned> member;
  for (auto [i, n] : llvm::enumerate(group.nodes)) {
    member[dag[n].generic] = i;
  }

  SmallVector<Edge> edges(group.edges);
  bool readsDram = false;
  for (OpOperand &operand : generic.getInputsMutable()) {
    bool readsMember =
        llvm::any_of(producerGenerics(operand.get(), block),
                     [&](Operation *p) { return member.contains(p); });
    if (!readsMember) {
      readsDram |= isDRAMLayout(operand.get().getType());
      continue;
    }
    OpResult src = traceEdgeSource(operand.get());
    if (!src || !member.contains(src.getOwner())) {
      return "input " + std::to_string(operand.getOperandNumber()) +
             " reads the group through ops that cannot be an edge";
    }
    Type dst = operand.get().getType();
    if (!isL1Layout(src.getType()) ||
        (!isL1Layout(dst) && !isDRAMLayout(dst))) {
      return "input " + std::to_string(operand.getOperandNumber()) +
             " is not an L1 tensor at both ends (an edge stays in L1)";
    }
    if (!sameLayoutExceptMemorySpace(src.getType(), dst)) {
      return "input " + std::to_string(operand.getOperandNumber()) +
             " needs a layout change from its producer";
    }
    Edge edge{member[src.getOwner()], src.getResultNumber(),
              static_cast<unsigned>(group.nodes.size())};
    if (llvm::none_of(edges, [&](const Edge &e) {
          return e.producer == edge.producer && e.result == edge.result &&
                 e.consumer == edge.consumer;
        })) {
      edges.push_back(edge);
    }
  }

  SmallVector<unsigned> nodes(group.nodes);
  nodes.push_back(node);
  int64_t cbs = 0;
  for (unsigned n : nodes) {
    cbs += costs[n].cbs;
  }
  if (cbs > maxCbs) {
    return "the spatial would need " + std::to_string(cbs) + " CBs (limit " +
           std::to_string(maxCbs) + ")";
  }
  int64_t sems = maxSemaphoresPerCore(nodes, edges, costs);
  if (sems > maxSemaphores) {
    return "a core would need " + std::to_string(sems) +
           " local semaphores (limit " + std::to_string(maxSemaphores) + ")";
  }

  Group trial = group;
  std::string tried;
  if (!fit(trial, node, readsDram, dag, costs, tried)) {
    return "no free " + std::to_string(costs[node].grid[0]) + "x" +
           std::to_string(costs[node].grid[1]) + " core range is left" +
           triedTargets(tried);
  }
  trial.nodes = std::move(nodes);
  trial.edges = std::move(edges);
  group = std::move(trial);
  return "";
}

void D2MSpatialPlanningPass::planBlock(const GenericDAG &dag, Block &block,
                                       const GridAnalysis &gridAnalysis) {
  SmallVector<NodeCost> costs(dag.size());
  SmallVector<bool> eligible(dag.size(), false);
  for (auto [i, node] : llvm::enumerate(dag)) {
    std::string why;
    if (llvm::is_contained(skipGenerics, nextGenericIndex + int64_t(i))) {
      why = "it is in skip-generics";
    } else if (auto cost = costOf(node.generic, gridAnalysis, why)) {
      costs[i] = *cost;
      eligible[i] = true;
    }
    if (!eligible[i] && remarks) {
      node.generic->emitRemark("generic #")
          << nextGenericIndex + int64_t(i) << " is not grouped: " << why;
    }
  }

  auto genericName = [&](unsigned n) {
    return "#" + std::to_string(nextGenericIndex + n);
  };
  auto close = [&](Group &group) {
    if (group.nodes.size() >= 2) {
      int64_t id = nextGroupId++;
      Builder b(&getContext());
      std::string names, shrunk;
      int64_t cores = 0, cbs = 0;
      for (auto [n, range, grid] :
           llvm::zip(group.nodes, group.ranges, group.grids)) {
        GenericOp generic = dag[n].generic;
        generic->setAttr(kSpatialGroupAttr, b.getI64IntegerAttr(id));
        generic->setAttr(kSpatialRangeAttr, range);
        names += " " + genericName(n);
        cores += grid[0] * grid[1];
        cbs += costs[n].cbs;
        shrunk += " " + genericName(n) + " " +
                  std::to_string(costs[n].grid[0]) + "x" +
                  std::to_string(costs[n].grid[1]) + "->" +
                  std::to_string(grid[0]) + "x" + std::to_string(grid[1]);
      }
      for (auto [n, range, dram, first] : llvm::zip(
               group.nodes, group.ranges, group.readsDram, group.firstFit)) {
        if (!remarks || !dram) {
          continue;
        }
        auto where = [](ttcore::CoreRangeAttr r) {
          int64_t y = r.getStartCoord().getY(), x = r.getStartCoord().getX();
          return "(" + std::to_string(y) + "," + std::to_string(x) + ") " +
                 std::to_string(r.getEndCoord().getY() - y + 1) + "x" +
                 std::to_string(r.getEndCoord().getX() - x + 1);
        };
        dag[n].generic->emitRemark("spatial group ")
            << id << ": generic " << genericName(n) << " reads DRAM: region "
            << where(range) << " covers " << dramCoresIn(range) << " of "
            << dramCores.size() << " DRAM-near cores (first fit "
            << where(first.first) << " covers " << first.second << ")";
      }
      if (remarks && group.target != 0) {
        dag[group.nodes.front()].generic->emitRemark("spatial group ")
            << id << " shrinks to target " << targets[group.target][0] << "x"
            << targets[group.target][1] << ":" << shrunk;
      }
      if (remarks) {
        dag[group.nodes.front()].generic->emitRemark("spatial group ")
            << id << ": generics" << names << ", " << group.edges.size()
            << " edges, " << cores << " cores, " << cbs << " CBs, "
            << maxSemaphoresPerCore(group.nodes, group.edges, costs)
            << " semaphores per core at most";
      }
    }
    group = emptyGroup();
  };

  Group group = emptyGroup();
  // Generics outside the group that depend on one of its members.
  llvm::DenseSet<unsigned> tainted;
  for (unsigned i = 0; i < dag.size(); ++i) {
    bool fromGroup = llvm::any_of(dag[i].predecessors, [&](unsigned p) {
      return llvm::is_contained(group.nodes, p) || tainted.contains(p);
    });
    if (!eligible[i]) {
      if (fromGroup) {
        tainted.insert(i);
      }
      continue;
    }
    std::string why;
    if (!group.nodes.empty()) {
      auto taintedPred = llvm::find_if(
          dag[i].predecessors, [&](unsigned p) { return tainted.contains(p); });
      why = taintedPred != dag[i].predecessors.end()
                ? "not convex: it reads generic " + genericName(*taintedPred) +
                      ", which is outside the group and reads it"
                : tryAdd(group, i, dag, block, costs);
      if (why.empty()) {
        continue;
      }
      if (remarks) {
        dag[i].generic->emitRemark("generic ")
            << genericName(i) << " starts a new group: " << why;
      }
      close(group);
      tainted.clear();
    }
    std::string tried;
    bool readsDram =
        llvm::any_of(GenericOp(dag[i].generic).getInputs(),
                     [](Value v) { return isDRAMLayout(v.getType()); });
    if (fit(group, i, readsDram, dag, costs, tried)) {
      group.nodes.push_back(i);
    } else if (remarks) {
      dag[i].generic->emitRemark("generic ")
          << genericName(i) << " is not grouped: its " << costs[i].grid[0]
          << "x" << costs[i].grid[1]
          << " grid does not fit around the excluded cores"
          << triedTargets(tried);
    }
  }
  close(group);
}

} // namespace
} // namespace mlir::tt::d2m
