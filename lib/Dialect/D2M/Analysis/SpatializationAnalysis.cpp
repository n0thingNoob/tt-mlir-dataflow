// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Analysis/SpatializationAnalysis.h"

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MTraits.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"

#include <algorithm>

namespace mlir::tt::d2m {
namespace {

// Tensor allocation defines a fresh value; it does not constrain the relative
// execution of existing generics. In particular d2m.empty declares Allocate,
// so isMemoryEffectFree alone would split almost every TTIRToD2M graph.
static bool isGroupingBarrier(Operation &op) {
  if (isa<d2m::EmptyOp, tensor::EmptyOp>(op) &&
      llvm::all_of(op.getResultTypes(),
                   [](Type type) { return isa<TensorType>(type); })) {
    return false;
  }
  return op.getNumRegions() != 0 || !isMemoryEffectFree(&op);
}

template <typename T>
static void appendUnique(SmallVectorImpl<T> &values, const T &value) {
  if (!llvm::is_contained(values, value)) {
    values.push_back(value);
  }
}

static void classifyGeneric(SpatialGenericNode &node) {
  SmallVector<std::string> categories;
  node.generic->walk([&](Operation *op) {
    if (!op->hasTrait<D2MGenericRegionComputeOpTrait>()) {
      return;
    }
    appendUnique(node.computeOps, op->getName().getStringRef().str());
    std::string category = "unknown";
    std::string resource = "unknown";
    if (isa<TileMatmulOp, TileMatmulBlockOp>(op)) {
      category = "matmul";
      resource = "FPU";
    } else if (isa<TileReluOp, TileSigmoidOp, TileGeluOp>(op)) {
      category = "activation";
      resource = "SFPU";
    } else if (isa<TileAddOp, TileSubOp, TileMulOp>(op)) {
      category = "elementwise";
      // These ops have FPU and SFPU lowerings depending on types and operands.
      resource = "FPU_or_SFPU";
    }
    appendUnique(categories, category);
    appendUnique(node.resourceHints, resource);
  });
  node.opType = categories.empty()       ? "unknown"
                : categories.size() == 1 ? categories.front()
                                         : "mixed";
  if (node.resourceHints.empty()) {
    node.resourceHints.push_back("unknown");
  }
}

static void
traceDependency(SpatialGenericDAG &dag, Block &block,
                const llvm::DenseMap<Operation *, unsigned> &indices,
                unsigned consumer, Value target, unsigned targetIndex,
                bool captured) {
  struct WorkItem {
    Value value;
    SmallVector<Operation *> via;
  };
  SmallVector<WorkItem> worklist;
  worklist.push_back({target, {}});
  llvm::DenseSet<Value> visited;
  while (!worklist.empty()) {
    WorkItem item = worklist.pop_back_val();
    if (!visited.insert(item.value).second) {
      continue;
    }
    Operation *def = item.value.getDefiningOp();
    auto producer = indices.find(def);
    if (producer != indices.end()) {
      dag.dependencies.push_back({producer->second, consumer, item.value,
                                  target, targetIndex, captured,
                                  std::move(item.via)});
      appendUnique(dag.nodes[consumer].predecessors, producer->second);
      appendUnique(dag.nodes[producer->second].successors, consumer);
      continue;
    }
    // Do not look through control flow or a generic that was excluded from the
    // tensor DAG. Ordinary region-free operations preserve SSA dependencies,
    // even when they are barriers to candidate formation.
    if (!def || def->getBlock() != &block || def->getNumRegions() != 0 ||
        def->getNumOperands() == 0) {
      dag.dependencies.push_back({std::nullopt, consumer, item.value, target,
                                  targetIndex, captured, std::move(item.via)});
      continue;
    }
    item.via.push_back(def);
    // Reverse insertion preserves operand order in this depth-first traversal.
    for (Value operand : llvm::reverse(def->getOperands())) {
      worklist.push_back({operand, item.via});
    }
  }
}

static Operation *topLevelUser(Operation *op, Block &block) {
  while (op && op->getBlock() != &block) {
    op = op->getParentOp();
  }
  return op;
}

static void
collectExternalUses(SpatialGenericDAG &dag, Block &block,
                    const llvm::DenseMap<Operation *, unsigned> &indices) {
  // Walk users in IR order, rather than relying on the SSA use-list ordering.
  llvm::DenseMap<Value, unsigned> results;
  for (auto [id, node] : llvm::enumerate(dag.nodes)) {
    for (Value result : node.results) {
      results[result] = id;
    }
  }
  block.getParentOp()->walk([&](Operation *user) {
    if (indices.contains(topLevelUser(user, block))) {
      return;
    }
    for (OpOperand &operand : user->getOpOperands()) {
      auto producer = results.find(operand.get());
      if (producer != results.end()) {
        dag.externalUses.push_back({producer->second, operand.get(), user,
                                    operand.getOperandNumber()});
      }
    }
  });
}

static void computeReachability(SpatialGenericDAG &dag) {
  for (unsigned root = 0; root < dag.nodes.size(); ++root) {
    llvm::BitVector reachable(dag.nodes.size());
    SmallVector<unsigned> worklist(dag.nodes[root].successors);
    while (!worklist.empty()) {
      unsigned next = worklist.pop_back_val();
      if (reachable.test(next)) {
        continue;
      }
      reachable.set(next);
      llvm::append_range(worklist, dag.nodes[next].successors);
    }
    dag.reachable.push_back(std::move(reachable));
  }
}

static bool sameInterval(const SpatialGenericDAG &dag, unsigned a, unsigned b) {
  return dag.nodes[a].interval == dag.nodes[b].interval;
}

static bool isChainEdge(const SpatialGenericDAG &dag, unsigned a, unsigned b) {
  return sameInterval(dag, a, b) && dag.nodes[a].successors.size() == 1 &&
         dag.nodes[b].predecessors.size() == 1;
}

static void addCandidate(SmallVectorImpl<RegionCandidate> &candidates,
                         SmallVector<unsigned> members,
                         RegionCandidateKind kind, StringRef reason,
                         RegionCandidateContext context = {}) {
  llvm::sort(members);
  members.erase(std::unique(members.begin(), members.end()), members.end());
  auto found = llvm::find_if(candidates, [&](const RegionCandidate &candidate) {
    return candidate.members == members;
  });
  if (found == candidates.end()) {
    candidates.emplace_back();
    found = std::prev(candidates.end());
    found->members = std::move(members);
  }
  appendUnique(found->kinds, kind);
  appendUnique(found->reasons, reason.str());
  if (context.producer || context.consumer || context.externalInput) {
    auto equal = [&](const RegionCandidateContext &other) {
      return other.producer == context.producer &&
             other.consumer == context.consumer &&
             other.externalInput == context.externalInput;
    };
    if (!llvm::any_of(found->contexts, equal)) {
      found->contexts.push_back(context);
    }
  }
}

static void collectChains(const SpatialGenericDAG &dag,
                          SmallVectorImpl<RegionCandidate> &candidates) {
  for (unsigned id = 0; id < dag.nodes.size(); ++id) {
    const auto &node = dag.nodes[id];
    if (node.predecessors.size() == 1 &&
        isChainEdge(dag, node.predecessors.front(), id)) {
      continue;
    }
    SmallVector<unsigned> members{id};
    unsigned current = id;
    while (dag.nodes[current].successors.size() == 1) {
      unsigned next = dag.nodes[current].successors.front();
      if (!isChainEdge(dag, current, next)) {
        break;
      }
      members.push_back(next);
      current = next;
    }
    if (members.size() >= 2) {
      addCandidate(candidates, std::move(members),
                   RegionCandidateKind::ProducerConsumerChain,
                   "maximal non-branching producer-consumer chain");
    }
  }
}

static void groupIndependent(const SpatialGenericDAG &dag,
                             ArrayRef<unsigned> members,
                             RegionCandidateContext context,
                             SmallVectorImpl<RegionCandidate> &candidates) {
  SmallVector<unsigned> ordered(members);
  llvm::sort(ordered);
  ordered.erase(std::unique(ordered.begin(), ordered.end()), ordered.end());
  SmallVector<SmallVector<unsigned>> groups;
  for (unsigned id : ordered) {
    // Only members must share an interval. The context is outside the
    // candidate: e.g. two consumers after a barrier may share a producer
    // before it without placing that producer in their region.
    auto group = llvm::find_if(groups, [&](const auto &existing) {
      return sameInterval(dag, id, existing.front()) &&
             llvm::all_of(existing, [&](unsigned member) {
               return dag.independent(id, member);
             });
    });
    if (group == groups.end()) {
      groups.push_back({id});
    } else {
      group->push_back(id);
    }
  }
  for (auto &group : groups) {
    if (group.size() >= 2) {
      addCandidate(candidates, std::move(group),
                   RegionCandidateKind::ParallelBranches,
                   "shared dataflow context with no dependency path between "
                   "any two members",
                   context);
    }
  }
}

static void
collectParallelBranches(const SpatialGenericDAG &dag,
                        SmallVectorImpl<RegionCandidate> &candidates) {
  for (unsigned id = 0; id < dag.nodes.size(); ++id) {
    groupIndependent(dag, dag.nodes[id].successors, {id, std::nullopt, {}},
                     candidates);
    groupIndependent(dag, dag.nodes[id].predecessors, {std::nullopt, id, {}},
                     candidates);
  }
  // Keep root values in encounter order: DenseMap iteration is not stable.
  llvm::SetVector<Value> roots;
  llvm::DenseMap<Value, SmallVector<unsigned>> users;
  for (const SpatialDependency &dependency : dag.dependencies) {
    const auto &node = dag.nodes[dependency.consumer];
    if (dependency.producer || !isa<TensorType>(dependency.source.getType())) {
      continue;
    }
    // DPS initializers are dependencies but not shared read-input context.
    if (!dependency.captured && dependency.targetIndex >= node.inputs.size()) {
      continue;
    }
    // Explicit operands also appear as captures in generic bodies. Do not
    // accidentally count an output initializer as a read through that route.
    if (dependency.captured &&
        llvm::is_contained(node.outputs, dependency.target) &&
        !llvm::is_contained(node.inputs, dependency.target)) {
      continue;
    }
    roots.insert(dependency.source);
    appendUnique(users[dependency.source], dependency.consumer);
  }
  for (Value root : roots) {
    groupIndependent(dag, users[root], {std::nullopt, std::nullopt, root},
                     candidates);
  }
}

static void collectForkJoins(const SpatialGenericDAG &dag,
                             SmallVectorImpl<RegionCandidate> &candidates) {
  for (unsigned fork = 0; fork < dag.nodes.size(); ++fork) {
    if (dag.nodes[fork].successors.size() < 2) {
      continue;
    }
    SmallVector<unsigned> members{fork};
    SmallVector<unsigned> ends;
    std::optional<unsigned> join;
    llvm::BitVector seen(dag.nodes.size());
    bool valid = true;
    for (unsigned first : dag.nodes[fork].successors) {
      unsigned current = first;
      unsigned previous = fork;
      while (true) {
        if (!sameInterval(dag, fork, current)) {
          valid = false;
          break;
        }
        const auto &node = dag.nodes[current];
        if (node.predecessors.size() > 1) {
          // Each branch must have an internal node, not just a fork->join edge.
          if (previous == fork || (join && *join != current)) {
            valid = false;
          }
          join = current;
          ends.push_back(previous);
          break;
        }
        if (node.predecessors.size() != 1 ||
            node.predecessors.front() != previous ||
            node.successors.size() != 1 || seen.test(current)) {
          valid = false;
          break;
        }
        seen.set(current);
        members.push_back(current);
        previous = current;
        current = node.successors.front();
      }
      if (!valid) {
        break;
      }
    }
    if (!valid || !join) {
      continue;
    }
    llvm::sort(ends);
    if (ends != dag.nodes[*join].predecessors) {
      continue;
    }
    members.push_back(*join);
    addCandidate(
        candidates, std::move(members), RegionCandidateKind::ForkJoin,
        "disjoint linear branches from one fork reconverge at one join",
        {fork, *join, {}});
  }
}

static void populateCandidateDetails(const SpatialGenericDAG &dag,
                                     RegionCandidate &candidate) {
  llvm::BitVector members(dag.nodes.size());
  for (unsigned member : candidate.members) {
    members.set(member);
  }
  for (auto [index, edge] : llvm::enumerate(dag.dependencies)) {
    bool sourceInside = edge.producer && members.test(*edge.producer);
    bool targetInside = members.test(edge.consumer);
    if (sourceInside && targetInside) {
      candidate.internalDependencies.push_back(index);
    } else if (targetInside) {
      candidate.incomingDependencies.push_back(index);
    } else if (sourceInside) {
      candidate.outgoingDependencies.push_back(index);
    }
  }
  for (auto [index, use] : llvm::enumerate(dag.externalUses)) {
    if (members.test(use.producer)) {
      candidate.externalUses.push_back(index);
    }
  }
  for (auto [index, a] : llvm::enumerate(candidate.members)) {
    for (unsigned b : ArrayRef(candidate.members).drop_front(index + 1)) {
      if (dag.independent(a, b)) {
        candidate.independentPairs.emplace_back(a, b);
      }
    }
  }
}

static StringRef kindName(RegionCandidateKind kind) {
  switch (kind) {
  case RegionCandidateKind::ProducerConsumerChain:
    return "producer_consumer_chain";
  case RegionCandidateKind::ParallelBranches:
    return "parallel_branches";
  case RegionCandidateKind::ForkJoin:
    return "fork_join";
  case RegionCandidateKind::Singleton:
    return "singleton";
  }
  llvm_unreachable("unknown candidate kind");
}

static void printIDs(llvm::raw_ostream &os, ArrayRef<unsigned> ids,
                     StringRef prefix) {
  os << '[';
  llvm::interleaveComma(ids, os, [&](unsigned id) { os << prefix << id; });
  os << ']';
}

static void printValues(llvm::raw_ostream &os, ArrayRef<Value> values,
                        AsmState &state) {
  os << '[';
  llvm::interleaveComma(values, os, [&](Value value) {
    value.printAsOperand(os, state);
    os << " : " << value.getType();
  });
  os << ']';
}

} // namespace

bool SpatialGenericDAG::independent(unsigned a, unsigned b) const {
  return a != b && !reachable[a].test(b) && !reachable[b].test(a);
}

SpatialGenericDAG buildSpatialGenericDAG(Block &block) {
  SpatialGenericDAG dag;
  llvm::DenseMap<Operation *, unsigned> indices;
  unsigned interval = 0;
  for (Operation &op : block) {
    auto generic = dyn_cast<GenericOp>(op);
    // Use the concrete DPS tensor check: the bufferization TensorLikeType
    // external model need not be loaded when this pass runs standalone.
    if (!generic || !generic.hasPureTensorSemantics()) {
      if (isGroupingBarrier(op)) {
        dag.barriers.push_back(&op);
        ++interval;
      }
      continue;
    }
    indices[&op] = dag.nodes.size();
    SpatialGenericNode node;
    node.generic = generic;
    node.interval = interval;
    llvm::append_range(node.inputs, generic.getInputs());
    llvm::append_range(node.outputs, generic.getOutputs());
    llvm::append_range(node.results, generic->getResults());
    llvm::SetVector<Value> captures;
    getUsedValuesDefinedAbove(generic->getRegions(), captures);
    llvm::append_range(node.captures, captures);
    classifyGeneric(node);
    dag.nodes.push_back(std::move(node));
  }
  for (unsigned id = 0; id < dag.nodes.size(); ++id) {
    GenericOp generic = dag.nodes[id].generic;
    for (auto [index, operand] : llvm::enumerate(generic->getOperands())) {
      traceDependency(dag, block, indices, id, operand, index, false);
    }
    for (auto [index, capture] : llvm::enumerate(dag.nodes[id].captures)) {
      // The metadata retains all captures, but explicit operands already have
      // a binding. Only implicit captures need additional dependency records.
      if (!llvm::is_contained(generic->getOperands(), capture)) {
        traceDependency(dag, block, indices, id, capture, index, true);
      }
    }
  }
  for (auto &node : dag.nodes) {
    llvm::sort(node.predecessors);
    llvm::sort(node.successors);
  }
  collectExternalUses(dag, block, indices);
  computeReachability(dag);
  return dag;
}

SmallVector<RegionCandidate, 0>
formSpatialRegionCandidates(const SpatialGenericDAG &dag) {
  SmallVector<RegionCandidate, 0> candidates;
  collectChains(dag, candidates);
  collectParallelBranches(dag, candidates);
  collectForkJoins(dag, candidates);
  llvm::BitVector covered(dag.nodes.size());
  for (const auto &candidate : candidates) {
    for (unsigned member : candidate.members) {
      covered.set(member);
    }
  }
  for (unsigned id = 0; id < dag.nodes.size(); ++id) {
    if (!covered.test(id)) {
      addCandidate(candidates, {id}, RegionCandidateKind::Singleton,
                   "not covered by a local multi-generic candidate");
    }
  }
  llvm::sort(candidates, [](const auto &a, const auto &b) {
    return std::lexicographical_compare(a.members.begin(), a.members.end(),
                                        b.members.begin(), b.members.end());
  });
  for (auto &candidate : candidates) {
    populateCandidateDetails(dag, candidate);
  }
  return candidates;
}

SpatializationAnalysisResult analyzeSpatialization(Block &block) {
  SpatializationAnalysisResult result;
  result.dag = buildSpatialGenericDAG(block);
  result.candidates = formSpatialRegionCandidates(result.dag);
  return result;
}

void printSpatializationAnalysis(llvm::raw_ostream &os, Block &block,
                                 const SpatializationAnalysisResult &result) {
  auto func = cast<func::FuncOp>(block.getParentOp());
  AsmState state(func);
  unsigned blockID = std::distance(func.getBody().begin(), block.getIterator());
  const auto &dag = result.dag;
  os << "spatial_region_candidates @" << func.getSymName() << " block"
     << blockID << " candidate_only {\n";
  os << "  barriers = [";
  llvm::interleaveComma(dag.barriers, os, [&](Operation *op) {
    os << op->getName() << " at " << op->getLoc();
  });
  os << "]\n";
  for (auto [id, node] : llvm::enumerate(dag.nodes)) {
    os << "  G" << id << " type=" << node.opType
       << " interval=" << node.interval << " producers=";
    printIDs(os, node.predecessors, "G");
    os << " consumers=";
    printIDs(os, node.successors, "G");
    os << "\n    location=" << node.generic->getLoc() << " compute_ops=[";
    llvm::interleaveComma(node.computeOps, os);
    os << "] resource_hints=[";
    llvm::interleaveComma(node.resourceHints, os);
    os << "]\n    inputs=";
    printValues(os, node.inputs, state);
    os << "\n    output_inits=";
    printValues(os, node.outputs, state);
    os << "\n    results=";
    printValues(os, node.results, state);
    os << "\n    captures=";
    printValues(os, node.captures, state);
    os << '\n';
  }
  for (auto [id, edge] : llvm::enumerate(dag.dependencies)) {
    os << "  E" << id << ' ';
    if (edge.producer) {
      os << 'G' << *edge.producer;
    } else {
      os << "external";
    }
    os << " -> G" << edge.consumer << (edge.captured ? ".capture" : ".operand")
       << edge.targetIndex << " source=";
    edge.source.printAsOperand(os, state);
    os << " target=";
    edge.target.printAsOperand(os, state);
    os << " connection=" << (edge.via.empty() ? "direct" : "projected")
       << " via=[";
    llvm::interleaveComma(edge.via, os,
                          [&](Operation *op) { os << op->getName(); });
    os << "]\n";
  }
  for (auto [id, use] : llvm::enumerate(dag.externalUses)) {
    os << "  U" << id << " G" << use.producer << " value=";
    use.value.printAsOperand(os, state);
    os << " user=" << use.user->getName() << ".operand" << use.operandIndex
       << " at " << use.user->getLoc() << '\n';
  }
  for (auto [id, candidate] : llvm::enumerate(result.candidates)) {
    os << "  candidate R" << id << " {\n    members=";
    printIDs(os, candidate.members, "G");
    os << " kinds=[";
    llvm::interleaveComma(candidate.kinds, os, [&](RegionCandidateKind kind) {
      os << kindName(kind);
    });
    os << "]\n    dependencies=";
    printIDs(os, candidate.internalDependencies, "E");
    os << "\n    independent=[";
    llvm::interleaveComma(candidate.independentPairs, os, [&](auto pair) {
      os << "(G" << pair.first << ", G" << pair.second << ')';
    });
    os << "]\n    incoming=";
    printIDs(os, candidate.incomingDependencies, "E");
    os << " outgoing=";
    printIDs(os, candidate.outgoingDependencies, "E");
    os << " external_uses=";
    printIDs(os, candidate.externalUses, "U");
    os << '\n';
    for (const auto &context : candidate.contexts) {
      os << "    context={";
      if (context.producer) {
        os << " producer=G" << *context.producer;
      }
      if (context.consumer) {
        os << " consumer=G" << *context.consumer;
      }
      if (context.externalInput) {
        os << " external_input=";
        context.externalInput.printAsOperand(os, state);
      }
      os << " }\n";
    }
    for (const auto &reason : candidate.reasons) {
      os << "    reason=" << reason << '\n';
    }
    os << "  }\n";
  }
  os << "}\n";
}

} // namespace mlir::tt::d2m
