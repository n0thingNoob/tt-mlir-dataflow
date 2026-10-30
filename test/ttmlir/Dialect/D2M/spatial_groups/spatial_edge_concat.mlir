// RUN: ttmlir-opt --d2m-fe-pipeline="execution-strategy=spatial spatial-policy=none spatial-groups=0,0,0,0,0,0,0,1,0,1,0,1,0,2,2,0,2,1,0,3,4,0,4,1" --mlir-print-ir-after=d2m-materialize-spatial-groups --mlir-print-local-scope --mlir-disable-threading %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=HAND
// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="ttnn-mode=false execution-strategy=spatial spatial-remarks=true" %s 2>&1 | FileCheck %s --check-prefix=AUTO

// A concatenation (ttir.concat -> a d2m.composite_view read by a copy generic)
// of two outputs of the same spatial group.

// By hand: exp (0,0), exp (0,1), the concatenating copy on (2,0)-(2,1), tanh on
// (4,0)-(4,1). Each concatenated input is its own edge into the copy's region,
// and the composite_view is rebuilt inside that region over the two block
// arguments (it used to be "reads another member ... through ops that cannot
// be an edge").
// HAND-LABEL: IR Dump After D2MMaterializeSpatialGroups
// HAND: d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 2:0>, #d2m.spatial_edge<1:0 -> 2:1>, #d2m.spatial_edge<2:0 -> 3:0>]
// HAND: ^bb0(%[[A:.*]]: tensor<1x1x1x4x!ttcore.tile<32x32, bf16>, {{.*}}>, %[[B:.*]]: tensor<1x1x1x4x!ttcore.tile<32x32, bf16>, {{.*}}>):
// HAND-NEXT: %[[C:.*]] = "d2m.composite_view"(%[[A]], %[[B]]) <{dim = 1 : si32}>
// HAND-NEXT: d2m.generic
// HAND-NEXT: ins(%[[C]] :

// Automatically the planner cuts at the concatenation, so the second group's
// copy reads a composite_view defined outside its spatial, which the spatial
// lists as an input. Expanding the view's reads used to fail ("Composite view
// has remaining uses after expansion"); the spatial now reads the view's
// inputs instead.
// AUTO: remark: generic #2 starts a new group: input 0 reads the group through ops that cannot be an edge
// AUTO: remark: spatial group 1: generics #2 #3, 1 edges
// AUTO-NOT: error
// AUTO: "ttmetal.enqueue_program"
func.func @concat(%x0: tensor<32x128xbf16>, %x1: tensor<32x128xbf16>) -> tensor<32x256xbf16> {
  %a = "ttir.exp"(%x0) : (tensor<32x128xbf16>) -> tensor<32x128xbf16>
  %b = "ttir.exp"(%x1) : (tensor<32x128xbf16>) -> tensor<32x128xbf16>
  %c = "ttir.concat"(%a, %b) <{dim = 1 : si32}> : (tensor<32x128xbf16>, tensor<32x128xbf16>) -> tensor<32x256xbf16>
  %y = "ttir.tanh"(%c) : (tensor<32x256xbf16>) -> tensor<32x256xbf16>
  return %y : tensor<32x256xbf16>
}
