# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# End-to-end test of a whole decode gated-MLP dataflow in one d2m.spatial: the
# weights are split by DRAM bank (slice j of Wg, Wu and Wd in bank j through
# d2m.dram_bank), each bank's gate matmul runs on the worker core next to that
# bank, up / silu(g) * u / down (split-K) run next to it, and a 12-input hub
# sums the partial results and writes y to DRAM. Every intermediate goes core
# to core through L1 (spatial edges). The groups are given by hand
# (spatial-groups); elementwise chains are fused before planning.
#
# Two designs: V1 as above, and V2, where a load region on the bank's core
# copies Wg[:, j] and Wu[:, j] into L1 (spatial-loads) and gate / up read them
# over edges from the next cores (down still reads Wd[j, :] itself).
#
# Two shapes: a scaled one (H = 512, F = 1536) and the main one (H = 2048,
# F = 8192 padded to 12 x 22 tiles = 8448 with zero weights; the matmuls are
# also blocked along N so each region's stream buffers fit its L1). V2 runs the
# scaled shape only: a main-shape load region would hold 2 x 1.4 MiB of
# weights at once (whole-tensor edges).

import re
import pytest
import torch
from typing import List, Tuple

from conftest import get_request_kwargs

from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from ttmlir.ir import ArrayAttr, DictAttr, IntegerAttr, IntegerType
from ttmlir.passes import ttir_to_ttmetal_backend_pipeline

pytestmark = pytest.mark.frontend("ttir")

BANKS = 12

# name -> (H, F, padded F, extra pipeline options for spatial mode)
SHAPES = {
    "scaled": (512, 1536, 1536, []),
    "main": (2048, 8192, 8448, ["test-buffer-size-policy=auto-mn"]),
}

# Worker cores (y, x) next to DRAM banks 0..11 (dram_bank_to_logical_worker_noc0
# of an n150). Another card only makes the placement less local.
BANK_CORES = [
    (7, 3),
    (0, 0),
    (4, 0),
    (1, 2),
    (0, 4),
    (7, 7),
    (1, 4),
    (6, 4),
    (5, 4),
    (2, 4),
    (4, 4),
    (3, 6),
]


def placement(loads: bool = False) -> Tuple[List[int], List[int]]:
    """spatial-groups and spatial-loads values: per bank j, generics
    4j..4j+3 = gate, up, silu(g) * u, down; generic 48 = the hub. V1: the gate
    takes the bank's core; up, down and silu(g) * u take the free core nearest
    to it. V2 (loads): the load region takes the bank's core and copies input 1
    (the weight) of gate and up; the four compute generics take the nearest
    free cores."""
    taken = set()

    def nearest(core):
        free = [
            (abs(y - core[0]) + abs(x - core[1]), (y, x))
            for y in range(8)
            for x in range(8)
            if (y, x) not in taken
        ]
        return min(free)[1]

    roles = ("gate", "up", "down", "glu")
    if loads:
        roles = ("load",) + roles
    where = {}
    for i, role in enumerate(roles):
        for j, core in enumerate(BANK_CORES):
            where[(j, role)] = core if i == 0 else nearest(core)
            taken.add(where[(j, role)])
    hub = nearest((5, 2))
    groups, load_values = [], []
    for j in range(BANKS):
        for k, role in enumerate(("gate", "up", "glu", "down")):
            y, x = where[(j, role)]
            groups += [0, 4 * j + k, y, x, y, x]
        if loads:
            y, x = where[(j, "load")]
            load_values += [4 * j, 1, y, x, y, x, 4 * j + 1, 1, y, x, y, x]
    groups += [0, 4 * BANKS, hub[0], hub[1], hub[0], hub[1]]
    return groups, load_values


def gated_mlp(b, x, *weights):
    wg, wu, wd = (
        weights[:BANKS],
        weights[BANKS : 2 * BANKS],
        weights[2 * BANKS :],
    )
    parts = []
    for j in range(BANKS):
        g = b.matmul(x, wg[j])
        u = b.matmul(x, wu[j])
        parts.append(b.matmul(b.multiply(b.silu(g), u), wd[j]))
    y = parts[0]
    for part in parts[1:]:
        y = b.add(y, part)
    return y


def make_inputs(h: int, f: int, f_padded: int) -> List[torch.Tensor]:
    """x and the per-bank slices of Wg, Wu (columns) and Wd (rows). The F
    positions past `f` are zero, so they add nothing: silu(0) * 0 = 0."""
    rand = lambda *shape: (torch.rand(*shape) * 2 - 1).to(torch.bfloat16)
    x, wg, wu, wd = rand(32, h), rand(h, f), rand(h, f), rand(f, h)
    pad = f_padded - f
    wg = torch.nn.functional.pad(wg, (0, pad))
    wu = torch.nn.functional.pad(wu, (0, pad))
    wd = torch.nn.functional.pad(wd, (0, 0, 0, pad))
    slices = f_padded // BANKS
    return (
        [x]
        + list(wg.split(slices, dim=1))
        + list(wu.split(slices, dim=1))
        + list(wd.split(slices, dim=0))
    )


def reference(inputs: List[torch.Tensor], f: int) -> torch.Tensor:
    """The unpadded layer in float32."""
    x = inputs[0].float()
    wg = torch.cat(inputs[1 : 1 + BANKS], dim=1)[:, :f].float()
    wu = torch.cat(inputs[1 + BANKS : 1 + 2 * BANKS], dim=1)[:, :f].float()
    wd = torch.cat(inputs[1 + 2 * BANKS :], dim=0)[:f].float()
    return (torch.nn.functional.silu(x @ wg) * (x @ wu)) @ wd


def create_program(inputs: List[torch.Tensor]):
    shapes = [tuple(t.shape) for t in inputs]

    def module(builder: TTIRBuilder):
        @builder.func(shapes, [torch.bfloat16] * len(shapes))
        def program(*args):
            operands, builder = args[: len(shapes)], args[len(shapes)]
            builder.set_goldens(inputs=dict(zip(operands, inputs)))
            return gated_mlp(builder, *operands)

    return module


def pipeline_with_dram_banks(module, options: str):
    """The TTMetal pipeline, after putting weight slice j (arguments 1 + j,
    13 + j, 25 + j) in DRAM bank j."""
    with module.context:
        i64 = IntegerType.get_signless(64)
        for op in module.body.operations:
            if op.operation.name != "func.func":
                continue
            attrs = [DictAttr.get({})]
            for _ in range(3):
                attrs += [
                    DictAttr.get({"d2m.dram_bank": IntegerAttr.get(i64, j)})
                    for j in range(BANKS)
                ]
            op.attributes["arg_attrs"] = ArrayAttr.get(attrs)
    ttir_to_ttmetal_backend_pipeline(module, options)


# (design, spatial): V2 only differs in spatial mode.
MODES = [("v1", False), ("v1", True), ("v2", True)]


@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize(
    "design,spatial", MODES, ids=["default", "spatial", "spatial_v2"]
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_spatial_dataflow_gated_mlp(
    shape: str, design: str, spatial: bool, target: str, request, device
):
    if design == "v2" and shape != "scaled":
        pytest.skip("V2 needs streaming edges at the main shape")
    h, f, f_padded, extra = SHAPES[shape]
    inputs = make_inputs(h, f, f_padded)
    # The zero padding leaves the layer unchanged (checked on the host; the
    # device result is checked against the padded program's golden below).
    padded = sum(
        (
            torch.nn.functional.silu(inputs[0].float() @ inputs[1 + j].float())
            * (inputs[0].float() @ inputs[1 + BANKS + j].float())
        )
        @ inputs[1 + 2 * BANKS + j].float()
        for j in range(BANKS)
    )
    ref = reference(inputs, f)
    torch.testing.assert_close(
        padded, ref, rtol=1e-4, atol=1e-4 * ref.abs().max().item()
    )

    options = ["dram-args-and-results=true"]
    if spatial:
        groups, loads = placement(loads=design == "v2")
        options += [
            "execution-strategy=spatial",
            "spatial-fuse-elementwise=true",
            "spatial-policy=none",
            "spatial-groups=" + ",".join(map(str, groups)),
        ] + extra
        if loads:
            options.append("spatial-loads=" + ",".join(map(str, loads)))
    kwargs = get_request_kwargs(request)
    kwargs["save_artifacts"] = True
    compiled = compile_and_execute_ttir(
        create_program(inputs),
        target=target,
        device=device,
        pcc=0.999,
        custom_pipeline=pipeline_with_dram_banks,
        pipeline_options=options,
        **kwargs,
    )
    with open(compiled) as f:
        final = f.read()
    regions = [
        len(set(re.findall(r"core_range<([^>]*)>", line)))
        for line in final.splitlines()
        if '"ttmetal.enqueue_program"' in line
    ]
    # Spatial: one program holds all 49 one-core regions (61 with the loads).
    expected = 4 * BANKS + 1 + (BANKS if design == "v2" else 0)
    assert [r for r in regions if r > 1] == ([expected] if spatial else [])
