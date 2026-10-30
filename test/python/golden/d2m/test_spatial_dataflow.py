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

H, F, BANKS = 512, 1536, 12

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


def placement() -> List[int]:
    """spatial-groups values: per bank j, generics 4j..4j+3 = gate, up,
    silu(g) * u, down; generic 48 = the hub. The gate takes the bank's core;
    up, down and silu(g) * u take the free core nearest to it."""
    taken = set()

    def nearest(core):
        free = [
            (abs(y - core[0]) + abs(x - core[1]), (y, x))
            for y in range(8)
            for x in range(8)
            if (y, x) not in taken
        ]
        return min(free)[1]

    where = {}
    for role in ("gate", "up", "down", "glu"):
        for j, core in enumerate(BANK_CORES):
            where[(j, role)] = core if role == "gate" else nearest(core)
            taken.add(where[(j, role)])
    hub = nearest((5, 2))
    values = []
    for j in range(BANKS):
        for k, role in enumerate(("gate", "up", "glu", "down")):
            y, x = where[(j, role)]
            values += [0, 4 * j + k, y, x, y, x]
    return values + [0, 4 * BANKS, hub[0], hub[1], hub[0], hub[1]]


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


def create_program(shapes: List[Tuple[int, ...]]):
    def module(builder: TTIRBuilder):
        @builder.func(shapes, [torch.bfloat16] * len(shapes))
        def program(*args):
            operands, builder = args[: len(shapes)], args[len(shapes)]
            builder.set_goldens(
                inputs={
                    operand: (torch.rand(shape) * 2 - 1).to(torch.bfloat16)
                    for operand, shape in zip(operands, shapes)
                }
            )
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


@pytest.mark.parametrize("spatial", [False, True], ids=["default", "spatial"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_spatial_dataflow_gated_mlp(spatial: bool, target: str, request, device):
    shapes = [(32, H)] + [(H, F // BANKS)] * (2 * BANKS) + [(F // BANKS, H)] * BANKS
    options = ["dram-args-and-results=true"]
    if spatial:
        options += [
            "execution-strategy=spatial",
            "spatial-fuse-elementwise=true",
            "spatial-policy=none",
            "spatial-groups=" + ",".join(map(str, placement())),
        ]
    kwargs = get_request_kwargs(request)
    kwargs["save_artifacts"] = True
    compiled = compile_and_execute_ttir(
        create_program(shapes),
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
    # Spatial: one program holds all 49 one-core regions.
    assert [r for r in regions if r > 1] == ([4 * BANKS + 1] if spatial else [])
