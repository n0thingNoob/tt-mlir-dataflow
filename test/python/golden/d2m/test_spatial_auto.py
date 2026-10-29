# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# End-to-end tests of automatic spatial grouping: small TTIR programs with no
# hand-written spatial or edge, compiled with and without
# `execution-strategy=spatial`. In spatial mode the test also checks that the
# expected number of multi-region programs reached the final TTMetal IR, so a
# silent fallback to one program per generic fails the test.

import re
import pytest
import torch
from typing import Callable, List, Tuple

from conftest import get_request_kwargs

from builder.base.builder_utils import Operand
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir

pytestmark = pytest.mark.frontend("ttir")


def mlp(b, x, w1, b1, w2):
    return b.matmul(b.relu(b.add(b.matmul(x, w1), b1)), w2)


def attn(b, q, kt, v):
    return b.matmul(b.softmax(b.matmul(q, kt), dimension=1), v)


def fanout(b, x, y):
    t = b.tanh(x)
    return b.add(b.exp(t), b.matmul(t, y))


def rms_mm(b, x, g, w):
    return b.matmul(b.rms_norm(x, normalized_shape=[128], weight=g, epsilon=1e-5), w)


def decode_mm(b, x, w1, w2, w3):
    return b.matmul(b.matmul(b.matmul(x, w1), w2), w3)


def decode_glu(b, x, wg, wu, wd):
    g = b.matmul(x, wg)
    u = b.matmul(x, wu)
    return b.matmul(b.multiply(b.multiply(g, b.sigmoid(g)), u), wd)


def tm(b, x, w, y):
    t = b.transpose(b.matmul(x, w), dim0=0, dim1=1)
    return b.relu(b.reshape(b.matmul(t, y), [64, 128]))


# name -> (input shapes, body, spatials expected in spatial mode). The big
# programs have 8x8 or 4x8 generics in the default flow; they only group at a
# smaller target, where their eltwise regions are also reblocked.
PROGRAMS = {
    "mlp": ([(64, 128), (128, 256), (64, 256), (256, 128)], mlp, 1),
    "fanout": ([(64, 64), (64, 64)], fanout, 1),
    "attn": ([(64, 64), (64, 64), (64, 64)], attn, 1),
    "rms_mm": ([(64, 128), (128,), (128, 128)], rms_mm, 2),
    "decode_mm": ([(32, 256), (256, 512), (512, 256), (256, 256)], decode_mm, 1),
    "decode_glu": ([(32, 256), (256, 512), (256, 512), (512, 256)], decode_glu, 1),
    "attn_odd": ([(50, 64), (64, 50), (50, 64)], attn, 2),
    "tm": ([(64, 128), (128, 128), (64, 64)], tm, 1),
    "big_fanout": ([(128, 256), (256, 256)], fanout, 1),
    "big_attn": ([(256, 256), (256, 256), (256, 256)], attn, 1),
    "big_mlp": ([(256, 256), (256, 512), (256, 512), (512, 256)], mlp, 1),
}


def create_program(shapes: List[Tuple[int, ...]], body: Callable):
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
            return body(builder, *operands)

    return module


def count_multi_region_programs(ttmetal_mlir: str) -> int:
    count = 0
    for line in ttmetal_mlir.splitlines():
        if '"ttmetal.enqueue_program"' in line:
            count += len(set(re.findall(r"core_range<([^>]*)>", line))) > 1
    return count


@pytest.mark.parametrize("program", PROGRAMS.keys())
@pytest.mark.parametrize("spatial", [False, True], ids=["default", "spatial"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_spatial_auto(program: str, spatial: bool, target: str, request, device):
    shapes, body, num_spatials = PROGRAMS[program]
    kwargs = get_request_kwargs(request)
    kwargs["save_artifacts"] = True
    compiled = compile_and_execute_ttir(
        create_program(shapes, body),
        target=target,
        device=device,
        pcc=0.999,
        pipeline_options=["execution-strategy=spatial"] if spatial else [],
        **kwargs,
    )
    with open(compiled) as f:
        multi_region = count_multi_region_programs(f.read())
    assert multi_region == (num_spatials if spatial else 0)
