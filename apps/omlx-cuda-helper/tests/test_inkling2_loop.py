# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 Mini stages (inkling2_loop) must reproduce the oMLX contract fixture.

A helper takes layer visits up to the end of the first core pass, where visit
``i`` is layer ``i``; the host runs the later passes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
family = pytest.importorskip("omlx_cuda_helper.families.inkling2_loop")

from omlx_cuda_helper.families.inkling2_loop.stage import load_stage  # noqa: E402

# Recorded by tests/fixtures/record_prefill_contract.py in the oMLX tree.
FIXTURE = Path(__file__).parents[3] / "tests/fixtures/prefill_contract/inkling2_loop"
pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason="the oMLX contract fixture is not next to this tree"
)


def _case():
    meta = json.loads((FIXTURE / "meta.json").read_text())
    data = np.load(FIXTURE / "case.npz")
    arrays = {name: (data[name], dtype) for name, dtype, _, _ in meta["manifest"]}
    return meta, arrays


def _group(arrays, tag):
    return {n[len(tag) :]: v for n, v in arrays.items() if n.startswith(tag)}


def _replay(bounds):
    meta, arrays = _case()
    k, sizes, skip = meta["layers"], meta["sizes"], meta["prefix_chunks"]
    assert meta["contract"] == family.CONTRACT
    stages = [load_stage(FIXTURE / "model", a, b) for a, b in zip(bounds, bounds[1:])]
    jobs = [s.new_job(min(s.last, k) - s.first) for s in stages]
    for stage, job in zip(stages, jobs):
        stage.enable_kernels()
        stage.load_prefix(job, _group(arrays, "p:"))
    tokens = np.asarray(meta["ids"], dtype=np.int32)[sum(sizes[:skip]) :]
    start = 0
    for index, size in enumerate(sizes[skip:]):
        boundary, state = None, {}
        for stage, job in zip(stages, jobs):
            if not job:
                continue
            boundary, own = stage.run(
                job,
                start=start,
                tokens=tokens[start : start + size],
                boundary=boundary,
                inputs={},
                following=0,
                capture=None,
            )
            state.update(own)
        start += size
        yield index, boundary, state


def _assert_matches(got, want, meta, tag):
    assert sorted(got) == sorted(want), tag
    for name, (data, dtype) in got.items():
        expected, expected_dtype = want[name]
        assert (dtype, data.dtype, data.shape) == (
            expected_dtype,
            expected.dtype,
            expected.shape,
        ), tag + name
        np.testing.assert_allclose(
            data, expected, rtol=meta["rtol"], atol=meta["atol"], err_msg=tag + name
        )


@pytest.mark.parametrize(
    "bounds",
    [(0, 9), (0, 6, 9), (0, 7, 9), (0, 3, 6, 9), (0, 12)],
    ids=["one-stage", "cut-at-core", "cut-in-core", "cuts-3-6", "holds-the-pass"],
)
def test_stages_reproduce_the_contract_fixture(bounds):
    meta, arrays = _case()
    for index, boundary, state in _replay(bounds):
        _assert_matches(boundary, _group(arrays, f"b{index}:"), meta, f"b{index}:")
        _assert_matches(state, _group(arrays, f"s{index}:"), meta, f"s{index}:")


def test_a_stage_ends_by_the_end_of_the_first_core_pass():
    with pytest.raises(ValueError, match="first core pass"):
        load_stage(FIXTURE / "model", 0, 13)


def test_the_plan_leaves_out_the_loop_controllers():
    assert not family.resident("model.layers.7.loop_down")
    assert not family.resident("model.layers.7.loop_message.query")
    assert family.resident("model.layers.7.mlp_norm.weight")
