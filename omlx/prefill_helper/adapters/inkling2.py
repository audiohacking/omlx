# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 (inkling2, inkling2_loop): split at any layer, or any layer visit.

The layers exchange a partial block sum and the completed attention-residual
blocks. The cut sends the partial sum ``h`` entering layer ``k`` and, as
``blk.<i>``, the partial sum entering each block boundary layer ``i`` below
``k``; each side normalizes those itself, as the boundary layer does. A
layer's state is the K/V rows the chunk appended and its four conv windows;
a sliding layer keeps only the rows its window can still reach.

A looped checkpoint (inkling2_loop) runs its core layers several times per
token, and every visit owns a cache entry, so the split counts visits. The
helper takes visits up to the end of the first core pass, which reads no
messages: up to there visit ``i`` is layer ``i``, and the cut carries what
a plain one does. This Mac continues the loop from the cut: the later passes
restart from the partial sum entering the core, which the cut already sends
as ``blk.<core_start>``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.utils import load_model

from ...patches.inkling2 import cache as windows
from ...patches.inkling2 import model as inkling2
from ...utils.model_loading import maybe_apply_pre_load_patches
from ..adapter import Arrays, Capture, HostArrays, StageAdapter

# Source names (model.llm.*) and the runtime names oQ writes (model.*).
_ROOTS = ("model.llm.", "model.")
_SHARED = ("embed.", "embed_norm.")
_LAYER = re.compile(r"^layers\.(\d+)\.")
# Source attention-residual queries: attn_queries[i] belongs to layer i + 1.
_QUERY = re.compile(r"^attn_res\.(attn|mlp)_queries\.(\d+)$")
_CONV_SLOTS = 4


def _layers(lm: Any) -> list:
    return lm.model.layers


def _loop_end(model_path: str | Path | None) -> int | None:
    """The visit that ends a looped checkpoint's first core pass, or None."""
    if model_path is None:
        return None
    try:
        config = json.loads((Path(model_path).expanduser() / "config.json").read_text())
    except (OSError, ValueError):
        return None
    loop = (config.get("text_config") or config).get("loop_config")
    return int(loop["core_end"]) if loop else None


def _keep_visits(model: Any, k: int) -> None:
    """A looped model that keeps only its first ``k`` layers runs them once."""
    if getattr(model, "_core_start", None) is not None:
        model._schedule = tuple(model._schedule[:k])


def _local_name(tensor: str) -> str:
    for root in _ROOTS:
        if tensor.startswith(root):
            return tensor[len(root) :]
    return tensor


def _owner(tensor: str) -> int | None | bool:
    """Layer of a helper tensor, None for the embedding, False otherwise."""
    name = _local_name(tensor)
    if name.startswith(_SHARED):
        return None
    if match := _LAYER.match(name):
        return int(match.group(1))
    if match := _QUERY.match(name):
        return int(match.group(2)) + (match.group(1) == "attn")
    return False


def _unit_rms(x: mx.array) -> mx.array:
    return mx.fast.rms_norm(x.astype(mx.float32), None, 1e-6)


def _run_layers(
    h: mx.array,
    blocks: list,
    layers: list,
    cache: list,
    first: int = 0,
    capture: Capture | None = None,
) -> tuple[mx.array, dict[int, mx.array]]:
    """``layers`` (global ids from ``first``) on the partial sum ``h``.

    ``blocks`` gains a block at each boundary layer. Returns the partial sum
    and what entered each boundary layer.
    """
    entered = {}
    for i, (layer, c) in enumerate(zip(layers, cache), first):
        # The drafter reads the partial sum entering a layer.
        if capture is not None:
            capture.take(i, h)
        if layer.is_boundary:
            entered[i] = h
        h = layer(h, blocks, cache=c)
    return h, entered


def _run_visits(
    model: Any,
    h: mx.array,
    blocks: list,
    boundary: Arrays,
    below: list[int],
    visits: list,
    k: int,
    capture: Capture | None = None,
) -> tuple[mx.array, list]:
    """A looped model's visits from ``k`` on, as ``Inkling2TextModel`` runs
    them. A cut inside the first core pass rebuilds the prelude the later
    passes restart from: the partial sum entering the core and the blocks
    completed before it."""
    core = model._core_start
    prelude = None
    if k > core:
        prelude = (
            boundary[f"blk.{core}"],
            [_unit_rms(boundary[f"blk.{i}"]) for i in below if i < core],
        )
    bank = tau = None
    for visit in range(k, len(model._schedule)):
        idx, loop_pass = model._schedule[visit]
        layer, c = model.layers[idx], visits[visit]
        if idx == core and prelude is None:
            prelude = (h, list(blocks))
        elif idx == core:
            bank = (blocks[len(prelude[1]) + 1 :], inkling2._unit_rms(h.astype(mx.float32)))
            h, blocks = prelude[0], list(prelude[1])
        if capture is not None:
            capture.take(visit, h)
        if not layer.sliding and tau is None:
            tau = layer.attn.log_tau(c[0], h.shape[1])
        h = layer(
            h,
            blocks,
            cache=c,
            bank=bank if loop_pass else None,
            tau=None if layer.sliding else tau,
        )
    return h, blocks


def _cache_state(i: int, visit: tuple, rows: int | None) -> Arrays:
    """Layer ``i``'s conv windows and its ``rows`` newest K/V rows (all it
    holds when None)."""
    kv, conv = visit
    if kv is None:
        base = windows.CONV
        if rows is None:
            keys, values = conv[windows.KEYS], conv[windows.VALUES]
        else:
            keys, values = windows.newest(conv, rows)
    else:
        base = 0
        start = 0 if rows is None else kv.offset - rows
        keys = kv.keys[..., start : kv.offset, :]
        values = kv.values[..., start : kv.offset, :]
    out = {f"{i}.k": keys, f"{i}.v": values}
    for slot in range(_CONV_SLOTS):
        if conv[base + slot] is not None:
            out[f"{i}.c{slot}"] = conv[base + slot]
    return out


class Inkling2Adapter(StageAdapter):
    model_types = ("inkling2", "inkling2_loop")
    contract = 1
    captures_layers = True

    def serves(self, engine_type: str | None, model_path: str | Path | None) -> bool:
        # Text-only conversions run on the LLM engine, which has no helper.
        return engine_type == "vlm"

    def layer_count(self, lm: Any) -> int:
        # A looped model's split and its cost count layer visits.
        return lm.model.num_visits

    def split_limit(self, n_layers: int, model_path: str | Path | None = None) -> int:
        end = _loop_end(model_path)
        return n_layers - 1 if end is None else min(n_layers - 1, end)

    def resident(self, tensor: str) -> bool:
        # The embedding and the layers; the head, the final mix and norm,
        # vision and the draft model stay on this Mac.
        return _owner(tensor) is not False

    def keep_first(self, lm: Any, k: int) -> None:
        lm.model.layers = _layers(lm)[:k]
        _keep_visits(lm.model, k)
        for name in ("lm_head", "vision_tower"):
            if name in lm:
                del lm[name]
        for name in ("final_input", "norm"):
            if name in lm.model:
                del lm.model[name]

    @contextmanager
    def use_first(self, lm: Any, k: int):
        layers, schedule = _layers(lm), getattr(lm.model, "_schedule", None)
        lm.model.layers = layers[:k]
        _keep_visits(lm.model, k)
        try:
            yield
        finally:
            lm.model.layers = layers
            if schedule is not None:
                lm.model._schedule = schedule

    def load_first(self, path: str, k: int, a8_min_tokens: int | None = None) -> Any:
        maybe_apply_pre_load_patches(path, None, for_vlm=True)
        adapter, loaded = self, set()

        class First(inkling2.Model):
            def __init__(self, args: inkling2.ModelArgs):
                super().__init__(args)
                adapter.keep_first(self, k)

            def sanitize(self, weights: dict) -> dict:
                # A helper holds whole files, which also carry later layers.
                held = {}
                for name, value in weights.items():
                    layer = _owner(name)
                    if layer is None or (layer is not False and layer < k):
                        held[name] = value
                out = super().sanitize(held)
                loaded.update(out)
                return out

        lm, _ = load_model(
            Path(path),
            lazy=True,
            strict=False,
            get_model_classes=lambda config: (First, inkling2.ModelArgs),
        )
        missing = [n for n, _ in tree_flatten(lm.parameters()) if n not in loaded]
        if missing:
            raise ValueError(
                f"{len(missing)} parameters of the helper's layers are not in "
                f"the checkpoint, such as {missing[0]}"
            )
        mx.eval(lm.parameters())
        return lm

    def run_first(
        self,
        lm: Any,
        cache: list,
        tokens: mx.array,
        next_tokens: int,
        inputs: HostArrays | None = None,
        capture: Capture | None = None,
    ) -> tuple[Arrays, Arrays]:
        visits = lm.model.visit_caches(cache)
        h = lm.model.embed_norm(lm.model.embed(tokens))
        h, entered = _run_layers(h, [], _layers(lm), visits, capture=capture)
        boundary = {"h": h, **{f"blk.{i}": v for i, v in entered.items()}}
        state: Arrays = {}
        for i, visit in enumerate(visits):
            state.update(_cache_state(i, visit, tokens.shape[1]))
        mx.eval(
            list(boundary.values()),
            list(state.values()),
            list(capture.out.values()) if capture else [],
        )
        return boundary, state

    def run_rest(
        self,
        prefill: Callable[..., Any],
        lm: Any,
        cache: list,
        tokens: mx.array,
        k: int,
        boundary: Arrays,
        state: Arrays,
        capture: Capture | None = None,
        **kwargs: Any,
    ) -> Any:
        model = lm.model
        below = sorted(int(n[4:]) for n in boundary if n.startswith("blk."))
        blocks = [_unit_rms(boundary[f"blk.{i}"]) for i in below]
        visits = model.visit_caches(cache)
        h = boundary["h"]
        core = getattr(model, "_core_start", None)
        if core is None:
            h, _ = _run_layers(h, blocks, _layers(lm)[k:], visits[k:], k, capture)
        else:
            h, blocks = _run_visits(model, h, blocks, boundary, below, visits, k, capture)
        end = model.num_visits
        if capture is not None and end in capture.layers:
            # The last draft tap is the final-norm output, of the kept rows
            # only: the mix and the norm work row by row.
            rows = slice(capture.start, None)
            sources = [b[:, rows] for b in blocks]
            x = model.final_input(sources, _unit_rms(h[:, rows]))[:, :, 0]
            capture.out[end] = model.norm(x).astype(h.dtype)
        return None

    def apply_state(self, cache: list, k: int, state: Arrays) -> None:
        for i, (kv, conv) in enumerate(windows.visits(cache)[:k]):
            keys = state.get(f"{i}.k")
            if keys is not None and keys.shape[2]:
                if kv is None:
                    windows.append(conv, keys, state[f"{i}.v"])
                else:
                    kv.update_and_fetch(keys, state[f"{i}.v"])
            base = windows.CONV if kv is None else 0
            for slot in range(_CONV_SLOTS):
                value = state.get(f"{i}.c{slot}")
                if value is not None:
                    conv[base + slot] = value

    def prefix_state(self, cache: list, k: int) -> Arrays:
        state: Arrays = {}
        for i, visit in enumerate(windows.visits(cache)[:k]):
            held = _cache_state(i, visit, None)
            state.update({n: v for n, v in held.items() if v is not None})
        mx.eval(list(state.values()))
        return state

    def load_prefix(self, cache: list, state: Arrays) -> None:
        self.apply_state(cache, len(windows.visits(cache)), state)
