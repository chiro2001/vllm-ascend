# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.ops import rope_dsv4
from vllm_ascend.ops.rope_dsv4 import (
    _ROPE_STATE,
    ComplexExpRotaryEmbedding,
    RopeDataProxy,
    _rope_gather_rows,
    _rope_index_1d,
    get_cos_and_sin_dsa,
    get_full_cos_and_sin_dsa_for_layer,
)


def test_full_rope_lookup_resolves_exact_layer_config(monkeypatch):
    first = (torch.randn(4, 1, 1, 8), torch.randn(4, 1, 1, 8))
    second = (torch.randn(4, 1, 1, 8), torch.randn(4, 1, 1, 8))
    monkeypatch.setattr(
        rope_dsv4._ROPE_STATE,
        "layer_info",
        {
            "model.layers.0.self_attn.attn": ("base", ["default"]),
            "model.layers.2.self_attn.attn": ("compressed", ["default"]),
        },
    )
    monkeypatch.setattr(
        rope_dsv4._ROPE_STATE,
        "full_rope_cache",
        {"base": first, "compressed": second},
    )

    actual = get_full_cos_and_sin_dsa_for_layer("model.layers.2.self_attn.attn")

    assert actual[0] is second[0]
    assert actual[1] is second[1]
    with pytest.raises(KeyError, match="not registered"):
        get_full_cos_and_sin_dsa_for_layer("missing")


def test_plain_rope_disables_yarn_explicitly():
    dim = 8
    base = 10000
    actual = ComplexExpRotaryEmbedding.precompute_freqs_cis(
        dim,
        seqlen=65536,
        original_seq_len=4096,
        apply_yarn_scaling=False,
        base=base,
        factor=16,
        beta_fast=32,
        beta_slow=1,
    )
    expected = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    torch.testing.assert_close(actual, expected)


# ──────────────────────────────────────────────
# Equivalence: pad_to + slice  vs  pad-positions + gather + slice
# ──────────────────────────────────────────────


def _gather_rope(
    positions: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor
) -> tuple[RopeDataProxy, RopeDataProxy]:
    """Index ``rope_cos`` / ``rope_sin`` by ``positions`` and wrap in ``RopeDataProxy``.

    This mirrors what ``get_cos_and_sin_dsa`` does internally — lookup the
    RoPE table at the given positions — without depending on the global
    ``_ROPE_STATE`` singleton.
    """
    cos_t = rope_cos[positions]  # [N, 1, 1, D]
    sin_t = rope_sin[positions]
    data_map = {"_test": {"default": (cos_t, sin_t)}}
    return RopeDataProxy(data_map, is_cos=True), RopeDataProxy(data_map, is_cos=False)


def _extract_tensor(proxy: RopeDataProxy) -> torch.Tensor:
    """Extract the raw tensor from a singled-group proxy for comparison."""
    for groups in proxy._data.values():
        for tensors in groups.values():
            return tensors[proxy.idx]
    raise AssertionError("empty proxy")


def _tp_params(num_input_tokens: int, tp_size: int):
    """Yield ``(tp_rank, local_start, local_end, num_tokens_pad)`` for each TP rank."""
    num_tokens_pad = ((num_input_tokens + tp_size - 1) // tp_size) * tp_size
    tokens_per_rank = num_tokens_pad // tp_size
    for tp_rank in range(tp_size):
        local_start = tp_rank * tokens_per_rank
        local_end = local_start + tokens_per_rank
        yield tp_rank, local_start, local_end, num_tokens_pad


class TestEquivalenceWithGatherSemantics:
    """Verify that ``proxy.pad_to(N)[s:e]`` is semantically equivalent to
    the original ``get_cos_and_sin_dsa`` approach of padding positions first.
    """

    # (num_input_tokens, tp_size)
    CASES = [
        (32, 8),
        (33, 8),
        (39, 8),
        (40, 8),
        (45, 8),
        (1, 2),
        (2, 4),
        (7, 8),
        (100, 16),
        (102, 16),
    ]

    def test_all_cases(self):
        for num_input_tokens, tp_size in self.CASES:
            self._run_equivalence(num_input_tokens, tp_size)

    def _run_equivalence(self, num_input_tokens: int, tp_size: int):
        """Run the equivalence check for one (N, tp_size) combination."""
        max_pos = num_input_tokens + tp_size + 5  # rope table large enough
        rotary_dim = 32
        rng = torch.Generator().manual_seed(42)
        rope_cos = torch.randn(max_pos, 1, 1, rotary_dim, generator=rng)
        rope_sin = torch.randn(max_pos, 1, 1, rotary_dim, generator=rng)
        input_positions = torch.randint(0, max_pos - 1, (num_input_tokens,), generator=rng)

        # Gather from UNPADDED positions — this is what the optimised path does.
        ref_cos_proxy, ref_sin_proxy = _gather_rope(input_positions, rope_cos, rope_sin)

        for tp_rank, local_start, local_end, num_tokens_pad in _tp_params(num_input_tokens, tp_size):
            # ── Original path: pad positions → gather → slice ──
            padded_pos = torch.nn.functional.pad(input_positions, (0, num_tokens_pad - num_input_tokens), value=0)
            orig_cos_p, orig_sin_p = _gather_rope(padded_pos, rope_cos, rope_sin)
            orig_cos = _extract_tensor(orig_cos_p[local_start:local_end])
            orig_sin = _extract_tensor(orig_sin_p[local_start:local_end])

            # ── Optimised path: gather → pad_to → slice ──
            opt_cos_p = ref_cos_proxy.pad_to(num_tokens_pad)
            opt_sin_p = ref_sin_proxy.pad_to(num_tokens_pad)
            opt_cos = _extract_tensor(opt_cos_p[local_start:local_end])
            opt_sin = _extract_tensor(opt_sin_p[local_start:local_end])

            # ── Compare ──
            # Real-token region: exact match expected.
            real_end = min(local_end, num_input_tokens) - local_start
            if real_end > 0:
                assert torch.equal(orig_cos[:real_end], opt_cos[:real_end]), (
                    f"cos mismatch in real region, N={num_input_tokens}, tp_size={tp_size}, rank={tp_rank}"
                )
                assert torch.equal(orig_sin[:real_end], opt_sin[:real_end]), (
                    f"sin mismatch in real region, N={num_input_tokens}, tp_size={tp_size}, rank={tp_rank}"
                )


# ──────────────────────────────────────────────
# Indexed lookup: index_select + out=  vs  advanced indexing / 4-D gather
# ──────────────────────────────────────────────

# Pre-allocated runtime buffers use these fill values, so a row that the lookup
# did not write shows up in the assertions below.
_COS_SENTINEL = 7.0
_SIN_SENTINEL = -7.0

_STATE_FIELDS = (
    "full_rope_cache",
    "runtime_buffer",
    "spec_runtime_buffer",
    "registry_summary",
    "layer_info",
)


def _make_rope_table(max_pos: int, rotary_dim: int, seed: int = 1234) -> tuple[torch.Tensor, torch.Tensor]:
    """Build RoPE tables shaped like the ones ``ComplexExpRotaryEmbedding`` registers."""
    rng = torch.Generator().manual_seed(seed)
    cos = torch.randn(max_pos, 1, 1, rotary_dim, generator=rng)
    sin = torch.randn(max_pos, 1, 1, rotary_dim, generator=rng)
    return cos, sin


def _make_gather_idx(positions: torch.Tensor, rotary_dim: int) -> torch.Tensor:
    """The 4-D index the pre-optimisation code handed to ``torch.gather``."""
    return positions.to(torch.long).reshape(-1, 1, 1, 1).expand(positions.size(0), 1, 1, rotary_dim)


def _register_group(
    state,
    *,
    max_pos: int = 64,
    rotary_dim: int = 32,
    num_slots: int = 16,
    num_speculative_tokens: int = 0,
    config_key: str = "cfg",
    group_name: str = "default",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror the shapes ``ComplexExpRotaryEmbedding`` stores in ``_ROPE_STATE``.

    ``full_rope_cache`` holds ``[max_pos, 1, 1, rotary_dim]`` tables and the
    runtime buffers hold ``[max_num_batched_tokens, 1, 1, rotary_dim]`` slots,
    so ``buf[:num_tokens]`` has exactly the shape of ``full_rope[pos_tensor]``.
    """
    full_cos, full_sin = _make_rope_table(max_pos, rotary_dim)
    state.full_rope_cache[config_key] = (full_cos, full_sin)
    state.registry_summary[config_key] = {group_name}
    state.layer_info["test.attn"] = (config_key, [group_name])
    state.runtime_buffer[config_key] = {
        group_name: (
            torch.full((num_slots, 1, 1, rotary_dim), _COS_SENTINEL),
            torch.full((num_slots, 1, 1, rotary_dim), _SIN_SENTINEL),
        )
    }
    if num_speculative_tokens:
        state.spec_runtime_buffer[config_key] = {
            group_name: (
                [torch.full((num_slots, 1, 1, rotary_dim), _COS_SENTINEL) for _ in range(num_speculative_tokens)],
                [torch.full((num_slots, 1, 1, rotary_dim), _SIN_SENTINEL) for _ in range(num_speculative_tokens)],
            )
        }
    return full_cos, full_sin


def _entry(proxy: RopeDataProxy, config_key: str, group_name: str) -> torch.Tensor:
    """Read one ``(cos | sin)`` tensor back out of a proxy."""
    return proxy._data[config_key][group_name][proxy.idx]


def _lookup(config_key: str, group_name: str, positions: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ``get_cos_and_sin_dsa`` and return the ``(cos, sin)`` pair it produced."""
    cos_proxy, sin_proxy = get_cos_and_sin_dsa(positions, **kwargs)
    return _entry(cos_proxy, config_key, group_name), _entry(sin_proxy, config_key, group_name)


@pytest.fixture
def rope_state():
    """Clear the module-level ``_ROPE_STATE`` singleton and restore it afterwards."""
    saved = {name: dict(getattr(_ROPE_STATE, name)) for name in _STATE_FIELDS}
    for name in _STATE_FIELDS:
        getattr(_ROPE_STATE, name).clear()
    try:
        yield _ROPE_STATE
    finally:
        for name, value in saved.items():
            target = getattr(_ROPE_STATE, name)
            target.clear()
            target.update(value)


class TestRopeIndex1D:
    """``_rope_index_1d`` must flatten without disturbing an already-int64 dtype."""

    def test_int64_positions_are_returned_as_is(self):
        positions = torch.tensor([0, 5, 7, 3], dtype=torch.int64)
        index = _rope_index_1d(positions)
        assert index.dtype == torch.int64
        assert index.shape == (4,)
        assert torch.equal(index, positions)
        # No cast and no copy: both sides point at the same storage.
        assert index.untyped_storage().data_ptr() == positions.untyped_storage().data_ptr()

    def test_non_int64_positions_are_cast_to_int64(self):
        for dtype in (torch.int32, torch.int16, torch.uint8):
            positions = torch.tensor([0, 5, 7, 3], dtype=dtype)
            index = _rope_index_1d(positions)
            assert index.dtype == torch.int64
            assert torch.equal(index, torch.tensor([0, 5, 7, 3], dtype=torch.int64))

    def test_non_contiguous_positions_are_flattened(self):
        positions = torch.arange(12, dtype=torch.int64).reshape(3, 4)
        transposed = positions.t()
        index = _rope_index_1d(transposed)
        assert index.dtype == torch.int64
        assert index.shape == (12,)
        assert torch.equal(index, transposed.reshape(-1))

    def test_int32_positions_select_the_same_rows_as_int64(self, rope_state):
        full_cos, full_sin = _register_group(rope_state)
        positions64 = torch.tensor([0, 5, 7, 3], dtype=torch.int64)

        cos64, sin64 = _lookup("cfg", "default", positions64, use_cache=True)
        cos32, sin32 = _lookup("cfg", "default", positions64.to(torch.int32), use_cache=True)

        assert torch.equal(cos32, cos64)
        assert torch.equal(sin32, sin64)
        assert torch.equal(cos32, full_cos[positions64])
        assert torch.equal(sin32, full_sin[positions64])


class TestIndexSelectLookupEquivalence:
    """``index_select`` must be bit-identical to the lookups it replaced."""

    # Unsorted on purpose: row order and repeated positions must be preserved.
    POSITIONS = [0, 5, 63, 7, 7, 3]
    ROTARY_DIM = 32

    def test_helper_matches_advanced_indexing_and_4d_gather(self):
        full_cos, full_sin = _make_rope_table(max_pos=64, rotary_dim=self.ROTARY_DIM)
        positions = torch.tensor(self.POSITIONS)
        shape = (len(self.POSITIONS), 1, 1, self.ROTARY_DIM)

        ref_cos, ref_sin = full_cos[positions], full_sin[positions]
        assert ref_cos.shape == shape

        gather_cos, gather_sin = full_cos.new_empty(shape), full_sin.new_empty(shape)
        gather_idx = _make_gather_idx(positions, self.ROTARY_DIM)
        torch.gather(full_cos, 0, gather_idx, out=gather_cos)
        torch.gather(full_sin, 0, gather_idx, out=gather_sin)

        select_cos, select_sin = full_cos.new_empty(shape), full_sin.new_empty(shape)
        index_1d = _rope_index_1d(positions)
        _rope_gather_rows(full_cos, index_1d, None, select_cos)
        _rope_gather_rows(full_sin, index_1d, None, select_sin)

        # torch.equal covers shape, dtype and every single bit of the payload.
        assert torch.equal(select_cos, ref_cos)
        assert torch.equal(select_sin, ref_sin)
        assert torch.equal(gather_cos, ref_cos)
        assert torch.equal(gather_sin, ref_sin)
        assert select_cos.shape == gather_cos.shape == ref_cos.shape
        assert select_cos.dtype == gather_cos.dtype == ref_cos.dtype

    def test_helper_keeps_the_4d_gather_path(self):
        """An explicit ``gather_idx`` still routes through ``torch.gather``."""
        full_cos, _ = _make_rope_table(max_pos=64, rotary_dim=self.ROTARY_DIM)
        positions = torch.tensor(self.POSITIONS)
        out = full_cos.new_empty((len(self.POSITIONS), 1, 1, self.ROTARY_DIM))

        _rope_gather_rows(full_cos, None, _make_gather_idx(positions, self.ROTARY_DIM), out)

        assert torch.equal(out, full_cos[positions])

    def test_get_cos_and_sin_dsa_matches_advanced_indexing(self, rope_state):
        full_cos, full_sin = _register_group(rope_state, rotary_dim=self.ROTARY_DIM)
        positions = torch.tensor(self.POSITIONS)

        cached_cos, cached_sin = _lookup("cfg", "default", positions, use_cache=True)
        direct_cos, direct_sin = _lookup("cfg", "default", positions, use_cache=False)

        assert cached_cos.shape == (len(self.POSITIONS), 1, 1, self.ROTARY_DIM)
        assert torch.equal(cached_cos, full_cos[positions])
        assert torch.equal(cached_sin, full_sin[positions])
        assert torch.equal(direct_cos, full_cos[positions])
        assert torch.equal(direct_sin, full_sin[positions])
        # The cached and the uncached lookup must agree with each other too.
        assert torch.equal(cached_cos, direct_cos)
        assert torch.equal(cached_sin, direct_sin)

    def test_cache_path_writes_only_the_token_rows(self, rope_state):
        num_slots = 16
        full_cos, full_sin = _register_group(rope_state, rotary_dim=self.ROTARY_DIM, num_slots=num_slots)
        positions = torch.tensor(self.POSITIONS)
        num_tokens = len(self.POSITIONS)
        buf_cos, buf_sin = rope_state.runtime_buffer["cfg"]["default"]

        cos_t, sin_t = _lookup("cfg", "default", positions, use_cache=True)

        # The lookup writes into the pre-allocated buffers, which keeps their
        # addresses stable across graph capture/replay.
        assert cos_t.untyped_storage().data_ptr() == buf_cos.untyped_storage().data_ptr()
        assert sin_t.untyped_storage().data_ptr() == buf_sin.untyped_storage().data_ptr()
        assert cos_t.storage_offset() == 0
        assert cos_t.shape == (num_tokens, 1, 1, self.ROTARY_DIM)
        assert torch.equal(cos_t, full_cos[positions])
        assert torch.equal(sin_t, full_sin[positions])
        # Rows past num_tokens keep the fill value they were registered with.
        assert torch.equal(buf_cos[num_tokens:], torch.full_like(buf_cos[num_tokens:], _COS_SENTINEL))
        assert torch.equal(buf_sin[num_tokens:], torch.full_like(buf_sin[num_tokens:], _SIN_SENTINEL))

        cos_again, _ = _lookup("cfg", "default", positions, use_cache=True)
        assert cos_again.data_ptr() == cos_t.data_ptr()


class TestDraftIndexPath:
    """The spec-decode path writes into ``spec_runtime_buffer[draft_index - 1]``."""

    POSITIONS = [1, 4, 9, 2]
    ROTARY_DIM = 32
    NUM_DRAFTS = 3

    def test_draft_index_selects_the_matching_spec_buffer(self, rope_state):
        num_slots = 16
        full_cos, full_sin = _register_group(
            rope_state,
            rotary_dim=self.ROTARY_DIM,
            num_slots=num_slots,
            num_speculative_tokens=self.NUM_DRAFTS,
        )
        positions = torch.tensor(self.POSITIONS)
        num_tokens = len(self.POSITIONS)
        spec_cos, spec_sin = rope_state.spec_runtime_buffer["cfg"]["default"]
        runtime_cos, runtime_sin = rope_state.runtime_buffer["cfg"]["default"]

        for draft_index in range(1, self.NUM_DRAFTS + 1):
            cos_t, sin_t = _lookup("cfg", "default", positions, use_cache=True, draft_index=draft_index)
            assert cos_t.shape == (num_tokens, 1, 1, self.ROTARY_DIM)
            assert torch.equal(cos_t, full_cos[positions])
            assert torch.equal(sin_t, full_sin[positions])
            # draft_index is 1-based and selects its own pre-allocated buffer.
            assert cos_t.data_ptr() == spec_cos[draft_index - 1].data_ptr()
            assert sin_t.data_ptr() == spec_sin[draft_index - 1].data_ptr()

        for cos_buf, sin_buf in zip(spec_cos, spec_sin):
            assert torch.equal(cos_buf[:num_tokens], full_cos[positions])
            assert torch.equal(sin_buf[:num_tokens], full_sin[positions])
            assert torch.equal(cos_buf[num_tokens:], torch.full_like(cos_buf[num_tokens:], _COS_SENTINEL))
            assert torch.equal(sin_buf[num_tokens:], torch.full_like(sin_buf[num_tokens:], _SIN_SENTINEL))

        # The non-speculative buffer is not touched by the draft path.
        assert torch.equal(runtime_cos, torch.full_like(runtime_cos, _COS_SENTINEL))
        assert torch.equal(runtime_sin, torch.full_like(runtime_sin, _SIN_SENTINEL))


class TestNon1DPositions:
    """Non-1-D positions must keep the pre-optimisation code paths."""

    POSITIONS = [0, 5, 7, 3]
    ROTARY_DIM = 32

    def test_cache_path_keeps_the_4d_gather_result(self, rope_state):
        full_cos, full_sin = _register_group(rope_state, rotary_dim=self.ROTARY_DIM)
        positions = torch.tensor(self.POSITIONS).reshape(-1, 1)
        shape = (len(self.POSITIONS), 1, 1, self.ROTARY_DIM)

        # Reference: the exact call the pre-optimisation code made for these ranks.
        ref_cos = full_cos.new_empty(shape)
        ref_sin = full_sin.new_empty(shape)
        gather_idx = _make_gather_idx(positions, self.ROTARY_DIM)
        torch.gather(full_cos, 0, gather_idx, out=ref_cos)
        torch.gather(full_sin, 0, gather_idx, out=ref_sin)

        cos_t, sin_t = _lookup("cfg", "default", positions, use_cache=True)

        assert torch.equal(cos_t, ref_cos)
        assert torch.equal(sin_t, ref_sin)
        assert torch.equal(cos_t, full_cos[positions.reshape(-1)])

    def test_no_cache_path_keeps_advanced_indexing(self, rope_state):
        full_cos, full_sin = _register_group(rope_state, rotary_dim=self.ROTARY_DIM)
        positions = torch.tensor(self.POSITIONS).reshape(-1, 1)

        ref_cos, ref_sin = full_cos[positions], full_sin[positions]
        cos_t, sin_t = _lookup("cfg", "default", positions, use_cache=False)

        # torch.equal also compares the shape: a [N, 1] index still adds its
        # dimension instead of being flattened to N rows, unlike the 1-D path.
        assert torch.equal(cos_t, ref_cos)
        assert torch.equal(sin_t, ref_sin)
        flat_cos = torch.index_select(full_cos, 0, _rope_index_1d(positions))
        assert cos_t.shape != flat_cos.shape


class _StubSchedulerConfig:
    def __init__(self, max_num_batched_tokens: int) -> None:
        self.max_num_batched_tokens = max_num_batched_tokens


class _StubSpeculativeConfig:
    def __init__(self, num_speculative_tokens: int) -> None:
        self.num_speculative_tokens = num_speculative_tokens

    def use_eagle(self) -> bool:
        return True


class _StubVllmConfig:
    """Just enough of ``VllmConfig`` for ``ComplexExpRotaryEmbedding`` to register buffers."""

    def __init__(self, max_num_batched_tokens: int, num_speculative_tokens: int = 0) -> None:
        self.scheduler_config = _StubSchedulerConfig(max_num_batched_tokens)
        self.speculative_config = _StubSpeculativeConfig(num_speculative_tokens) if num_speculative_tokens else None


class TestRegisteredBufferShapes:
    """The buffers the lookup writes into, as the real constructor registers them."""

    ROTARY_DIM = 32
    MAX_POS = 64
    MAX_NUM_BATCHED_TOKENS = 24
    NUM_SPECULATIVE_TOKENS = 3

    def test_registered_shapes_feed_the_optimised_lookup(self, rope_state, monkeypatch):
        # The constructor allocates on current_platform.device_type; pin it to
        # CPU so that this test needs no NPU.
        monkeypatch.setattr(rope_dsv4.current_platform, "device_type", "cpu")

        rope_dsv4.ComplexExpRotaryEmbedding(
            vllm_config=_StubVllmConfig(self.MAX_NUM_BATCHED_TOKENS, self.NUM_SPECULATIVE_TOKENS),
            layername="test.attn",
            head_size=self.ROTARY_DIM,
            rotary_dim=self.ROTARY_DIM,
            max_position_embeddings=self.MAX_POS,
            base=10000.0,
            scaling_factor=1.0,
            rope_groups=["default"],
        )

        config_key = next(iter(rope_state.registry_summary))
        full_cos, full_sin = rope_state.full_rope_cache[config_key]
        assert full_cos.shape == (self.MAX_POS, 1, 1, self.ROTARY_DIM)
        assert full_sin.shape == (self.MAX_POS, 1, 1, self.ROTARY_DIM)

        buf_cos, buf_sin = rope_state.runtime_buffer[config_key]["default"]
        assert buf_cos.shape == (self.MAX_NUM_BATCHED_TOKENS, 1, 1, self.ROTARY_DIM)
        assert buf_sin.shape == (self.MAX_NUM_BATCHED_TOKENS, 1, 1, self.ROTARY_DIM)

        spec_cos, spec_sin = rope_state.spec_runtime_buffer[config_key]["default"]
        assert len(spec_cos) == self.NUM_SPECULATIVE_TOKENS
        assert all(t.shape == buf_cos.shape for t in spec_cos)
        assert all(t.shape == buf_sin.shape for t in spec_sin)

        positions = torch.tensor([0, 3, 7, 11])
        cached_cos, cached_sin = _lookup(config_key, "default", positions, use_cache=True)
        assert torch.equal(cached_cos, full_cos[positions])
        assert torch.equal(cached_sin, full_sin[positions])
        assert cached_cos.data_ptr() == buf_cos.data_ptr()

        draft_cos, draft_sin = _lookup(
            config_key, "default", positions, use_cache=True, draft_index=self.NUM_SPECULATIVE_TOKENS
        )
        assert torch.equal(draft_cos, full_cos[positions])
        assert torch.equal(draft_sin, full_sin[positions])
        assert draft_cos.data_ptr() == spec_cos[-1].data_ptr()
