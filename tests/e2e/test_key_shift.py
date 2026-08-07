"""Subtracting a constant from post-RoPE K on its way into the cache.

The write path can remove a per-layer constant from K before quantizing it,
which shrinks the block scale and so the quantization step. Nothing adds the
constant back, and the reason that is allowed is an identity rather than an
approximation: every key a query sees moved by the same vector, so every score
moved by the same amount and softmax is unchanged.

That identity is what the read path's silence depends on, so it is tested here
alongside the loader. The rest of the tests are about the loader refusing a
vector that does not belong to the model, because a silently mis-shaped or
mis-scaled shift would not crash — it would quietly cache the wrong keys.

The constant can also be measured from the model at startup instead of loaded,
and the tests for that are about the arithmetic of the accumulator and about
its one ordering hazard: measuring while a shift is applied would return the
residual mean rather than the mean.

CPU only: nothing here runs a kernel.
"""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from nvfp4_decode_kernel.reference import nvfp4_round_trip
from nvfp4_vllm.calibrate import _long_prompt
from nvfp4_vllm.runtime import KEY_SHIFT_ENV, LayerRuntime, load_key_shift


LAYERS = 3
HEADS_KV = 8
HEAD_DIM = 128
# What the tail is allocated as, and so what the shift is held in.
DTYPE = torch.bfloat16


def _write(tmp_path, tensors) -> str:
    path = tmp_path / "key_shift.safetensors"
    save_file(tensors, str(path))
    return str(path)


def test_no_env_means_no_shift(monkeypatch) -> None:
    monkeypatch.delenv(KEY_SHIFT_ENV, raising=False)
    assert load_key_shift(LAYERS, HEADS_KV, HEAD_DIM, torch.device("cpu"), DTYPE) is None


def test_shift_arrives_in_the_dtype_the_write_path_subtracts_in(
    monkeypatch, tmp_path
) -> None:
    """A sidecar in any precision, converted once here rather than per step."""
    shift = torch.randn(LAYERS, HEADS_KV, HEAD_DIM, dtype=torch.float32)
    monkeypatch.setenv(KEY_SHIFT_ENV, _write(tmp_path, {"key_shift": shift}))

    loaded = load_key_shift(LAYERS, HEADS_KV, HEAD_DIM, torch.device("cpu"), DTYPE)

    assert loaded.dtype is DTYPE
    assert loaded.shape == (LAYERS, HEADS_KV, HEAD_DIM)


def test_shift_for_another_model_is_refused(monkeypatch, tmp_path) -> None:
    shift = torch.zeros(LAYERS + 1, HEADS_KV, HEAD_DIM)
    monkeypatch.setenv(KEY_SHIFT_ENV, _write(tmp_path, {"key_shift": shift}))

    with pytest.raises(ValueError, match="needs"):
        load_key_shift(LAYERS, HEADS_KV, HEAD_DIM, torch.device("cpu"), DTYPE)


def test_file_without_the_tensor_is_refused(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(
        KEY_SHIFT_ENV, _write(tmp_path, {"layers.0.c": torch.ones(HEAD_DIM)})
    )

    with pytest.raises(ValueError, match="no 'key_shift' tensor"):
        load_key_shift(LAYERS, HEADS_KV, HEAD_DIM, torch.device("cpu"), DTYPE)


def test_shifting_every_key_leaves_attention_unchanged() -> None:
    generator = torch.Generator().manual_seed(0x5417)
    query = torch.randn(4, HEAD_DIM, generator=generator)
    key = torch.randn(256, HEAD_DIM, generator=generator)
    value = torch.randn(256, HEAD_DIM, generator=generator)
    shift = torch.randn(HEAD_DIM, generator=generator) * 8.0

    def attend(keys):
        scores = query @ keys.T / HEAD_DIM**0.5
        return scores.softmax(dim=-1) @ value

    torch.testing.assert_close(attend(key), attend(key - shift))


def test_removing_the_offset_shrinks_the_quantization_error() -> None:
    """Why the shift is worth its plumbing, on a K-shaped distribution.

    Post-RoPE K sits far from zero per channel, and the block scale comes from
    the group's largest magnitude, so the offset sets the step size for the
    variation around it. Here the offset is 20x the variation, which is the
    regime the real tensors are in.
    """
    generator = torch.Generator().manual_seed(0x5418)
    offset = torch.randn(HEAD_DIM, generator=generator) * 20.0
    key = torch.randn(256, HEAD_DIM, generator=generator) + offset

    plain = (nvfp4_round_trip(key) - key).square().mean()
    shifted = (
        nvfp4_round_trip(key - offset) - (key - offset)
    ).square().mean()

    assert shifted < plain / 10


def _runtime(monkeypatch) -> LayerRuntime:
    monkeypatch.delenv(KEY_SHIFT_ENV, raising=False)
    return LayerRuntime(
        num_layers=LAYERS,
        num_slots=2,
        num_heads=HEADS_KV,
        num_kv_heads=HEADS_KV,
        head_dim=HEAD_DIM,
        device=torch.device("cpu"),
    )


def _feed(runtime: LayerRuntime, keys: torch.Tensor) -> None:
    """What the write path does to the accumulator for one step."""
    for layer in range(LAYERS):
        runtime.key_shift_sum[layer] += keys.double().sum(dim=0)
    runtime.key_shift_tokens += keys.shape[0]


def test_measured_shift_is_the_mean_over_every_step(monkeypatch) -> None:
    """Steps differ in length, so a mean of means would be the wrong answer."""
    generator = torch.Generator().manual_seed(0x5419)
    steps = [
        torch.randn(n, HEADS_KV, HEAD_DIM, generator=generator) + 7.0
        for n in (300, 1, 1, 64)
    ]

    runtime = _runtime(monkeypatch)
    runtime.begin_key_shift_measurement()
    for step in steps:
        _feed(runtime, step)
    shift = runtime.finish_key_shift_measurement()

    expected = torch.cat(steps).mean(dim=0)
    assert shift.shape == (LAYERS, HEADS_KV, HEAD_DIM)
    assert shift.dtype is DTYPE
    for layer in range(LAYERS):
        torch.testing.assert_close(shift[layer].float(), expected, atol=0.05, rtol=0)
    assert shift is runtime.key_shift


def test_measuring_drops_a_shift_already_in_force(monkeypatch, tmp_path) -> None:
    """Otherwise the second measurement returns the residual, not the mean.

    A shift is applied to what gets cached but not to the K handed to the
    accumulator, so the hazard is not arithmetic — it is that leaving the old
    shift loaded would have the write path subtract a stale constant from
    every key stored during the measurement.
    """
    loaded = torch.full((LAYERS, HEADS_KV, HEAD_DIM), 3.0)
    monkeypatch.setenv(KEY_SHIFT_ENV, _write(tmp_path, {"key_shift": loaded}))
    runtime = LayerRuntime(
        num_layers=LAYERS,
        num_slots=2,
        num_heads=HEADS_KV,
        num_kv_heads=HEADS_KV,
        head_dim=HEAD_DIM,
        device=torch.device("cpu"),
    )
    assert runtime.key_shift is not None

    runtime.begin_key_shift_measurement()
    assert runtime.key_shift is None

    _feed(runtime, torch.full((10, HEADS_KV, HEAD_DIM), 5.0))
    shift = runtime.finish_key_shift_measurement()

    torch.testing.assert_close(shift, torch.full_like(shift, 5.0))
    assert not torch.isclose(shift.float(), loaded).all()


def test_finishing_without_any_tokens_is_refused(monkeypatch) -> None:
    runtime = _runtime(monkeypatch)
    runtime.begin_key_shift_measurement()

    with pytest.raises(ValueError, match="no tokens were measured"):
        runtime.finish_key_shift_measurement()


def test_a_loaded_shift_starts_enabled(monkeypatch, tmp_path) -> None:
    shift = torch.randn(LAYERS, HEADS_KV, HEAD_DIM)
    monkeypatch.setenv(KEY_SHIFT_ENV, _write(tmp_path, {"key_shift": shift}))
    runtime = LayerRuntime(
        num_layers=LAYERS,
        num_slots=2,
        num_heads=HEADS_KV,
        num_kv_heads=HEADS_KV,
        head_dim=HEAD_DIM,
        device=torch.device("cpu"),
    )

    assert runtime.active_key_shift is not None


def test_the_runtime_holds_the_shift_in_its_cache_dtype(
    monkeypatch, tmp_path
) -> None:
    """A mismatch puts an upcast and a downcast into every write, which is
    12% of an eager decode step."""
    shift = torch.randn(LAYERS, HEADS_KV, HEAD_DIM, dtype=torch.float32)
    monkeypatch.setenv(KEY_SHIFT_ENV, _write(tmp_path, {"key_shift": shift}))
    runtime = LayerRuntime(
        num_layers=LAYERS,
        num_slots=2,
        num_heads=HEADS_KV,
        num_kv_heads=HEADS_KV,
        head_dim=HEAD_DIM,
        device=torch.device("cpu"),
    )

    assert runtime.key_shift.dtype is runtime.tail_key.dtype
    key = torch.randn(4, HEADS_KV, HEAD_DIM, dtype=runtime.tail_key.dtype)
    assert torch.sub(key, runtime.key_shift[0]).dtype is key.dtype


def test_a_coarse_constant_is_still_exactly_invariant() -> None:
    """Why rounding mu is a question about error removed, not correctness."""
    generator = torch.Generator().manual_seed(0x541A)
    query = torch.randn(4, HEAD_DIM, generator=generator)
    key = torch.randn(256, HEAD_DIM, generator=generator)
    value = torch.randn(256, HEAD_DIM, generator=generator)
    coarse = (torch.randn(HEAD_DIM, generator=generator) * 8.0).to(DTYPE)

    def attend(keys):
        scores = query @ keys.T / HEAD_DIM**0.5
        return scores.softmax(dim=-1) @ value

    torch.testing.assert_close(attend(key), attend(key - coarse.float()))


def test_turning_the_shift_off_keeps_the_vector(monkeypatch) -> None:
    """Both arms from one loaded model is the point of the switch."""
    runtime = _runtime(monkeypatch)
    runtime.begin_key_shift_measurement()
    _feed(runtime, torch.full((8, HEADS_KV, HEAD_DIM), 2.0))
    measured = runtime.finish_key_shift_measurement()

    runtime.set_key_shift_enabled(False)
    assert runtime.active_key_shift is None
    assert runtime.key_shift is measured

    runtime.set_key_shift_enabled(True)
    assert runtime.active_key_shift is measured


def test_measuring_leaves_the_shift_enabled(monkeypatch) -> None:
    """Otherwise calibrating after a disable would silently do nothing."""
    runtime = _runtime(monkeypatch)
    runtime.set_key_shift_enabled(False)

    runtime.begin_key_shift_measurement()
    _feed(runtime, torch.full((8, HEADS_KV, HEAD_DIM), 2.0))
    runtime.finish_key_shift_measurement()

    assert runtime.active_key_shift is not None


def test_enabling_a_shift_that_was_never_measured_is_refused(
    monkeypatch,
) -> None:
    runtime = _runtime(monkeypatch)

    with pytest.raises(ValueError, match="nothing to enable"):
        runtime.set_key_shift_enabled(True)


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


class _LLM:
    def get_tokenizer(self):
        return _Tokenizer()


def test_calibration_prompt_fills_the_window_it_was_given() -> None:
    """Reaching high positions is the whole reason this prompt exists."""
    prompt = _long_prompt(_LLM(), ["abc", "de"], 5000)

    assert len(prompt) == 5000


def test_calibration_prompt_repeats_rather_than_truncates() -> None:
    prompt = _long_prompt(_LLM(), ["abcd"], 20)
    unit = _Tokenizer().encode("abcd\n\n")

    assert prompt == (unit * 4)[:20]


def test_calibration_prompt_needs_something_to_tile() -> None:
    with pytest.raises(ValueError, match="no calibration text"):
        _long_prompt(_LLM(), ["", "  \n "], 128)
