"""The one-call way to get an engine on this cache with the shift applied.

Every test here stops before an engine is built. What is worth checking is the
arguments, because the failures this function exists to prevent are all quiet
ones: a missing scheduler setting serves from another cache, and a shift that
was asked for but never measured is the baseline under a different name.

CPU only: nothing here loads a model.
"""

from __future__ import annotations

import pytest

from nvfp4_vllm import engine
from nvfp4_vllm.backend import PAGE_SIZE
from nvfp4_vllm.calibrate import WORKER_EXTENSION_CLS
from nvfp4_vllm.guards import MAX_SLOTS
from nvfp4_vllm.runtime import KEY_SHIFT_ENV


MODEL = "/models/some-checkpoint"


class _Recorder:
    def __init__(self) -> None:
        self.engine_kwargs: dict = {}
        self.calibrated: list = []


@pytest.fixture
def recorder(monkeypatch):
    """Capture what would reach vLLM, and whether calibration was run."""
    recorded = _Recorder()

    class _FakeLLM:
        def __init__(self, **kwargs):
            recorded.engine_kwargs.update(kwargs)

    def _fake_calibrate(llm, **kwargs):
        recorded.calibrated.append(llm)
        return {"tokens": 32768}

    monkeypatch.setattr("vllm.LLM", _FakeLLM)
    monkeypatch.setattr(engine, "calibrate_key_shift", _fake_calibrate)
    # Recorded either way, so the ambient value is restored after the test.
    monkeypatch.delenv(KEY_SHIFT_ENV, raising=False)
    return recorded


def test_every_setting_the_backend_needs_is_bound(recorder) -> None:
    """Carrying only some of these is how an engine serves the wrong cache."""
    engine.build_llm(MODEL, key_shift=None)

    kwargs = recorder.engine_kwargs
    assert kwargs["model"] == MODEL
    assert kwargs["kv_cache_dtype"] == "nvfp4"
    assert kwargs["attention_config"] == {"backend": "CUSTOM"}
    assert kwargs["block_size"] == PAGE_SIZE
    assert kwargs["enable_prefix_caching"] is False
    assert kwargs["enable_chunked_prefill"] is False


def test_contradicting_a_required_setting_is_refused(recorder) -> None:
    with pytest.raises(ValueError, match="different cache"):
        engine.build_llm(MODEL, key_shift=None, enable_prefix_caching=True)

    assert recorder.engine_kwargs == {}


def test_other_arguments_reach_vllm_untouched(recorder) -> None:
    engine.build_llm(
        MODEL, key_shift=None, tensor_parallel_size=4, enforce_eager=True
    )

    assert recorder.engine_kwargs["tensor_parallel_size"] == 4
    assert recorder.engine_kwargs["enforce_eager"] is True


def test_the_batch_is_kept_inside_the_tail_slots(recorder) -> None:
    """vLLM defaults to 1024 here, which the guard refuses outright."""
    engine.build_llm(MODEL, key_shift=None)

    assert recorder.engine_kwargs["max_num_seqs"] == MAX_SLOTS


def test_a_caller_may_run_fewer_sequences(recorder) -> None:
    engine.build_llm(MODEL, key_shift=None, max_num_seqs=4)

    assert recorder.engine_kwargs["max_num_seqs"] == 4


def test_a_prefill_is_given_room_to_arrive_whole(recorder) -> None:
    """Chunking is off, so a batch too small for the window cannot schedule."""
    engine.build_llm(MODEL, key_shift=None, max_model_len=32768)

    assert recorder.engine_kwargs["max_num_batched_tokens"] == 32768


def test_a_caller_who_sized_the_batch_keeps_their_size(recorder) -> None:
    engine.build_llm(
        MODEL,
        key_shift=None,
        max_model_len=32768,
        max_num_batched_tokens=65536,
    )

    assert recorder.engine_kwargs["max_num_batched_tokens"] == 65536


def test_auto_measures_the_shift(recorder) -> None:
    engine.build_llm(MODEL, key_shift="auto")

    assert recorder.engine_kwargs["worker_extension_cls"] == (
        WORKER_EXTENSION_CLS
    )
    assert len(recorder.calibrated) == 1


def test_a_sidecar_is_loaded_rather_than_measured(recorder, tmp_path) -> None:
    """Paying seven seconds for a vector already on disk is the waste."""
    import os

    path = tmp_path / "key_shift.safetensors"
    engine.build_llm(MODEL, key_shift=str(path))

    assert os.environ[KEY_SHIFT_ENV] == str(path)
    assert recorder.calibrated == []


def test_asking_for_no_shift_overrides_a_stale_environment(
    recorder, monkeypatch
) -> None:
    """Otherwise the baseline arm silently inherits the previous arm's mean."""
    import os

    monkeypatch.setenv(KEY_SHIFT_ENV, "/left/over/from/an/earlier/run")

    engine.build_llm(MODEL, key_shift=None)

    assert KEY_SHIFT_ENV not in os.environ
    assert recorder.calibrated == []


def test_measuring_needs_the_extension_this_repository_ships(
    recorder,
) -> None:
    """vLLM takes one, so someone else's is a conflict rather than an extra."""
    with pytest.raises(ValueError, match="accepts one"):
        engine.build_llm(
            MODEL, key_shift="auto", worker_extension_cls="other.Extension"
        )


def test_the_switch_is_installed_even_when_nothing_is_measured(
    recorder,
) -> None:
    """A/B-ing both arms from one loaded model depends on it being there."""
    engine.build_llm(MODEL, key_shift=None)

    assert recorder.engine_kwargs["worker_extension_cls"] == (
        WORKER_EXTENSION_CLS
    )


def test_an_engine_without_the_nvfp4_cache_is_not_reported_as_calibrated(
    monkeypatch,
) -> None:
    """calibrate_key_shift returns None there, and None must not pass for done."""

    class _FakeLLM:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr("vllm.LLM", _FakeLLM)
    monkeypatch.setattr(engine, "calibrate_key_shift", lambda llm, **_: None)
    monkeypatch.delenv(KEY_SHIFT_ENV, raising=False)

    with pytest.raises(RuntimeError, match="no K mean"):
        engine.build_llm(MODEL, key_shift="auto")
