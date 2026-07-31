"""What a live engine computes when decode reads the NVFP4 cache.

Every decode step of a real generation is checked against
``_torch_fp4_decode``, the independent PyTorch simulation the kernel suite
already validates against, on the same inputs. Not against a BF16 run: pages
0..n-1 hold FP4 by then, so a quantization difference is expected and comparing
generated text would only say how far into the sequence the two happened to
agree.

The oracle takes BF16 pages and quantizes them itself, so it needs the BF16 the
model produced rather than what the cache holds. That is captured at write
time, reusing the interception ``test_write_path.py`` established. The chain
being compared is therefore "same BF16 in, two independent quantize-then-attend
pipelines out", which is the same shape as the kernel suite's own comparison —
hence the same cosine floor rather than a new one.

Promotion is not implemented yet, so the prompt length and generation length
are chosen to keep every row inside one page for the whole run, and that
premise is asserted rather than assumed.

Requires ``NVFP4_RUN_VLLM_E2E=1``.

Environment overrides:

- ``NVFP4_TEST_MODEL``: model to serve.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


RUN_E2E = os.environ.get("NVFP4_RUN_VLLM_E2E") == "1"
MODEL = os.environ.get(
    "NVFP4_TEST_MODEL",
    "NousResearch/Meta-Llama-3.1-8B-Instruct",
)

PAGE_SIZE = 128
MAX_MODEL_LEN = 4096
MAX_NUM_SEQS = 8

# 7 * 128 + 1. The prefill quantizes seven pages and leaves one token in the
# tail, so seqused_fp4 stays at 896 and the tail grows by one token per step.
PROMPT_TOKENS = 897
GENERATED_TOKENS = 100
FP4_TOKENS = ((PROMPT_TOKENS - 1) // PAGE_SIZE) * PAGE_SIZE

# Tail lengths to compare at. The residual is the only thing changing from step
# to step, so it is the axis worth sampling: the two shortest tails, one in the
# middle, and the last step of the run. A decode step's tail is
# ``1 + tokens generated so far``, so 100 is the final step of 100 tokens.
CHECKED_RESIDUALS = (2, 3, 64, 100)

# Same floor as tests/kernel. Both sides quantize independently from the same
# BF16, which is exactly the comparison that constant was chosen for.
MIN_COSINE = 0.99


pytestmark = pytest.mark.skipif(
    not RUN_E2E,
    reason="set NVFP4_RUN_VLLM_E2E=1 to run the vLLM integration test",
)


@dataclass
class DecodeRecord:
    """One layer's decode on one step, with everything needed to redo it."""

    layer_index: int
    seq_len: int
    seqused_fp4: int
    softmax_scale: float
    query: torch.Tensor
    output: torch.Tensor

    @property
    def residual(self) -> int:
        return self.seq_len - self.seqused_fp4


def _require_sm100() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability[0] != 10:
        pytest.skip(f"SM100 is required, found compute capability {capability}")


def _oracle():
    """The kernel suite's decode oracle, loaded by path.

    ``tests/`` has no package structure, so the two directories cannot import
    each other. Loading by path keeps one definition of the oracle instead of a
    copy here that could drift away from the kernel it is meant to pin down.
    """
    path = (
        Path(__file__).resolve().parents[1]
        / "kernel"
        / "test_fp4_decode_correctness.py"
    )
    spec = importlib.util.spec_from_file_location("_fp4_decode_oracle", path)
    module = importlib.util.module_from_spec(spec)
    # Registered before execution because the module defines dataclasses, and
    # dataclasses resolves annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class Session:
    """Everything one generation left behind."""

    records: list[DecodeRecord]
    history_key: dict[int, torch.Tensor]
    history_value: dict[int, torch.Tensor]
    decode_calls: int
    fp4_prefix_seen: bool
    promotion_flags: list[torch.Tensor]
    generated_ids: list[int]
    generated_text: str


@pytest.fixture(scope="module")
def session() -> Session:
    _require_sm100()
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.model_executor.layers.attention.attention import (
        get_attention_context,
    )

    import nvfp4_vllm.impl as impl_module

    chunks: dict[int, list[torch.Tensor]] = {}
    records: list[DecodeRecord] = []
    promotion_flags: list[torch.Tensor] = []
    counters = {"decode_calls": 0, "fp4_prefix": False}

    original_update = impl_module.NVFP4Impl.do_kv_cache_update
    original_decode = impl_module.NVFP4Impl._decode
    original_fp4_decode = impl_module.fp4_decode

    def record_update(self, layer, key, value, kv_cache, slot_mapping):
        original_update(self, layer, key, value, kv_cache, slot_mapping)
        metadata, _, _, _ = get_attention_context(layer.layer_name)
        if self.runtime is None or metadata is None:
            return
        tokens = metadata.num_actual_tokens
        chunks.setdefault(self.layer_index, []).append(
            (key[:tokens].clone(), value[:tokens].clone())
        )
        if self.layer_index == 0:
            # Kept as a device tensor and read once at the end: this runs on
            # every step, and .item() here would synchronize on every one.
            promotion_flags.append(metadata.promotion_mask.any())

    def record_decode(self, rows, query, kv_cache, attn_metadata, output):
        original_decode(self, rows, query, kv_cache, attn_metadata, output)
        # The write for this layer has already run, so the history is the whole
        # sequence the kernel just attended over, current token included.
        seq_len = sum(chunk[0].shape[0] for chunk in chunks[self.layer_index])
        records.append(
            DecodeRecord(
                layer_index=self.layer_index,
                seq_len=seq_len,
                seqused_fp4=FP4_TOKENS,
                softmax_scale=self.scale,
                query=query[:rows].clone(),
                output=output[:rows].clone(),
            )
        )

    def count_fp4_decode(**kwargs):
        result = original_fp4_decode(**kwargs)
        counters["decode_calls"] += 1
        if not counters["fp4_prefix"]:
            # Reading a device tensor synchronizes, so this happens once and
            # then never again for the rest of the run.
            counters["fp4_prefix"] = bool((kwargs["seqused_fp4"] > 0).all())
        return result

    impl_module.NVFP4Impl.do_kv_cache_update = record_update
    impl_module.NVFP4Impl._decode = record_decode
    impl_module.fp4_decode = count_fp4_decode
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        kv_cache_dtype="nvfp4",
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=MAX_MODEL_LEN,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        block_size=PAGE_SIZE,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        attention_config={"backend": "CUSTOM"},
    )
    try:
        prompt = _prompt_of_exact_length(llm, PROMPT_TOKENS)
        chunks.clear()
        records.clear()
        promotion_flags.clear()
        counters.update(decode_calls=0, fp4_prefix=False)
        completions = llm.generate(
            TokensPrompt(prompt_token_ids=prompt),
            SamplingParams(
                max_tokens=GENERATED_TOKENS, ignore_eos=True, temperature=0.0
            ),
        )
        completion = completions[0].outputs[0]
        yield Session(
            records=records,
            history_key={
                layer: torch.cat([chunk[0] for chunk in layer_chunks])
                for layer, layer_chunks in chunks.items()
            },
            history_value={
                layer: torch.cat([chunk[1] for chunk in layer_chunks])
                for layer, layer_chunks in chunks.items()
            },
            decode_calls=counters["decode_calls"],
            fp4_prefix_seen=counters["fp4_prefix"],
            promotion_flags=promotion_flags,
            generated_ids=list(completion.token_ids),
            generated_text=completion.text,
        )
    finally:
        impl_module.NVFP4Impl.do_kv_cache_update = original_update
        impl_module.NVFP4Impl._decode = original_decode
        impl_module.fp4_decode = original_fp4_decode
        # Another engine follows in this session and sizes its cache from free
        # memory; only shutdown() releases what startup froze.
        llm.llm_engine.engine_core.shutdown()
        del llm


def _prompt_of_exact_length(llm, length: int) -> list[int]:
    """Natural text trimmed to exactly ``length`` tokens.

    The length has to be exact for the page arithmetic above to hold, and the
    text has to be real for the non-degeneracy check below to mean anything: a
    prompt of consecutive token ids would invite the model to repeat whatever
    it liked, and that would look like the failure this is watching for.
    """
    tokenizer = llm.get_tokenizer()
    passage = " ".join(
        [
            "The harbour town wakes early, and the fishing boats leave "
            "before the light reaches the far side of the bay.",
            "Its market sells oranges from the valley, salt from the flats "
            "east of the road, and rope wound by hand in the old sheds.",
            "A railway arrived in 1868 and closed a century later, leaving "
            "an embankment that children now use as a shortcut to school.",
            "Winters are mild but wet, and the wind that crosses the "
            "headland has bent every pine along the cliff path inland.",
            "The lighthouse keeper's cottage became a museum, and its "
            "logbooks record every storm since the year the tower was lit.",
            "Farmers on the terraces above grow olives in soil so thin that "
            "each tree stands in a pocket of earth built up by hand.",
            "In late summer the whole coast smells of thyme, and the "
            "cicadas make a noise you stop hearing after the first day.",
            "Visitors come for the beaches, but the people who return come "
            "for the long evenings when nothing at all is expected of them.",
        ]
    )
    ids: list[int] = []
    while len(ids) < length:
        ids.extend(tokenizer.encode(passage, add_special_tokens=not ids))
    return ids[:length]


def test_decode_ran_on_the_fp4_cache(session: Session):
    """The kernel has to be what served these tokens.

    Every other assertion here would also pass if attention had quietly fallen
    back to something else, so this is the one that makes them about the FP4
    read path rather than about the model.
    """
    assert session.decode_calls > 0, "no decode step reached fp4_decode"
    assert session.fp4_prefix_seen, (
        "fp4_decode was called but never with FP4 pages behind it, so only "
        "the BF16 tail was exercised"
    )


def test_no_row_crossed_a_page_boundary(session: Session):
    """The premise the prompt length was chosen for.

    Crossing a boundary sets ``promotion_mask``, and nothing consumes it yet,
    so the tail would start dropping tokens. Per-step cosine would not
    necessarily notice, because the oracle reads the same tail. Changing
    ``PROMPT_TOKENS`` or ``GENERATED_TOKENS`` without redoing the arithmetic
    lands here.
    """
    assert session.promotion_flags, "no step was observed"
    promoted = torch.stack(session.promotion_flags).any().item()
    assert not promoted, (
        "a row filled a page during the run, which S7 has no answer for"
    )


def test_every_layer_matches_the_torch_oracle(session: Session):
    """Each layer's decode output, against the same attention done in PyTorch.

    All layers rather than a sample: each one attends over its own K/V, so a
    failure names the layer instead of only saying the model drifted. The steps
    are sampled along the residual length, which is the only input that changes
    from one decode step to the next.
    """
    oracle = _oracle()
    wanted = set(CHECKED_RESIDUALS)
    layers = set(session.history_key)
    assert layers, "no layer recorded a write"

    # Every layer-step is measured before anything is asserted, so a failure
    # reports how far the disagreement spreads rather than whichever layer
    # happened to be compared first.
    checked: set[tuple[int, int]] = set()
    cosines: list[tuple[float, int, int]] = []
    for record in session.records:
        if record.residual not in wanted:
            continue
        expected = _replay(oracle, session, record)
        cosine = F.cosine_similarity(
            record.output.float().flatten(),
            expected.float().flatten(),
            dim=0,
        ).item()
        checked.add((record.layer_index, record.residual))
        cosines.append((cosine, record.layer_index, record.residual))

    cosines.sort()
    print(
        f"\nworst cosines over {len(cosines)} layer-steps: "
        + ", ".join(
            f"{c:.4f} (layer {layer}, residual {residual})"
            for c, layer, residual in cosines[:5]
        )
    )
    below = [entry for entry in cosines if entry[0] < MIN_COSINE]
    assert not below, (
        f"{len(below)} of {len(cosines)} layer-steps are below {MIN_COSINE} "
        f"against the PyTorch oracle, worst "
        f"{below[0][0]:.6f} at layer {below[0][1]} residual {below[0][2]}"
    )
    assert checked == {
        (layer, residual) for layer in layers for residual in wanted
    }, (
        "some layer or residual length was never compared, so this passed "
        "without looking at it"
    )


def test_the_generation_did_not_collapse(session: Session):
    """The steps the cosine check did not sample still have to be sane.

    Cosine looks at four steps out of ninety-nine. A fault that only shows up
    after many steps — a tail offset that creeps, a slot that changes owner —
    would leave those four intact and wreck everything after them, and the only
    evidence left is the text.
    """
    assert len(session.generated_ids) == GENERATED_TOKENS, (
        f"asked for {GENERATED_TOKENS} tokens, got "
        f"{len(session.generated_ids)}"
    )
    assert session.generated_text.strip(), "the model produced no text"
    assert len(set(session.generated_ids)) > 1, (
        "the model emitted one token a hundred times: "
        f"{session.generated_text!r}"
    )
    unhealthy = [
        (record.layer_index, record.seq_len, record.query.isfinite().all().item())
        for record in session.records
        if not record.output.isfinite().all()
    ]
    assert not unhealthy, (
        "non-finite decode outputs at (layer, sequence, query was finite): "
        f"{unhealthy[:20]} of {len(unhealthy)}"
    )
    print(f"\ngenerated: {session.generated_text!r}")


def _replay(oracle, session: Session, record: DecodeRecord) -> torch.Tensor:
    """The oracle's answer for one recorded decode.

    The cache the kernel read is rebuilt from the BF16 the model wrote: whole
    pages the oracle will quantize itself, and a last page it keeps in BF16,
    which is what the tail is.
    """
    seq = record.seq_len
    pages = -(-seq // PAGE_SIZE)
    device = record.query.device

    def paged(history: torch.Tensor) -> torch.Tensor:
        tokens = history[:seq]
        padding = pages * PAGE_SIZE - seq
        if padding:
            tokens = F.pad(tokens, (0, 0, 0, 0, 0, padding))
        return tokens.reshape(pages, PAGE_SIZE, *tokens.shape[1:])

    def ints(*values: int) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.int32, device=device)

    with torch.no_grad():
        expected = oracle._torch_fp4_decode(
            record.query.unsqueeze(0),
            paged(session.history_key[record.layer_index]),
            paged(session.history_value[record.layer_index]),
            torch.arange(pages, dtype=torch.int32, device=device).reshape(
                1, pages
            ),
            ints(seq),
            ints(pages - 1),
            ints(record.residual),
            record.softmax_scale,
        )
    return expected[:, 0]
