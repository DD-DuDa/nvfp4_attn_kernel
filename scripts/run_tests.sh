#!/usr/bin/env bash
#
# Run one of this repository's test suites.
#
# Exists because the python on PATH cannot run any of them: it has a CuTeDSL
# without iket and no compiled vLLM. See README.md.
#
#   scripts/run_tests.sh kernel                    # kernel numerics
#   scripts/run_tests.sh e2e                       # vLLM integration, no soak
#   scripts/run_tests.sh soak                      # slot reuse, many requests
#   scripts/run_tests.sh all
#   scripts/run_tests.sh kernel -- -k residual -x  # after --, pytest's own
#
# Overridable: NVFP4_ENV, NVFP4_TEST_MODEL, NVFP4_SOAK_ROUNDS,
# CUDA_VISIBLE_DEVICES.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEV="$(dirname "$ROOT")"

: "${NVFP4_ENV:=$DEV/BitKV_nvfp4/_local/envs/vllm-nvfp4}"
: "${NVFP4_TEST_MODEL:=$DEV/models/Meta-Llama-3.1-8B-Instruct}"
: "${CUDA_VISIBLE_DEVICES:=0}"

PYTHON="$NVFP4_ENV/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "no interpreter at $PYTHON" >&2
    echo "point NVFP4_ENV at an environment with a compiled vLLM and an" >&2
    echo "iket-capable CuTeDSL; see README.md" >&2
    exit 1
fi

suite="${1:-all}"
shift || true
if [[ "${1:-}" == "--" ]]; then
    shift
fi

export NVFP4_TEST_MODEL CUDA_VISIBLE_DEVICES
# In-process engine core. Without it a test that builds a second engine hits
# the first one's leftovers.
export VLLM_ENABLE_V1_MULTIPROCESSING=0

failed=0

run() {
    local label="$1"
    shift
    echo
    echo "=== $label (GPU $CUDA_VISIBLE_DEVICES) ==="
    if ! (cd "$ROOT" && "$PYTHON" -m pytest -q "$@"); then
        failed=1
    fi
}

kernel() {
    # tests/kernel never builds an engine, so the e2e switch means nothing to
    # it either way.
    run kernel tests/kernel "$@"
}

e2e() {
    export NVFP4_RUN_VLLM_E2E=1
    run e2e tests/e2e --ignore=tests/e2e/test_soak.py "$@"
}

soak() {
    export NVFP4_RUN_VLLM_E2E=1
    run soak tests/e2e/test_soak.py "$@"
}

case "$suite" in
    kernel) kernel "$@" ;;
    e2e) e2e "$@" ;;
    soak) soak "$@" ;;
    all)
        # Cheapest first, so a broken kernel is reported before four minutes
        # of engine startup.
        kernel "$@"
        e2e "$@"
        soak "$@"
        ;;
    *)
        echo "unknown suite: $suite (want kernel, e2e, soak, or all)" >&2
        exit 2
        ;;
esac

exit "$failed"
