"""Public API for the NVFP4 paged-decode kernel."""

import os

os.environ.setdefault("CUTE_DSL_ENABLE_TVM_FFI", "1")

from .interface import RESIDUAL_ROW_TILE, KernelNotAvailableError, fp4_decode

__all__ = ["RESIDUAL_ROW_TILE", "KernelNotAvailableError", "fp4_decode"]
