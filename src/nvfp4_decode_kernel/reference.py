"""PyTorch reference operations for the kernel's NVFP4 quantization."""

from __future__ import annotations

import torch


def round_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round values to the nearest representable E2M1 value."""
    magnitudes = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=x.device,
    )
    midpoints = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
        dtype=torch.float32,
        device=x.device,
    )
    absolute = x.abs()
    indices = torch.bucketize(
        absolute, midpoints, right=False, out_int32=True
    )
    indices = torch.where(
        torch.isfinite(absolute), indices, torch.zeros_like(indices)
    )
    return torch.sign(x) * magnitudes[indices]


def nvfp4_round_trip(x: torch.Tensor, group: int = 16) -> torch.Tensor:
    """Quantize and dequantize groups on the last axis with NVFP4."""
    if group <= 0:
        raise ValueError(f"group must be positive, got {group}")
    if x.shape[-1] % group != 0:
        raise ValueError(
            f"last dimension {x.shape[-1]} must be divisible by group {group}"
        )

    groups = x.float().reshape(*x.shape[:-1], -1, group)
    scale = groups.abs().amax(dim=-1, keepdim=True) / 6.0
    scale = scale.to(torch.float8_e4m3fn).float()
    zero_scale = scale == 0
    safe_scale = torch.where(zero_scale, torch.ones_like(scale), scale)
    quantized = round_e2m1(groups / safe_scale) * scale
    quantized = torch.where(zero_scale, groups, quantized)
    return quantized.reshape_as(x)
