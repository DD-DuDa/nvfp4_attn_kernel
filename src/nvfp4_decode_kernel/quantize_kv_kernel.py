"""CuTeDSL K/V page quantization."""

from typing import Tuple

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.runtime import make_ptr

from .quantize_q_kernel import _pack_e2m1, _pack_e4m3


PAGE_SIZE = 128
HEAD_DIM = 128
SF_VEC_SIZE = 16


@cute.kernel
def _quantize_key_pages_kernel(
    key_pages: cute.Tensor,
    key_pages_fp4: cute.Tensor,
    key_scales: cute.Tensor,
    sf_vec_size: cutlass.Constexpr[int],
    rest_k: cutlass.Constexpr[int],
):
    thread, _, _ = cute.arch.thread_idx()
    head, page, _ = cute.arch.block_idx()
    lane = thread % 32
    warp = thread // 32
    sequence = warp * 32 + lane

    source = key_pages[page, sequence, head, None]
    destination = key_pages_fp4[page, sequence, head, None]
    scale_e4m3 = cute.make_rmem_tensor(
        cute.make_layout(4), cutlass.Float8E4M3FN
    )
    scale_u32 = cute.recast_tensor(scale_e4m3, cutlass.Int32)
    inv6 = Float32(1.0 / 6.0)

    for k_atom in cutlass.range_constexpr(rest_k):
        scales_f32 = [
            Float32(0.0),
            Float32(0.0),
            Float32(0.0),
            Float32(0.0),
        ]
        for group in cutlass.range_constexpr(4):
            group_index = k_atom * 4 + group
            start = group_index * sf_vec_size
            maximum = Float32(0.0)
            for element in cutlass.range_constexpr(sf_vec_size):
                value = Float32(source[start + element])
                maximum = cute.arch.fmax(
                    maximum, cute.arch.fmax(value, -value)
                )
            scales_f32[group] = maximum * inv6

        packed_scale = _pack_e4m3(
            scales_f32[0],
            scales_f32[1],
            scales_f32[2],
            scales_f32[3],
        )
        scale_tile = key_scales[
            page, lane, warp, 0, None, k_atom, head
        ]
        scale_word = cute.make_tensor(
            cute.recast_ptr(scale_tile.iterator, dtype=cutlass.Int32),
            cute.make_layout(1),
        )
        scale_word[0] = packed_scale
        scale_u32[0] = packed_scale
        rounded_scales = scale_e4m3.load().to(Float32)

        for group in cutlass.range_constexpr(4):
            group_index = k_atom * 4 + group
            start = group_index * sf_vec_size
            inverse_scale = Float32(1.0) / (
                rounded_scales[group] + Float32(1.0e-30)
            )
            scaled = [Float32(0.0)] * 16
            for element in cutlass.range_constexpr(sf_vec_size):
                scaled[element] = (
                    Float32(source[start + element]) * inverse_scale
                )

            packed_low = _pack_e2m1(
                scaled[0],
                scaled[1],
                scaled[2],
                scaled[3],
                scaled[4],
                scaled[5],
                scaled[6],
                scaled[7],
            )
            packed_high = _pack_e2m1(
                scaled[8],
                scaled[9],
                scaled[10],
                scaled[11],
                scaled[12],
                scaled[13],
                scaled[14],
                scaled[15],
            )
            byte_offset = group_index * (sf_vec_size // 2)
            low_word = cute.make_tensor(
                cute.recast_ptr(
                    destination.iterator + byte_offset,
                    dtype=cutlass.Int32,
                ),
                cute.make_layout(1),
            )
            high_word = cute.make_tensor(
                cute.recast_ptr(
                    destination.iterator + byte_offset + 4,
                    dtype=cutlass.Int32,
                ),
                cute.make_layout(1),
            )
            low_word[0] = packed_low
            high_word[0] = packed_high


@cute.kernel
def _quantize_value_pages_kernel(
    value_pages: cute.Tensor,
    value_pages_fp4: cute.Tensor,
    value_scales: cute.Tensor,
    sf_vec_size: cutlass.Constexpr[int],
    page_rest_k: cutlass.Constexpr[int],
):
    thread, _, _ = cute.arch.thread_idx()
    head, page, _ = cute.arch.block_idx()
    lane = thread % 32
    warp = thread // 32
    head_dim = warp * 32 + lane

    source = value_pages[page, None, head, head_dim]
    destination = value_pages_fp4[page, head, head_dim, None]
    scale_e4m3 = cute.make_rmem_tensor(
        cute.make_layout(4), cutlass.Float8E4M3FN
    )
    scale_u32 = cute.recast_tensor(scale_e4m3, cutlass.Int32)
    inv6 = Float32(1.0 / 6.0)

    for k_atom in cutlass.range_constexpr(page_rest_k):
        scales_f32 = [
            Float32(0.0),
            Float32(0.0),
            Float32(0.0),
            Float32(0.0),
        ]
        for group in cutlass.range_constexpr(4):
            group_index = k_atom * 4 + group
            start = group_index * sf_vec_size
            maximum = Float32(0.0)
            for element in cutlass.range_constexpr(sf_vec_size):
                value = Float32(source[start + element])
                maximum = cute.arch.fmax(
                    maximum, cute.arch.fmax(value, -value)
                )
            scales_f32[group] = maximum * inv6

        packed_scale = _pack_e4m3(
            scales_f32[0],
            scales_f32[1],
            scales_f32[2],
            scales_f32[3],
        )
        scale_tile = value_scales[
            page, lane, warp, 0, None, k_atom, head
        ]
        scale_word = cute.make_tensor(
            cute.recast_ptr(scale_tile.iterator, dtype=cutlass.Int32),
            cute.make_layout(1),
        )
        scale_word[0] = packed_scale
        scale_u32[0] = packed_scale
        rounded_scales = scale_e4m3.load().to(Float32)

        for group in cutlass.range_constexpr(4):
            group_index = k_atom * 4 + group
            start = group_index * sf_vec_size
            inverse_scale = Float32(1.0) / (
                rounded_scales[group] + Float32(1.0e-30)
            )
            scaled = [Float32(0.0)] * 16
            for element in cutlass.range_constexpr(sf_vec_size):
                scaled[element] = (
                    Float32(source[start + element]) * inverse_scale
                )

            packed_low = _pack_e2m1(
                scaled[0],
                scaled[1],
                scaled[2],
                scaled[3],
                scaled[4],
                scaled[5],
                scaled[6],
                scaled[7],
            )
            packed_high = _pack_e2m1(
                scaled[8],
                scaled[9],
                scaled[10],
                scaled[11],
                scaled[12],
                scaled[13],
                scaled[14],
                scaled[15],
            )
            byte_offset = group_index * (sf_vec_size // 2)
            low_word = cute.make_tensor(
                cute.recast_ptr(
                    destination.iterator + byte_offset,
                    dtype=cutlass.Int32,
                ),
                cute.make_layout(1),
            )
            high_word = cute.make_tensor(
                cute.recast_ptr(
                    destination.iterator + byte_offset + 4,
                    dtype=cutlass.Int32,
                ),
                cute.make_layout(1),
            )
            low_word[0] = packed_low
            high_word[0] = packed_high


@cute.jit
def _launch_key_pages(
    key_pages_ptr: cute.Pointer,
    key_pages_fp4_ptr: cute.Pointer,
    key_scales_ptr: cute.Pointer,
    pages_shape: Tuple[Int32, Int32, Int32, Int32],
    fp4_shape: Tuple[Int32, Int32, Int32, Int32],
    scales_shape: Tuple[
        Int32, Int32, Int32, Int32, Int32, Int32, Int32
    ],
    scales_strides: Tuple[
        Int32, Int32, Int32, Int32, Int32, Int32, Int32
    ],
    heads: cutlass.Constexpr[int],
    pages: Int32,
    rest_k: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    key_pages = cute.make_tensor(
        key_pages_ptr,
        cute.make_ordered_layout(
            pages_shape,
            order=tuple(range(len(pages_shape) - 1, -1, -1)),
        ),
    )
    key_pages_fp4 = cute.make_tensor(
        key_pages_fp4_ptr,
        cute.make_ordered_layout(
            fp4_shape,
            order=tuple(range(len(fp4_shape) - 1, -1, -1)),
        ),
    )
    key_scales = cute.make_tensor(
        key_scales_ptr,
        cute.make_layout(scales_shape, stride=scales_strides),
    )
    _quantize_key_pages_kernel(
        key_pages,
        key_pages_fp4,
        key_scales,
        const_expr(SF_VEC_SIZE),
        const_expr(rest_k),
    ).launch(
        grid=(heads, pages, 1),
        block=(128, 1, 1),
        stream=stream,
    )


@cute.jit
def _launch_value_pages(
    value_pages_ptr: cute.Pointer,
    value_pages_fp4_ptr: cute.Pointer,
    value_scales_ptr: cute.Pointer,
    pages_shape: Tuple[Int32, Int32, Int32, Int32],
    fp4_shape: Tuple[Int32, Int32, Int32, Int32],
    scales_shape: Tuple[
        Int32, Int32, Int32, Int32, Int32, Int32, Int32
    ],
    scales_strides: Tuple[
        Int32, Int32, Int32, Int32, Int32, Int32, Int32
    ],
    heads: cutlass.Constexpr[int],
    pages: Int32,
    page_rest_k: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    value_pages = cute.make_tensor(
        value_pages_ptr,
        cute.make_ordered_layout(
            pages_shape,
            order=tuple(range(len(pages_shape) - 1, -1, -1)),
        ),
    )
    value_pages_fp4 = cute.make_tensor(
        value_pages_fp4_ptr,
        cute.make_ordered_layout(
            fp4_shape,
            order=tuple(range(len(fp4_shape) - 1, -1, -1)),
        ),
    )
    value_scales = cute.make_tensor(
        value_scales_ptr,
        cute.make_layout(scales_shape, stride=scales_strides),
    )
    _quantize_value_pages_kernel(
        value_pages,
        value_pages_fp4,
        value_scales,
        const_expr(SF_VEC_SIZE),
        const_expr(page_rest_k),
    ).launch(
        grid=(heads, pages, 1),
        block=(128, 1, 1),
        stream=stream,
    )


_launch_key_pages.compile_cache = {}
_launch_value_pages.compile_cache = {}


def _validate_pages(pages: torch.Tensor, name: str) -> tuple[int, int]:
    if (
        pages.dtype is not torch.bfloat16
        or not pages.is_cuda
        or not pages.is_contiguous()
        or pages.ndim != 4
    ):
        raise ValueError(
            f"{name} must be contiguous BF16 CUDA with shape "
            "[pages, 128, heads, 128]"
        )
    page_count, page_size, heads, head_dim = pages.shape
    if page_count < 1 or page_size != PAGE_SIZE or head_dim != HEAD_DIM:
        raise ValueError(
            f"{name} must have shape [pages, 128, heads, 128]"
        )
    return page_count, heads


def _allocate_scales(
    page_count: int,
    heads: int,
    rest_m: int,
    rest_k: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    atom_bytes = 512
    m_block_stride = rest_k * atom_bytes
    head_stride = rest_m * m_block_stride
    page_stride = heads * head_stride
    shape = (page_count, 32, 4, rest_m, 4, rest_k, heads)
    strides = (
        page_stride,
        16,
        4,
        m_block_stride,
        1,
        atom_bytes,
        head_stride,
    )
    storage = torch.empty(
        page_count * page_stride,
        dtype=torch.uint8,
        device=device,
    )
    return storage, storage.as_strided(shape, strides)


def _compile_and_launch(
    launcher,
    cache: dict,
    pages: torch.Tensor,
    pages_fp4: torch.Tensor,
    scale_storage: torch.Tensor,
    scales: torch.Tensor,
    *,
    heads: int,
    rest_k: int,
) -> None:
    page_count = pages.shape[0]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    pages_ptr = make_ptr(
        cutlass.BFloat16,
        pages.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    fp4_ptr = make_ptr(
        cutlass.Uint8,
        pages_fp4.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    scales_ptr = make_ptr(
        cutlass.Uint8,
        scale_storage.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    pages_shape = tuple(Int32(value) for value in pages.shape)
    fp4_shape = tuple(Int32(value) for value in pages_fp4.shape)
    scales_shape = tuple(Int32(value) for value in scales.shape)
    scales_strides = tuple(Int32(value) for value in scales.stride())
    compile_args = (
        pages_ptr,
        fp4_ptr,
        scales_ptr,
        pages_shape,
        fp4_shape,
        scales_shape,
        scales_strides,
        heads,
        Int32(page_count),
        rest_k,
        stream,
    )
    cache_key = (
        torch.cuda.current_device(),
        heads,
        rest_k,
    )
    compiled = cache.get(cache_key)
    if compiled is None:
        compiled = cute.compile(launcher, *compile_args)
        cache[cache_key] = compiled
    compiled(
        pages_ptr,
        fp4_ptr,
        scales_ptr,
        pages_shape,
        fp4_shape,
        scales_shape,
        scales_strides,
        Int32(page_count),
        stream,
    )


def quantize_key_pages(
    key_pages: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 K pages into packed E2M1 and E4M3 scales."""
    page_count, heads = _validate_pages(key_pages, "key_pages")
    rest_k = HEAD_DIM // 64
    key_pages_fp4 = torch.empty(
        page_count,
        PAGE_SIZE,
        heads,
        HEAD_DIM // 2,
        dtype=torch.uint8,
        device=key_pages.device,
    )
    scale_storage, key_scales = _allocate_scales(
        page_count,
        heads,
        1,
        rest_k,
        key_pages.device,
    )
    _compile_and_launch(
        _launch_key_pages,
        _launch_key_pages.compile_cache,
        key_pages,
        key_pages_fp4,
        scale_storage,
        key_scales,
        heads=heads,
        rest_k=rest_k,
    )
    return key_pages_fp4, key_scales.view(torch.float8_e4m3fn)


def quantize_value_pages(
    value_pages: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 V pages into packed E2M1 and E4M3 scales."""
    page_count, heads = _validate_pages(value_pages, "value_pages")
    rest_m = HEAD_DIM // 128
    page_rest_k = PAGE_SIZE // 64
    value_pages_fp4 = torch.empty(
        page_count,
        heads,
        HEAD_DIM,
        PAGE_SIZE // 2,
        dtype=torch.uint8,
        device=value_pages.device,
    )
    scale_storage, value_scales = _allocate_scales(
        page_count,
        heads,
        rest_m,
        page_rest_k,
        value_pages.device,
    )
    _compile_and_launch(
        _launch_value_pages,
        _launch_value_pages.compile_cache,
        value_pages,
        value_pages_fp4,
        scale_storage,
        value_scales,
        heads=heads,
        rest_k=page_rest_k,
    )
    return value_pages_fp4, value_scales.view(torch.float8_e4m3fn)
