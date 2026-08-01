"""CuTeDSL K/V page quantization."""

from typing import Tuple

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, const_expr
from cutlass.cute.runtime import make_ptr

from .quantize_q_kernel import _pack_e2m1, _pack_e4m3


PAGE_SIZE = 128
HEAD_DIM = 128
SF_VEC_SIZE = 16


@cute.jit
def _resolve_work(
    source_tokens: cute.Tensor,
    destination_pages: cute.Tensor,
    work: Int32,
) -> Tuple[Int32, Int32]:
    """Map a grid slot to the token it reads and the page it writes.

    Without index tensors the grid is the page array itself: work item ``w``
    quantizes source page ``w`` into destination page ``w``. With them the two
    ends are independent, which is what a paged cache needs — the tokens of one
    page are somewhere in a flat activation buffer and the page they belong to
    is wherever the block table says. A negative destination marks a grid slot
    with no work, so the launch shape can be fixed while the batch varies.
    """
    if const_expr(source_tokens is None):
        source_token = work * PAGE_SIZE
    else:
        source_token = source_tokens[work]
    if const_expr(destination_pages is None):
        destination_page = work
    else:
        destination_page = destination_pages[work]
    return source_token, destination_page


@cute.jit
def _rebase(tensor: cute.Tensor, bases, layer: Int32) -> cute.Tensor:
    """The same page layout, based wherever this layer's copy happens to be.

    A caller with one destination passes no table and gets the tensor back
    untouched. A caller with one destination per layer passes their base
    addresses, because independently allocated destinations share only their
    interior layout — their addresses are not a stride apart, so no single
    tensor can reach all of them.
    """
    if const_expr(bases is None):
        return tensor
    return cute.make_tensor(
        cute.make_ptr(
            cutlass.Uint8,
            bases[layer],
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        tensor.layout,
    )


@cute.jit
def _quantize_vector(
    source: cute.Tensor,
    destination: cute.Tensor,
    scales: cute.Tensor,
    page: Int32,
    lane: Int32,
    warp: Int32,
    head: Int32,
    sf_vec_size: cutlass.Constexpr[int],
    rest_k: cutlass.Constexpr[int],
):
    """Quantize one thread's vector to E2M1 with a per-16-element E4M3 scale.

    K and V reduce to this once their page slicing is done: K walks the head
    dimension of one token, V walks one head dimension across the page. The
    scale tile is addressed identically in both, which is what keeps the two
    directions numerically the same.
    """
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
        scale_tile = scales[page, lane, warp, 0, None, k_atom, head]
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
def _quantize_key_pages_kernel(
    key_pages: cute.Tensor,
    key_pages_fp4: cute.Tensor,
    key_scales: cute.Tensor,
    source_tokens: cute.Tensor,
    destination_pages: cute.Tensor,
    packed_bases: cute.Tensor,
    scale_bases: cute.Tensor,
    source_layer_stride: Int32,
    sf_vec_size: cutlass.Constexpr[int],
    rest_k: cutlass.Constexpr[int],
):
    thread, _, _ = cute.arch.thread_idx()
    head, work, layer = cute.arch.block_idx()
    lane = thread % 32
    warp = thread // 32
    sequence = warp * 32 + lane

    source_token, page = _resolve_work(source_tokens, destination_pages, work)
    if page >= 0:
        _quantize_vector(
            key_pages[
                source_token + layer * source_layer_stride, sequence, head, None
            ],
            _rebase(key_pages_fp4, packed_bases, layer)[
                page, sequence, head, None
            ],
            _rebase(key_scales, scale_bases, layer),
            page,
            lane,
            warp,
            head,
            sf_vec_size,
            rest_k,
        )


@cute.kernel
def _quantize_value_pages_kernel(
    value_pages: cute.Tensor,
    value_pages_fp4: cute.Tensor,
    value_scales: cute.Tensor,
    source_tokens: cute.Tensor,
    destination_pages: cute.Tensor,
    packed_bases: cute.Tensor,
    scale_bases: cute.Tensor,
    source_layer_stride: Int32,
    sf_vec_size: cutlass.Constexpr[int],
    page_rest_k: cutlass.Constexpr[int],
):
    thread, _, _ = cute.arch.thread_idx()
    head, work, layer = cute.arch.block_idx()
    lane = thread % 32
    warp = thread // 32
    head_dim = warp * 32 + lane

    source_token, page = _resolve_work(source_tokens, destination_pages, work)
    if page >= 0:
        _quantize_vector(
            value_pages[
                source_token + layer * source_layer_stride, None, head, head_dim
            ],
            _rebase(value_pages_fp4, packed_bases, layer)[
                page, head, head_dim, None
            ],
            _rebase(value_scales, scale_bases, layer),
            page,
            lane,
            warp,
            head,
            sf_vec_size,
            page_rest_k,
        )


@cute.jit
def _index_tensor(pointer, length: Int32):
    if const_expr(pointer is None):
        return None
    return cute.make_tensor(pointer, cute.make_layout(length))


@cute.jit
def _launch_key_pages(
    key_pages_ptr: cute.Pointer,
    key_pages_fp4_ptr: cute.Pointer,
    key_scales_ptr: cute.Pointer,
    source_tokens_ptr,
    destination_pages_ptr,
    packed_bases_ptr,
    scale_bases_ptr,
    pages_shape: Tuple[Int32, Int32, Int32, Int32],
    pages_strides: Tuple[Int32, Int32, Int32, Int32],
    fp4_shape: Tuple[Int64, Int64, Int64, Int64],
    fp4_strides: Tuple[Int64, Int64, Int64, Int64],
    scales_shape: Tuple[
        Int64, Int64, Int64, Int64, Int64, Int64, Int64
    ],
    scales_strides: Tuple[
        Int64, Int64, Int64, Int64, Int64, Int64, Int64
    ],
    heads: cutlass.Constexpr[int],
    work: Int32,
    layers: Int32,
    source_layer_stride: Int32,
    rest_k: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    key_pages = cute.make_tensor(
        key_pages_ptr,
        cute.make_layout(pages_shape, stride=pages_strides),
    )
    key_pages_fp4 = cute.make_tensor(
        key_pages_fp4_ptr,
        cute.make_layout(fp4_shape, stride=fp4_strides),
    )
    key_scales = cute.make_tensor(
        key_scales_ptr,
        cute.make_layout(scales_shape, stride=scales_strides),
    )
    _quantize_key_pages_kernel(
        key_pages,
        key_pages_fp4,
        key_scales,
        _index_tensor(source_tokens_ptr, work),
        _index_tensor(destination_pages_ptr, work),
        _index_tensor(packed_bases_ptr, layers),
        _index_tensor(scale_bases_ptr, layers),
        source_layer_stride,
        const_expr(SF_VEC_SIZE),
        const_expr(rest_k),
    ).launch(
        grid=(heads, work, layers),
        block=(128, 1, 1),
        stream=stream,
    )


@cute.jit
def _launch_value_pages(
    value_pages_ptr: cute.Pointer,
    value_pages_fp4_ptr: cute.Pointer,
    value_scales_ptr: cute.Pointer,
    source_tokens_ptr,
    destination_pages_ptr,
    packed_bases_ptr,
    scale_bases_ptr,
    pages_shape: Tuple[Int32, Int32, Int32, Int32],
    pages_strides: Tuple[Int32, Int32, Int32, Int32],
    fp4_shape: Tuple[Int64, Int64, Int64, Int64],
    fp4_strides: Tuple[Int64, Int64, Int64, Int64],
    scales_shape: Tuple[
        Int64, Int64, Int64, Int64, Int64, Int64, Int64
    ],
    scales_strides: Tuple[
        Int64, Int64, Int64, Int64, Int64, Int64, Int64
    ],
    heads: cutlass.Constexpr[int],
    work: Int32,
    layers: Int32,
    source_layer_stride: Int32,
    page_rest_k: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    value_pages = cute.make_tensor(
        value_pages_ptr,
        cute.make_layout(pages_shape, stride=pages_strides),
    )
    value_pages_fp4 = cute.make_tensor(
        value_pages_fp4_ptr,
        cute.make_layout(fp4_shape, stride=fp4_strides),
    )
    value_scales = cute.make_tensor(
        value_scales_ptr,
        cute.make_layout(scales_shape, stride=scales_strides),
    )
    _quantize_value_pages_kernel(
        value_pages,
        value_pages_fp4,
        value_scales,
        _index_tensor(source_tokens_ptr, work),
        _index_tensor(destination_pages_ptr, work),
        _index_tensor(packed_bases_ptr, layers),
        _index_tensor(scale_bases_ptr, layers),
        source_layer_stride,
        const_expr(SF_VEC_SIZE),
        const_expr(page_rest_k),
    ).launch(
        grid=(heads, work, layers),
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


def _validate_destination(
    pages_fp4: torch.Tensor,
    scales: torch.Tensor,
    *,
    heads: int,
    inner_shape: tuple[int, ...],
    rest_m: int,
    rest_k: int,
) -> None:
    """Reject a destination whose page interior is not the quantizer's layout.

    Only the page pitch is free. Everything inside a page is the layout the
    decode kernel reads back, so a mismatch there is silent corruption.
    """
    if (
        pages_fp4.dtype is not torch.uint8
        or not pages_fp4.is_cuda
        or tuple(pages_fp4.shape[1:]) != inner_shape
        or pages_fp4.stride()[1:] != _dense_strides(inner_shape)
    ):
        raise ValueError(
            f"packed destination must be uint8 CUDA pages of {inner_shape}, "
            "dense within a page"
        )
    expected_scale_shape = (32, 4, rest_m, 4, rest_k, heads)
    atom_bytes = 512
    m_block_stride = rest_k * atom_bytes
    head_stride = rest_m * m_block_stride
    expected_scale_strides = (
        16,
        4,
        m_block_stride,
        1,
        atom_bytes,
        head_stride,
    )
    if (
        scales.element_size() != 1
        or not scales.is_cuda
        or tuple(scales.shape[1:]) != expected_scale_shape
        or scales.stride()[1:] != expected_scale_strides
    ):
        raise ValueError(
            "scale destination must use the quantizer's own scale layout"
        )
    if pages_fp4.shape[0] != scales.shape[0]:
        raise ValueError("packed and scale destinations must have equal pages")


def _dense_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides, running = [], 1
    for extent in reversed(shape):
        strides.append(running)
        running *= extent
    return tuple(reversed(strides))


def _validate_tokens(tokens: torch.Tensor, name: str) -> int:
    if (
        tokens.dtype is not torch.bfloat16
        or not tokens.is_cuda
        or tokens.ndim != 3
        or tokens.shape[2] != HEAD_DIM
        or tokens.stride()[2] != 1
        or tokens.stride()[1] != HEAD_DIM
    ):
        raise ValueError(
            f"{name} must be BF16 CUDA of shape [tokens, heads, 128], dense "
            "within a token"
        )
    return tokens.shape[1]


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


def _int_pointer(tensor: torch.Tensor | None):
    if tensor is None:
        return None
    if (
        tensor.dtype is not torch.int32
        or not tensor.is_cuda
        or not tensor.is_contiguous()
        or tensor.ndim != 1
    ):
        raise ValueError("work indices must be contiguous 1-D int32 CUDA")
    return make_ptr(
        Int32,
        tensor.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=4,
    )


def _base_pointers(bases: torch.Tensor | None):
    """Split the ``[2, layers]`` address table into one pointer per region.

    Addresses rather than a tensor because there is nothing to make a tensor
    of: the destinations are separate allocations. Entries must all be real —
    a table short of a layer is a caller bug that the kernel would turn into a
    null dereference, which is the loud failure this deliberately does not
    soften into a silent skip.
    """
    if bases is None:
        return None, None, 1
    if (
        bases.dtype is not torch.int64
        or not bases.is_cuda
        or not bases.is_contiguous()
        or bases.ndim != 2
        or bases.shape[0] != 2
        or bases.shape[1] < 1
    ):
        raise ValueError(
            "destination_bases must be a contiguous int64 CUDA tensor of "
            "shape [2, layers] holding the packed and scale base addresses"
        )
    layers = bases.shape[1]

    def pointer(row: int):
        return make_ptr(
            Int64,
            bases[row].data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=8,
        )

    return pointer(0), pointer(1), layers


def _compile_and_launch(
    launcher,
    cache: dict,
    pages: torch.Tensor,
    pages_fp4: torch.Tensor,
    scales: torch.Tensor,
    source_tokens: torch.Tensor | None,
    destination_pages: torch.Tensor | None,
    destination_bases: torch.Tensor | None,
    *,
    heads: int,
    rest_k: int,
    work: int,
) -> None:
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
        scales.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    source_ptr = _int_pointer(source_tokens)
    destination_ptr = _int_pointer(destination_pages)
    packed_bases_ptr, scale_bases_ptr, layers = _base_pointers(destination_bases)
    # One destination per layer means one source per layer too, stacked in that
    # order along the token axis. Deriving the stride from the buffer keeps the
    # two ends from disagreeing about how long a layer is.
    if pages.shape[0] % layers:
        raise ValueError(
            f"a source of {pages.shape[0]} tokens does not divide into "
            f"{layers} layers"
        )
    source_layer_stride = pages.shape[0] // layers
    # The source is flat in tokens, and the kernel wants a (page, token, head,
    # dim) view of it. Giving the page axis a one-token stride turns the leading
    # index into a token offset, so a work item can start anywhere rather than
    # only on a page boundary. The axes overlap, which is harmless for a read.
    token_stride = pages.stride()[0]
    pages_shape = tuple(
        Int32(value) for value in (work, PAGE_SIZE) + tuple(pages.shape[1:])
    )
    pages_strides = tuple(
        Int32(value)
        for value in (token_stride, token_stride) + pages.stride()[1:]
    )
    # 64-bit on the destination side only. A page index times a block pitch
    # passes 2^31 at 14563 blocks, which is 2 GiB of a cache that a served
    # model sizes in the hundreds; the wrapped offset then writes somewhere
    # else entirely. The source is a batch of activations or one layer's tail,
    # neither of which comes close.
    fp4_shape = tuple(Int64(value) for value in pages_fp4.shape)
    fp4_strides = tuple(Int64(value) for value in pages_fp4.stride())
    scales_shape = tuple(Int64(value) for value in scales.shape)
    scales_strides = tuple(Int64(value) for value in scales.stride())
    tensor_args = (
        pages_ptr,
        fp4_ptr,
        scales_ptr,
        source_ptr,
        destination_ptr,
        packed_bases_ptr,
        scale_bases_ptr,
    )
    layout_args = (
        pages_shape,
        pages_strides,
        fp4_shape,
        fp4_strides,
        scales_shape,
        scales_strides,
    )
    grid_args = (Int32(work), Int32(layers), Int32(source_layer_stride))
    cache_key = (
        torch.cuda.current_device(),
        heads,
        rest_k,
        source_tokens is None,
        destination_pages is None,
        destination_bases is None,
    )
    compiled = cache.get(cache_key)
    if compiled is None:
        compiled = cute.compile(
            launcher,
            *tensor_args,
            *layout_args,
            heads,
            *grid_args,
            rest_k,
            stream,
        )
        cache[cache_key] = compiled
    compiled(*tensor_args, *layout_args, *grid_args, stream)


def quantize_key_tokens_into(
    key_tokens: torch.Tensor,
    key_pages_fp4: torch.Tensor,
    key_scales: torch.Tensor,
    source_tokens: torch.Tensor,
    destination_pages: torch.Tensor,
    destination_bases: torch.Tensor | None = None,
) -> None:
    """Quantize K pages out of a flat token buffer into a paged destination.

    Work item ``w`` reads the 128 tokens starting at ``source_tokens[w]`` and
    writes page ``destination_pages[w]``; a negative destination skips the item,
    which is how a fixed launch shape covers a varying batch. Both destinations
    may be strided along their page axis, so they can be regions of a cache page
    that carries more than one of them.

    ``destination_bases`` repeats the whole thing once per layer: an int64
    ``[2, layers]`` table of packed and scale base addresses, against a source
    holding the layers back to back in the same order. The destinations keep
    the layout given here and differ only in where they start, because a KV
    cache is allocated one layer at a time and the layers are not a stride
    apart. The indices are shared, so one launch covers every layer.
    """
    heads = _validate_tokens(key_tokens, "key_tokens")
    _validate_destination(
        key_pages_fp4,
        key_scales,
        heads=heads,
        inner_shape=(PAGE_SIZE, heads, HEAD_DIM // 2),
        rest_m=1,
        rest_k=HEAD_DIM // 64,
    )
    work = source_tokens.numel()
    if destination_pages.numel() != work:
        raise ValueError("source_tokens and destination_pages must agree")
    _compile_and_launch(
        _launch_key_pages,
        _launch_key_pages.compile_cache,
        key_tokens,
        key_pages_fp4,
        key_scales,
        source_tokens,
        destination_pages,
        destination_bases,
        heads=heads,
        rest_k=HEAD_DIM // 64,
        work=work,
    )


def quantize_value_tokens_into(
    value_tokens: torch.Tensor,
    value_pages_fp4: torch.Tensor,
    value_scales: torch.Tensor,
    source_tokens: torch.Tensor,
    destination_pages: torch.Tensor,
    destination_bases: torch.Tensor | None = None,
) -> None:
    """Quantize V pages out of a flat token buffer into a paged destination.

    The K counterpart documents the indexing contract.
    """
    heads = _validate_tokens(value_tokens, "value_tokens")
    _validate_destination(
        value_pages_fp4,
        value_scales,
        heads=heads,
        inner_shape=(heads, HEAD_DIM, PAGE_SIZE // 2),
        rest_m=HEAD_DIM // 128,
        rest_k=PAGE_SIZE // 64,
    )
    work = source_tokens.numel()
    if destination_pages.numel() != work:
        raise ValueError("source_tokens and destination_pages must agree")
    _compile_and_launch(
        _launch_value_pages,
        _launch_value_pages.compile_cache,
        value_tokens,
        value_pages_fp4,
        value_scales,
        source_tokens,
        destination_pages,
        destination_bases,
        heads=heads,
        rest_k=PAGE_SIZE // 64,
        work=work,
    )


def quantize_key_pages(
    key_pages: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 K pages into freshly allocated E2M1 and E4M3 scales."""
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
    _, key_scales = _allocate_scales(
        page_count,
        heads,
        1,
        rest_k,
        key_pages.device,
    )
    _compile_and_launch(
        _launch_key_pages,
        _launch_key_pages.compile_cache,
        key_pages.reshape(-1, heads, HEAD_DIM),
        key_pages_fp4,
        key_scales,
        None,
        None,
        None,
        heads=heads,
        rest_k=rest_k,
        work=page_count,
    )
    return key_pages_fp4, key_scales.view(torch.float8_e4m3fn)


def quantize_value_pages(
    value_pages: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 V pages into freshly allocated E2M1 and E4M3 scales."""
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
    _, value_scales = _allocate_scales(
        page_count,
        heads,
        rest_m,
        page_rest_k,
        value_pages.device,
    )
    _compile_and_launch(
        _launch_value_pages,
        _launch_value_pages.compile_cache,
        value_pages.reshape(-1, heads, HEAD_DIM),
        value_pages_fp4,
        value_scales,
        None,
        None,
        None,
        heads=heads,
        rest_k=page_rest_k,
        work=page_count,
    )
    return value_pages_fp4, value_scales.view(torch.float8_e4m3fn)
