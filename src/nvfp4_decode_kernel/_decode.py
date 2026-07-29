"""Narrow host dispatch for the extracted SM100 FP4 paged-decode kernel."""

from __future__ import annotations

import math
from typing import Any

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from cutlass.cute.runtime import from_dlpack, make_ptr

from .fp4_decode_kernel import FP4DecodeKernel


PAGE_SIZE = 128
HEAD_DIM = 128
_decode_compile_cache: dict[tuple[Any, ...], Any] = {}


def _to_cute_tensor(
    tensor: torch.Tensor,
    *,
    assumed_align: int,
    leading_dim: int,
) -> cute.Tensor:
    result = from_dlpack(
        tensor.detach(),
        assumed_align=assumed_align,
        enable_tvm_ffi=True,
    )
    return result.mark_layout_dynamic(leading_dim=leading_dim)


def _require_cuda_tensor(
    tensor: torch.Tensor,
    name: str,
    *,
    device: torch.device,
) -> None:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")


def _as_e4m3(scales: torch.Tensor, name: str) -> torch.Tensor:
    if scales.dtype is torch.float8_e4m3fn:
        return scales
    if scales.dtype is torch.uint8:
        return scales.view(torch.float8_e4m3fn)
    raise ValueError(f"{name} must contain E4M3 scale-factor bytes")


def _as_scale_bytes(scales: torch.Tensor, name: str) -> torch.Tensor:
    if scales.dtype is torch.uint8:
        return scales
    if scales.dtype is torch.float8_e4m3fn:
        return scales.view(torch.uint8)
    raise ValueError(f"{name} must contain E4M3 scale-factor bytes")


def _page_stride_bytes(pages: torch.Tensor, name: str) -> int:
    """Validate a paged tensor's strides and return its page pitch in bytes.

    Everything below the page axis must be densely packed, because that is the
    shape the quantizer writes and the MMA reads. The page axis itself is free:
    the quantizer's own output packs pages back to back, while a page of a vLLM
    cache is one region of a wider page that also carries the scale factors and
    the other of K/V. TMA only requires that every stride above the contiguous
    axis be a whole number of 16-byte lines.
    """
    itemsize = pages.element_size()
    dense_stride = 1
    for axis in range(pages.ndim - 1, 0, -1):
        if pages.stride(axis) != dense_stride:
            raise ValueError(
                f"{name} must be densely packed within a page; axis {axis} has "
                f"stride {pages.stride(axis)}, expected {dense_stride}"
            )
        dense_stride *= pages.shape[axis]
    page_stride = pages.stride(0)
    if page_stride < dense_stride or (page_stride * itemsize) % 16:
        raise ValueError(
            f"{name} needs a page stride of at least {dense_stride} elements, "
            f"aligned to 16 bytes, got {page_stride}"
        )
    return page_stride * itemsize


def _page_scales_for_kernel(
    scales: torch.Tensor,
    name: str,
) -> torch.Tensor:
    """Expose a quantizer scale slab in the kernel's seven-axis layout."""
    pages, _, _, rest_m, _, rest_k, heads = scales.shape
    atom_bytes = 512
    m_block_stride = rest_k * atom_bytes
    head_stride = rest_m * m_block_stride
    dense_page_stride = heads * head_stride
    # Only the swizzle below the page axis is fixed. The page pitch comes from
    # the caller so that a scale region carved out of a paged cache works, and
    # as_strided keeps that region's storage offset.
    page_stride = scales.stride()[0]
    expected_strides = (
        page_stride,
        16,
        4,
        m_block_stride,
        1,
        atom_bytes,
        head_stride,
    )
    if scales.stride() != expected_strides:
        raise ValueError(
            f"{name} must use the scale layout returned by the K/V quantizer"
        )
    # 512 is the scale-factor atom the SFB TMA descriptor is built around, and
    # the kernel is told the page pitch keeps that alignment.
    if page_stride < dense_page_stride or page_stride % 512:
        raise ValueError(
            f"{name} needs a page stride of at least {dense_page_stride} "
            f"bytes, aligned to 512, got {page_stride}"
        )
    return scales.as_strided(
        (32, 4, rest_m, 4, rest_k, heads, pages),
        (
            16,
            4,
            m_block_stride,
            1,
            atom_bytes,
            head_stride,
            page_stride,
        ),
    )


def _check_device_values(condition: torch.Tensor, message: str) -> None:
    if bool(torch.any(condition).item()):
        raise ValueError(message)


def _compile_decode(
    *,
    query_fp4: torch.Tensor,
    query_scales: torch.Tensor,
    query_padded_bf16: torch.Tensor | None,
    key_pages_fp4: torch.Tensor,
    key_scales: torch.Tensor,
    value_pages_fp4: torch.Tensor,
    value_scales: torch.Tensor,
    fp4_page_table: torch.Tensor,
    seqused_fp4: torch.Tensor,
    output: torch.Tensor,
    softmax_scale: float,
    residual_key_pages_bf16: torch.Tensor | None,
    residual_value_pages_bf16: torch.Tensor | None,
    residual_page_ids: torch.Tensor | None,
    seqused_residual: torch.Tensor | None,
    out_indices: torch.Tensor | None,
) -> Any:
    heads_q = query_fp4.shape[2]
    heads_kv = key_pages_fp4.shape[2]
    qhead_per_kvhead = heads_q // heads_kv
    has_residual = residual_key_pages_bf16 is not None
    cache_key = (
        query_fp4.device.index,
        heads_q,
        heads_kv,
        has_residual,
        out_indices is not None,
    )
    compiled = _decode_compile_cache.get(cache_key)
    if compiled is not None:
        return compiled

    operation = FP4DecodeKernel(
        HEAD_DIM,
        HEAD_DIM,
        qhead_per_kvhead=qhead_per_kvhead,
        is_causal=False,
        is_local=False,
        is_split_kv=False,
        pack_gqa=True,
        m_block_size=PAGE_SIZE,
        n_block_size=PAGE_SIZE,
        is_persistent=not has_residual,
        score_mod=None,
        mask_mod=None,
        has_aux_tensors=False,
        paged_kv_non_tma=False,
        is_varlen_q=False,
        sf_dtype=cutlass.Float8E4M3FN,
        sf_vec_size=16,
        bf16_q_input=False,
        fused_residual_first_block=has_residual,
        residual_source="paged_bf16",
        use_out_indices=out_indices is not None,
        seqlen_q_static_one=True,
    )
    fake_stream = cute.runtime.make_fake_stream()
    q_pointer = make_ptr(
        cutlass.Float4E2M1FN,
        0,
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    k_pointer = make_ptr(
        cutlass.Float4E2M1FN,
        0,
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    v_pointer = make_ptr(
        cutlass.Float4E2M1FN,
        0,
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    output_tensor = _to_cute_tensor(
        output, assumed_align=16, leading_dim=3
    )
    page_table_tensor = _to_cute_tensor(
        fp4_page_table, assumed_align=4, leading_dim=1
    )
    seqused_fp4_tensor = _to_cute_tensor(
        seqused_fp4, assumed_align=4, leading_dim=0
    )
    query_scales_tensor = _to_cute_tensor(
        query_scales, assumed_align=16, leading_dim=3
    )
    key_scales_tensor = _to_cute_tensor(
        key_scales, assumed_align=16, leading_dim=3
    )
    value_scales_tensor = _to_cute_tensor(
        value_scales, assumed_align=16, leading_dim=3
    )
    symbolic_q_shape = tuple(Int32(0) for _ in query_fp4.shape)
    symbolic_k_shape = tuple(Int32(0) for _ in key_pages_fp4.shape)
    symbolic_v_shape = (
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
    )
    compile_args = (
        operation,
        q_pointer,
        k_pointer,
        v_pointer,
        output_tensor,
        None,
        Float32(softmax_scale),
        fake_stream,
        None,
        None,
        None,
        seqused_fp4_tensor,
        page_table_tensor,
        None,
        None,
        None,
        None,
        None,
        query_scales_tensor,
        key_scales_tensor,
        value_scales_tensor,
        symbolic_q_shape,
        symbolic_k_shape,
        symbolic_v_shape,
    )
    compile_kwargs: dict[str, Any] = {
        # Always dynamic, so one compiled kernel serves both a densely packed
        # page array and a page carved out of a wider cache page.
        "k_page_stride": Int32(0),
        "v_page_stride": Int32(0),
        "k_sf_page_stride": Int32(0),
        "v_sf_page_stride": Int32(0),
        "mOutIndices": (
            _to_cute_tensor(
                out_indices,
                assumed_align=4,
                leading_dim=0,
            )
            if out_indices is not None
            else None
        )
    }
    if has_residual:
        assert residual_key_pages_bf16 is not None
        assert residual_value_pages_bf16 is not None
        assert residual_page_ids is not None
        assert seqused_residual is not None
        compile_kwargs.update(
            mResidualQ=_to_cute_tensor(
                query_padded_bf16,
                assumed_align=16,
                leading_dim=3,
            ),
            mResidualK=None,
            mResidualV=None,
            mResidualSeqUsedK=_to_cute_tensor(
                seqused_residual,
                assumed_align=4,
                leading_dim=0,
            ),
            mResidualKCache=_to_cute_tensor(
                residual_key_pages_bf16,
                assumed_align=16,
                leading_dim=3,
            ),
            mResidualVCache=_to_cute_tensor(
                residual_value_pages_bf16,
                assumed_align=16,
                leading_dim=3,
            ),
            mResidualBlockIds=_to_cute_tensor(
                residual_page_ids,
                assumed_align=4,
                leading_dim=0,
            ),
        )
    compiled = cute.compile(
        *compile_args,
        **compile_kwargs,
        options="--enable-tvm-ffi",
    )
    _decode_compile_cache[cache_key] = compiled
    return compiled


def decode_fp4(
    *,
    query_fp4: torch.Tensor,
    query_scales: torch.Tensor,
    query_padded_bf16: torch.Tensor | None,
    key_pages_fp4: torch.Tensor,
    key_scales: torch.Tensor,
    value_pages_fp4: torch.Tensor,
    value_scales: torch.Tensor,
    fp4_page_table: torch.Tensor,
    seqused_fp4: torch.Tensor,
    residual_key_pages_bf16: torch.Tensor | None = None,
    residual_value_pages_bf16: torch.Tensor | None = None,
    residual_page_ids: torch.Tensor | None = None,
    seqused_residual: torch.Tensor | None = None,
    has_bf16: torch.Tensor | None = None,
    softmax_scale: float,
    out: torch.Tensor | None = None,
    out_indices: torch.Tensor | None = None,
    trusted_metadata: bool = False,
) -> torch.Tensor:
    """Validate, compile/cache, and launch FP4 paged decode on SM100."""
    if not torch.cuda.is_available():
        raise RuntimeError("decode_fp4 requires CUDA")
    if not math.isfinite(softmax_scale) or softmax_scale <= 0.0:
        raise ValueError("softmax_scale must be finite and positive")
    if query_fp4.ndim != 4:
        raise ValueError(
            "query_fp4 must have shape [rows, 1, heads_q, 64]"
        )
    device = query_fp4.device
    _require_cuda_tensor(query_fp4, "query_fp4", device=device)
    if torch.cuda.get_device_capability(device) != (10, 0):
        raise RuntimeError("decode_fp4 requires an SM100 CUDA device")
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if query_fp4.dtype is not fp4_dtype or not query_fp4.is_contiguous():
        raise ValueError("query_fp4 must be contiguous packed E2M1 FP4")

    rows, query_length, heads_q, packed_head_dim = query_fp4.shape
    if (
        rows < 1
        or query_length != 1
        or packed_head_dim * 2 != HEAD_DIM
    ):
        raise ValueError(
            "query_fp4 must have shape [rows, 1, heads_q, 64]"
        )
    expected_padded_shape = (rows, PAGE_SIZE, heads_q, HEAD_DIM)

    for tensor, name in (
        (key_pages_fp4, "key_pages_fp4"),
        (value_pages_fp4, "value_pages_fp4"),
        (query_scales, "query_scales"),
        (key_scales, "key_scales"),
        (value_scales, "value_scales"),
        (fp4_page_table, "fp4_page_table"),
        (seqused_fp4, "seqused_fp4"),
    ):
        _require_cuda_tensor(tensor, name, device=device)

    packed_types = (torch.uint8, fp4_dtype)
    if (
        key_pages_fp4.dtype not in packed_types
        or key_pages_fp4.ndim != 4
        or key_pages_fp4.shape[1] != PAGE_SIZE
        or key_pages_fp4.shape[3] * 2 != HEAD_DIM
    ):
        raise ValueError(
            "key_pages_fp4 must be packed FP4 with shape "
            "[pages, 128, heads_kv, 64]"
        )
    page_count, _, heads_kv, _ = key_pages_fp4.shape
    if (
        value_pages_fp4.dtype not in packed_types
        or tuple(value_pages_fp4.shape)
        != (page_count, heads_kv, HEAD_DIM, PAGE_SIZE // 2)
    ):
        raise ValueError(
            "value_pages_fp4 must be packed FP4 with shape "
            "[pages, heads_kv, 128, 64]"
        )
    _page_stride_bytes(key_pages_fp4, "key_pages_fp4")
    _page_stride_bytes(value_pages_fp4, "value_pages_fp4")
    if heads_kv < 1 or heads_q % heads_kv:
        raise ValueError("heads_q must be divisible by heads_kv")
    qhead_per_kvhead = heads_q // heads_kv
    if PAGE_SIZE % qhead_per_kvhead:
        raise ValueError(
            "128 must be divisible by the query-heads-per-KV-head ratio"
        )

    expected_query_sf = (32, 4, 1, 4, 2, heads_kv, rows)
    expected_page_sf = (page_count, 32, 4, 1, 4, 2, heads_kv)
    if tuple(query_scales.shape) != expected_query_sf:
        raise ValueError(
            f"query_scales must have shape {expected_query_sf}"
        )
    expected_query_sf_strides = (
        16,
        4,
        heads_q * 1024,
        1,
        512,
        1024,
        heads_q * 1024,
    )
    if query_scales.stride() != expected_query_sf_strides:
        raise ValueError(
            "query_scales must use the layout returned by quantize_query"
        )
    if tuple(key_scales.shape) != expected_page_sf:
        raise ValueError(f"key_scales must have shape {expected_page_sf}")
    if tuple(value_scales.shape) != expected_page_sf:
        raise ValueError(
            f"value_scales must have shape {expected_page_sf}"
        )
    query_scales = _as_scale_bytes(query_scales, "query_scales")
    key_scales = _as_e4m3(key_scales, "key_scales")
    value_scales = _as_e4m3(value_scales, "value_scales")
    if query_scales.storage_offset() != 0:
        raise ValueError("query_scales must start at storage offset zero")
    key_scales = _page_scales_for_kernel(key_scales, "key_scales")
    value_scales = _page_scales_for_kernel(value_scales, "value_scales")

    if (
        fp4_page_table.dtype is not torch.int32
        or fp4_page_table.ndim != 2
        or fp4_page_table.shape[0] != rows
        or fp4_page_table.stride(-1) != 1
    ):
        raise ValueError(
            "fp4_page_table must be row-major INT32 with shape "
            "[rows, max_pages]"
        )
    if (
        seqused_fp4.dtype is not torch.int32
        or tuple(seqused_fp4.shape) != (rows,)
        or not seqused_fp4.is_contiguous()
    ):
        raise ValueError(
            "seqused_fp4 must be contiguous INT32 with shape [rows]"
        )
    max_fp4_tokens = fp4_page_table.shape[1] * PAGE_SIZE
    # The tagged-metadata producer owns consumed-prefix ID validation. Do not
    # scan unused columns here: persistent vLLM scratch may retain poison or
    # stale IDs beyond the prefix selected by seqused_fp4.
    if not trusted_metadata:
        _check_device_values(
            (seqused_fp4 < 0)
            | (seqused_fp4 > max_fp4_tokens)
            | (seqused_fp4 % PAGE_SIZE != 0),
            "seqused_fp4 values must be page-aligned and in "
            f"[0, {max_fp4_tokens}]",
        )

    residual_values = (
        residual_key_pages_bf16,
        residual_value_pages_bf16,
        residual_page_ids,
        seqused_residual,
    )
    has_residual = any(value is not None for value in residual_values)
    if has_residual and not all(
        value is not None for value in residual_values
    ):
        raise ValueError(
            "residual_key_pages_bf16, residual_value_pages_bf16, "
            "residual_page_ids, and seqused_residual must be provided together"
        )
    if has_residual:
        if query_padded_bf16 is None:
            raise ValueError(
                "pre-quantized query with a BF16 residual requires "
                "query_padded_bf16 support; use the BF16 query path"
            )
        _require_cuda_tensor(
            query_padded_bf16, "query_padded_bf16", device=device
        )
        if (
            query_padded_bf16.dtype is not torch.bfloat16
            or tuple(query_padded_bf16.shape) != expected_padded_shape
            or not query_padded_bf16.is_contiguous()
        ):
            raise ValueError(
                f"query_padded_bf16 must be contiguous BF16 with shape "
                f"{expected_padded_shape}"
            )
        assert residual_key_pages_bf16 is not None
        assert residual_value_pages_bf16 is not None
        assert residual_page_ids is not None
        assert seqused_residual is not None
        for tensor, name in (
            (residual_key_pages_bf16, "residual_key_pages_bf16"),
            (residual_value_pages_bf16, "residual_value_pages_bf16"),
            (residual_page_ids, "residual_page_ids"),
            (seqused_residual, "seqused_residual"),
        ):
            _require_cuda_tensor(tensor, name, device=device)
        if residual_key_pages_bf16.ndim != 4:
            raise ValueError(
                "residual_key_pages_bf16 must be contiguous BF16 with "
                "shape [pages, 128, heads_kv, 128]"
            )
        residual_pages = residual_key_pages_bf16.shape[0]
        expected_residual_shape = (
            residual_pages,
            PAGE_SIZE,
            heads_kv,
            HEAD_DIM,
        )
        if (
            residual_key_pages_bf16.dtype is not torch.bfloat16
            or tuple(residual_key_pages_bf16.shape)
            != expected_residual_shape
            or not residual_key_pages_bf16.is_contiguous()
        ):
            raise ValueError(
                "residual_key_pages_bf16 must be contiguous BF16 with "
                "shape [pages, 128, heads_kv, 128]"
            )
        if (
            residual_value_pages_bf16.dtype is not torch.bfloat16
            or tuple(residual_value_pages_bf16.shape)
            != expected_residual_shape
            or not residual_value_pages_bf16.is_contiguous()
        ):
            raise ValueError(
                "residual_value_pages_bf16 must match the residual K cache"
            )
        if (
            residual_page_ids.dtype is not torch.int32
            or tuple(residual_page_ids.shape) != (rows,)
            or not residual_page_ids.is_contiguous()
        ):
            raise ValueError(
                "residual_page_ids must be contiguous INT32 with shape [rows]"
            )
        if (
            seqused_residual.dtype is not torch.int32
            or tuple(seqused_residual.shape) != (rows,)
            or not seqused_residual.is_contiguous()
        ):
            raise ValueError(
                "seqused_residual must be contiguous INT32 with shape [rows]"
            )
        if not trusted_metadata:
            _check_device_values(
                (residual_page_ids < 0)
                | (residual_page_ids >= residual_pages),
                "residual_page_ids contains an out-of-range physical page ID",
            )
            _check_device_values(
                (seqused_residual < 0)
                | (seqused_residual > PAGE_SIZE),
                "seqused_residual values must be in [0, 128]",
            )
        if has_bf16 is not None:
            _require_cuda_tensor(has_bf16, "has_bf16", device=device)
            if (
                has_bf16.dtype is not torch.bool
                or tuple(has_bf16.shape) != (rows,)
                or not has_bf16.is_contiguous()
            ):
                raise ValueError(
                    "has_bf16 must be contiguous BOOL with shape [rows]"
                )
            if not trusted_metadata:
                _check_device_values(
                    has_bf16 != (seqused_residual > 0),
                    "has_bf16 must agree with seqused_residual > 0",
                )
    elif has_bf16 is not None:
        raise ValueError(
            "has_bf16 requires the BF16 residual cache arguments"
        )

    packed_query_scales = query_scales
    use_out_indices = out is not None or out_indices is not None
    if use_out_indices and (out is None or out_indices is None):
        raise ValueError("out and out_indices must be provided together")
    if out is not None and out_indices is not None:
        _require_cuda_tensor(out, "out", device=device)
        _require_cuda_tensor(out_indices, "out_indices", device=device)
        if (
            out.dtype is not torch.bfloat16
            or out.ndim != 3
            or tuple(out.shape[1:]) != (heads_q, HEAD_DIM)
            or not out.is_contiguous()
        ):
            raise ValueError(
                "out must be contiguous BF16 with shape "
                "[output_rows, heads_q, 128]"
            )
        if (
            out_indices.dtype is not torch.int32
            or tuple(out_indices.shape) != (rows,)
            or not out_indices.is_contiguous()
        ):
            raise ValueError(
                "out_indices must be contiguous INT32 with shape [rows]"
            )
        if not trusted_metadata:
            _check_device_values(
                (out_indices < 0) | (out_indices >= out.shape[0]),
                "out_indices contains an out-of-range output row",
            )
        output_4d = out.unsqueeze(1)
    else:
        output_4d = torch.empty(
            rows,
            1,
            heads_q,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
    with torch.cuda.device(device):
        compiled = _compile_decode(
            query_fp4=query_fp4,
            query_scales=packed_query_scales,
            query_padded_bf16=query_padded_bf16,
            key_pages_fp4=key_pages_fp4,
            key_scales=key_scales,
            value_pages_fp4=value_pages_fp4,
            value_scales=value_scales,
            fp4_page_table=fp4_page_table,
            seqused_fp4=seqused_fp4,
            output=output_4d,
            softmax_scale=softmax_scale,
            residual_key_pages_bf16=residual_key_pages_bf16,
            residual_value_pages_bf16=residual_value_pages_bf16,
            residual_page_ids=residual_page_ids,
            seqused_residual=seqused_residual,
            out_indices=out_indices,
        )
        stream = cuda.CUstream(
            torch.cuda.current_stream(device).cuda_stream
        )
        q_pointer = make_ptr(
            cutlass.Float4E2M1FN,
            query_fp4.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        k_pointer = make_ptr(
            cutlass.Float4E2M1FN,
            key_pages_fp4.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        v_pointer = make_ptr(
            cutlass.Float4E2M1FN,
            value_pages_fp4.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        call_args = (
            q_pointer,
            k_pointer,
            v_pointer,
            output_4d,
            None,
            softmax_scale,
            stream,
            None,
            None,
            None,
            seqused_fp4,
            fp4_page_table,
            None,
            None,
            None,
            None,
            None,
            packed_query_scales,
            key_scales,
            value_scales,
            (
                rows,
                1,
                heads_q,
                HEAD_DIM,
            ),
            (
                page_count,
                PAGE_SIZE,
                heads_kv,
                HEAD_DIM,
            ),
            (
                page_count,
                PAGE_SIZE,
                heads_kv,
                HEAD_DIM,
            ),
        )
        # Packed pitches scale by two because one byte holds two FP4. The scale
        # slabs are byte-per-element, and _page_scales_for_kernel moved their
        # page axis last.
        call_kwargs: dict[str, Any] = {
            "k_page_stride": _page_stride_bytes(
                key_pages_fp4, "key_pages_fp4"
            ) * 2,
            "v_page_stride": _page_stride_bytes(
                value_pages_fp4, "value_pages_fp4"
            ) * 2,
            "k_sf_page_stride": key_scales.stride()[-1],
            "v_sf_page_stride": value_scales.stride()[-1],
            "mOutIndices": out_indices,
        }
        if has_residual:
            call_kwargs.update(
                mResidualQ=query_padded_bf16,
                mResidualK=None,
                mResidualV=None,
                mResidualSeqUsedK=seqused_residual,
                mResidualKCache=residual_key_pages_bf16,
                mResidualVCache=residual_value_pages_bf16,
                mResidualBlockIds=residual_page_ids,
            )
        compiled(*call_args, **call_kwargs)
    if out is not None:
        return out
    return output_4d[:, 0]
