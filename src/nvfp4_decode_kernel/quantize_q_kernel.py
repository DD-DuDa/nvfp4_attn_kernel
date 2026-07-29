"""CuTeDSL decode-query quantization."""

from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import T


PAGE_SIZE = 128
SF_VEC_SIZE = 16


@cute.jit
def _pack_e4m3(
    f0: Float32,
    f1: Float32,
    f2: Float32,
    f3: Float32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    packed = llvm.inline_asm(
        T.i32(),
        [
            Float32(f0).ir_value(loc=loc, ip=ip),
            Float32(f1).ir_value(loc=loc, ip=ip),
            Float32(f2).ir_value(loc=loc, ip=ip),
            Float32(f3).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t"
        ".reg .b16 lo;\n\t"
        ".reg .b16 hi;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3;\n\t"
        "mov.b32 $0, {lo, hi};\n\t"
        "}\n",
        "=r,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(packed)


@cute.jit
def _pack_e2m1(
    f0: Float32,
    f1: Float32,
    f2: Float32,
    f3: Float32,
    f4: Float32,
    f5: Float32,
    f6: Float32,
    f7: Float32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    packed = llvm.inline_asm(
        T.i32(),
        [
            Float32(f0).ir_value(loc=loc, ip=ip),
            Float32(f1).ir_value(loc=loc, ip=ip),
            Float32(f2).ir_value(loc=loc, ip=ip),
            Float32(f3).ir_value(loc=loc, ip=ip),
            Float32(f4).ir_value(loc=loc, ip=ip),
            Float32(f5).ir_value(loc=loc, ip=ip),
            Float32(f6).ir_value(loc=loc, ip=ip),
            Float32(f7).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t"
        ".reg .b8 byte0;\n\t"
        ".reg .b8 byte1;\n\t"
        ".reg .b8 byte2;\n\t"
        ".reg .b8 byte3;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 byte0, $2, $1;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 byte1, $4, $3;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 byte2, $6, $5;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 byte3, $8, $7;\n\t"
        "mov.b32 $0, {byte0, byte1, byte2, byte3};\n\t"
        "}\n",
        "=r,f,f,f,f,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(packed)


@cute.kernel
def _quantize_query_kernel(
    query: cute.Tensor,
    row_indices: cute.Tensor,
    query_fp4: cute.Tensor,
    query_scales: cute.Tensor,
    query_padded: cute.Tensor,
    sf_vec_size: cutlass.Constexpr[int],
    rest_k: cutlass.Constexpr[int],
    use_row_indices: cutlass.Constexpr[bool],
    write_padded: cutlass.Constexpr[bool],
    qhead_per_kvhead: cutlass.Constexpr[int],
):
    row, head, _ = cute.arch.block_idx()
    source_row = (
        row_indices[row] if const_expr(use_row_indices) else row
    )
    source = query[source_row, head, None]
    destination = query_fp4[row, 0, head, None]

    if const_expr(write_padded):
        padded = query_padded[row, 0, head, None]
        head_dim = const_expr(rest_k * 4 * sf_vec_size)
        words = const_expr(head_dim // 2)
        padded_u32 = cute.make_tensor(
            cute.recast_ptr(padded.iterator, dtype=cutlass.Int32),
            cute.make_layout(words),
        )
        source_u32 = cute.make_tensor(
            cute.recast_ptr(source.iterator, dtype=cutlass.Int32),
            cute.make_layout(words),
        )
        for word in cutlass.range(words, unroll_full=True):
            padded_u32[word] = source_u32[word]

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
        packed_row = head % qhead_per_kvhead
        scale_m1 = packed_row % 32
        scale_m2 = packed_row // 32
        scale_head = head // qhead_per_kvhead
        scale_tile = query_scales[
            scale_m1, scale_m2, 0, None, k_atom, scale_head, row
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
def _launch_quantize_query(
    query_ptr: cute.Pointer,
    row_indices_ptr: cute.Pointer,
    query_fp4_ptr: cute.Pointer,
    query_scales_ptr: cute.Pointer,
    query_padded_ptr: cute.Pointer,
    query_shape: Tuple[Int32, Int32, Int32],
    query_strides: Tuple[Int32, Int32, Int32],
    row_indices_shape: Tuple[Int32],
    fp4_shape: Tuple[Int32, Int32, Int32, Int32],
    scales_shape: Tuple[
        Int32, Int32, Int32, Int32, Int32, Int32, Int32
    ],
    scales_strides: Tuple[
        Int32, Int32, Int32, Int32, Int32, Int32, Int32
    ],
    padded_shape: Tuple[Int32, Int32, Int32, Int32],
    rows: Int32,
    heads: cutlass.Constexpr[int],
    rest_k: cutlass.Constexpr[int],
    use_row_indices: cutlass.Constexpr[bool],
    write_padded: cutlass.Constexpr[bool],
    qhead_per_kvhead: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    query = cute.make_tensor(
        query_ptr,
        cute.make_layout(query_shape, stride=query_strides),
    )
    row_indices = cute.make_tensor(
        row_indices_ptr,
        cute.make_ordered_layout(row_indices_shape, order=(0,)),
    )
    query_fp4 = cute.make_tensor(
        query_fp4_ptr,
        cute.make_ordered_layout(
            fp4_shape,
            order=tuple(range(len(fp4_shape) - 1, -1, -1)),
        ),
    )
    query_scales = cute.make_tensor(
        query_scales_ptr,
        cute.make_layout(scales_shape, stride=scales_strides),
    )
    query_padded = cute.make_tensor(
        query_padded_ptr,
        cute.make_ordered_layout(
            padded_shape,
            order=tuple(range(len(padded_shape) - 1, -1, -1)),
        ),
    )

    _quantize_query_kernel(
        query,
        row_indices,
        query_fp4,
        query_scales,
        query_padded,
        const_expr(SF_VEC_SIZE),
        const_expr(rest_k),
        const_expr(use_row_indices),
        const_expr(write_padded),
        const_expr(qhead_per_kvhead),
    ).launch(
        grid=(rows, heads, 1),
        block=(1, 1, 1),
        stream=stream,
    )


_launch_quantize_query.compile_cache = {}


def quantize_decode_q_to_padded_fp4(
    query: torch.Tensor,
    query_fp4_out: torch.Tensor,
    query_scales_out: torch.Tensor,
    query_padded_out: Optional[torch.Tensor] = None,
    *,
    row_indices: Optional[torch.Tensor] = None,
    heads_kv: Optional[int] = None,
) -> None:
    """Quantize selected BF16 query rows into preallocated outputs."""
    if query.dtype is not torch.bfloat16 or not query.is_cuda:
        raise ValueError("query must be a BF16 CUDA tensor")
    if query.ndim != 3 or query.stride(-1) != 1:
        raise ValueError("query must have shape [rows, heads, head_dim]")

    source_rows, heads, head_dim = query.shape
    heads_kv = heads if heads_kv is None else heads_kv
    if heads_kv < 1 or heads % heads_kv:
        raise ValueError("query heads must be divisible by heads_kv")
    qhead_per_kvhead = heads // heads_kv
    rows = query_fp4_out.shape[0]
    if head_dim % 64 != 0:
        raise ValueError("query head_dim must be divisible by 64")
    if row_indices is None:
        if source_rows != rows:
            raise ValueError("query rows must match output rows")
    elif (
        row_indices.dtype is not torch.int32
        or not row_indices.is_cuda
        or row_indices.device != query.device
        or not row_indices.is_contiguous()
        or row_indices.shape != (rows,)
    ):
        raise ValueError(
            "row_indices must be contiguous INT32 CUDA with shape [rows]"
        )

    expected_fp4_shape = (
        rows,
        1,
        heads,
        head_dim // 2,
    )
    expected_scales_shape = (
        32,
        4,
        1,
        4,
        head_dim // 64,
        heads_kv,
        rows,
    )
    if (
        tuple(query_fp4_out.shape) != expected_fp4_shape
        or query_fp4_out.dtype is not torch.float4_e2m1fn_x2
        or not query_fp4_out.is_cuda
        or query_fp4_out.device != query.device
        or not query_fp4_out.is_contiguous()
    ):
        raise ValueError(
            f"query_fp4_out must have shape {expected_fp4_shape}"
        )
    if (
        tuple(query_scales_out.shape) != expected_scales_shape
        or query_scales_out.dtype is not torch.uint8
        or not query_scales_out.is_cuda
        or query_scales_out.device != query.device
    ):
        raise ValueError(
            f"query_scales_out must have shape {expected_scales_shape}"
        )

    write_padded = query_padded_out is not None
    if write_padded:
        if (
            query_padded_out.ndim != 4
            or query_padded_out.shape[0] < rows
            or tuple(query_padded_out.shape[1:])
            != (PAGE_SIZE, heads, head_dim)
            or query_padded_out.dtype is not torch.bfloat16
            or not query_padded_out.is_cuda
            or query_padded_out.device != query.device
            or not query_padded_out.is_contiguous()
        ):
            raise ValueError(
                "query_padded_out must have shape "
                f"[at least {rows}, {PAGE_SIZE}, {heads}, {head_dim}]"
            )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    query_ptr = make_ptr(
        cutlass.BFloat16,
        query.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    if row_indices is None:
        row_indices_ptr = make_ptr(
            cutlass.Int32,
            query.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4,
        )
    else:
        row_indices_ptr = make_ptr(
            cutlass.Int32,
            row_indices.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4,
        )
    fp4_ptr = make_ptr(
        cutlass.Uint8,
        query_fp4_out.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    scales_ptr = make_ptr(
        cutlass.Uint8,
        query_scales_out.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    if query_padded_out is None:
        padded_ptr = query_ptr
        padded_shape = (
            Int32(rows),
            Int32(1),
            Int32(heads),
            Int32(head_dim),
        )
    else:
        padded_ptr = make_ptr(
            cutlass.BFloat16,
            query_padded_out.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        padded_shape = tuple(
            Int32(value) for value in query_padded_out.shape
        )

    query_shape = tuple(Int32(value) for value in query.shape)
    query_strides = tuple(Int32(value) for value in query.stride())
    row_indices_shape = (Int32(rows),)
    fp4_shape = tuple(Int32(value) for value in expected_fp4_shape)
    scales_shape = tuple(Int32(value) for value in expected_scales_shape)
    scales_strides = tuple(
        Int32(value) for value in query_scales_out.stride()
    )
    rest_k = head_dim // 64
    use_row_indices = row_indices is not None

    compile_args = (
        query_ptr,
        row_indices_ptr,
        fp4_ptr,
        scales_ptr,
        padded_ptr,
        query_shape,
        query_strides,
        row_indices_shape,
        fp4_shape,
        scales_shape,
        scales_strides,
        padded_shape,
        Int32(rows),
        heads,
        rest_k,
        use_row_indices,
        write_padded,
        qhead_per_kvhead,
        stream,
    )
    compile_key = (
        torch.cuda.current_device(),
        heads,
        rest_k,
        use_row_indices,
        write_padded,
        qhead_per_kvhead,
    )
    compiled = _launch_quantize_query.compile_cache.get(compile_key)
    if compiled is None:
        compiled = cute.compile(_launch_quantize_query, *compile_args)
        _launch_quantize_query.compile_cache[compile_key] = compiled

    compiled(
        query_ptr,
        row_indices_ptr,
        fp4_ptr,
        scales_ptr,
        padded_ptr,
        query_shape,
        query_strides,
        row_indices_shape,
        fp4_shape,
        scales_shape,
        scales_strides,
        padded_shape,
        Int32(rows),
        stream,
    )
