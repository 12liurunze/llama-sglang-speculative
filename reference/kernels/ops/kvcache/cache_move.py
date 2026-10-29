import torch
import triton
import triton.language as tl

from sglang.srt.utils import is_cpu

_is_cpu = is_cpu()

if _is_cpu:
    from sgl_kernel import copy_all_layer_kv_cache_cpu


@triton.jit
def set_kv_buffer_prefix_valid_tiled(
    src_k_ptr,
    src_v_ptr,
    dst_k_ptr,
    dst_v_ptr,
    loc_2d_ptr,
    commit_len_ptr,
    src_k_row_stride,
    src_v_row_stride,
    dst_k_row_stride,
    dst_v_row_stride,
    block_size,
    ROW_BYTES: tl.constexpr,
    BYTES_PER_TILE: tl.constexpr,
):
    bid = tl.program_id(0)
    row = tl.program_id(1)
    tid = tl.program_id(2)

    commit_len = tl.load(commit_len_ptr + bid)
    if row >= commit_len:
        return

    byte_off = tid * BYTES_PER_TILE + tl.arange(0, BYTES_PER_TILE)
    mask_byte = byte_off < ROW_BYTES
    tl.multiple_of(byte_off, 16)

    loc = tl.load(loc_2d_ptr + bid * block_size + row)
    src_row = bid * block_size + row

    src_k_ptr = tl.cast(src_k_ptr, tl.pointer_type(tl.uint8))
    src_v_ptr = tl.cast(src_v_ptr, tl.pointer_type(tl.uint8))
    dst_k_ptr = tl.cast(dst_k_ptr, tl.pointer_type(tl.uint8))
    dst_v_ptr = tl.cast(dst_v_ptr, tl.pointer_type(tl.uint8))

    src_k_row_ptr = src_k_ptr + src_row * src_k_row_stride + byte_off
    src_v_row_ptr = src_v_ptr + src_row * src_v_row_stride + byte_off
    dst_k_row_ptr = dst_k_ptr + loc * dst_k_row_stride + byte_off
    dst_v_row_ptr = dst_v_ptr + loc * dst_v_row_stride + byte_off

    k_val = tl.load(src_k_row_ptr, mask=mask_byte, other=0)
    v_val = tl.load(src_v_row_ptr, mask=mask_byte, other=0)
    tl.store(dst_k_row_ptr, k_val, mask=mask_byte)
    tl.store(dst_v_row_ptr, v_val, mask=mask_byte)


@triton.jit
def copy_all_layer_kv_cache_tiled(
    data_ptrs, strides, tgt_loc_ptr, src_loc_ptr, num_locs,
    num_locs_upper: tl.constexpr, BYTES_PER_TILE: tl.constexpr,
):
    # Every byte tile owns ALL dependent rows, retaining snapshot-copy semantics.
    bid = tl.program_id(0)
    tid = tl.program_id(1)
    stride = tl.load(strides + bid)
    address = tl.load(data_ptrs + bid)
    loc_idx = tl.arange(0, num_locs_upper)
    valid = loc_idx < num_locs
    src = tl.load(src_loc_ptr + loc_idx, mask=valid, other=0)
    tgt = tl.load(tgt_loc_ptr + loc_idx, mask=valid, other=0)
    if True:
        valid = valid & (src != tgt)
    # Integer bit copies: no floating conversion or arithmetic on K/V values.
    # Runtime alignment check keeps generic unaligned rows on the original path.
    if (stride % 4 == 0) & (address % 4 == 0):
        words = tid * (BYTES_PER_TILE // 4) + tl.arange(0, BYTES_PER_TILE // 4)
        word_stride = stride // 4
        base32 = tl.cast(address, tl.pointer_type(tl.uint32))
        mask32 = valid[:, None] & (words[None, :] < word_stride)
        src32 = base32 + src[:, None] * word_stride + words[None, :]
        dst32 = base32 + tgt[:, None] * word_stride + words[None, :]
        values32 = tl.load(src32, mask=mask32, other=0)
        tl.debug_barrier()
        tl.store(dst32, values32, mask=mask32)
    else:
        byte_off = tid * BYTES_PER_TILE + tl.arange(0, BYTES_PER_TILE)
        base8 = tl.cast(address, tl.pointer_type(tl.uint8))
        mask8 = valid[:, None] & (byte_off[None, :] < stride)
        src8 = base8 + src[:, None] * stride + byte_off[None, :]
        dst8 = base8 + tgt[:, None] * stride + byte_off[None, :]
        values8 = tl.load(src8, mask=mask8, other=0)
        tl.debug_barrier()
        tl.store(dst8, values8, mask=mask8)


def copy_all_layer_kv_cache_func(
    data_ptrs: torch.Tensor,
    strides: torch.Tensor,
    tgt_loc: torch.Tensor,
    src_loc: torch.Tensor,
    num_locs: int,
    num_locs_upper: int,
    kv_copy_config: dict,
):
    if _is_cpu:
        copy_all_layer_kv_cache_cpu(
            data_ptrs,
            strides,
            tgt_loc[:num_locs],
            src_loc[:num_locs],
        )
        return
    if 1 <= num_locs <= 256:
        # Keep all source rows within the same CTA to preserve in-place dependencies.
        tile = 128
        kv_copy_config = dict(kv_copy_config, bytes_per_tile=tile,
                             byte_tiles=(kv_copy_config["byte_tiles"] * kv_copy_config["bytes_per_tile"] + tile - 1) // tile,
                             num_warps=8)
    grid = (data_ptrs.numel(), kv_copy_config["byte_tiles"])
    copy_all_layer_kv_cache_tiled[grid](
        data_ptrs,
        strides,
        tgt_loc,
        src_loc,
        num_locs,
        num_locs_upper,
        BYTES_PER_TILE=kv_copy_config["bytes_per_tile"],
        num_warps=kv_copy_config["num_warps"],
        num_stages=2,
    )
