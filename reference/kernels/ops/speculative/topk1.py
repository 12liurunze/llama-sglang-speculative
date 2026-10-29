from __future__ import annotations

import torch
import triton
import triton.language as tl

_DRAFT_TOPK1_BLOCK = 8192


@triton.jit
def _draft_topk1_partial_argmax_kernel(
    logits,
    partial_vals,
    partial_indices,
    logits_row_stride,
    vocab_size: tl.constexpr,
    num_splits: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # int64 row base: row * stride overflows int32 once bs * vocab reaches 2^31.
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    offsets = split * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < vocab_size
    vals = tl.load(
        logits + row * logits_row_stride + offsets,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)
    # Keep NaNs on valid lanes from selecting the masked tail.
    vals = tl.where(vals == vals, vals, -1e30)

    max_val = tl.max(vals, axis=0)
    local_index = tl.argmax(vals, axis=0)
    out_offset = row * num_splits + split
    tl.store(partial_vals + out_offset, max_val)
    tl.store(partial_indices + out_offset, split * BLOCK + local_index)


@triton.jit
def _draft_topk1_finalize_kernel(
    partial_vals,
    partial_indices,
    topk_p,
    topk_index,
    positions,
    draft_tokens,
    draft_tokens_stride,
    draft_token_column,
    num_splits: tl.constexpr,
    WRITE_DRAFT_TOKEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < num_splits
    vals = tl.load(
        partial_vals + row * num_splits + offsets,
        mask=mask,
        other=-float("inf"),
    )

    split = tl.argmax(vals, axis=0)
    index = tl.load(partial_indices + row * num_splits + split).to(tl.int64)
    tl.store(topk_index + row, index)
    tl.store(topk_p + row, 1.0)
    if WRITE_DRAFT_TOKEN:
        tl.store(draft_tokens + row * draft_tokens_stride + draft_token_column, index)

    position = tl.load(positions + row)
    tl.store(positions + row, position + 1)


def draft_topk1_postprocess(
    next_token_logits: torch.Tensor,
    positions: torch.Tensor,
    draft_tokens: torch.Tensor | None = None,
    draft_token_column: int = 0,
):
    """Argmax draft logits for topk=1 and advance positions.

    PyTorch eager argmax reduces each row with too little parallelism for the
    GLM/DSV4 vocab widths in CUDA graph replay. This split reduction exposes
    the vocab dimension across CTAs, then finalizes one token per row.

    If ``draft_tokens`` is given, the finalize kernel also stores the argmax
    into ``draft_tokens[:, draft_token_column]``, mutating the caller-owned
    buffer in place. ``topk_p`` is returned as constant 1.0: topk=1 drafting
    is greedy and the chain probabilities are unused downstream.
    """
    assert next_token_logits.ndim == 2
    assert next_token_logits.stride(1) == 1
    assert positions.ndim == 1
    assert positions.is_contiguous()
    assert positions.shape[0] == next_token_logits.shape[0]
    assert positions.device == next_token_logits.device
    write_draft_token = draft_tokens is not None
    if write_draft_token:
        assert draft_tokens.ndim == 2
        assert draft_tokens.dtype == torch.long
        assert draft_tokens.device == next_token_logits.device
        assert draft_tokens.shape[0] == next_token_logits.shape[0]
        assert draft_tokens.stride(1) == 1
        assert 0 <= draft_token_column < draft_tokens.shape[1]

    bs, vocab_size = next_token_logits.shape
    topk_p = torch.empty((bs, 1), dtype=torch.float32, device=next_token_logits.device)
    topk_index = torch.empty(
        (bs, 1), dtype=torch.int64, device=next_token_logits.device
    )
    if bs == 0:
        return topk_p, topk_index

    block = _DRAFT_TOPK1_BLOCK
    num_splits = triton.cdiv(vocab_size, block)
    partial_vals = torch.empty(
        (bs, num_splits), dtype=torch.float32, device=next_token_logits.device
    )
    partial_indices = torch.empty(
        (bs, num_splits), dtype=torch.int32, device=next_token_logits.device
    )

    _draft_topk1_partial_argmax_kernel[(bs, num_splits)](
        next_token_logits,
        partial_vals,
        partial_indices,
        next_token_logits.stride(0),
        vocab_size,
        num_splits,
        BLOCK=block,
        num_warps=8,
    )
    # Dummy operand for the disabled draft-token slot: the pointer must be
    # valid even though the kernel never dereferences it (gated off by
    # WRITE_DRAFT_TOKEN).
    _draft_topk1_finalize_kernel[(bs,)](
        partial_vals,
        partial_indices,
        topk_p,
        topk_index,
        positions,
        draft_tokens if write_draft_token else topk_index,
        draft_tokens.stride(0) if write_draft_token else 0,
        draft_token_column,
        num_splits,
        WRITE_DRAFT_TOKEN=write_draft_token,
        BLOCK=triton.next_power_of_2(num_splits),
        num_warps=1,
    )
    return topk_p, topk_index


# Top2 probabilities use the full vocabulary normalizer, unlike greedy Top1.
@triton.jit
def _draft_topk2_partial_kernel(X, PV, PI, PL, STRIDE,
                                 V: tl.constexpr, NS: tl.constexpr,
                                 BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    ids = split * BLOCK + tl.arange(0, BLOCK)
    valid = ids < V
    x = tl.load(X + row * STRIDE + ids, valid, other=-float('inf')).to(tl.float32)
    m = tl.max(x, 0)
    # An entirely masked block contributes zero to the finite-row denominator.
    z = tl.sum(tl.exp(x - tl.where(m == -float('inf'), 0., m)), 0)
    i1 = tl.min(tl.where(valid & (x == m), ids, 2147483647), 0)
    rest = tl.where(valid & (ids != i1), x, -float('inf'))
    v2 = tl.max(rest, 0)
    i2 = tl.min(tl.where(valid & (ids != i1) & (rest == v2), ids, 2147483647), 0)
    base = (row * NS + split) * 2
    tl.store(PV + base, m)
    tl.store(PV + base + 1, v2)
    tl.store(PI + base, i1)
    tl.store(PI + base + 1, i2)
    tl.store(PL + row * NS + split, z)


@triton.jit
def _draft_topk2_merge_kernel(PV, PI, PL, P, I,
                               V: tl.constexpr, NS: tl.constexpr,
                               BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    k = tl.arange(0, BLOCK)
    values = tl.load(PV + row * NS * 2 + k, k < NS * 2, other=-float('inf'))
    ids = tl.load(PI + row * NS * 2 + k, k < NS * 2, other=2147483647)
    m = tl.max(values, 0)
    # Each even entry is the corresponding block maximum.
    local_sum = tl.load(PL + row * NS + k // 2, (k < NS * 2) & (k % 2 == 0), other=0.)
    terms = tl.where((k < NS * 2) & (k % 2 == 0),
                     local_sum * tl.exp(values - m), 0.)
    denom = tl.sum(terms, 0)
    i1 = tl.min(tl.where((ids < V) & (values == m), ids, 2147483647), 0)
    v2 = tl.max(tl.where(ids != i1, values, -float('inf')), 0)
    i2 = tl.min(tl.where((ids < V) & (ids != i1) & (values == v2), ids, 2147483647), 0)
    p1 = tl.exp(m - m) / denom
    p2 = tl.exp(v2 - m) / denom
    # CUDA torch.topk's two-element sort reverses equal gathered values.
    # Match the tested reference: first gather increasing IDs, then swap ties.
    swap = p1 == p2
    tl.store(I + row * 2, tl.minimum(tl.where(swap, i2, i1), V - 1))
    tl.store(I + row * 2 + 1, tl.minimum(tl.where(swap, i1, i2), V - 1))
    tl.store(P + row * 2, p1)
    tl.store(P + row * 2 + 1, p2)


def draft_topk2_probs(next_token_logits: torch.Tensor):
    """Full-vocabulary probabilities of Top2; two launches for FP32 CUDA logits.

    Logit ties gather increasing token IDs and reverse equal output values,
    matching the tested CUDA torch.topk two-value sort. Low precision uses the reference
    path because rounding the full softmax can create additional ties.
    No positions or speculative tree state are modified here.
    """
    x = next_token_logits
    if (x.is_cuda and x.dtype == torch.bfloat16 and x.ndim == 2
            and x.stride(1) == 1 and x.shape[1] == 262144):
        return _draft_topk2_bf16_probs(x)
    if (not x.is_cuda or x.dtype != torch.float32 or x.ndim != 2
            or x.stride(1) != 1 or x.shape[1] < 2):
        return torch.topk(torch.softmax(x, dim=-1), 2, dim=-1)
    rows, vocab = x.shape
    p = torch.empty((rows, 2), device=x.device, dtype=x.dtype)
    ids = torch.empty((rows, 2), device=x.device, dtype=torch.int64)
    if rows == 0:
        return p, ids
    block = 4096
    splits = triton.cdiv(vocab, block)
    pv = torch.empty((rows, splits, 2), device=x.device, dtype=torch.float32)
    pi = torch.empty((rows, splits, 2), device=x.device, dtype=torch.int32)
    pl = torch.empty((rows, splits), device=x.device, dtype=torch.float32)
    _draft_topk2_partial_kernel[(rows, splits)](
        x, pv, pi, pl, x.stride(0), vocab, splits, BLOCK=block, num_warps=4)
    _draft_topk2_merge_kernel[(rows,)](
        pv, pi, pl, p, ids, vocab, splits,
        BLOCK=triton.next_power_of_2(splits * 2), num_warps=4)
    return p, ids


# BF16 softmax rounds its output BEFORE TopK. Select on rounded probabilities,
# including ties created by rounding. This path is scoped to Gemma's 262144 vocab.
@triton.jit
def _draft_bf16_norm_partial_kernel(X, M, Z, STRIDE,
                                   V: tl.constexpr, NS: tl.constexpr,
                                   BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    ids = split * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + row * STRIDE + ids, ids < V, other=-float('inf')).to(tl.float32)
    maximum = tl.max(x, 0)
    total = tl.sum(tl.exp(x - tl.where(maximum == -float('inf'), 0., maximum)), 0)
    tl.store(M + row * NS + split, maximum)
    tl.store(Z + row * NS + split, total)


@triton.jit
def _draft_bf16_top2_partial_kernel(X, M, Z, PV, PI, STRIDE,
                                   V: tl.constexpr, NS: tl.constexpr,
                                   NBLOCK: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    k = tl.arange(0, NBLOCK)
    maxima = tl.load(M + row * NS + k, k < NS, other=-float('inf'))
    sums = tl.load(Z + row * NS + k, k < NS, other=0.)
    maximum = tl.max(maxima, 0)
    denominator = tl.sum(sums * tl.exp(maxima - maximum), 0)
    ids = split * BLOCK + tl.arange(0, BLOCK)
    valid = ids < V
    x = tl.load(X + row * STRIDE + ids, valid, other=-float('inf')).to(tl.float32)
    p = (tl.exp(x - maximum) / denominator).to(tl.bfloat16).to(tl.float32)
    p = tl.where(valid, p, -float('inf'))
    p1 = tl.max(p, 0)
    i1 = tl.min(tl.where(valid & (p == p1), ids, 2147483647), 0)
    p2 = tl.max(tl.where(ids != i1, p, -float('inf')), 0)
    i2 = tl.min(tl.where(valid & (ids != i1) & (p == p2), ids, 2147483647), 0)
    base = (row * NS + split) * 2
    tl.store(PV + base, p1)
    tl.store(PV + base + 1, p2)
    tl.store(PI + base, i1)
    tl.store(PI + base + 1, i2)


@triton.jit
def _draft_bf16_top2_merge_kernel(PV, PI, P, I,
                                 V: tl.constexpr, NS: tl.constexpr,
                                 BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    k = tl.arange(0, BLOCK)
    p = tl.load(PV + row * NS * 2 + k, k < NS * 2, other=-float('inf')).to(tl.float32)
    ids = tl.load(PI + row * NS * 2 + k, k < NS * 2, other=2147483647)
    p1 = tl.max(p, 0)
    i1 = tl.min(tl.where((ids < V) & (p == p1), ids, 2147483647), 0)
    p2 = tl.max(tl.where(ids != i1, p, -float('inf')), 0)
    i2 = tl.min(tl.where((ids < V) & (ids != i1) & (p == p2), ids, 2147483647), 0)
    swap = p1 == p2
    tl.store(P + row * 2, p1)
    tl.store(P + row * 2 + 1, p2)
    tl.store(I + row * 2, tl.where(swap, i2, i1))
    tl.store(I + row * 2 + 1, tl.where(swap, i1, i2))


def _draft_topk2_bf16_probs(x):
    rows, vocab = x.shape
    p = torch.empty((rows, 2), device=x.device, dtype=torch.bfloat16)
    ids = torch.empty((rows, 2), device=x.device, dtype=torch.int64)
    if rows == 0:
        return p, ids
    block = 4096
    splits = triton.cdiv(vocab, block)
    maxima = torch.empty((rows, splits), device=x.device, dtype=torch.float32)
    sums = torch.empty_like(maxima)
    pv = torch.empty((rows, splits, 2), device=x.device, dtype=torch.bfloat16)
    pi = torch.empty((rows, splits, 2), device=x.device, dtype=torch.int32)
    _draft_bf16_norm_partial_kernel[(rows, splits)](
        x, maxima, sums, x.stride(0), vocab, splits, BLOCK=block, num_warps=4)
    _draft_bf16_top2_partial_kernel[(rows, splits)](
        x, maxima, sums, pv, pi, x.stride(0), vocab, splits,
        NBLOCK=triton.next_power_of_2(splits), BLOCK=block, num_warps=4)
    _draft_bf16_top2_merge_kernel[(rows,)](
        pv, pi, p, ids, vocab, splits,
        BLOCK=triton.next_power_of_2(splits * 2), num_warps=4)
    return p, ids
