# Triton implementation of flash cosine similarity attention.
#
# The cosine similarity formulation (l2 normalized queries and keys) makes the
# attention logits bounded in [-scale, scale], which means the online softmax
# machinery (running row maximums, logsumexp) is unnecessary. The forward pass
# simply accumulates unnormalized attention probabilities and row sums, and
# normalizes at the very end - a single division by the row sum.
#
# Modeled after the flash attention triton kernel from
# https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_triton.py
# modernized for triton 3.x and adapted for cosine similarity attention.

from math import ceil

import torch
from torch import Tensor
from torch.autograd import Function

from flash_cosine_sim_attention.flash_cosine_sim_attention import l2norm_tensors

try:
    import triton
    import triton.language as tl
    from importlib.metadata import version as pkg_version

    def _parse_version(v):
        return tuple(int(x) for x in v.split('.')[:2])

    assert _parse_version(pkg_version('triton')) >= (2, 1), 'triton must be version 2.1.0 or above'
except (ImportError, AssertionError):
    triton = None
    tl = None

# helper functions

def exists(v):
    return v is not None

def is_contiguous(x: Tensor):
    return x.stride(-1) == 1

def round_up_multiple(x, multiple):
    return ceil(x / multiple) * multiple

def assert_triton_available():
    if not exists(triton):
        raise ImportError('triton must be installed to use the triton flash cosine sim attention - please run `pip install triton`')

# kernels

@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    Mask,
    Out,
    InvL,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_mb,
    stride_ob,
    stride_oh,
    stride_om,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    HAS_MASK: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    STORE_INV_L: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    FP32_DOTS: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])

    # load queries

    if EVEN_M:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
        else:
            q = tl.load(q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)

    # accumulate attention output and row sums

    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    if HAS_MASK:
        mask_ptrs = Mask + off_b * stride_mb + offs_n

    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)

    for start_n in range(0, end_n, BLOCK_N):
        # load keys and values

        if EVEN_N & EVEN_M:
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn)
                v = tl.load(v_ptrs + start_n * stride_vn)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_d[None, :] < headdim, other=0.0)
                v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=(start_n + offs_n)[:, None] < seqlen_k, other=0.0)
                v = tl.load(v_ptrs + start_n * stride_vn, mask=(start_n + offs_n)[:, None] < seqlen_k, other=0.0)
            else:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                )

        # cosine similarity scores, scaled and offset so that the exponentiation
        # is well conditioned for normalized queries and keys (logits in [-2*scale, 0])

        qk = tl.dot(q, tl.trans(k), input_precision=INPUT_PRECISION)

        if HAS_MASK:
            m = tl.load(mask_ptrs + start_n)
            qk = tl.where(m[None, :], qk, float("-inf"))

        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], qk, float("-inf"))

        if not EVEN_N:
            qk = tl.where((start_n + offs_n)[None, :] < seqlen_k, qk, float("-inf"))

        logits = qk * softmax_scale - softmax_scale

        p = tl.exp(logits)
        l_i += tl.sum(p, 1)

        # for very long sequences, keep the attention probabilities in fp32
        # to avoid rounding error compounding over the many accumulated terms

        if FP32_DOTS:
            acc_o += tl.dot(p, v.to(tl.float32), input_precision=INPUT_PRECISION)
        else:
            acc_o += tl.dot(p.to(v.dtype), v, input_precision=INPUT_PRECISION)

    # normalize by row sums

    inv_l = 1.0 / tl.maximum(l_i, 1e-10)

    out_ptrs = Out + off_b * stride_ob + off_h * stride_oh + (offs_m[:, None] * stride_om + offs_d[None, :])

    acc_o = acc_o * inv_l[:, None]

    if EVEN_M:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o)
        else:
            tl.store(out_ptrs, acc_o, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen_q)
        else:
            tl.store(out_ptrs, acc_o, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim))

    if STORE_INV_L:
        inv_l_ptrs = InvL + off_hb * seqlen_q_rounded + offs_m

        tl.store(inv_l_ptrs, inv_l, mask=offs_m < seqlen_q)

# delta = rowsum(do * o), precomputed for the backward pass

@triton.jit
def _bwd_preprocess_do_o_dot(
    Out,
    DO,
    Delta,
    stride_ob,
    stride_oh,
    stride_om,
    stride_dob,
    stride_doh,
    stride_dom,
    nheads,
    seqlen_q,
    seqlen_q_rounded,
    headdim,
    BLOCK_M: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    o = tl.load(
        Out + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :],
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
        other=0.0,
    ).to(tl.float32)

    do = tl.load(
        DO + off_b * stride_dob + off_h * stride_doh + offs_m[:, None] * stride_dom + offs_d[None, :],
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
        other=0.0,
    ).to(tl.float32)

    delta = tl.sum(o * do, axis=1)

    tl.store(Delta + off_hb * seqlen_q_rounded + offs_m, delta)

@triton.jit
def _bwd_kernel_one_col_block(
    start_n,
    Q,
    K,
    V,
    Mask,
    DO,
    DQ,
    DK,
    DV,
    InvL,
    D,
    softmax_scale,
    off_b,
    off_h,
    stride_qm,
    stride_kn,
    stride_vn,
    stride_mb,
    stride_dom,
    stride_dqm,
    stride_dkn,
    stride_dvn,
    seqlen_q,
    seqlen_k,
    headdim,
    HAS_MASK: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    ATOMIC_ADD_DQ: tl.constexpr,
    ATOMIC_ADD_KV: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    FP32_DOTS: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # causal: only rows after the start of the column block attend to it

    begin_m = 0 if not IS_CAUSAL else ((start_n * BLOCK_N) // BLOCK_M) * BLOCK_M

    offs_qm = begin_m + tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    q_ptrs = Q + offs_qm[:, None] * stride_qm + offs_d[None, :]
    k_ptrs = K + offs_n[:, None] * stride_kn + offs_d[None, :]
    v_ptrs = V + offs_n[:, None] * stride_vn + offs_d[None, :]
    do_ptrs = DO + offs_qm[:, None] * stride_dom + offs_d[None, :]
    dq_ptrs = DQ + offs_qm[:, None] * stride_dqm + offs_d[None, :]

    if HAS_MASK:
        mask_ptrs = Mask + off_b * stride_mb + offs_n

    # keys and values stay in shared memory throughout the row loop

    if EVEN_N:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
        else:
            k = tl.load(k_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
            v = tl.load(v_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
            v = tl.load(v_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
        else:
            k = tl.load(k_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
            v = tl.load(v_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)

    # accumulate dk and dv for this column block in registers

    dv = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)

    num_block_m = tl.cdiv(seqlen_q, BLOCK_M)

    for start_m in range(begin_m, num_block_m * BLOCK_M, BLOCK_M):
        offs_m_curr = start_m + offs_m

        # load queries and output gradients

        if EVEN_M & EVEN_HEADDIM:
            q = tl.load(q_ptrs)
            do = tl.load(do_ptrs)
        else:
            if EVEN_M:
                q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
                do = tl.load(do_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
            elif EVEN_HEADDIM:
                q = tl.load(q_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0)
                do = tl.load(do_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0)
            else:
                q = tl.load(
                    q_ptrs,
                    mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
                do = tl.load(
                    do_ptrs,
                    mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                    other=0.0,
                )

        # recompute the normalized attention probabilities

        qk = tl.dot(q, tl.trans(k), input_precision=INPUT_PRECISION)

        if HAS_MASK:
            m = tl.load(mask_ptrs)
            qk = tl.where(m[None, :], qk, float("-inf"))

        if IS_CAUSAL:
            qk = tl.where(offs_m_curr[:, None] >= offs_n[None, :], qk, float("-inf"))

        if not EVEN_N:
            qk = tl.where(offs_n[None, :] < seqlen_k, qk, float("-inf"))

        if not EVEN_M:
            # out of bounds rows are masked out, as their stored inverse
            # row sums are never computed and the uninitialized memory
            # must not leak into the value and key gradients
            qk = tl.where(offs_m_curr[:, None] < seqlen_q, qk, float("-inf"))

        logits = qk * softmax_scale - softmax_scale

        inv_l_i = tl.load(InvL + offs_m_curr)

        p = tl.exp(logits) * inv_l_i[:, None]

        # dv = p^T @ do

        if FP32_DOTS:
            dv += tl.dot(tl.trans(p), do.to(tl.float32), input_precision=INPUT_PRECISION)
        else:
            dv += tl.dot(tl.trans(p.to(do.dtype)), do, input_precision=INPUT_PRECISION)

        # dp = do @ v^T

        if FP32_DOTS:
            dp = tl.dot(do.to(tl.float32), tl.trans(v.to(tl.float32)), input_precision=INPUT_PRECISION)
        else:
            dp = tl.dot(do, tl.trans(v), input_precision=INPUT_PRECISION)

        # ds = p * (dp - delta)

        Di = tl.load(D + offs_m_curr)

        ds = p * (dp - Di[:, None]) * softmax_scale

        if not EVEN_M:
            # the deltas of out of bounds rows are never computed, so the
            # uninitialized memory must not leak into the gradients
            ds = tl.where(offs_m_curr[:, None] < seqlen_q, ds, 0.0)

        if FP32_DOTS:
            ds32 = ds
        else:
            ds32 = ds.to(q.dtype)

        # dk = ds^T @ q

        if FP32_DOTS:
            dk += tl.dot(tl.trans(ds32), q.to(tl.float32), input_precision=INPUT_PRECISION)
        else:
            dk += tl.dot(tl.trans(ds32), q, input_precision=INPUT_PRECISION)

        # dq: either atomic accumulation across column blocks, or a
        # read-modify-write when a single block handles all column blocks

        if FP32_DOTS:
            dq = tl.dot(ds32, k.to(tl.float32), input_precision=INPUT_PRECISION)
        else:
            dq = tl.dot(ds32, k, input_precision=INPUT_PRECISION)

        if ATOMIC_ADD_DQ:
            if EVEN_M & EVEN_HEADDIM:
                tl.atomic_add(dq_ptrs, dq)
            else:
                if EVEN_M:
                    tl.atomic_add(dq_ptrs, dq, mask=offs_d[None, :] < headdim)
                elif EVEN_HEADDIM:
                    tl.atomic_add(dq_ptrs, dq, mask=offs_m_curr[:, None] < seqlen_q)
                else:
                    tl.atomic_add(
                        dq_ptrs,
                        dq,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                    )
        else:
            if FP32_DOTS:
                k32 = k.to(tl.float32)

            if EVEN_M & EVEN_HEADDIM:
                dq = tl.load(dq_ptrs, eviction_policy="evict_last")
                dq += tl.dot(ds32, k32 if FP32_DOTS else k, input_precision=INPUT_PRECISION)
                tl.store(dq_ptrs, dq, eviction_policy="evict_last")
            else:
                if EVEN_M:
                    dq = tl.load(dq_ptrs, mask=offs_d[None, :] < headdim, other=0.0, eviction_policy="evict_last")
                    dq += tl.dot(ds32, k32 if FP32_DOTS else k, input_precision=INPUT_PRECISION)
                    tl.store(dq_ptrs, dq, mask=offs_d[None, :] < headdim, eviction_policy="evict_last")
                elif EVEN_HEADDIM:
                    dq = tl.load(dq_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0, eviction_policy="evict_last")
                    dq += tl.dot(ds32, k32 if FP32_DOTS else k, input_precision=INPUT_PRECISION)
                    tl.store(dq_ptrs, dq, mask=offs_m_curr[:, None] < seqlen_q, eviction_policy="evict_last")
                else:
                    dq = tl.load(
                        dq_ptrs,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                        other=0.0,
                        eviction_policy="evict_last",
                    )
                    dq += tl.dot(ds32, k32 if FP32_DOTS else k, input_precision=INPUT_PRECISION)
                    tl.store(
                        dq_ptrs,
                        dq,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                        eviction_policy="evict_last",
                    )

        # increment pointers

        dq_ptrs += BLOCK_M * stride_dqm
        q_ptrs += BLOCK_M * stride_qm
        do_ptrs += BLOCK_M * stride_dom

    # write back dk and dv for this column block

    dv_ptrs = DV + offs_n[:, None] * stride_dvn + offs_d[None, :]
    dk_ptrs = DK + offs_n[:, None] * stride_dkn + offs_d[None, :]

    if ATOMIC_ADD_KV:
        if EVEN_N & EVEN_HEADDIM:
            tl.atomic_add(dv_ptrs, dv)
            tl.atomic_add(dk_ptrs, dk)
        else:
            if EVEN_N:
                tl.atomic_add(dv_ptrs, dv, mask=offs_d[None, :] < headdim)
                tl.atomic_add(dk_ptrs, dk, mask=offs_d[None, :] < headdim)
            elif EVEN_HEADDIM:
                tl.atomic_add(dv_ptrs, dv, mask=offs_n[:, None] < seqlen_k)
                tl.atomic_add(dk_ptrs, dk, mask=offs_n[:, None] < seqlen_k)
            else:
                tl.atomic_add(dv_ptrs, dv, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))
                tl.atomic_add(dk_ptrs, dk, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))
    else:
        if EVEN_N & EVEN_HEADDIM:
            tl.store(dv_ptrs, dv)
            tl.store(dk_ptrs, dk)
        else:
            if EVEN_N:
                tl.store(dv_ptrs, dv, mask=offs_d[None, :] < headdim)
                tl.store(dk_ptrs, dk, mask=offs_d[None, :] < headdim)
            elif EVEN_HEADDIM:
                tl.store(dv_ptrs, dv, mask=offs_n[:, None] < seqlen_k)
                tl.store(dk_ptrs, dk, mask=offs_n[:, None] < seqlen_k)
            else:
                tl.store(dv_ptrs, dv, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))
                tl.store(dk_ptrs, dk, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))

@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _bwd_kernel(
    Q,
    K,
    V,
    Mask,
    DO,
    DQ,
    DK,
    DV,
    InvL,
    D,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_mb,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    stride_dkb,
    stride_dkh,
    stride_dkn,
    stride_dvb,
    stride_dvh,
    stride_dvn,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    HAS_MASK: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SEQUENCE_PARALLEL: tl.constexpr,
    ATOMIC_ADD_KV: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    FP32_DOTS: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads

    # offset pointers for batch / head

    Q += off_b * stride_qb + off_h * stride_qh
    K += off_b * stride_kb + off_h * stride_kh
    V += off_b * stride_vb + off_h * stride_vh
    DO += off_b * stride_dob + off_h * stride_doh
    DQ += off_b * stride_dqb + off_h * stride_dqh
    DK += off_b * stride_dkb + off_h * stride_dkh
    DV += off_b * stride_dvb + off_h * stride_dvh

    # pointer to row-wise quantities (inverse row sums, deltas)

    InvL += off_hb * seqlen_q_rounded
    D += off_hb * seqlen_q_rounded

    if SEQUENCE_PARALLEL:
        # one block per column block - the lack of logsumexp bookkeeping means
        # each column block can independently recompute the normalized
        # attention probabilities from the stored inverse row sums

        start_n = tl.program_id(0)

        _bwd_kernel_one_col_block(
            start_n,
            Q,
            K,
            V,
            Mask,
            DO,
            DQ,
            DK,
            DV,
            InvL,
            D,
            softmax_scale,
            off_b,
            off_h,
            stride_qm,
            stride_kn,
            stride_vn,
            stride_mb,
            stride_dom,
            stride_dqm,
            stride_dkn,
            stride_dvn,
            seqlen_q,
            seqlen_k,
            headdim,
            HAS_MASK=HAS_MASK,
            IS_CAUSAL=IS_CAUSAL,
            ATOMIC_ADD_DQ=True,
            ATOMIC_ADD_KV=ATOMIC_ADD_KV,
            INPUT_PRECISION=INPUT_PRECISION,
            FP32_DOTS=FP32_DOTS,
            BLOCK_HEADDIM=BLOCK_HEADDIM,
            EVEN_M=EVEN_M,
            EVEN_N=EVEN_N,
            EVEN_HEADDIM=EVEN_HEADDIM,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
    else:
        # a single block per (batch, head) iterates over all column blocks,
        # accumulating the query gradients in place

        num_block_n = tl.cdiv(seqlen_k, BLOCK_N)

        for start_n in range(num_block_n):
            _bwd_kernel_one_col_block(
                start_n,
                Q,
                K,
                V,
                Mask,
                DO,
                DQ,
                DK,
                DV,
                InvL,
                D,
                softmax_scale,
                off_b,
                off_h,
                stride_qm,
                stride_kn,
                stride_vn,
                stride_mb,
                stride_dom,
                stride_dqm,
                stride_dkn,
                stride_dvn,
                seqlen_q,
                seqlen_k,
                headdim,
                HAS_MASK=HAS_MASK,
                IS_CAUSAL=IS_CAUSAL,
                ATOMIC_ADD_DQ=False,
                ATOMIC_ADD_KV=ATOMIC_ADD_KV,
                INPUT_PRECISION=INPUT_PRECISION,
                FP32_DOTS=FP32_DOTS,
                BLOCK_HEADDIM=BLOCK_HEADDIM,
                EVEN_M=EVEN_M,
                EVEN_N=EVEN_N,
                EVEN_HEADDIM=EVEN_HEADDIM,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )

# shape helpers shared between forward and backward

def _prepare_shapes(q, k, v):
    is_merged_batch_head = q.ndim == 3
    single_head_kv = k.ndim == 3

    if is_merged_batch_head:
        assert k.ndim == 3 and v.ndim == 3, 'if batch and heads are merged for queries, keys and values must also similarly have only 3 dimensions'
        q = q[:, None, ...]

    if single_head_kv:
        # zero stride views, so that every head reads the same keys and values
        k = k[:, None, ...].expand(q.shape[0], q.shape[1], *k.shape[1:])
        v = v[:, None, ...].expand(q.shape[0], q.shape[1], *v.shape[1:])

    return q, k, v, is_merged_batch_head, single_head_kv

# forward entry

def _triton_forward(
    q,
    k,
    v,
    mask,
    scale,
    causal
):
    assert_triton_available()

    # computed from the original inputs, as the contiguity passes inside the
    # function run under no grad and would lose the requires grad flag of
    # non leaf tensors

    should_backwards = q.requires_grad or k.requires_grad or v.requires_grad

    q, k, v, is_merged_batch_head, single_head_kv = _prepare_shapes(q, k, v)

    batch, heads, seqlen_q, d = q.shape
    _, _, seqlen_k, _ = k.shape

    assert d <= 128, 'flash attention only supports head dimensions up to 128'
    assert q.dtype == k.dtype == v.dtype, 'all tensors must have the same type'
    assert q.dtype in [torch.float16, torch.bfloat16, torch.float32], 'only fp16, bf16 and fp32 are supported'
    assert q.is_cuda and k.is_cuda and v.is_cuda, 'tensors must be on the cuda device'

    # single headed keys and values stay as zero stride views, so that every
    # head reads the same keys and values without any extra memory

    q, k, v = map(lambda t: t.contiguous() if not single_head_kv else t, (q, k, v))

    # mask

    has_mask = exists(mask)

    if has_mask:
        mask = mask.contiguous()
        assert mask.ndim == 2 and mask.shape == (batch, seqlen_k), 'mask must be of shape (batch, seqlen_k)'

    # derived constants

    seqlen_q_rounded = round_up_multiple(seqlen_q, 128)

    o = torch.empty_like(q)
    inv_l = torch.zeros((batch * heads, seqlen_q_rounded), device=q.device, dtype=torch.float32) if should_backwards else torch.empty((0,), device=q.device, dtype=torch.float32)

    softmax_scale = float(scale)

    # for very long sequences the per-element rounding of the fp16 attention
    # probabilities would compound over the accumulated terms, so the dot
    # products are carried out in full fp32 precision instead

    use_fp32_dots = seqlen_k >= 65536

    input_precision = "ieee" if (q.dtype == torch.float32 or use_fp32_dots) else None

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)

    if q.dtype == torch.float32 or use_fp32_dots:
        BLOCK_M, BLOCK_N = (64, 64) if d <= 64 else (32, 32)
    else:
        BLOCK_M, BLOCK_N = (128, 128) if d <= 64 else (64, 64)

    num_warps = 4 if d <= 64 else 8

    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * heads)

    mask_strides = (mask.stride(0),) if has_mask else (0,)

    # python int strides are widened to int64 by triton when they exceed the
    # int32 range, so very long sequences (500k to 1 million tokens) do not
    # overflow the pointer arithmetic

    _fwd_kernel[grid](
        q,
        k,
        v,
        mask,
        o,
        inv_l,
        softmax_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        *mask_strides,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        heads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        d,
        has_mask,
        causal,
        should_backwards,
        input_precision,
        use_fp32_dots,
        BLOCK_HEADDIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=1,
    )

    if is_merged_batch_head:
        o = o.squeeze(1)

    return o, inv_l, should_backwards

# backward entry

def _triton_backward(
    do,
    o,
    inv_l,
    q,
    k,
    v,
    mask,
    scale,
    causal,
    sequence_parallel = True
):
    assert_triton_available()

    q, k, v, is_merged_batch_head, single_head_kv = _prepare_shapes(q, k, v)

    if is_merged_batch_head:
        o = o[:, None, ...]
        do = do[:, None, ...]

    batch, heads, seqlen_q, d = q.shape
    _, _, seqlen_k, _ = k.shape

    do, o = map(lambda t: t.contiguous(), (do, o))

    has_mask = exists(mask)

    if has_mask:
        mask = mask.contiguous()

    # derived constants

    seqlen_q_rounded = round_up_multiple(seqlen_q, 128)

    softmax_scale = float(scale)

    # for very long sequences the per-element rounding of the fp16 attention
    # probabilities would compound over the accumulated terms, so the dot
    # products are carried out in full fp32 precision instead

    use_fp32_dots = seqlen_k >= 65536

    input_precision = "ieee" if (q.dtype == torch.float32 or use_fp32_dots) else None

    # deltas

    delta = torch.zeros((batch * heads, seqlen_q_rounded), device=q.device, dtype=torch.float32)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)

    grid_preprocess = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * heads)

    _bwd_preprocess_do_o_dot[grid_preprocess](
        o,
        do,
        delta,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        do.stride(0),
        do.stride(1),
        do.stride(2),
        heads,
        seqlen_q,
        seqlen_q_rounded,
        d,
        BLOCK_M=128,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
    )

    # outputs

    dq_accum = torch.zeros_like(q, dtype=torch.float32)
    dk = torch.zeros_like(k, dtype=torch.float32) if single_head_kv else torch.empty_like(k, dtype=torch.float32)
    dv = torch.zeros_like(v, dtype=torch.float32) if single_head_kv else torch.empty_like(v, dtype=torch.float32)

    if q.dtype == torch.float32 or use_fp32_dots:
        BLOCK_M, BLOCK_N = (64, 64) if d <= 64 else (32, 32)
    else:
        BLOCK_M, BLOCK_N = (128, 128) if d <= 64 else (64, 64)

    num_warps = 4 if d <= 64 else 8

    # parallelize across column blocks, atomically accumulating the query gradients

    grid = lambda META: (triton.cdiv(seqlen_k, META["BLOCK_N"]) if sequence_parallel else 1, batch * heads)

    mask_strides = (mask.stride(0),) if has_mask else (0,)

    # python int strides are widened to int64 by triton when they exceed the
    # int32 range, so very long sequences (500k to 1 million tokens) do not
    # overflow the pointer arithmetic

    _bwd_kernel[grid](
        q,
        k,
        v,
        mask,
        do,
        dq_accum,
        dk,
        dv,
        inv_l,
        delta,
        softmax_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        *mask_strides,
        do.stride(0),
        do.stride(1),
        do.stride(2),
        dq_accum.stride(0),
        dq_accum.stride(1),
        dq_accum.stride(2),
        dk.stride(0),
        dk.stride(1),
        dk.stride(2),
        dv.stride(0),
        dv.stride(1),
        dv.stride(2),
        heads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        d,
        has_mask,
        causal,
        sequence_parallel,
        single_head_kv,
        input_precision,
        use_fp32_dots,
        BLOCK_HEADDIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=1,
    )

    if single_head_kv:
        # each head accumulates into its own slice of the expanded buffer,
        # so the total is the sum over the heads
        dk = dk.sum(dim = 1).to(k.dtype)
        dv = dv.sum(dim = 1).to(v.dtype)
    else:
        dk = dk.to(k.dtype)
        dv = dv.to(v.dtype)

    dq = dq_accum.to(q.dtype)

    if is_merged_batch_head:
        dq = dq.squeeze(1)
        dk = dk.squeeze(1)
        dv = dv.squeeze(1)

    return dq, dk, dv, None

# autograd function

class TritonFlashCosineSimAttention(Function):
    @staticmethod
    def forward(
        ctx,
        q, k, v,
        mask,
        scale,
        causal,
        sequence_parallel
    ):
        o, inv_l, should_backwards = _triton_forward(
            q, k, v,
            mask,
            scale,
            causal
        )

        if not should_backwards:
            return o

        ctx.should_backwards = should_backwards

        ctx.save_for_backward(o, inv_l, q, k, v, mask)

        ctx.params = (
            scale,
            causal,
            sequence_parallel
        )

        return o

    @staticmethod
    def backward(ctx, do):
        assert ctx.should_backwards

        o, inv_l, q, k, v, mask = ctx.saved_tensors

        scale, causal, sequence_parallel = ctx.params

        dq, dk, dv, db = _triton_backward(
            do, o, inv_l,
            q, k, v,
            mask,
            scale,
            causal,
            sequence_parallel
        )

        return dq, dk, dv, None, None, None, None

triton_flash_cosine_sim_attention_cuda = TritonFlashCosineSimAttention.apply

# main entry

def triton_flash_cosine_sim_attention(
    q,
    k,
    v,
    mask = None,
    scale = 8,
    groups = 1,
    causal = False,
    l2norm_qk = True,
    sequence_parallel = True
):
    assert_triton_available()

    if l2norm_qk:
        q, k = l2norm_tensors(q, k, groups = groups)

    return triton_flash_cosine_sim_attention_cuda(
        q, k, v,
        mask,
        scale,
        causal,
        sequence_parallel
    )
