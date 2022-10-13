import os
import math
import importlib
from functools import partial, wraps

import torch
from torch import einsum
import torch.nn.functional as F
from torch.autograd import Function

exec(open(os.path.dirname(os.path.abspath(__file__)) + '/version.py').read())

# try to import cuda

try:
    cuda_pkg = importlib.import_module(__cuda_pkg_name__)

    forward = cuda_pkg.forward
    backward = cuda_pkg.backward
    debug = cuda_pkg.debug

except ImportError:
    print('CUDA extension for flash-cosine-sim-attention was not compiled correctly - please run `pip install flash-cosine-sim-attention --force-reinstall --no-cache-dir`')

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def divisible_by(numer, denom):
    return (numer % denom) == 0

def l2norm_cpu(t):
    eps = 1e-12 if t.dtype == torch.float32 else 1e-3
    norm = t.norm(dim = -1)
    norm_clamped = torch.where(norm > eps, norm, eps)
    return t / norm_clamped[..., None]

def l2norm(t):
    if t.data.is_cuda:
        return F.normalize(t, dim = -1)

    return l2norm_cpu(t)

def l2norm_tensors(*tensors):
    assert len(tensors) > 0
    dtype = tensors[0].dtype

    tensors = tuple(map(l2norm, tensors))
    tensors = tuple(map(lambda t: t.type(dtype), tensors))
    return tensors

# original cosine sim attention

# b - batch
# h - heads
# i - src sequence length
# j - target sequence length
# d - feature dimension

def plain_cosine_sim_attention(
    q,
    k,
    v,
    mask = None,
    attn_bias = None,
    scale = 8,
    causal = False,
    l2norm_qk = True,
    attn_bias_batch_dim = False

):
    assert not (causal and exists(mask)), 'mask should not be supplied if causality is needed'

    is_merged_batch_heads_query = q.ndim == 3
    single_head_kv = k.ndim == 3

    if is_merged_batch_heads_query:
        assert k.ndim == 3 and v.ndim ==3, 'if batch and heads are merged for queries, keys and values must also similarly have only 3 dimensions'

        attn_bias_batch_dim = True
        q = q[:, None, ...]

    if l2norm_qk:
        q, k = l2norm_tensors(q, k)

    kv_einsum_eq = 'b j d' if single_head_kv else 'b h j d'
    sim = einsum(f'b h i d, {kv_einsum_eq} -> b h i j', q, k)
    sim = sim * scale

    if exists(attn_bias):
        attn_bias = attn_bias.unsqueeze(1 if attn_bias_batch_dim else 0)
        sim = sim + attn_bias

    mask_value = -torch.finfo(sim.dtype).max

    if causal:
        i, j = sim.shape[-2:]
        causal_mask = torch.ones((i, j), device = q.device, dtype = torch.bool).triu(j - i + 1)
        sim = sim.masked_fill(causal_mask, mask_value)

    if exists(mask):
        sim = sim.masked_fill(~mask[:, None, None, :], mask_value)

    attn = sim.softmax(dim = -1)
    out = einsum(f'b h i j, {kv_einsum_eq} -> b h i d', attn, v)

    if is_merged_batch_heads_query:
        out = out.squeeze(1)

    return out

# cpu forwards

def flash_cosine_sim_attention_cpu(
    q, k, v,
    mask,
    attn_bias,
    scale,
    causal,
    attn_bias_batch_dim,
    row_tile_size = 512,
    col_tile_size = 512
):
    needs_backwards = any([exists(t) and t.requires_grad for t in (q, k, v, attn_bias)])

    assert not needs_backwards, 'cpu version does not support backwards'
    assert not (causal and exists(mask)), 'mask should not be supplied if causality is needed'

    dtype = q.dtype
    q, k, v = q.float(), k.float(), v.float()

    is_merged_batch_heads_query = q.ndim == 3
    single_head_kv = k.ndim == 3

    shape = q.shape
    col_seq_len = k.shape[-2]
    row_seq_len = q.shape[-2]
    seq_len_diff = col_seq_len - row_seq_len
    row_tiles = math.ceil(row_seq_len / row_tile_size)
    col_tiles = math.ceil(col_seq_len / col_tile_size)
    max_neg_value = -torch.finfo(q.dtype).max

    if is_merged_batch_heads_query:
        assert k.ndim == 3 and v.ndim ==3, 'if batch and heads are merged for queries, keys and values must also similarly have only 3 dimensions'

        attn_bias_batch_dim = True
        q = q.unsqueeze(1)

    if exists(attn_bias):
        attn_bias = attn_bias.unsqueeze(1 if attn_bias_batch_dim else 0)

    kv_einsum_eq = 'b j d' if single_head_kv else 'b h j d'

    # loop over rows and columns

    o = torch.zeros_like(q)
    l = torch.zeros((*q.shape[:-1], 1))

    # prepare mask

    if not exists(mask):
        mask = (None,) * col_tiles
    else:
        mask = mask[:, None, None, :]
        mask = mask.split(col_tile_size, dim = -1)

    if not exists(attn_bias):
        attn_bias = (None,) * row_tiles
    else:
        attn_bias = attn_bias.split(row_tile_size, dim = -2)

    row_splits = zip(
        q.split(row_tile_size, dim = -2),
        o.split(row_tile_size, dim = -2),
        l.split(row_tile_size, dim = -2),
        attn_bias
    )

    for ind, (qc, oc, lc, bc) in enumerate(row_splits):
        row_chunk_size = qc.shape[-2]
        q_start_index = ind * row_tile_size + seq_len_diff

        if not exists(bc):
            bc = (None,) * col_tiles
        else:
            bc = bc.split(col_tile_size, dim = -1)

        col_splits = zip(
            k.split(col_tile_size, dim = -2),
            v.split(col_tile_size, dim = -2),
            mask,
            bc
        )

        for k_ind, (kc, vc, maskc, bias) in enumerate(col_splits):
            col_chunk_size = kc.shape[-2]
            k_start_index = k_ind * col_tile_size

            if causal and q_start_index >= (k_start_index + col_tile_size - 1):
                continue

            attn_weights = einsum(f'b h i d, {kv_einsum_eq} -> b h i j', qc, kc) * scale

            if exists(bias):
                attn_weights += bias

            if exists(maskc):
                attn_weights.masked_fill_(~maskc, max_neg_value)

            if causal and q_start_index < (k_start_index + col_tile_size - 1):
                causal_mask = torch.ones((row_chunk_size, col_chunk_size), dtype = torch.bool).triu(q_start_index - k_start_index + 1)
                attn_weights.masked_fill_(causal_mask, max_neg_value)

            exp_weights = torch.exp(attn_weights - scale)

            if exists(maskc):
                exp_weights.masked_fill_(~maskc, 0.)

            exp_values = einsum(f'b h i j, {kv_einsum_eq} -> b h i d', exp_weights, vc)

            oc.add_(exp_values)
            lc.add_(exp_weights.sum(dim = -1, keepdim = True))
    
    o.div_(l.clamp(min = 1e-12))
    return o.reshape(shape).type(dtype)

# main class

class FlashCosineSimAttention(Function):
    @staticmethod
    def forward(
        ctx,
        q, k, v,
        mask,
        attn_bias,
        scale,
        causal,
        attn_bias_batch_dim
    ):
        o, inv_l, should_backwards = forward(
            q, k, v,
            mask,
            attn_bias,
            attn_bias_batch_dim,
            scale,
            causal
        )

        if not should_backwards:
            return o

        ctx.should_backwards = should_backwards

        ctx.save_for_backward(o, inv_l, q, k, v, mask, attn_bias)

        ctx.params = (
            scale,
            causal,
            attn_bias_batch_dim
        )

        return o

    @staticmethod
    def backward(ctx, do):
        assert ctx.should_backwards

        o, inv_l, q, k, v, mask, attn_bias = ctx.saved_tensors

        (
            scale,
            causal,
            attn_bias_batch_dim
        ) = ctx.params

        dq, dk, dv, db = backward(
            do, o, inv_l,
            q, k, v,
            mask,
            attn_bias,
            attn_bias_batch_dim,
            scale,
            causal
        )

        return dq, dk, dv, None, db, None, None, None, None, None, None, None, None, None, None

flash_cosine_sim_attention_cuda = FlashCosineSimAttention.apply

# wrapper function

def flash_cosine_sim_attention(
    q,
    k,
    v,
    mask = None,
    attn_bias = None,
    scale = 8,
    causal = False,
    l2norm_qk = True,
    attn_bias_batch_dim = False
):
    if l2norm_qk:
        q, k = l2norm_tensors(q, k)

    fn = flash_cosine_sim_attention_cuda if q.data.is_cuda else flash_cosine_sim_attention_cpu

    o = fn(
        q, k, v,
        mask,
        attn_bias,
        scale,
        causal,
        attn_bias_batch_dim
    )

    return o
