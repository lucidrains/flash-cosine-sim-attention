import importlib
from typing import Optional

import torch
from torch import einsum
import torch.nn.functional as F
from torch.autograd import Function

exec(open('flash_cosine_sim_attention/version.py').read())

# try to import cuda

try:
    cuda_pkg = importlib.import_module(__cuda_pkg_name__)

    forward = cuda_pkg.forward
    backward = cuda_pkg.backward

except ImportError:
    print('CUDA extension for flash-cosine-sim-attention was not compiled correctly - please run `pip install flash-cosine-sim-attention --force-reinstall --no-cache-dir`')

# constants

ALLOWED_DIMS = (32, 64, 128)
ALLOWED_HALF_DIMS = (32, 64, 128)

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def divisible_by(numer, denom):
    return (numer % denom) == 0

def l2norm(t):
    return F.normalize(t, dim = -1)

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
    scale = 10,
    causal = False,
    l2norm_qk = True,
    attn_bias_batch_dim = False

):
    assert not (causal and exists(mask)), 'mask should not be supplied if causality is needed'
    single_head_kv = k.ndim == 3

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
        causal_mask = torch.ones(sim.shape[-2:], device = q.device, dtype = torch.bool).triu(1)
        sim = sim.masked_fill(causal_mask, mask_value)

    if exists(mask):
        sim = sim.masked_fill(~mask[:, None, None, :], mask_value)

    attn = sim.softmax(dim = -1)
    return einsum(f'b h i j, {kv_einsum_eq} -> b h i d', attn, v)

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
        batch, heads, seq, dim_qk, dim, device, dtype = *q.shape, v.shape[-1], q.device, q.dtype
        assert dim_qk == dim, 'query / key head dimension must be equal to value head dimension for now'

        single_head_kv = k.ndim == 3
        is_half = dtype == torch.float16

        assert is_half or (dim in ALLOWED_DIMS), f'query key dimension must be one of {ALLOWED_DIMS}'
        assert (not is_half) or (dim in ALLOWED_HALF_DIMS), f'half dimensions must be one of {ALLOWED_HALF_DIMS}'

        mask = default(mask, lambda: torch.empty(q.shape[0], 0, device = q.device, dtype = torch.bool))

        attn_bias = default(attn_bias, torch.empty(1, 0, 0, device = q.device, dtype = dtype))

        should_backwards = any([*map(lambda t: t.requires_grad, (q, k, v, attn_bias))])

        if single_head_kv:
            k, v = map(lambda t: t.unsqueeze(1), (k, v))

        o, l = forward(
            q, k, v,
            mask,
            attn_bias,
            attn_bias_batch_dim,
            scale,
            causal,
            should_backwards
        )

        ctx.should_backwards = should_backwards

        if not ctx.should_backwards:
            return o

        ctx.save_for_backward(o, l, q, k, v, mask, attn_bias)

        ctx.params = (
            dtype,
            scale,
            causal,
            attn_bias_batch_dim,
            single_head_kv
        )

        return o

    @staticmethod
    def backward(ctx, do):
        assert ctx.should_backwards

        o, l, q, k, v, mask, attn_bias = ctx.saved_tensors

        batch, heads, src_seq, tgt_seq, device, dtype = *q.shape[:3], k.shape[2], q.device, q.dtype

        (
            dtype,
            scale,
            causal,
            attn_bias_batch_dim,
            single_head_kv
        ) = ctx.params

        dq, dk, dv, db = backward(
            do, o, l,
            q, k, v,
            mask,
            attn_bias,
            attn_bias_batch_dim,
            scale,
            causal,
            attn_bias.requires_grad
        )

        db = db if attn_bias.requires_grad else None

        if single_head_kv:
            dk, dv = map(lambda t: t.squeeze(1), (dk, dv))

        dq = dq.type(dtype)
        dk = dk.type(dtype)
        dv = dv.type(dtype)

        if exists(db):
            db = db.type(dtype)

        return dq, dk, dv, None, db, None, None, None, None, None, None, None, None, None, None

# wrapper function

def flash_cosine_sim_attention(
    q,
    k,
    v,
    mask = None,
    attn_bias = None,
    scale = 10,
    causal = False,
    l2norm_qk = True,
    attn_bias_batch_dim = False
):
    assert not (causal and exists(mask)), 'mask should not be supplied if causality is needed'

    if l2norm_qk:
        q, k = l2norm_tensors(q, k)

    o = FlashCosineSimAttention.apply(
        q, k, v,
        mask,
        attn_bias,
        scale,
        causal,
        attn_bias_batch_dim
    )

    return o
