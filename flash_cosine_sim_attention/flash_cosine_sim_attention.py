import importlib
from functools import partial, wraps
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

# decorators

def merged_batch_head_queries(fn):
    @wraps(fn)
    def inner(
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
        is_merged_batch_heads_query = q.ndim == 3

        if is_merged_batch_heads_query:
            assert k.ndim == 3 and v.ndim ==3, 'if batch and heads are merged for queries, keys and values must also similarly have only 3 dimensions'

            attn_bias_batch_dim = True
            q = q[:, None, ...]

        out = fn(
            q,
            k,
            v,
            mask = mask,
            attn_bias = attn_bias,
            scale = scale,
            causal = causal,
            l2norm_qk = l2norm_qk,
            attn_bias_batch_dim = attn_bias_batch_dim
        )

        if is_merged_batch_heads_query:
            out = out.squeeze(1)

        return out

    return inner

# original cosine sim attention

# b - batch
# h - heads
# i - src sequence length
# j - target sequence length
# d - feature dimension

@merged_batch_head_queries
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

@merged_batch_head_queries
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
    if l2norm_qk:
        q, k = l2norm_tensors(q, k)    

    o = flash_cosine_sim_attention_cuda(
        q, k, v,
        mask,
        attn_bias,
        scale,
        causal,
        attn_bias_batch_dim
    )

    return o
