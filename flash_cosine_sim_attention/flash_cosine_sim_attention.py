import torch
from typing import Optional
from torchtyping import TensorType
from torch import einsum
import torch.nn.functional as F
from torch.autograd import Function

# try to import cuda

try:
    from flash_cosine_sim_attention_cuda import (
        forward,
        backward
    )
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

# original cosine sim attention

# b - batch
# h - heads
# i - src sequence length
# j - target sequence length
# d - feature dimension

def plain_cosine_sim_attention(
    q: TensorType['b', 'h', 'i', 'd'],
    k: TensorType['b', 'h', 'j', 'd'],
    v: TensorType['b', 'h', 'j', 'e'],
    mask: Optional[TensorType['b', 'j']] = None,
    attn_bias: Optional[TensorType['h', 'i', 'j']] = None,
    scale = 10,
    causal = False,
    l2norm_qk = True

) -> TensorType['b', 'h', 'i', 'e']:
    assert not (causal and exists(mask)), 'mask should not be supplied if causality is needed'

    if l2norm_qk:
        q, k = map(l2norm, (q, k))

    sim = einsum('... i d, ... j d -> ... i j', q, k)
    sim = sim * scale

    if exists(attn_bias):
        sim = sim + attn_bias[None, ...]

    mask_value = -torch.finfo(sim.dtype).max

    if causal:
        causal_mask = torch.ones(sim.shape[-2:], device = q.device, dtype = torch.bool).triu(1)
        sim = sim.masked_fill(causal_mask, mask_value)

    if exists(mask):
        sim = sim.masked_fill(~mask[:, None, None, :], mask_value)

    attn = sim.softmax(dim = -1)
    return einsum('... i j, ... j d -> ... i d', attn, v)

# main class

class FlashCosineSimAttention(Function):
    @staticmethod
    def forward(
        ctx,
        q, k, v,
        mask,
        attn_bias,
        scale,
        causal
    ):
        qk_dim, v_dim = q.shape[-1], v.shape[-1]

        batch, heads, seq, _, dim, device, dtype = *q.shape, v.shape[-1], q.device, q.dtype

        mask = default(mask, lambda: torch.ones(q.shape[0], 0, device = q.device, dtype = torch.bool))

        attn_bias = default(attn_bias, torch.empty(1, 0, 0, device = q.device, dtype = dtype))

        should_backwards = any([*map(lambda t: t.requires_grad, (q, k, v, attn_bias))])

        o, l = forward(
            q, k, v,
            mask,
            attn_bias,
            scale,
            causal,
            should_backwards
        )

        ctx.should_backwards = should_backwards

        if not ctx.should_backwards:
            return o

        ctx.save_for_backward(o, l, q, k, v, mask, attn_bias)

        ctx.params = (
            scale,
            causal
        )

        return o

    @staticmethod
    def backward(ctx, do):
        assert ctx.should_backwards

        o, l, q, k, v, mask, attn_bias = ctx.saved_tensors

        batch, heads, src_seq, tgt_seq, device, dtype = *q.shape[:3], k.shape[2], q.device, q.dtype

        (
            scale,
            causal
        ) = ctx.params

        dq, dk, dv, db = backward(
            do, o, l,
            q, k, v,
            mask,
            attn_bias,
            scale,
            causal,
            attn_bias.requires_grad
        )

        db = db if attn_bias.requires_grad else None

        return dq, dk, dv, None, db, None, None, None, None, None, None, None, None, None, None

# wrapper function

def flash_cosine_sim_attention(
    q: TensorType['b', 'h', 'i', 'd'],
    k: TensorType['b', 'h', 'j', 'd'],
    v: TensorType['b', 'h', 'j', 'e'],
    mask: Optional[TensorType['b', 'j']] = None,
    attn_bias: Optional[TensorType['h', 'i', 'j']] = None,
    scale = 10,
    causal = False,
    l2norm_qk = True
) -> TensorType['b', 'h', 'i', 'e']:

    assert not (causal and exists(mask)), 'mask should not be supplied if causality is needed'

    if l2norm_qk:
        q, k = map(l2norm, (q, k))

    o = FlashCosineSimAttention.apply(
        q, k, v,
        mask,
        attn_bias,
        scale,
        causal
    )

    return o
