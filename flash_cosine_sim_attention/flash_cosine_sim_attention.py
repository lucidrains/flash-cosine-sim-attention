import importlib
from typing import Optional

import torch
from torchtyping import TensorType
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

ALLOWED_QK_DIMS = (64,)
ALLOWED_V_DIMS = (64,)

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
    l2norm_qk = True,
    attn_bias_batch_dim = False

) -> TensorType['b', 'h', 'i', 'e']:
    assert not (causal and exists(mask)), 'mask should not be supplied if causality is needed'

    if l2norm_qk:
        dtype = q.dtype
        q, k = map(l2norm, (q, k))
        q, k = map(lambda t: t.type(dtype), (q, k))

    sim = einsum('... i d, ... j d -> ... i j', q, k)
    sim = sim * scale

    if exists(attn_bias):
        attn_bias = attn_bias[:, None, ...] if attn_bias_batch_dim else attn_bias[None, ...]
        sim = sim + attn_bias

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
        causal,
        attn_bias_batch_dim
    ):
        qk_dim, v_dim = q.shape[-1], v.shape[-1]

        assert (qk_dim in ALLOWED_QK_DIMS), f'query key dimension must be one of {ALLOWED_QK_DIMS}'
        assert (v_dim in ALLOWED_V_DIMS), f'value dimension must be one of {ALLOWED_V_DIMS}'

        batch, heads, seq, _, dim, device, dtype = *q.shape, v.shape[-1], q.device, q.dtype

        mask = default(mask, lambda: torch.empty(q.shape[0], 0, device = q.device, dtype = torch.bool))

        attn_bias = default(attn_bias, torch.empty(1, 0, 0, device = q.device, dtype = dtype))

        should_backwards = any([*map(lambda t: t.requires_grad, (q, k, v, attn_bias))])

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
            scale,
            causal,
            attn_bias_batch_dim
        )

        return o

    @staticmethod
    def backward(ctx, do):
        assert ctx.should_backwards

        o, l, q, k, v, mask, attn_bias = ctx.saved_tensors

        batch, heads, src_seq, tgt_seq, device, dtype = *q.shape[:3], k.shape[2], q.device, q.dtype

        (
            scale,
            causal,
            attn_bias_batch_dim
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
    l2norm_qk = True,
    attn_bias_batch_dim = False
) -> TensorType['b', 'h', 'i', 'e']:

    assert not (causal and exists(mask)), 'mask should not be supplied if causality is needed'

    if l2norm_qk:
        dtype = q.dtype
        q, k = map(l2norm, (q, k))
        q, k = map(lambda t: t.type(dtype), (q, k))

    o = FlashCosineSimAttention.apply(
        q, k, v,
        mask,
        attn_bias,
        scale,
        causal,
        attn_bias_batch_dim
    )

    return o
