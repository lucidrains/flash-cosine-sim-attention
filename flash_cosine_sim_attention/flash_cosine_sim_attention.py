import torch
from typing import Optional
from torchtyping import TensorType
from torch import einsum
import torch.nn.functional as F
from torch.autograd import Function

# try to import cuda

try:
    from flash_cosine_sim_attention_cuda import forward, backward
except ImportError:
    print('CUDA extension for flash-cosine-sim-attention was not compiled correctly - please run `pip install flash-cosine-sim-attention --force-reinstall --no-cache-dir`')

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

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
    v: TensorType['b', 'h', 'j', 'd'],
    mask: Optional[TensorType['b', 'j']] = None,
    attn_bias: Optional[TensorType['h', 'i', 'j']] = None,
    scale = 8,
    causal = False,
    l2norm_qk = True

) -> TensorType['b', 'h', 'i', 'd']:

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
        causal,
        q_block_size,
        k_block_size,
        tile_size,
        backward_tile_size
    ):
        batch, heads, seq, _, dim, device, dtype = *q.shape, v.shape[-1], q.device, q.dtype

        mask = default(mask, lambda: torch.ones(q.shape[0], q.shape[2], device = q.device, dtype = torch.bool))

        o = torch.zeros((batch, heads, seq, dim), device = device, dtype = torch.float32)
        l = torch.zeros((batch, heads, seq), device = device, dtype = torch.float32)

        attn_bias = default(attn_bias, torch.zeros(1, 0, 0, device = q.device, dtype = dtype))

        forward(q, k, v, o, l, mask, attn_bias, scale, causal, q_block_size, k_block_size, tile_size)

        o.div_(l[..., None].clamp(min = 1e-20))

        ctx.save_for_backward(o, l, q, k, v, mask, attn_bias)

        ctx.scale = scale
        ctx.causal = causal
        ctx.q_block_size = q_block_size
        ctx.k_block_size = k_block_size
        ctx.tile_size = tile_size
        ctx.backward_tile_size = backward_tile_size
        return o

    @staticmethod
    def backward(ctx, do):
        o, l, q, k, v, mask, attn_bias = ctx.saved_tensors

        batch, heads, src_seq, tgt_seq, device, dtype = *q.shape[:3], k.shape[2], q.device, q.dtype

        scale = ctx.scale
        causal = ctx.causal
        q_block_size = ctx.q_block_size
        k_block_size = ctx.k_block_size
        tile_size = ctx.tile_size
        backward_tile_size = ctx.backward_tile_size

        dq, dk, dv = map(torch.zeros_like, (q, k, v))

        db = torch.zeros((heads, src_seq, tgt_seq), device = device, dtype = dtype) if attn_bias.requires_grad else torch.zeros((heads, 0, 0), device = device, dtype = dtype)
        do_scaled = torch.zeros_like(l)

        backward(do, o, l, q, k, v, dq, dk, dv, db, do_scaled, mask, attn_bias, scale, causal, q_block_size, k_block_size, backward_tile_size)

        db = db if attn_bias.requires_grad else None

        return dq, dk, dv, None, db, None, None, None, None, None, None

# wrapper function

def flash_cosine_sim_attention(
    q: TensorType['b', 'h', 'i', 'd'],
    k: TensorType['b', 'h', 'j', 'd'],
    v: TensorType['b', 'h', 'j', 'd'],
    mask: Optional[TensorType['b', 'j']] = None,
    attn_bias: Optional[TensorType['h', 'i', 'j']] = None,
    scale = 8,
    causal = False,
    q_block_size = 64,
    k_block_size = 64,
    l2norm_qk = True,
    tile_size = 16,
    backward_tile_size = 16
) -> TensorType['b', 'h', 'i', 'd']:

    if l2norm_qk:
        q, k = map(l2norm, (q, k))

    return FlashCosineSimAttention.apply(q, k, v, mask, attn_bias, scale, causal, q_block_size, k_block_size, tile_size, backward_tile_size)
