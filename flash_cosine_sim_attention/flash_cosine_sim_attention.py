import torch
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

def plain_cosine_sim_attention(
    q,
    k,
    v,
    scale = 8,
    causal = False,
    mask = None,
    attn_bias = None,
    l2norm_qk = True
):
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
        scale,
        causal,
        mask,
        attn_bias,
        q_block_size,
        k_block_size
    ):
        batch, heads, seq, _, dim, device, dtype = *q.shape, v.shape[-1], q.device, q.dtype

        mask = default(mask, lambda: torch.ones(q.shape[0], q.shape[2], device = q.device, dtype = torch.bool))

        o = torch.zeros((batch, heads, seq, dim), device = device, dtype = torch.float32)
        l = torch.zeros((batch, heads, seq), device = device, dtype = torch.float32)

        attn_bias = default(attn_bias, torch.zeros(1, 0, 0, device = q.device, dtype = dtype))

        forward(q, k, v, o, l, mask, attn_bias, scale, causal, q_block_size, k_block_size)

        o.div_(l[..., None].clamp(min = 1e-20))

        ctx.save_for_backward(o, l, q, k, v, mask, attn_bias)

        ctx.scale = scale
        ctx.causal = causal
        ctx.q_block_size = q_block_size
        ctx.k_block_size = k_block_size
        return o

    @staticmethod
    def backward(ctx, do):
        o, l, q, k, v, mask, attn_bias = ctx.saved_tensors

        heads, src_seq, tgt_seq, device, dtype = q.shape[1], q.shape[2], k.shape[2], q.device, q.dtype

        scale = ctx.scale
        causal = ctx.causal
        q_block_size = ctx.q_block_size
        k_block_size = ctx.k_block_size

        dq, dk, dv = map(torch.zeros_like, (q, k, v))

        d_attn_bias_input = torch.zeros((heads, src_seq, tgt_seq), device = device, dtype = dtype) if attn_bias.requires_grad else torch.zeros((heads, 0, 0), device = device, dtype = dtype)

        dq, dk, dv = backward(do, o, l, q, k, v, dq, dk, dv, d_attn_bias_input, mask, attn_bias, scale, causal, q_block_size, k_block_size)

        db = d_attn_bias_input if attn_bias.requires_grad else None

        return dq, dk, dv, None, None, None, db, None, None

# wrapper function

def flash_cosine_sim_attention(
    q, k, v,
    scale = 8,
    causal = False,
    mask = None,
    attn_bias = None,
    q_block_size = 16,
    k_block_size = 16,
    l2norm_qk = True
):
    if l2norm_qk:
        q, k = map(l2norm, (q, k))

    o = FlashCosineSimAttention.apply(q, k, v, scale, causal, mask, attn_bias, q_block_size, k_block_size)
    return o
