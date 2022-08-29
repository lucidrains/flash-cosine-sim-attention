import torch
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

# main class

class FlashCosineSimAttention(Function):
    @staticmethod
    def forward(
        ctx,
        q, k, v,
        scale,
        causal,
        mask,
        q_block_size,
        k_block_size
    ):
        mask = default(mask, lambda: torch.ones(q.shape[0], q.shape[2], device = q.device, dtype = torch.bool))

        o, l = forward(q, k, v, mask, scale, causal, q_block_size, k_block_size)

        ctx.save_for_backward(o, l, q, k, v, mask)

        ctx.scale = scale
        ctx.causal = causal
        ctx.q_block_size = q_block_size
        ctx.k_block_size = k_block_size
        return o

    @staticmethod
    def backward(ctx, do):
        o, l, q, k, v, mask = ctx.saved_tensors

        scale = ctx.scale
        causal = ctx.causal
        q_block_size = ctx.q_block_size
        k_block_size = ctx.k_block_size


        dq, dk, dv = backward(do, o, l, q, k, v, mask, scale, causal, q_block_size, k_block_size)
        return dq, dk, dv, None, None, None, None, None

# wrapper function

def flash_cosine_sim_attention(
    q, k, v,
    scale = 8,
    causal = False,
    mask = None,
    q_block_size = 32,
    k_block_size = 32
):
    q, k = map(l2norm, (q, k))
    o = FlashCosineSimAttention.apply(q, k, v, scale, causal, mask, q_block_size, k_block_size)
    return o
