import torch
import torch.nn.functional as F
from torch.autograd import Function

# try to import cuda

try:
    from flash_cosine_sim_attention_cuda import forward, backward
except ImportError:
    print('CUDA extension for flash-cosine-sim-attention was not compiled correctly - please run `pip install flash-cosine-sim-attention --force-reinstall --no-cache-dir`')

# helper functions

def l2norm(t):
    return F.normalize(t, dim = -1)

# main class

class FlashCosineSimAttention(Function):
    @staticmethod
    def forward(
        ctx,
        q, k, v,
        scale,
        q_block_size,
        k_block_size
    ):
        o, l = forward(q, k, v, scale, q_block_size, k_block_size)

        ctx.save_for_backward(o, l, q, k, v)

        ctx.scale = scale
        ctx.q_block_size = q_block_size
        ctx.k_block_size = k_block_size
        return o

    @staticmethod
    def backward(ctx, do):
        o, l, q, k, v = ctx.saved_tensors

        scale = ctx.scale
        q_block_size = ctx.q_block_size
        k_block_size = ctx.k_block_size

        dq, dk, dv = backward(do, o, l, q, k, v, scale, q_block_size, k_block_size)
        return dq, dk, dv, None, None, None

# wrapper function

def flash_cosine_sim_attention(
    q, k, v,
    scale = 8,
    q_block_size = 128,
    k_block_size = 128
):
    q, k = map(l2norm, (q, k))
    o = FlashCosineSimAttention.apply(q, k, v, scale, q_block_size, k_block_size)
    return o
