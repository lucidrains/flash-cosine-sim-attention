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
        scale
    ):
        o, l = forward(q, k, v, scale)

        ctx.save_for_backward(o, l, q, k, v)
        ctx.scale = scale
        return o

    @staticmethod
    def backward(ctx, do):
        o, l, q, k, v = ctx.saved_tensors
        scale = ctx.scale

        dq, dk, dv = backward(do, o, l, q, k, v, scale)
        return dq, dk, dv, None

# wrapper function

def flash_cosine_sim_attention(q, k, v, scale = 8):
    q, k = map(l2norm, (q, k))
    o = FlashCosineSimAttention.apply(q, k, v, scale)
    return o
