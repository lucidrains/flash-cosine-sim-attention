import torch
import torch.nn.functional as F
from torch.autograd import Function

# try to import cuda

from .flash_cosine_sim_attention_cuda import forward, backward

# helper functions

def l2norm(t):
    return F.normalize(t, dim = -1)

# main class

class FlashCosineSimAttention(Function):
    @staticmethod
    def forward(
        ctx,
        q, k, v
    ):
        o, l = forward(q, k, v)
        ctx.save_for_backward(o, l, q, k, v)
        return q

    @staticmethod
    def backward(ctx, do):
        o, l, q, k, v = ctx.saved_tensors
        dq, dk, dv = backward(do, o, l, q, k, v)
        return dq, dk, dv

# wrapper function

def flash_cosine_sim_attention(q, k, v):
    q, k = map(l2norm, (q, k))
    o = FlashCosineSimAttention.apply(q, k, v)
    return o
