import torch
import torch.nn.functional as F
from torch.autograd import Function

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
        ctx.save_for_backward(q, k, v)
        return q

    @staticmethod
    def backward(ctx, do):
        q, k, v = ctx.saved_tensors
        return q, k, v

# wrapper function

def flash_cosine_sim_attention(q, k, v):
    q, k = map(l2norm, (q, k))
    o = FlashCosineSimAttention.apply(q, k, v)
    return o
