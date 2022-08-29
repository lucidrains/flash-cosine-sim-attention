import torch
from torch import einsum
import torch.nn.functional as F
from flash_cosine_sim_attention import flash_cosine_sim_attention

# regular attention

def l2norm(t):
    return F.normalize(t, dim = -1)

def plain_cosine_sim_attention(q, k, v, scale = 8, causal = False):
    q, k = map(l2norm, (q, k))
    sim = einsum('... i d, ... j d -> ... i j', q, k)
    sim = sim * scale

    if causal:
        causal_mask = torch.ones(sim.shape[-2:], device = q.device).triu(1)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

    attn = sim.softmax(dim = -1)
    return einsum('... i j, ... j d -> ... i d', attn, v)

# helper functions

def allclose(a, b, atol = 1e-6):
    diff = (a - b).abs().amax()
    return diff <= atol

# tests

def test_output_equal():
    q = torch.randn(2, 4, 1024, 512)
    k = torch.randn(2, 4, 1024, 512)
    v = torch.randn(2, 4, 1024, 512)
    plain_output = plain_cosine_sim_attention(q, k, v)
    flash_output = flash_cosine_sim_attention(q, k, v)
    assert allclose(plain_output, flash_output)

def test_grad_equal():
    q = torch.randn(2, 4, 1024, 512).requires_grad_()
    k = torch.randn(2, 4, 1024, 512).requires_grad_()
    v = torch.randn(2, 4, 1024, 512).requires_grad_()

    plain_output = plain_cosine_sim_attention(q, k, v)
    plain_output.sum().backward()

    dq, dk, dv = q.grad, k.grad, v.grad

    q.grad, k.grad, v.grad = None, None, None

    flash_output = flash_cosine_sim_attention(q, k, v)
    flash_output.sum().backward()

    fdq, fdk, fdv = q.grad, k.grad, v.grad

    assert allclose(dq, fdq)
    assert allclose(dk, fdk)
    assert allclose(dv, fdv)
