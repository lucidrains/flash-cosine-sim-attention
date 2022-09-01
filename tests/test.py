import torch
from flash_cosine_sim_attention import plain_cosine_sim_attention, flash_cosine_sim_attention

# helper functions

def allclose(a, b, atol = 1e-6):
    diff = (a - b).abs().amax()
    return diff <= atol

# tests

def test_output_equal():
    q = torch.randn(2, 4, 63, 63).cuda()
    k = torch.randn(2, 4, 63, 63).cuda()
    v = torch.randn(2, 4, 63, 63).cuda()
    plain_output = plain_cosine_sim_attention(q, k, v)
    flash_output = flash_cosine_sim_attention(q, k, v)
    assert allclose(plain_output, flash_output)

def test_grad_equal():
    q = torch.randn(2, 4, 63, 63).cuda().requires_grad_()
    k = torch.randn(2, 4, 63, 63).cuda().requires_grad_()
    v = torch.randn(2, 4, 63, 63).cuda().requires_grad_()

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
