import torch
import pytest
from flash_cosine_sim_attention import plain_cosine_sim_attention, flash_cosine_sim_attention

assert torch.cuda.is_available(), 'cuda must be available'

# helper functions

def allclose(a, b, atol = 1e-3):
    diff = (a - b).abs().amax()
    return diff <= atol

# tests

@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('dim_head', [15, 31, 63])
def test_output_equal(causal, dim_head):
    q = torch.randn(2, 4, 63, dim_head).cuda()
    k = torch.randn(2, 4, 63, dim_head).cuda()
    v = torch.randn(2, 4, 63, dim_head).cuda()
    plain_output = plain_cosine_sim_attention(q, k, v, causal = causal)
    flash_output = flash_cosine_sim_attention(q, k, v, causal = causal)
    assert allclose(plain_output, flash_output)

@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('dim_head', [15, 31, 63])
def test_grad_equal(causal, dim_head):
    q = torch.randn(2, 4, 63, dim_head).cuda().requires_grad_()
    k = torch.randn(2, 4, 63, dim_head).cuda().requires_grad_()
    v = torch.randn(2, 4, 63, dim_head).cuda().requires_grad_()

    plain_output = plain_cosine_sim_attention(q, k, v, causal = causal)
    plain_output.sum().backward()

    dq, dk, dv = q.grad, k.grad, v.grad

    q.grad, k.grad, v.grad = None, None, None

    flash_output = flash_cosine_sim_attention(q, k, v, causal = causal)
    flash_output.sum().backward()

    fdq, fdk, fdv = q.grad, k.grad, v.grad

    assert allclose(dq, fdq)
    assert allclose(dk, fdk)
    assert allclose(dv, fdv)
