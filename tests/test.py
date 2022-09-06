import torch
import pytest
from flash_cosine_sim_attention import plain_cosine_sim_attention, flash_cosine_sim_attention

assert torch.cuda.is_available(), 'cuda must be available'

# helper functions

def allclose(a, b, atol = 1e-4):
    diff = (a - b).abs().amax()
    return diff <= atol

# tests

@pytest.mark.parametrize('causal,mask', [(True, False), (False, True), (False, False)])
@pytest.mark.parametrize('attn_bias', [True, False])
@pytest.mark.parametrize('dim_head', [15, 31, 63])
def test_output_equal(
    causal,
    mask,
    attn_bias,
    dim_head
):
    q = torch.randn(2, 4, 63, dim_head).cuda()
    k = torch.randn(2, 4, 63, dim_head).cuda()
    v = torch.randn(2, 4, 63, dim_head).cuda()

    attn_mask = torch.randint(0, 2, (2, 63), dtype = torch.bool).cuda() if mask else None
    bias = torch.randn(4, 63, 63).cuda() if attn_bias else None

    plain_output = plain_cosine_sim_attention(q, k, v, causal = causal, mask = attn_mask, attn_bias = bias)
    flash_output = flash_cosine_sim_attention(q, k, v, causal = causal, mask = attn_mask, attn_bias = bias)

    assert allclose(plain_output, flash_output)

@pytest.mark.parametrize('causal,mask', [(True, False), (False, True), (False, False)])
@pytest.mark.parametrize('attn_bias', [True, False])
@pytest.mark.parametrize('dim_head', [15, 31, 63])
def test_grad_equal(
    causal,
    mask,
    attn_bias,
    dim_head
):
    q = torch.randn(2, 4, 63, dim_head).cuda().requires_grad_()
    k = torch.randn(2, 4, 63, dim_head).cuda().requires_grad_()
    v = torch.randn(2, 4, 63, dim_head).cuda().requires_grad_()

    attn_mask = torch.randint(0, 2, (2, 63), dtype = torch.bool).cuda() if mask else None
    bias = torch.randn(4, 63, 63).requires_grad_().cuda() if attn_bias else None

    plain_output = plain_cosine_sim_attention(q, k, v, causal = causal, mask = attn_mask, attn_bias = bias)
    plain_output.sum().backward()

    dq, dk, dv = q.grad, k.grad, v.grad

    q.grad, k.grad, v.grad = None, None, None

    flash_output = flash_cosine_sim_attention(q, k, v, causal = causal, mask = attn_mask, attn_bias = bias)
    flash_output.sum().backward()

    fdq, fdk, fdv = q.grad, k.grad, v.grad

    assert allclose(dq, fdq)
    assert allclose(dk, fdk)
    assert allclose(dv, fdv)
