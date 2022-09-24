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
@pytest.mark.parametrize('seq_len', [63, 127])
@pytest.mark.parametrize('qk_dim_head', [32, 64])
@pytest.mark.parametrize('v_dim_head', [64, 32])
def test_output_equal(
    causal,
    mask,
    attn_bias,
    seq_len,
    qk_dim_head,
    v_dim_head
):
    q = torch.randn(4, 8, seq_len, qk_dim_head).cuda()
    k = torch.randn(4, 8, seq_len, qk_dim_head).cuda()
    v = torch.randn(4, 8, seq_len, v_dim_head).cuda()

    attn_mask = torch.randint(0, 2, (4, seq_len), dtype = torch.bool).cuda() if mask else None
    bias = torch.randn(8, seq_len, seq_len).cuda() if attn_bias else None

    plain_output = plain_cosine_sim_attention(q, k, v, causal = causal, mask = attn_mask, attn_bias = bias)
    flash_output = flash_cosine_sim_attention(q, k, v, causal = causal, mask = attn_mask, attn_bias = bias)

    assert allclose(plain_output, flash_output)

@pytest.mark.parametrize('causal,mask', [(True, False), (False, True), (False, False)])
@pytest.mark.parametrize('attn_bias', [True, False])
@pytest.mark.parametrize('seq_len', [63, 127])
@pytest.mark.parametrize('qk_dim_head', [64, 32])
@pytest.mark.parametrize('v_dim_head', [64, 32])
def test_grad_equal(
    causal,
    mask,
    attn_bias,
    seq_len,
    qk_dim_head,
    v_dim_head
):
    q = torch.randn(4, 8, seq_len, qk_dim_head).cuda().requires_grad_()
    k = torch.randn(4, 8, seq_len, qk_dim_head).cuda().requires_grad_()
    v = torch.randn(4, 8, seq_len, v_dim_head).cuda().requires_grad_()

    attn_mask = torch.randint(0, 2, (4, seq_len), dtype = torch.bool).cuda() if mask else None
    bias = torch.randn(8, seq_len, seq_len).cuda().requires_grad_() if attn_bias else None

    plain_output = plain_cosine_sim_attention(q, k, v, causal = causal, mask = attn_mask, attn_bias = bias)
    plain_output.sum().backward()

    dq, dk, dv = q.grad, k.grad, v.grad

    db = bias.grad if attn_bias else None

    q.grad, k.grad, v.grad = None, None, None

    if attn_bias:
        bias.grad = None

    flash_output = flash_cosine_sim_attention(q, k, v, causal = causal, mask = attn_mask, attn_bias = bias)
    flash_output.sum().backward()

    fdq, fdk, fdv = q.grad, k.grad, v.grad

    fdb = bias.grad if attn_bias else None

    assert allclose(dv, fdv)

    if attn_bias:
        assert allclose(db, fdb)

    assert allclose(dk, fdk)
    assert allclose(dq, fdq)
