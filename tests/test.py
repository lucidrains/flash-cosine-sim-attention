import torch
import pytest
from flash_cosine_sim_attention import plain_cosine_sim_attention, flash_cosine_sim_attention

assert torch.cuda.is_available(), 'cuda must be available'

# helper functions

def not_nan_or_infs(t):
    return not (torch.any(torch.isnan(t)) or torch.any(torch.isinf(t)))

def allclose(a, b, atol = 1e-4):
    diff = (a - b).abs().amax()

    if torch.any(diff > atol):
        print(f'diff: {diff}')

    return diff <= atol

# tests

@pytest.mark.parametrize('causal,mask', [(True, False), (False, True), (False, False)])
@pytest.mark.parametrize('attn_bias', [True, False])
@pytest.mark.parametrize('seq_len', [63, 127])
@pytest.mark.parametrize('dim_head', [32, 64, 96, 128])
@pytest.mark.parametrize('float16', [False, True])
@pytest.mark.parametrize('attn_bias_batch_dim', [False, True])
@pytest.mark.parametrize('single_head_kv', [False, True])
def test_output_equal(
    causal,
    mask,
    attn_bias,
    seq_len,
    dim_head,
    float16,
    attn_bias_batch_dim,
    single_head_kv
):
    batch, heads = 4, 8
    dtype, atol = (torch.float16, 1e-1) if float16 else (torch.float32, 1e-4)

    kv_shape = (batch, heads, seq_len, dim_head) if not single_head_kv else (batch, seq_len, dim_head)

    q = torch.randn(batch, heads, seq_len, dim_head, dtype = dtype).cuda()
    k = torch.randn(kv_shape, dtype = dtype).cuda()
    v = torch.randn(kv_shape, dtype = dtype).cuda()

    attn_mask = torch.randint(0, 2, (batch, seq_len), dtype = torch.bool).cuda() if mask else None
    bias = torch.randn(batch if attn_bias_batch_dim else heads, seq_len, seq_len, dtype = dtype).cuda() if attn_bias else None

    plain_output = plain_cosine_sim_attention(q, k, v, causal = causal, mask = attn_mask, attn_bias = bias, attn_bias_batch_dim = attn_bias_batch_dim)
    flash_output = flash_cosine_sim_attention(q, k, v, causal = causal, mask = attn_mask, attn_bias = bias, attn_bias_batch_dim = attn_bias_batch_dim)

    assert not_nan_or_infs(flash_output)
    assert allclose(plain_output, flash_output, atol = atol)

@pytest.mark.parametrize('causal,mask', [(True, False), (False, True), (False, False)])
@pytest.mark.parametrize('attn_bias', [True, False])
@pytest.mark.parametrize('seq_len', [63, 127])
@pytest.mark.parametrize('dim_head', [32, 64, 96, 128])
@pytest.mark.parametrize('float16', [False, True])
@pytest.mark.parametrize('attn_bias_batch_dim', [False, True])
@pytest.mark.parametrize('single_head_kv', [False, True])
def test_grad_equal(
    causal,
    mask,
    attn_bias,
    seq_len,
    dim_head,
    float16,
    attn_bias_batch_dim,
    single_head_kv
):
    batch, heads = 4, 8
    dtype, atol = (torch.float16, 1e-1) if float16 else (torch.float32, 1e-4)

    kv_shape = (batch, heads, seq_len, dim_head)

    q = torch.randn(batch, heads, seq_len, dim_head, dtype = dtype).cuda().requires_grad_()
    k = torch.randn(kv_shape, dtype = dtype).cuda().requires_grad_()
    v = torch.randn(kv_shape, dtype = dtype).cuda().requires_grad_()

    attn_mask = torch.randint(0, 2, (batch, seq_len), dtype = torch.bool).cuda() if mask else None
    bias = torch.randn(batch if attn_bias_batch_dim else heads, seq_len, seq_len, dtype = dtype).cuda().requires_grad_() if attn_bias else None

    plain_output = plain_cosine_sim_attention(q, k, v, causal = causal, mask = attn_mask, attn_bias = bias, attn_bias_batch_dim = attn_bias_batch_dim)
    plain_output.sum().backward()

    dq, dk, dv = q.grad, k.grad, v.grad

    db = bias.grad if attn_bias else None

    q.grad, k.grad, v.grad = None, None, None

    if attn_bias:
        bias.grad = None

    flash_output = flash_cosine_sim_attention(q, k, v, causal = causal, mask = attn_mask, attn_bias = bias, attn_bias_batch_dim = attn_bias_batch_dim)
    flash_output.sum().backward()

    fdq, fdk, fdv = q.grad, k.grad, v.grad

    fdb = bias.grad if attn_bias else None

    assert not_nan_or_infs(fdv)
    assert not_nan_or_infs(fdk)
    assert not_nan_or_infs(fdq)

    assert allclose(dv, fdv, atol = atol)

    assert allclose(dk, fdk, atol = atol)
    assert allclose(dq, fdq, atol = atol)

    if attn_bias:
        assert not_nan_or_infs(fdb)
        assert allclose(db, fdb, atol = atol)
