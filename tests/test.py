import torch
import pytest
from flash_cosine_sim_attention import plain_cosine_sim_attention, flash_cosine_sim_attention

pytest.importorskip('triton')
from flash_cosine_sim_attention.triton_flash_cosine_sim_attention import triton_flash_cosine_sim_attention

assert torch.cuda.is_available(), 'cuda must be available'

# helper functions

def not_nan_or_infs(t):
    return not (torch.any(torch.isnan(t)) or torch.any(torch.isinf(t)))

def allclose(a, b, atol = 1e-4):
    diff = (a - b).abs().amax()

    if torch.any(diff > atol):
        print(f'diff: {diff}')

    return diff <= atol

def exists(t):
    return t is not None

def maybe_cpu(t):
    if not exists(t):
        return None

    return t.cpu()

# tests

@pytest.mark.parametrize('causal,mask', [(True, False), (False, True), (False, False)])
@pytest.mark.parametrize('seq_len', [63, 127])
@pytest.mark.parametrize('dim_head', [32, 64, 96, 128])
@pytest.mark.parametrize('float16', [False, True])
@pytest.mark.parametrize('single_head_kv', [False, True])
def test_output_equal(
    causal,
    mask,
    seq_len,
    dim_head,
    float16,
    single_head_kv
):
    batch, heads = 4, 8
    dtype, atol = (torch.float16, 1e-1) if float16 else (torch.float32, 1e-4)

    kv_shape = (batch, heads, seq_len, dim_head) if not single_head_kv else (batch, seq_len, dim_head)

    q = torch.randn(batch, heads, seq_len, dim_head, dtype = dtype).cuda()
    k = torch.randn(kv_shape, dtype = dtype).cuda()
    v = torch.randn(kv_shape, dtype = dtype).cuda()

    attn_mask = torch.randint(0, 2, (batch, seq_len), dtype = torch.bool).cuda() if mask else None

    plain_output = plain_cosine_sim_attention(q, k, v, causal = causal, mask = attn_mask)
    flash_output = flash_cosine_sim_attention(q, k, v, causal = causal, mask = attn_mask)

    assert not_nan_or_infs(flash_output)
    assert allclose(plain_output, flash_output, atol = atol)

@pytest.mark.parametrize('causal,mask', [(True, False), (False, True), (False, False)])
@pytest.mark.parametrize('seq_len', [63, 127])
@pytest.mark.parametrize('dim_head', [32, 64, 96, 128])
@pytest.mark.parametrize('float16', [False, True])
@pytest.mark.parametrize('single_head_kv', [False, True])
def test_grad_equal(
    causal,
    mask,
    seq_len,
    dim_head,
    float16,
    single_head_kv
):
    batch, heads = 4, 8
    dtype, atol = (torch.float16, 1e-1) if float16 else (torch.float32, 1e-4)

    kv_shape = (batch, heads, seq_len, dim_head)

    q = torch.randn(batch, heads, seq_len, dim_head, dtype = dtype).cuda().requires_grad_()
    k = torch.randn(kv_shape, dtype = dtype).cuda().requires_grad_()
    v = torch.randn(kv_shape, dtype = dtype).cuda().requires_grad_()

    attn_mask = torch.randint(0, 2, (batch, seq_len), dtype = torch.bool).cuda() if mask else None

    plain_output = plain_cosine_sim_attention(q, k, v, causal = causal, mask = attn_mask)
    plain_output.sum().backward()

    dq, dk, dv = q.grad, k.grad, v.grad

    q.grad, k.grad, v.grad = None, None, None

    flash_output = flash_cosine_sim_attention(q, k, v, causal = causal, mask = attn_mask)
    flash_output.sum().backward()

    fdq, fdk, fdv = q.grad, k.grad, v.grad

    assert not_nan_or_infs(fdv)
    assert not_nan_or_infs(fdk)
    assert not_nan_or_infs(fdq)

    assert allclose(dv, fdv, atol = atol)

    assert allclose(dk, fdk, atol = atol)
    assert allclose(dq, fdq, atol = atol)

# test cpu

@pytest.mark.parametrize('causal,mask', [(True, False), (False, True), (False, False)])
@pytest.mark.parametrize('seq_len', [63, 127])
@pytest.mark.parametrize('dim_head', [32, 64, 96, 128])
@pytest.mark.parametrize('float16', [False, True])
@pytest.mark.parametrize('single_head_kv', [False, True])
def test_output_equal_cuda_and_cpu_forward(
    causal,
    mask,
    seq_len,
    dim_head,
    float16,
    single_head_kv
):
    batch, heads = 4, 8
    dtype, atol = (torch.float16, 1e-1) if float16 else (torch.float32, 1e-4)

    kv_shape = (batch, heads, seq_len, dim_head) if not single_head_kv else (batch, seq_len, dim_head)

    q = torch.randn(batch, heads, seq_len, dim_head, dtype = dtype).cuda()
    k = torch.randn(kv_shape, dtype = dtype).cuda()
    v = torch.randn(kv_shape, dtype = dtype).cuda()

    attn_mask = torch.randint(0, 2, (batch, seq_len), dtype = torch.bool).cuda() if mask else None

    flash_output = flash_cosine_sim_attention(q, k, v, causal = causal, mask = attn_mask)
    flash_output_cpu = flash_cosine_sim_attention(q.cpu(), k.cpu(), v.cpu(), causal = causal, mask = maybe_cpu(attn_mask))

    assert allclose(flash_output.cpu(), flash_output_cpu, atol = atol)


# triton flash cosine sim attention
# output and gradients must be equal to the plain implementation, for both
# the sequence parallel and the sequential backward kernels

@pytest.mark.parametrize('causal,mask', [(True, False), (False, True), (False, False)])
@pytest.mark.parametrize('seq_len', [63, 127, 256])
@pytest.mark.parametrize('dim_head', [32, 64, 96, 128])
@pytest.mark.parametrize('float16', [False, True])
@pytest.mark.parametrize('single_head_kv', [False, True])
def test_triton_output_equal(
    causal,
    mask,
    seq_len,
    dim_head,
    float16,
    single_head_kv
):
    batch, heads = 4, 8
    dtype, atol = (torch.float16, 1e-1) if float16 else (torch.float32, 1e-4)

    kv_shape = (batch, heads, seq_len, dim_head) if not single_head_kv else (batch, seq_len, dim_head)

    q = torch.randn(batch, heads, seq_len, dim_head, dtype = dtype).cuda()
    k = torch.randn(kv_shape, dtype = dtype).cuda()
    v = torch.randn(kv_shape, dtype = dtype).cuda()

    attn_mask = torch.randint(0, 2, (batch, seq_len), dtype = torch.bool).cuda() if mask else None

    plain_output = plain_cosine_sim_attention(q, k, v, causal = causal, mask = attn_mask)
    triton_output = triton_flash_cosine_sim_attention(q, k, v, causal = causal, mask = attn_mask)

    assert not_nan_or_infs(triton_output)
    assert allclose(plain_output, triton_output, atol = atol)

@pytest.mark.parametrize('causal,mask', [(True, False), (False, True), (False, False)])
@pytest.mark.parametrize('seq_len', [63, 127, 256])
@pytest.mark.parametrize('dim_head', [32, 64, 96, 128])
@pytest.mark.parametrize('float16', [False, True])
@pytest.mark.parametrize('single_head_kv', [False, True])
@pytest.mark.parametrize('sequence_parallel', [True, False])
def test_triton_grad_equal(
    causal,
    mask,
    seq_len,
    dim_head,
    float16,
    single_head_kv,
    sequence_parallel
):
    batch, heads = 4, 8
    dtype, atol = (torch.float16, 1e-1) if float16 else (torch.float32, 1e-4)

    kv_shape = (batch, heads, seq_len, dim_head) if not single_head_kv else (batch, seq_len, dim_head)

    q = torch.randn(batch, heads, seq_len, dim_head, dtype = dtype).cuda().requires_grad_()
    k = torch.randn(kv_shape, dtype = dtype).cuda().requires_grad_()
    v = torch.randn(kv_shape, dtype = dtype).cuda().requires_grad_()

    attn_mask = torch.randint(0, 2, (batch, seq_len), dtype = torch.bool).cuda() if mask else None

    plain_output = plain_cosine_sim_attention(q, k, v, causal = causal, mask = attn_mask)
    plain_output.sum().backward()

    dq, dk, dv = q.grad, k.grad, v.grad

    q.grad, k.grad, v.grad = None, None, None

    triton_output = triton_flash_cosine_sim_attention(q, k, v, causal = causal, mask = attn_mask, sequence_parallel = sequence_parallel)
    triton_output.sum().backward()

    fdq, fdk, fdv = q.grad, k.grad, v.grad

    assert not_nan_or_infs(fdv)
    assert not_nan_or_infs(fdk)
    assert not_nan_or_infs(fdq)

    assert allclose(dv, fdv, atol = atol)
    assert allclose(dk, fdk, atol = atol)
    assert allclose(dq, fdq, atol = atol)

# triton and cuda kernels should agree with each other

@pytest.mark.parametrize('causal,mask', [(True, False), (False, True), (False, False)])
@pytest.mark.parametrize('seq_len', [63, 127, 256])
@pytest.mark.parametrize('dim_head', [32, 64, 96, 128])
@pytest.mark.parametrize('float16', [False, True])
@pytest.mark.parametrize('single_head_kv', [False, True])
@pytest.mark.parametrize('sequence_parallel', [True, False])
def test_triton_cuda_equal(
    causal,
    mask,
    seq_len,
    dim_head,
    float16,
    single_head_kv,
    sequence_parallel
):
    batch, heads = 4, 8
    dtype, atol = (torch.float16, 1e-1) if float16 else (torch.float32, 1e-4)

    kv_shape = (batch, heads, seq_len, dim_head) if not single_head_kv else (batch, seq_len, dim_head)

    q = torch.randn(batch, heads, seq_len, dim_head, dtype = dtype).cuda().requires_grad_()
    k = torch.randn(kv_shape, dtype = dtype).cuda().requires_grad_()
    v = torch.randn(kv_shape, dtype = dtype).cuda().requires_grad_()

    attn_mask = torch.randint(0, 2, (batch, seq_len), dtype = torch.bool).cuda() if mask else None

    cuda_output = flash_cosine_sim_attention(q, k, v, causal = causal, mask = attn_mask)
    cuda_output.sum().backward()

    dq, dk, dv = q.grad, k.grad, v.grad

    q.grad, k.grad, v.grad = None, None, None

    triton_output = triton_flash_cosine_sim_attention(q, k, v, causal = causal, mask = attn_mask, sequence_parallel = sequence_parallel)
    triton_output.sum().backward()

    fdq, fdk, fdv = q.grad, k.grad, v.grad

    assert not_nan_or_infs(triton_output)
    assert allclose(cuda_output, triton_output, atol = atol)

    assert not_nan_or_infs(fdv)
    assert not_nan_or_infs(fdk)
    assert not_nan_or_infs(fdq)

    assert allclose(dv, fdv, atol = atol)
    assert allclose(dk, fdk, atol = atol)
    assert allclose(dq, fdq, atol = atol)

# the use_triton flag on the main entry point

def test_use_triton_flag():
    batch, heads, seq_len, dim_head = 2, 4, 127, 64

    q = torch.randn(batch, heads, seq_len, dim_head, dtype = torch.float16).cuda().requires_grad_()
    k = torch.randn(batch, heads, seq_len, dim_head, dtype = torch.float16).cuda().requires_grad_()
    v = torch.randn(batch, heads, seq_len, dim_head, dtype = torch.float16).cuda().requires_grad_()

    o1 = flash_cosine_sim_attention(q, k, v, causal = True, use_triton = True)
    o1.sum().backward()

    dq1, dk1, dv1 = q.grad, k.grad, v.grad

    q.grad, k.grad, v.grad = None, None, None

    o2 = flash_cosine_sim_attention(q, k, v, causal = True, use_triton = False)
    o2.sum().backward()

    assert allclose(o1, o2, atol = 1e-1)
    assert allclose(dq1, q.grad, atol = 1e-1)
    assert allclose(dk1, k.grad, atol = 1e-1)
    assert allclose(dv1, v.grad, atol = 1e-1)

# merged batch and head dimension

def test_triton_merged_batch_head():
    batch_heads, seq_len, dim_head = 8, 127, 64

    q = torch.randn(batch_heads, seq_len, dim_head, dtype = torch.float16).cuda().requires_grad_()
    k = torch.randn(batch_heads, seq_len, dim_head, dtype = torch.float16).cuda().requires_grad_()
    v = torch.randn(batch_heads, seq_len, dim_head, dtype = torch.float16).cuda().requires_grad_()

    o1 = plain_cosine_sim_attention(q, k, v, causal = True)
    o1.sum().backward()

    dq1, dk1, dv1 = q.grad, k.grad, v.grad

    q.grad, k.grad, v.grad = None, None, None

    o2 = triton_flash_cosine_sim_attention(q, k, v, causal = True)
    o2.sum().backward()

    assert o1.shape == o2.shape
    assert allclose(o1, o2, atol = 1e-1)
    assert allclose(dq1, q.grad, atol = 1e-1)
    assert allclose(dk1, k.grad, atol = 1e-1)
    assert allclose(dv1, v.grad, atol = 1e-1)

# bfloat16

@pytest.mark.parametrize('causal', [True, False])
def test_triton_bf16(causal):
    batch, heads, seq_len, dim_head = 2, 4, 127, 64
    dtype = torch.bfloat16

    q = torch.randn(batch, heads, seq_len, dim_head, dtype = dtype).cuda().requires_grad_()
    k = torch.randn(batch, heads, seq_len, dim_head, dtype = dtype).cuda().requires_grad_()
    v = torch.randn(batch, heads, seq_len, dim_head, dtype = dtype).cuda().requires_grad_()

    o1 = plain_cosine_sim_attention(q, k, v, causal = causal)
    o1.sum().backward()

    dq1, dk1, dv1 = q.grad, k.grad, v.grad

    q.grad, k.grad, v.grad = None, None, None

    o2 = triton_flash_cosine_sim_attention(q, k, v, causal = causal)
    o2.sum().backward()

    assert not_nan_or_infs(o2)
    assert allclose(o1, o2, atol = 1e-1)
    assert allclose(dq1, q.grad, atol = 1e-1)
    assert allclose(dk1, k.grad, atol = 1e-1)
    assert allclose(dv1, v.grad, atol = 1e-1)

# l2norm disabled - queries and keys must already be unit norm, as is the
# case for example after applying a norm-preserving rotation (rotary)

def test_triton_l2norm_disabled():
    batch, heads, seq_len, dim_head = 2, 4, 127, 64

    q = torch.randn(batch, heads, seq_len, dim_head, dtype = torch.float16).cuda()
    k = torch.randn(batch, heads, seq_len, dim_head, dtype = torch.float16).cuda()
    v = torch.randn(batch, heads, seq_len, dim_head, dtype = torch.float16).cuda().requires_grad_()

    # queries and keys must already be unit norm when l2norm is disabled

    q = torch.nn.functional.normalize(q, dim = -1).detach().requires_grad_()
    k = torch.nn.functional.normalize(k, dim = -1).detach().requires_grad_()

    o1 = plain_cosine_sim_attention(q, k, v, causal = True, l2norm_qk = False)
    o1.sum().backward()

    dq1, dk1, dv1 = q.grad, k.grad, v.grad

    q.grad, k.grad, v.grad = None, None, None

    o2 = triton_flash_cosine_sim_attention(q, k, v, causal = True, l2norm_qk = False)
    o2.sum().backward()

    assert not_nan_or_infs(o2)
    assert allclose(o1, o2, atol = 1e-1)
    assert allclose(dq1, q.grad, atol = 1e-1)
    assert allclose(dk1, k.grad, atol = 1e-1)
    assert allclose(dv1, v.grad, atol = 1e-1)

# transformer with the use_triton flag

def test_transformer_use_triton():
    from flash_cosine_sim_attention.transformer import CosineSimCausalTransformer

    torch.manual_seed(0)

    model = CosineSimCausalTransformer(
        num_tokens = 256,
        dim = 128,
        depth = 2,
        attn_scale = 1,
        attn_l2norm_groups = 8,
        dim_head = 64,
        heads = 4,
        pre_norm = True,
        use_triton = True,
        max_seq_len = 127
    ).cuda()

    x = torch.randint(0, 256, (2, 127)).cuda()

    loss = model(x, return_loss = True)
    loss.backward()

    assert not_nan_or_infs(loss)
    assert all(not_nan_or_infs(p.grad) for p in model.parameters() if p.grad is not None)
