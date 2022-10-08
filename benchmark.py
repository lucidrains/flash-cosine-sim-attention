import argparse
from itertools import product

import torch
from torch import einsum
assert torch.cuda.is_available(), 'cuda must be available to run benchmark'

from flash_cosine_sim_attention.benchmark import benchmark
from flash_cosine_sim_attention import flash_cosine_sim_attention, l2norm_tensors

# helper functions

def exists(t):
    return t is not None

def cast_tuple(t):
    return t if isinstance(t, tuple) else (t,)

# argparse

parser = argparse.ArgumentParser()
parser.add_argument('--causal', default = False, action = 'store_true')
parser.add_argument('--only-forwards', default = False, action = 'store_true')
parser.add_argument('--only-backwards', default = False, action = 'store_true')
args = parser.parse_args()

assert not (args.only_forwards and args.only_backwards)

# constants

BATCH_SIZES = 4
HEADS = 8
DIM = 64
CAUSAL = args.causal

TEST_SEQUENCE_LENGTHS = (128, 256, 512, 1024, 2048, 4096, 8192)

TEST_FORWARDS = not args.only_backwards
TEST_BACKWARDS = not args.only_forwards

# simplified cosine sim attention for benchmarking

def simplified_cosine_sim_attention(
    q,
    k,
    v,
    scale = 10,
    l2norm_qk = True,
    causal_mask = None
):
    if l2norm_qk:
        q, k = l2norm_tensors(q, k)

    sim = einsum(f'b h i d, b h j d -> b h i j', q, k)
    sim = sim * scale

    if exists(causal_mask):
        mask_value = -torch.finfo(sim.dtype).max
        sim = sim.masked_fill(causal_mask, mask_value)

    attn = sim.softmax(dim = -1)
    return einsum(f'b h i j, b h j d -> b h i d', attn, v)

# create benchmark function

fused_attention_fn = benchmark(
    flash_cosine_sim_attention,
    forwards = TEST_FORWARDS,
    backwards = TEST_BACKWARDS
)

attention_fn = benchmark(
    simplified_cosine_sim_attention,
    forwards = TEST_FORWARDS,
    backwards = TEST_BACKWARDS
)

# all permutations

params = dict((
    ('batch size', BATCH_SIZES),
    ('heads', HEADS),
    ('feature dimension', DIM)
))

permutations = list(product(*map(cast_tuple, params.values())))

for name, dtype in (('float32', torch.float32), ('float16', torch.float16)):

    for batch, heads, dim in permutations:
        print('-' * 60)
        print(f'{name}\t\tbatch: {batch}\theads: {heads}\tdim {dim}\t')
        print('-' * 60)

        for seq in TEST_SEQUENCE_LENGTHS:
            q = torch.randn(batch, heads, seq, dim, dtype = dtype).cuda().requires_grad_()
            k = torch.randn(batch, heads, seq, dim, dtype = dtype).cuda().requires_grad_()
            v = torch.randn(batch, heads, seq, dim, dtype = dtype).cuda().requires_grad_()

            causal_mask = torch.ones((seq, seq), dtype = torch.bool).cuda().triu(1)

            fused_args = dict(causal = CAUSAL)
            baseline_args = dict()

            if CAUSAL:
                baseline_args = {**baseline_args, 'causal_mask': causal_mask}

            # run benchmarks accounting for oom for baseline

            fused_time = fused_attention_fn(q, k, v, **fused_args)

            try:
                baseline_time = attention_fn(q, k, v, **baseline_args)
            except:
                torch.cuda.empty_cache()
                baseline_time = -1

            times_slower = (fused_time / baseline_time) if baseline_time != -1 else 0.
            baseline_time_str = 'oom' if baseline_time == -1 else f"{baseline_time:.2f}ms"

            print(f'seq_len: {seq}\tslower: {times_slower:.2f}x\tkernel: {fused_time:.2f}ms\tbaseline: {baseline_time_str}')
