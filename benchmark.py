import torch
import argparse
from itertools import product
from flash_cosine_sim_attention.benchmark import benchmark
from flash_cosine_sim_attention import plain_cosine_sim_attention, flash_cosine_sim_attention

# helper functions

def cast_tuple(t):
    return t if isinstance(t, tuple) else (t,)

# argparse

parser = argparse.ArgumentParser()
parser.add_argument('--only-forwards', default = False, action = 'store_true')
parser.add_argument('--only-backwards', default = False, action = 'store_true')
args = parser.parse_args()

assert not (args.only_forwards and args.only_backwards)

# constants

BATCH_SIZES = (2, 4, 8)
HEADS = 4
TEST_SEQUENCE_LENGTHS = (128, 256, 512, 1024, 2048)
DIM = 64

TEST_FORWARDS = not args.only_backwards
TEST_BACKWARDS = not args.only_forwards


fused_attention_fn = benchmark(
    flash_cosine_sim_attention,
    forwards = TEST_FORWARDS,
    backwards = TEST_BACKWARDS
)

attention_fn = benchmark(
    plain_cosine_sim_attention,
    forwards = TEST_FORWARDS,
    backwards = TEST_BACKWARDS
)

# all permutations

params = dict((
    ('batch size', BATCH_SIZES),
    ('heads', HEADS),
    ('sequence lengths', TEST_SEQUENCE_LENGTHS),
    ('feature dimension', DIM)
))

permutations = list(product(*map(cast_tuple, params.values())))

for batch, heads, seq, dim in permutations:
    print('-' * 60)
    print(f'batch: {batch}\theads: {heads}\tseq: {seq}\tdim {dim}\t')
    print('-' * 60)

    q = torch.randn(batch, heads, seq, dim).cuda().requires_grad_()
    k = torch.randn(batch, heads, seq, dim).cuda().requires_grad_()
    v = torch.randn(batch, heads, seq, dim).cuda().requires_grad_()

    fused_time = fused_attention_fn(q, k, v)
    baseline_time = attention_fn(q, k, v)

    times_slower = fused_time / baseline_time

    print(f'\nslower: {times_slower:.2f}x\tkernel: {fused_time:.2f}ms\tbaseline: {baseline_time:.2f}ms\n')
