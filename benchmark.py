import torch
import argparse
from itertools import product
from flash_cosine_sim_attention.benchmark import benchmark
from flash_cosine_sim_attention import plain_cosine_sim_attention, flash_cosine_sim_attention

assert torch.cuda.is_available(), 'cuda must be available to run benchmark'

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
DIM = 64

TEST_SEQUENCE_LENGTHS = (128, 256, 512, 1024, 2048, 4096, 8192)

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
    ('feature dimension', DIM)
))

permutations = list(product(*map(cast_tuple, params.values())))

for batch, heads, dim in permutations:
    print('-' * 40)
    print(f'batch: {batch}\theads: {heads}\tdim {dim}\t')
    print('-' * 40)

    for seq in TEST_SEQUENCE_LENGTHS:
        q = torch.randn(batch, heads, seq, dim).cuda().requires_grad_()
        k = torch.randn(batch, heads, seq, dim).cuda().requires_grad_()
        v = torch.randn(batch, heads, seq, dim).cuda().requires_grad_()

        fused_time = fused_attention_fn(q, k, v)

        try:
            baseline_time = attention_fn(q, k, v)
        except:
            torch.cuda.empty_cache()
            baseline_time = -1

        times_slower = (fused_time / baseline_time) if baseline_time != -1 else 0.
        baseline_time_str = 'oom' if baseline_time == -1 else f"{baseline_time:.2f}ms"

        print(f'seq_len: {seq}\tslower: {times_slower:.2f}x\tkernel: {fused_time:.2f}ms\tbaseline: {baseline_time_str}')
