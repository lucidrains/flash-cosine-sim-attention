import torch
import argparse
from flash_cosine_sim_attention.benchmark import benchmark
from flash_cosine_sim_attention import plain_cosine_sim_attention, flash_cosine_sim_attention

# argparse

parser = argparse.ArgumentParser()
parser.add_argument('--only-forwards', default = False, action = 'store_true')
parser.add_argument('--only-backwards', default = False, action = 'store_true')
args = parser.parse_args()

assert not (args.only_forwards and args.only_backwards)

# constants

TEST_SEQUENCE_LENGTHS = (128, 256, 512, 1024, 2048)

BATCH_SIZE = 1
HEADS = 4
DIM = 64

Q_BLOCK_SIZE = 64
K_BLOCK_SIZE = 64

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

for seq_len in TEST_SEQUENCE_LENGTHS:
    q = torch.randn(BATCH_SIZE, HEADS, seq_len, DIM).cuda().requires_grad_()
    k = torch.randn(BATCH_SIZE, HEADS, seq_len, DIM).cuda().requires_grad_()
    v = torch.randn(BATCH_SIZE, HEADS, seq_len, DIM).cuda().requires_grad_()

    fused_time = fused_attention_fn(q, k, v, q_block_size = Q_BLOCK_SIZE, k_block_size = K_BLOCK_SIZE)
    baseline_time = attention_fn(q, k, v)

    times_slower = fused_time / baseline_time

    print(f'slower: {times_slower:.3f}x\t seq_len: {seq_len}\tfused kernel: {fused_time:.3f}\tbaseline: {baseline_time:.3f}')
