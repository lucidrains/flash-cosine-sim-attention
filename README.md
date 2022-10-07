## Flash Cosine Similarity Attention (wip)

Implementation of fused cosine similarity attention in the same style as <a href="https://arxiv.org/abs/2205.14135">Flash Attention</a>. The observation is that by adopting l2 normalized queries and keys, you no longer need to keep track of the row maximums for numerical stability. This greatly simplifies the flash attention algorithm, assuming cosine similarity attention comes at no generalization cost.

In other words, potentially stable, fast, memory efficient, and longer context attention with no downsides.

## Status (wip)

- Forward kernel now only slightly behind baseline on GTX 2080Ti, but definitely faster on Ampere due to the greater amount of shared memory

- Backwards kernel is still 3x slower, 1.5x slower for autoregressive

- Thanks to Arthur, now at 2-4x faster for forwards autoregressive, 1-2x faster for backwards. float 16 causal still facing some numerical issues

- All tests now pass for F16 and the transformer trains! Big thanks to Arthur 🙏

- There is a bug with the atomicAdd for dQ for F16, so F16 is still not safe for use

- F16 is now fixed for dQ with a hack, but performance is no longer that great (1.3-3x slower on f16)

- F16 is all good now

## Todo

- [ ] bring in a CPU memory efficient version (only for inference, as training does not make sense) using just plain pytorch code
- [ ] support O(n) 1d dynamic positional bias
- [ ] bfloat16 support
- [ ] flexible which type is used for accumulation

- [x] support more standard head dimensions (wip)
- [x] debug and fix bias backwards gradients yet again for head size of 32
- [x] fix attention bias gradients
- [x] allow for single-headed key / values, as in PaLM
- [x] fix atomic add for f16
- [x] attention bias should be able to accept dimensions of an extra batch dimension, for Alphafold2 like attention biasing
- [x] automate cache-busting of kernel using version as suffix to package name
- [x] resolve f16 causal numerical issues
- [x] adopt all learnings from forward kernel to backwards kernel and make sure it outperforms at least on A100

## Supported head dimensions (wip)

- [x] 32
- [x] 64
- [x] 128

- [ ] 96 - f32 forwards
- [ ] 96 - f32 backwards
- [ ] 96 - f16 forwards
- [ ] 96 - f16 backwards

- [ ] 80 - f32 forwards
- [ ] 80 - f32 backwards
- [ ] 80 - f16 forwards
- [ ] 80 - f16 backwards

*80 for attention inside CLIP*

## Appreciation

- <a href="https://github.com/ahennequ">Arthur Hennequin</a> for coaching me through my first CUDA kernel, and for coding up a simple <a href="https://github.com/ahennequ/pytorch-custom-mma">reference implementation</a>, which helped me to bootstrap the first kernel that comes within reasonable performance to baseline. This work would not have been possible without his expertise.

- <a href="https://stability.ai/">Stability.ai</a> for the generous sponsorship to work on cutting edge artificial intelligence research

## Install

```bash
$ pip install flash-cosine-sim-attention
```

## Usage

Self Attention

```python
import torch
from flash_cosine_sim_attention import flash_cosine_sim_attention

q = torch.randn(1, 8, 1024, 64).cuda()
k = torch.randn(1, 8, 1024, 64).cuda()
v = torch.randn(1, 8, 1024, 64).cuda()

out = flash_cosine_sim_attention(q, k, v)  # (1, 8, 1024, 64)
```

Cross attention

```python
import torch
from flash_cosine_sim_attention import flash_cosine_sim_attention

q = torch.randn(1, 8, 1024, 64).cuda()
k = torch.randn(1, 8, 2048, 64).cuda()
v = torch.randn(1, 8, 2048, 64).cuda()

out = flash_cosine_sim_attention(q, k, v) # (1, 8, 1024, 64)
```

With key / value masking

```python
import torch
from flash_cosine_sim_attention import flash_cosine_sim_attention

q = torch.randn(1, 8, 1024, 64).cuda()
k = torch.randn(1, 8, 2048, 64).cuda()
v = torch.randn(1, 8, 2048, 64).cuda()

mask = torch.ones(1, 2048).bool().cuda()

out = flash_cosine_sim_attention(q, k, v, mask = mask) # (1, 8, 1024, 64)
```

Autoregressive

```python
import torch
from flash_cosine_sim_attention import flash_cosine_sim_attention

q = torch.randn(1, 8, 1024, 64).cuda()
k = torch.randn(1, 8, 1024, 64).cuda()
v = torch.randn(1, 8, 1024, 64).cuda()

out = flash_cosine_sim_attention(q, k, v, causal = True)  # (1, 8, 1024, 64)
```

## Description

So far cosine similarity attention is not widely used in industry. The only large model that has been trained with it so far is <a href="https://arxiv.org/abs/2111.09883">SwinV2</a>. If anyone can invalidate the approach, please open an issue or send me an email. You can run experiments against regular attention using the <a href="https://github.com/lucidrains/x-transformers#grouped-query-key-l2-normalization">x-transformers</a> repository.

Update: <a href="https://github.com/borisdayma">Boris Dayma</a> has graciously kicked off <a href="https://wandb.ai/dalle-mini/dalle-mini/reports/Fix-Swin-v2--VmlldzoyNDA4Mzc3">an experiment (blue with red as baseline)</a> to validate cosine similarity attention with a fixed scale of 10 in a real-world model setting. 🙏

Update 2: Cosine similarity attention has been proven out in a real-world text-to-image attention network, using a constant scale of `10`. No worse than regular attention. Credit goes to <a href="https://github.com/borisdayma">Boris Dayma</a> for investing the time to run the experiment and removing doubts surrounding the technique.

Update 3: <a href="https://github.com/rromb">Robin Rombach</a> has tested out the kernel in this repository with head size of 64 and fixed scale of 10 in a text-to-image model, observing no difference from regular attention. More evaluations pending.

## Testing

For testing output and gradients are equal for non-autoregressive and autoregressive scenarios

```bash
$ python setup.py test
```

## Benchmarking

Make sure to first install the CUDA kernel

```python
$ python setup.py install
```

Then

```python
$ python benchmark.py
```

For only benchmarking forwards or backwards, append either `--only-forwards` or `--only-backwards` flag to the above

## Training a small GPT on Enwik8

```bash
$ make train
```

Try 8192 sequence length. It'll be slow but will work (normal attention will break at > 2048, you'll see this if you remove the `--use-cuda-kernel` flag)

```python
$ python train.py --seq-len 8192 --use-cuda-kernel
```

## Citations

```bibtex
@article{Dao2022FlashAttentionFA,
    title   = {FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness},
    author  = {Tri Dao and Daniel Y. Fu and Stefano Ermon and Atri Rudra and Christopher R'e},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2205.14135}
}
```

```bibtex
@misc{rabe2021selfattention,
    title   = {Self-attention Does Not Need $O(n^2)$ Memory}, 
    author  = {Markus N. Rabe and Charles Staats},
    year    = {2021},
    eprint  = {2112.05682},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```

```bibtex
@inproceedings{Henry2020QueryKeyNF,
    title   = {Query-Key Normalization for Transformers},
    author  = {Alex Henry and Prudhvi Raj Dachapally and Shubham Vivek Pawar and Yuxuan Chen},
    booktitle = {FINDINGS},
    year    = {2020}
}
```

```bibtex
@article{Wang2022DeepNetST,
    title   = {DeepNet: Scaling Transformers to 1, 000 Layers},
    author  = {Hongyu Wang and Shuming Ma and Li Dong and Shaohan Huang and Dongdong Zhang and Furu Wei},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2203.00555}
}
```
