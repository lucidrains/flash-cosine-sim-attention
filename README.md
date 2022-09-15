## Flash Cosine Similarity Attention (wip)

Implementation of fused cosine similarity attention in the same style as <a href="https://arxiv.org/abs/2205.14135">Flash Attention</a>. The observation is that by adopting l2 normalized queries and keys, you no longer need to keep track of the row maximums for numerical stability. This greatly simplifies the flash attention algorithm, assuming cosine similarity attention comes at no generalization cost.

In other words, potentially stable, fast, memory efficient, and longer context attention with no downsides.

## Status (wip)

- Forward kernel now only slightly behind baseline on GTX 2080Ti, but definitely faster on Ampere due to the greater amount of shared memory

## Todo

- [ ] make sure works with f16
- [ ] adopt all learnings from forward kernel to backwards kernel and make sure it outperforms at least on A100
- [ ] make sure value dimensions can be 16, 32, 64, or 128 using the templating strategy recommended by Arthur
- [ ] attention bias should be able to accept dimensions of an extra batch dimension, for Alphafold2 like attention biasing

## Appreciation

- <a href="https://github.com/ahennequ">Arthur Hennequin</a> for coaching me through my first CUDA kernel, and for coding up a simple <a href="https://github.com/ahennequ/pytorch-custom-mma">reference implementation</a>, which helped me to bootstrap the first kernel that comes within reasonable performance to baseline. This work would not have been possible without his expertise.

- <a href="https://stability.ai/">Stability.ai</a> for the generous sponsorship to work on cutting edge artificial intelligence research

## Install

```bash
$ pip install flash-cosine-sim-attention
```

## Usage

```python
import torch
from flash_cosine_sim_attention import flash_cosine_sim_attention

q = torch.randn(1, 8, 1024, 64).cuda()
k = torch.randn(1, 8, 1024, 64).cuda()
v = torch.randn(1, 8, 1024, 64).cuda()

out = flash_cosine_sim_attention(q, k, v)  # (1, 8, 1024, 64)
```

## Description

So far cosine similarity attention is not widely used in industry. The only large model that has been trained with it so far is <a href="https://arxiv.org/abs/2111.09883">SwinV2</a>. If anyone can invalidate the approach, please open an issue or send me an email. You can run experiments against regular attention using the <a href="https://github.com/lucidrains/x-transformers#grouped-query-key-l2-normalization">x-transformers</a> repository.

Update: <a href="https://github.com/borisdayma">Boris Dayma</a> has graciously kicked off <a href="https://wandb.ai/dalle-mini/dalle-mini/reports/Fix-Swin-v2--VmlldzoyNDA4Mzc3">an experiment (blue with red as baseline)</a> to validate cosine similarity attention with a fixed scale of 10 in a real-world model setting. 🙏

Update 2: Cosine similarity attention has been proven out in a real-world text-to-image attention network, using a constant scale of `10`. No worse than full attention. Credit goes to <a href="https://github.com/borisdayma">Boris Dayma</a> for investing the time to run the experiment and removing doubts surrounding the technique.

## Testing

For testing output and gradients are equal for non-autoregressive and autoregressive scenarios

```bash
$ python setup.py test
```

For testing the cuda kernel on enwik8 training

```bash
$ pip install -r requirements.txt && python train.py --use-cuda-kernel
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
