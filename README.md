## Flash Cosine Similarity Attention (wip)

Implementation of fused cosine similarity attention in the same style as <a href="https://arxiv.org/abs/2205.14135">Flash Attention</a>. The observation is that by adopting l2 normalized queries and keys, you no longer need to keep track of the row maximums for numerical stability. This greatly simplifies the flash attention algorithm, assuming cosine similarity attention comes at no generalization cost.

In other words, potentially stable, fast, memory efficient, and longer context attention with no downsides.

So far cosine similarity attention is not widely used in industry. The only large model that has been trained with it so far is <a href="https://arxiv.org/abs/2111.09883">SwinV2</a>. If anyone can invalidate the approach, please open an issue or send me an email. You can run experiments against regular attention using the <a href="https://github.com/lucidrains/x-transformers#grouped-query-key-l2-normalization">x-transformers</a> repository.

This will be my first attempt at CUDA, so welcome any help or advice.

Update: Meta AI will be <a href="https://github.com/facebookresearch/xformers/pull/362#issuecomment-1212924962">considering merging the flash attention</a> implementation into Pytorch core. They have also done experiments on a variety of models and show that it is free from numerical issues. I suppose there is less a case for the cosine similarity variant other than that it could potentially be slightly faster. I will still complete it for my own GPGPU education.

## Testing

For testing output and gradients are equal for non-autoregressive and autoregressive scenarios

```bash
$ python setup.py test
```

For testing the cuda kernel on enwik8 training

```bash
$ pip install -r requirements.txt && python train.py --use-cuda-kernel
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
@inproceedings{Henry2020QueryKeyNF,
    title   = {Query-Key Normalization for Transformers},
    author  = {Alex Henry and Prudhvi Raj Dachapally and Shubham Vivek Pawar and Yuxuan Chen},
    booktitle = {FINDINGS},
    year    = {2020}
}
```
