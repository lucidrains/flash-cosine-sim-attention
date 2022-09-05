import torch
from functools import partial
from torch import nn
import torch.nn.functional as F

try:
    from einops import rearrange
except ImportError:
    print('pip install einops to use transformer')

from flash_cosine_sim_attention.flash_cosine_sim_attention import plain_cosine_sim_attention, flash_cosine_sim_attention

# helper function

def exists(val):
    return val is not None

def init_weight_xavier_normal_(module, beta):
    nn.init.xavier_normal_(module.weight.data, gain = beta)

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# top k filtering

def top_k(logits, thres = 0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# attention and feedforward

def FeedForward(dim, mult = 4):
    dim_hidden = int(dim * mult)
    return nn.Sequential(
        nn.Linear(dim, dim_hidden, bias = False),
        nn.GELU(),
        nn.Linear(dim_hidden, dim, bias = False)
    )

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        scale = 8,
        use_cuda_kernel = False,
        **kwargs
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = scale
        self.heads = heads

        self.attn_fn = plain_cosine_sim_attention if not use_cuda_kernel else partial(flash_cosine_sim_attention, **kwargs)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        h, scale = self.heads, self.scale

        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        o = self.attn_fn(q, k, v, causal = True, scale = scale)

        o = rearrange(o, 'b h n d -> b n (h d)')
        return self.to_out(o)

# transformer for testing

class CosineSimCausalTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        max_seq_len,
        depth,
        attn_scale = 8,
        heads = 8,
        dim_head = 64,
        use_cuda_kernel = False,
        **kwargs
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.residual_scale = (2 * depth) ** 0.25

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, dim_head = dim_head, heads = heads, use_cuda_kernel= use_cuda_kernel, scale = attn_scale, **kwargs),
                nn.LayerNorm(dim),
                FeedForward(dim),
                nn.LayerNorm(dim),
            ]))

        self.to_logits = nn.Linear(dim, num_tokens, bias = False)

        self.init_(depth)

    def init_(self, depth):
        nn.init.normal_(self.token_emb.weight, std = 1e-5)
        nn.init.normal_(self.pos_emb.weight, std = 1e-5)

        init_gain = (8 * depth) ** -0.25

        for attn, _, ff, _ in self.layers:
            init_weight_xavier_normal_(attn.to_q, 1.)
            init_weight_xavier_normal_(attn.to_k, 1.)
            init_weight_xavier_normal_(attn.to_v, init_gain)
            init_weight_xavier_normal_(attn.to_out, init_gain)
            init_weight_xavier_normal_(ff[0], init_gain)
            init_weight_xavier_normal_(ff[2], init_gain)

        init_weight_xavier_normal_(self.to_logits, 1)

    @torch.no_grad()
    @eval_decorator
    def generate(self, start_tokens, seq_len, temperature = 1., filter_thres = 0.9, **kwargs):
        b, n, device = *start_tokens.shape, start_tokens.device

        out = start_tokens

        for _ in range(seq_len):
            logits = self.forward(out[:, -self.max_seq_len:], **kwargs)[:, -1, :]
            filtered_logits = top_k(logits, thres = filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim = -1)
            sample = torch.multinomial(probs, 1)
            out = torch.cat((out, sample), dim = -1)

        return out[:, n:]

    def forward(self, x, return_loss = False):
        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        x = self.token_emb(x)
        x = x + self.pos_emb(torch.arange(x.shape[1], device = x.device))

        for attn, attn_norm, ff, ff_norm in self.layers:
            x = attn(x) + x * self.residual_scale
            x = attn_norm(x)
            x = ff(x) + x * self.residual_scale
            x = ff_norm(x)

        logits = self.to_logits(x)

        if not return_loss:
            return logits

        loss = F.cross_entropy(rearrange(logits, 'b c n -> b n c'), labels)
        return loss
