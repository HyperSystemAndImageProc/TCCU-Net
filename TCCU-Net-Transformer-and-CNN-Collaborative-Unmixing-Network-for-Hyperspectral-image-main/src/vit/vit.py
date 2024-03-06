from torch import einsum, nn


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, d_head, expansion, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, d_head, dropout)),
                PreNorm(dim, FeedForward(dim, expansion, dropout)),
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        x = self.fn(x, **kwargs)
        return self.fn(self.norm(x), **kwargs)


class Attention(nn.Module):
    def __init__(self, d_model, n_heads, d_head, dropout=0.):
        super().__init__()
        inner_dim = d_head * n_heads
        self.scaling = inner_dim**(-0.5)

        self.to_qkv = nn.Linear(d_model, inner_dim * 3, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        qkv = self.to_qkv(x)
        attn = self.attention(qkv)
        out = self.to_out(attn)
        return out

    def attention(self, qkv):
        q, k, v = qkv.chunk(3, dim=-1)
        scores = einsum('bhqd, bhkd -> bhqk', q, k) * self.scaling
        attn = einsum('bhad, bhdv -> bhav', self.softmax(scores), v)
        return attn


class FeedForward(nn.Sequential):
    def __init__(self, dim, expansion, dropout=0.):
        super().__init__(
            nn.Linear(dim, dim * expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout),
        )
