import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from module import Attention, PreNorm, FeedForward, CrossAttention
import numpy as np



class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MultiScaleTransformerEncoder(nn.Module):

    def __init__(self, spa_dim = 96, spa_depth = 4, spa_heads =3, spa_dim_head = 32, spa_mlp_dim = 384,
                 spe_dim = 192, spe_depth = 1, spe_heads = 3, spe_dim_head = 64, spe_mlp_dim = 768,
                 cross_attn_depth = 1, cross_attn_heads = 3, dropout = 0.):
        super().__init__()
        self.transformer_enc_spa = Transformer(spa_dim, spa_depth, spa_heads, spa_dim_head, spa_mlp_dim)
        self.transformer_enc_spe = Transformer(spe_dim, spe_depth, spe_heads, spe_dim_head, spe_mlp_dim)

        self.cross_attn_layers = nn.ModuleList([])
        for _ in range(cross_attn_depth):
            self.cross_attn_layers.append(nn.ModuleList([
                nn.Linear(spa_dim, spe_dim),
                nn.Linear(spe_dim, spa_dim),
                PreNorm(spe_dim, CrossAttention(spe_dim, heads = cross_attn_heads, dim_head = spe_dim_head, dropout = dropout)),
                nn.Linear(spe_dim, spa_dim),
                nn.Linear(spa_dim, spe_dim),
                PreNorm(spa_dim, CrossAttention(spa_dim, heads = cross_attn_heads, dim_head = spa_dim_head, dropout = dropout)),
            ]))

    def forward(self, xs, xl):

        xs = self.transformer_enc_spa(xs)
        xl = self.transformer_enc_spe(xl)

        for f_sl, g_ls, cross_attn_s, f_ls, g_sl, cross_attn_l in self.cross_attn_layers:
            spa_class = xs[:, 0]
            x_spa = xs[:, 1:]
            spe_class = xl[:, 0]
            x_spe = xl[:, 1:]
            cal_q = f_ls(spe_class.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_spa), dim=1)
            cal_out = cal_q + cross_attn_l(cal_qkv)
            cal_out = g_sl(cal_out)
            xl = torch.cat((cal_out, x_spe), dim=1)
            cal_q = f_sl(spa_class.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_spe), dim=1)
            cal_out = cal_q + cross_attn_s(cal_qkv)
            cal_out = g_ls(cal_out)
            xs = torch.cat((cal_out, x_spa), dim=1)

        return xs, xl





class CrossViT(nn.Module):
    def __init__(self, image_size, channels, num_classes, patch_size_spa = 14, patch_size_spe = 16, spa_dim = 96,
                 spe_dim = 192, spa_depth = 1, spe_depth = 4, cross_attn_depth = 1, multi_scale_enc_depth = 3,
                 heads = 3, pool = 'cls', dropout = 0., emb_dropout = 0., scale_dim = 4):
        super().__init__()

        assert image_size % patch_size_spa == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches_spa = (image_size // patch_size_spa) ** 2
        patch_dim_spa= channels * patch_size_spa ** 2

        assert image_size % patch_size_spe == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches_spe = (image_size // patch_size_spe) ** 2
        patch_dim_spe = channels * patch_size_spe ** 2
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'


        self.to_patch_embedding_spa = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size_spa, p2 = patch_size_spa),
            nn.Linear(patch_dim_spa, spa_dim),
        )

        self.to_patch_embedding_spe = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size_spe, p2=patch_size_spe),
            nn.Linear(patch_dim_spe, spe_dim),
        )

        self.pos_embedding_spa = nn.Parameter(torch.randn(1, num_patches_spa + 1, spa_dim))
        self.cls_token_spa = nn.Parameter(torch.randn(1, 1, spa_dim))
        self.dropout_spa = nn.Dropout(emb_dropout)

        self.pos_embedding_spe = nn.Parameter(torch.randn(1, num_patches_spe + 1, spe_dim))
        self.cls_token_spe= nn.Parameter(torch.randn(1, 1, spe_dim))
        self.dropout_spe = nn.Dropout(emb_dropout)

        self.multi_scale_transformers = nn.ModuleList([])
        for _ in range(multi_scale_enc_depth):
            self.multi_scale_transformers.append(MultiScaleTransformerEncoder(spa_dim=spa_dim, spa_depth=spa_depth,
                                                                              spa_heads=heads, spa_dim_head=spa_dim//heads,
                                                                              spa_mlp_dim=spa_dim*scale_dim,
                                                                              spe_dim=spe_dim, spe_depth=spe_depth,
                                                                              spe_heads=heads, spe_dim_head=spe_dim//heads,
                                                                              spe_mlp_dim=spe_dim*scale_dim,
                                                                              cross_attn_depth=cross_attn_depth, cross_attn_heads=heads,
                                                                              dropout=dropout))

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head_small = nn.Sequential(
            nn.LayerNorm(spa_dim),
            nn.Linear(spa_dim, num_classes)
        )

        self.mlp_head_spe = nn.Sequential(
            nn.LayerNorm(spe_dim),
            nn.Linear(spe_dim, num_classes)
        )


    def forward(self, img):

        xs = self.to_patch_embedding_spa(img)
        b, n, _ = xs.shape
        cls_token_spa = repeat(self.cls_token_spa, '() n d -> b n d', b=b)
        xl = self.to_patch_embedding_spe(img)
        b, n, _ = xl.shape
        cls_token_spe = repeat(self.cls_token_spe, '() n d -> b n d', b=b)

        xs = torch.cat((cls_token_spa, xl), dim=1)
        xl = torch.cat((cls_token_spe, xs), dim=1)
        xs = self.transformer_enc_spa(xs)
        xl = self.transformer_enc_spe(xl)

        for f_sl, g_ls, cross_attn_s, f_ls, g_sl, cross_attn_l in self.cross_attn_layers:
            spa_class = xs[:, 0]
            x_spa = xs[:, 1:]
            spe_class = xl[:, 0]
            x_spe = xl[:, 1:]


            cal_q = f_ls(spe_class.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_spa), dim=1)
            cal_out = cal_q + cross_attn_l(cal_qkv)
            cal_out = g_sl(cal_out)
            xl = torch.cat((cal_out, x_spe), dim=1)



            cal_q = f_sl(spa_class.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_spe), dim=1)
            cal_out = cal_q + cross_attn_s(cal_qkv)
            cal_out = g_ls(cal_out)
            xs = torch.cat((cal_out, x_spa), dim=1)


        xs += self.fea_embedding_spa[:, :(n + 1)]+xl
        xs = self.dropout_spa(xs)
        xl += self.fea_embedding_spe[:, :(n + 1)]+xs
        xl = self.dropout_spe(xl)

        for multi_scale_transformer in self.multi_scale_transformers:
            xs, xl = multi_scale_transformer(xs, xl)

        xs = xs.mean(dim = 1) if self.pool == 'mean' else xs[:, 0]
        xl = xl.mean(dim = 1) if self.pool == 'mean' else xl[:, 0]

        xs = self.mlp_head_spa(xs)
        xl = self.mlp_head_spe(xl)
        x = xs + xl
        return x
    
    
    

if __name__ == "__main__":
    
    img = torch.ones([1, 3, 224, 224])
    
    model = CrossViT(224, 3, 1000)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)
    
    out = model(img)
    
    print("Shape of out :", out.shape)      # [B, num_classes]

    
