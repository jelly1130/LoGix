from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


from einops import rearrange
from layers.Embed import DataEmbedding, DataEmbedding_wo_time
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath

class LocalMixer(nn.Module):
    def __init__(self, in_features, hidden_features, num_scales, drop=0.):
        super().__init__()
        
        self.convs = nn.ModuleList()
        kernel_sizes = [2**i - 1 for i in range(1, num_scales+1)]
        paddings = [(k - 1) // 2 for k in kernel_sizes]
        
        for k, p in zip(kernel_sizes, paddings):
            self.convs.append(nn.Conv1d(in_features, hidden_features, k, 1, padding=p))
        
        self.weights = nn.ParameterList([nn.Parameter(torch.rand(1)) for _ in range(2*(num_scales-1))])
        
        self.mixer = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()
        self.num_scales = num_scales


    def forward(self, x):
       
        x = x.transpose(1, 2)
        activations = [self.drop(self.act(conv(x))) for conv in self.convs]
        
        high2low = 0
        low2high = 0
        
        for i in range(self.num_scales - 1):
            high2low += self.weights[i] * self.convs[i](x) * activations[i + 1]
            low2high += self.weights[self.num_scales - 1 + i] * self.convs[-(i + 1)](x) * activations[-(i + 2)]
        
        y = self.mixer(high2low + low2high)
        y = y.transpose(1, 2)
        return y

class PatchEmbed(nn.Module):
    def __init__(self, seq_len, patch_size=8, in_chans=3, embed_dim=384, stride=1):
        super().__init__()
        num_patches = int((seq_len - patch_size) / stride + 1)
        # print(f'{num_patches=}, {seq_len=}, {patch_size=}, {stride=}')
        self.num_patches = num_patches
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        x_out = self.proj(x.transpose(1, 2)).flatten(2).transpose(1, 2)
        return x_out


class GlobalMixerNG(nn.Module):
    def __init__(self, num_patch, dim, device):
        super().__init__()
        
        if num_patch % 2 == 0:
            fft_len = int(num_patch / 2 + 1)
        else:
            fft_len = int((num_patch + 1) / 2)
        filter_width = fft_len // 2
        center_patch = fft_len // 2
        
        self.complex_weight_low = nn.Parameter(torch.randn(fft_len, dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight_mid = nn.Parameter(torch.randn(fft_len, dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight_high = nn.Parameter(torch.randn(fft_len, dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight_all = nn.Parameter(torch.randn(fft_len, dim, 2, dtype=torch.float32) * 0.02)

        with torch.no_grad():
            self.mask_low = torch.zeros(fft_len, dim, device=device)
            self.mask_low[0:filter_width, :] = 1

            self.mask_mid = torch.zeros(fft_len, dim, device=device)
            self.mask_mid[(center_patch - filter_width // 2):(center_patch + filter_width // 2), :] = 1

            self.mask_high = torch.zeros(fft_len, dim, device=device)
            self.mask_high[fft_len - filter_width:fft_len, :] = 1
        
        trunc_normal_(self.complex_weight_high, std=.02)
        trunc_normal_(self.complex_weight_mid, std=.02)
        trunc_normal_(self.complex_weight_low, std=.02)
        trunc_normal_(self.complex_weight_all, std=.02)
    
    def forward(self, x_in):
        B, N, C = x_in.shape
        
        dtype = x_in.dtype
        x = x_in.to(torch.float32)
        
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        
        weight_low = torch.view_as_complex(self.complex_weight_low) * self.mask_low
        x_low_filtered = x_fft * weight_low
        weight_mid = torch.view_as_complex(self.complex_weight_mid) * self.mask_mid
        x_mid_filtered = x_fft * weight_mid
        weight_high = torch.view_as_complex(self.complex_weight_high) * self.mask_high
        x_high_filtered = x_fft * weight_high
        weight_all = torch.view_as_complex(self.complex_weight_all)
        x_all_filtered = x_fft * weight_all
        
        x = torch.fft.irfft(x_low_filtered + x_mid_filtered + x_high_filtered + x_all_filtered, n=N, dim=1, norm='ortho')
        x = x.to(dtype)
        x = x.view(B, N, C)
        return x


class LoGix_layer(nn.Module):
    def __init__(self, dim, num_patches, device, num_scales, mlp_ratio=3., drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm = norm_layer(dim)
        self.local_mixer = LocalMixer(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop, num_scales=num_scales)
        self.global_mixer = GlobalMixerNG(num_patches, dim, device)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        y = self.norm(x)
        y = self.global_mixer(y) + self.local_mixer(y)
        
        y = x + self.drop_path(y)
        return y


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        
        device = torch.device('cuda:{}'.format(0))

        self.patch_embed = PatchEmbed(
            seq_len=args.seq_len, patch_size=args.patch_size,
            in_chans=args.num_channels, embed_dim=args.emb_dim, stride=args.stride
        )
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, args.emb_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=args.dropout_rate)

        # Layers/Networks
        # self.input_layer = nn.Linear(self.patch_size, args.emb_dim)

        dpr = [x.item() for x in torch.linspace(0, args.dropout, args.depth)]  # stochastic depth decay rule

        self.mixer_blocks = nn.ModuleList([
            LoGix_layer(dim=args.emb_dim, num_patches=num_patches, drop=args.dropout, drop_path=dpr[i], device=device, num_scales=args.num_scales)
            for i in range(args.depth)]
        )

        # Anomaly detect head
        self.head = nn.Linear(args.emb_dim, args.c_out, bias=True)


    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for mixer_blk in self.mixer_blocks:
            x = mixer_blk(x)

        outputs = self.head(x)
        return outputs


    
