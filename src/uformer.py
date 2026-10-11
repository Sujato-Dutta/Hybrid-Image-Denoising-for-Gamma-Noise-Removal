"""Native Uformer (CVPR 2022), linear projection / LeFF / decoder modulators.

Architecture reference: Wang et al., https://arxiv.org/abs/2106.03106.
Checked against author implementation revision 65fc970a8ffc09605faca74ed016ee93c9ad8a36.
No timm, einops, or upstream repository dependency is needed at runtime.
The RGB core retains upstream state-dict names for numerical fidelity checks.
Grayscale changes only the input/output channel counts, keeps the image residual,
and reflection-pads arbitrary images without resizing them.
Author-source license retained for the architectural code conventions:

MIT License. Copyright (c) 2022 Zhendong Wang.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


VARIANTS = {
    'tiny': (16, (2,) * 9),
    'small': (32, (2,) * 9),
    'base': (32, (1, 2, 8, 8, 2, 8, 8, 2, 1)),
}
HEADS = (1, 2, 4, 8, 16, 16, 8, 4, 2)


class DropPath(nn.Module):
    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        self.probability = probability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0:
            return x
        keep = 1 - self.probability
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


def window_partition(x: torch.Tensor, size: int) -> torch.Tensor:
    b, h, w, c = x.shape
    return x.reshape(b, h // size, size, w // size, size, c).permute(
        0, 1, 3, 2, 4, 5
    ).contiguous().view(-1, size, size, c)


def window_reverse(windows: torch.Tensor, size: int, h: int, w: int) -> torch.Tensor:
    b = windows.shape[0] // ((h // size) * (w // size))
    return windows.reshape(b, h // size, w // size, size, size, -1).permute(
        0, 1, 3, 2, 4, 5
    ).contiguous().view(b, h, w, -1)


def spatial_side(tokens: torch.Tensor) -> int:
    side = math.isqrt(tokens.shape[1])
    if side * side != tokens.shape[1]:
        raise ValueError('The Uformer core expects square spatial feature maps.')
    return side


class LinearProjection(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, dim)
        self.to_kv = nn.Linear(dim, 2 * dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        b, n, c = x.shape
        q = self.to_q(x).reshape(b, n, 1, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)[0]
        kv = self.to_kv(x).reshape(b, n, 2, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        return q, kv[0], kv[1]


class WindowAttention(nn.Module):
    def __init__(self, dim: int, size: int, heads: int) -> None:
        super().__init__()
        self.dim, self.size, self.num_heads = dim, size, heads
        self.scale = (dim // heads) ** -0.5
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * size - 1) ** 2, heads))
        coords = torch.stack(torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')).flatten(1)
        relative = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0).contiguous()
        relative[:, :, 0] += size - 1
        relative[:, :, 1] += size - 1
        relative[:, :, 0] *= 2 * size - 1
        self.register_buffer('relative_position_index', relative.sum(-1))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.qkv = LinearProjection(dim, heads)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(0)
        self.proj_drop = nn.Dropout(0)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, n, c = x.shape
        q, k, v = self.qkv(x)
        attention = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)]
        attention = attention + bias.reshape(n, n, self.num_heads).permute(2, 0, 1).unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attention = attention.view(b // nw, nw, self.num_heads, n, n)
            attention = (attention + mask.unsqueeze(0).unsqueeze(2)).view(b, self.num_heads, n, n)
        attention = self.attn_drop(attention.softmax(dim=-1))
        output = (attention @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(output))


class LeFF(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        hidden = 4 * dim
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden), nn.GELU())
        self.dwconv = nn.Sequential(nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden), nn.GELU())
        self.linear2 = nn.Sequential(nn.Linear(hidden, dim))
        self.eca = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        side = spatial_side(x)
        x = self.linear1(x)
        x = x.transpose(1, 2).reshape(x.shape[0], -1, side, side)
        x = self.dwconv(x).flatten(2).transpose(1, 2)
        return self.eca(self.linear2(x))


class LeWinTransformerBlock(nn.Module):
    def __init__(self, dim: int, resolution: int, heads: int, window: int,
                 shift: int, drop_path: float, modulator: bool) -> None:
        super().__init__()
        self.win_size = min(window, resolution)
        self.shift_size = 0 if resolution <= window else shift
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, self.win_size, heads)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = LeFF(dim)
        self.modulator = nn.Embedding(window * window, dim) if modulator else None

    def shift_mask(self, side: int, x: torch.Tensor) -> torch.Tensor | None:
        if not self.shift_size:
            return None
        mask = x.new_zeros((1, side, side, 1))
        slices = (slice(0, -self.win_size), slice(-self.win_size, -self.shift_size), slice(-self.shift_size, None))
        for index, (hs, ws) in enumerate((hs, ws) for hs in slices for ws in slices):
            mask[:, hs, ws, :] = index
        windows = window_partition(mask, self.win_size).view(-1, self.win_size ** 2)
        differences = windows.unsqueeze(1) - windows.unsqueeze(2)
        return differences.masked_fill(differences != 0, -100.0).masked_fill(differences == 0, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, c = x.shape
        side = spatial_side(x)
        shortcut = x
        y = self.norm1(x).view(b, side, side, c)
        if self.shift_size:
            y = torch.roll(y, (-self.shift_size, -self.shift_size), (1, 2))
        windows = window_partition(y, self.win_size).view(-1, self.win_size ** 2, c)
        if self.modulator is not None:
            windows = windows + self.modulator.weight
        windows = self.attn(windows, self.shift_mask(side, x))
        y = window_reverse(windows.view(-1, self.win_size, self.win_size, c), self.win_size, side, side)
        if self.shift_size:
            y = torch.roll(y, (self.shift_size, self.shift_size), (1, 2))
        x = shortcut + self.drop_path(y.view(b, side * side, c))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class BasicUformerLayer(nn.Module):
    def __init__(self, dim: int, resolution: int, depth: int, heads: int,
                 window: int, drop_paths: Sequence[float], modulator: bool = False) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            LeWinTransformerBlock(dim, resolution, heads, window,
                                  0 if index % 2 == 0 else window // 2, drop_paths[index], modulator)
            for index in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class Downsample(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(input_channels, output_channels, 4, stride=2, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        side = spatial_side(x)
        return self.conv(x.transpose(1, 2).reshape(x.shape[0], -1, side, side)).flatten(2).transpose(1, 2).contiguous()


class Upsample(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.deconv = nn.Sequential(nn.ConvTranspose2d(input_channels, output_channels, 2, stride=2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        side = spatial_side(x)
        return self.deconv(x.transpose(1, 2).reshape(x.shape[0], -1, side, side)).flatten(2).transpose(1, 2).contiguous()


class InputProj(nn.Module):
    def __init__(self, channels: int, dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(channels, dim, 3, padding=1), nn.LeakyReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2).contiguous()


class OutputProj(nn.Module):
    def __init__(self, dim: int, channels: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(dim, channels, 3, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        side = spatial_side(x)
        return self.proj(x.transpose(1, 2).reshape(x.shape[0], -1, side, side))


class Uformer(nn.Module):
    def __init__(self, variant: str = 'tiny', in_channels: int = 1,
                 img_size: int = 256, window_size: int = 8,
                 drop_path_rate: float = 0.1) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f'Unknown Uformer variant: {variant}')
        if img_size % (16 * window_size):
            raise ValueError('img_size must be a multiple of 16 * window_size.')
        embed, depths = VARIANTS[variant]
        self.variant, self.in_channels = variant, in_channels
        self.img_size, self.window_size = img_size, window_size
        self.input_proj = InputProj(in_channels, embed)
        self.output_proj = OutputProj(2 * embed, in_channels)
        self.pos_drop = nn.Dropout(0)
        encoder_rates = torch.linspace(0, drop_path_rate, sum(depths[:4])).tolist()
        decoder_rates = encoder_rates[::-1]
        position = 0
        for index in range(4):
            dim = embed * 2 ** index
            setattr(self, f'encoderlayer_{index}', BasicUformerLayer(
                dim, img_size // 2 ** index, depths[index], HEADS[index], window_size,
                encoder_rates[position:position + depths[index]],
            ))
            setattr(self, f'dowsample_{index}', Downsample(dim, 2 * dim))
            position += depths[index]
        self.conv = BasicUformerLayer(16 * embed, img_size // 16, depths[4], HEADS[4],
                                     window_size, [drop_path_rate] * depths[4])
        position = 0
        for index in range(4):
            dim = embed * 2 ** (4 - index)
            # After the first decoder block the concatenated representation has
            # twice as many channels as that spatial scale's encoder feature.
            input_dim = 16 * embed if index == 0 else 2 * dim
            setattr(self, f'upsample_{index}', Upsample(input_dim, dim // 2))
            setattr(self, f'decoderlayer_{index}', BasicUformerLayer(
                dim, img_size // 2 ** (3 - index), depths[5 + index], HEADS[5 + index],
                window_size, decoder_rates[position:position + depths[5 + index]], modulator=True,
            ))
            position += depths[5 + index]
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward_core(self, image: torch.Tensor) -> torch.Tensor:
        """Unclipped residual output on a square, hierarchy-compatible image."""
        features = self.pos_drop(self.input_proj(image))
        skips = []
        for index in range(4):
            features = getattr(self, f'encoderlayer_{index}')(features)
            skips.append(features)
            features = getattr(self, f'dowsample_{index}')(features)
        features = self.conv(features)
        for index in range(4):
            features = getattr(self, f'upsample_{index}')(features)
            features = torch.cat((features, skips[3 - index]), dim=-1)
            features = getattr(self, f'decoderlayer_{index}')(features)
        return image + self.output_proj(features)

    def padded_side(self, height: int, width: int) -> int:
        multiple = 16 * self.window_size
        return max(multiple, math.ceil(max(height, width) / multiple) * multiple)

    def forward(self, image: torch.Tensor, looks: object = None) -> torch.Tensor:
        del looks  # One fixed-L checkpoint per experiment; no added conditioning.
        if image.ndim != 4 or image.shape[1] != self.in_channels:
            raise ValueError(f'Expected B x {self.in_channels} x H x W input.')
        height, width = image.shape[-2:]
        side = self.padded_side(height, width)
        padded = image
        # Multiple steps retain reflection semantics when a small image needs
        # padding larger than its original extent. Single-pixel axes replicate.
        while padded.shape[-2] < side or padded.shape[-1] < side:
            h, w = padded.shape[-2:]
            ph, pw = min(side - h, max(1, h - 1)), min(side - w, max(1, w - 1))
            padded = F.pad(padded, (0, pw, 0, ph), mode='reflect' if min(h, w) > 1 else 'replicate')
        return self.forward_core(padded)[..., :height, :width]
