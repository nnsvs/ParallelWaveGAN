# NOTE: code was adapted from https://github.com/MoonInTheRiver/DiffSinger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, weight_norm

from .nsf import SourceModuleHnNSF

LRELU_SLOPE = 0.1


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class ResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
            ]
        )
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class Conv1d1x1(Conv1d):
    """1x1 Conv1d with customized initialization."""

    def __init__(self, in_channels, out_channels, bias):
        """Initialize 1x1 Conv1d module."""
        super(Conv1d1x1, self).__init__(
            in_channels, out_channels, kernel_size=1, padding=0, dilation=1, bias=bias
        )


class HnSincHifiGanGenerator(torch.nn.Module):
    def __init__(
        self,
        cin_channels=67,
        out_channels=1,
        sample_rate=24000,
        out_lf0_idx=60,
        out_lf0_mean=5.953093881972361,
        out_lf0_scale=0.23435173188961034,
        out_vuv_idx=61,
        out_vuv_mean=0.8027627522670242,
        out_vuv_scale=0.39791295007789834,
        vuv_threshold=0.3,
        aux_context_window=0,
        resblock="1",
        resblock_kernel_sizes=[3, 7, 11],
        upsample_rates=[8, 4, 2, 2],
        upsample_kernel_sizes=[16, 16, 4, 4],
        upsample_initial_channel=512,
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        melf0vuv=False,
    ):
        super(HnSincHifiGanGenerator, self).__init__()
        self.out_lf0_idx = out_lf0_idx
        self.out_lf0_mean = out_lf0_mean
        self.out_lf0_scale = out_lf0_scale
        self.out_vuv_idx = out_vuv_idx
        self.out_vuv_mean = out_vuv_mean
        self.out_vuv_scale = out_vuv_scale
        self.vuv_threshold = vuv_threshold
        self.aux_context_window = aux_context_window
        self.melf0vuv = melf0vuv

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        self.harmonic_num = 8
        # Upsampling for F0: don't smooth up-sampled F0
        self.f0_upsamp = torch.nn.Upsample(scale_factor=np.prod(upsample_rates))
        self.m_source = SourceModuleHnNSF(
            sampling_rate=sample_rate, harmonic_num=self.harmonic_num
        )
        self.noise_convs = nn.ModuleList()
        self.conv_pre = weight_norm(
            Conv1d(cin_channels, upsample_initial_channel, 7, 1, padding=3)
        )
        resblock = ResBlock1 if resblock == "1" else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(c_cur * 2, c_cur, k, u, padding=(k - u) // 2)
                )
            )
            if i + 1 < len(upsample_rates):
                stride_f0 = np.prod(upsample_rates[i + 1 :])
                self.noise_convs.append(
                    Conv1d(
                        1,
                        c_cur,
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=stride_f0 // 2,
                    )
                )
            else:
                self.noise_convs.append(Conv1d(1, c_cur, kernel_size=1))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, out_channels, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x, f0=None):
        # (B, C, T) -> (B, T, C)
        x = x.transpose(1, 2)
        if f0 is not None:
            # NOTE: DiffSinger's case
            f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)
        else:
            lf0 = (
                x[:, :, self.out_lf0_idx].unsqueeze(-1) * self.out_lf0_scale
                + self.out_lf0_mean
            )
            vuv = (
                x[:, :, self.out_vuv_idx].unsqueeze(-1) * self.out_vuv_scale
                + self.out_vuv_mean
            )

            if self.aux_context_window > 0:
                lf0 = lf0[:, self.aux_context_window : -self.aux_context_window]
                vuv = vuv[:, self.aux_context_window : -self.aux_context_window]

            f0 = torch.exp(lf0)
            f0[vuv < self.vuv_threshold] = 0

            # harmonic-source signal, noise-source signal, uv flag
            f0 = self.f0_upsamp(f0.permute(0, 2, 1)).permute(0, 2, 1)

            # Drop lf0 and vuv
            if self.melf0vuv:
                x = x[:, :, : self.out_lf0_idx]

        har_source, _, _ = self.m_source(f0)
        har_source = har_source.transpose(1, 2)

        x = x.transpose(1, 2)
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            x_source = self.noise_convs[i](har_source)
            x = x + x_source
            xs = 0
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
