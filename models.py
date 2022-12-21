# hifi-gan https://arxiv.org/abs/2010.05646
# implementation inspired by https://github.com/jik876/hifi-gan/blob/master/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import init_weights, get_padding
from configs import model_config


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()

        def get_conv(channels, kernel_size, dilation):
            return nn.utils.weight_norm(
                nn.Conv1d(
                    channels, channels,
                    kernel_size=kernel_size, stride=1, dilation=dilation,
                    padding=(kernel_size - 1) * dilation // 2,
                )
            )

        self.convs = nn.ModuleList([
            get_conv(channels, kernel_size, d)
            for d in dilation
        ])
        self.convs.apply(init_weights)
    
    def forward(self, x):
        for layer in self.convs:
            x = x + layer(F.leaky_relu(x, model_config.leaky_relu_slope))
        return x

    def remove_weight_norm(self):
        for layer in self.convs:
            nn.utils.remove_weight_norm(layer)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.pre_conv = nn.utils.weight_norm(nn.Conv1d(80, model_config.upsample_initial_channel, 7, 1, padding=3))

        self.upscales = nn.ModuleList([
            nn.utils.weight_norm(
                nn.ConvTranspose1d(
                    model_config.upsample_initial_channel // (2 ** i),
                    model_config.upsample_initial_channel // (2 ** (i + 1)),
                    k, u, padding=(k - u) // 2
                )
            )
            for i, (u, k) in enumerate(zip(model_config.upsample_rates, model_config.upsample_kernel_sizes))
        ])
        self.upscales.apply(init_weights)

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = model_config.upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(model_config.resblock_kernel_sizes, model_config.resblock_dilation_sizes):
                self.resblocks.append(ResBlock(ch, k, d))

        self.post_conv = nn.utils.weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))
        self.post_conv.apply(init_weights)

    def forward(self, x):
        x = self.pre_conv(x)
        for i in range(len(model_config.upsample_rates)):
            x = F.leaky_relu(x, model_config.leaky_relu_slope)
            x = self.upscales[i](x)
            xs = None
            for j in range(len(model_config.resblock_kernel_sizes)):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / len(model_config.resblock_kernel_sizes)
        x = F.leaky_relu(x)
        x = self.post_conv(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for layer in self.ups:
            nn.utils.remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()
        nn.utils.remove_weight_norm(self.pre_conv)
        nn.utils.remove_weight_norm(self.post_conv)


class OnePeriodDiscriminator(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            nn.utils.weight_norm(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            nn.utils.weight_norm(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            nn.utils.weight_norm(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            nn.utils.weight_norm(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.post_conv = nn.utils.weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
    
    def forward(self, x):
        fmap = []

        # перекладываем по выбранному периоду сэмплы построчно в 2д тензор
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), 'reflect')
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for layer in self.convs:
            x = F.leaky_relu(layer(x), model_config.leaky_relu_slope)
            fmap.append(x)
        x = self.post_conv(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            OnePeriodDiscriminator(period) for period in model_config.descriminitor_periods
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class OneScaleDiscriminator(nn.Module):
    def __init__(self, weight_norm=nn.utils.weight_norm):
        super().__init__()
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv1d(1, 128, 15, 1, padding=7)),
            weight_norm(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            weight_norm(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            weight_norm(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            weight_norm(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            weight_norm(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            weight_norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.post_conv = weight_norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, model_config.leaky_relu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            OneScaleDiscriminator(nn.utils.spectral_norm),
            OneScaleDiscriminator(),
            OneScaleDiscriminator(),
        ])
        self.poolings = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.poolings[i - 1](y)
                y_hat = self.poolings[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2 


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses