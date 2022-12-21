# hifi-gan https://arxiv.org/abs/2010.05646
# implementation inspired by https://github.com/jik876/hifi-gan/blob/master/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import init_weights, get_padding
from configs import model_config


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
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
        for i in range(len(self.upscales)):
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
                    xs = self.resblocks[i * len(model_config.resblock_kernel_sizes) + j](x)
                else:
                    xs += self.resblocks[i * len(model_config.resblock_kernel_sizes) + j](x)
            x = xs / len(model_config.resblock_kernel_sizes)
        x = F.leaky_relu(x)
        x = self.post_conv(x)
        x = torch.tanh(x)
        return x.squeeze(dim=1)


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
        b, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), 'reflect')
            t = t + n_pad
        x = x.view(b, 1, t // self.period, self.period)

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
            OnePeriodDiscriminator(period)
            for period in model_config.discriminator_periods
        ])

    def forward(self, real, fake):
        real_results = []
        fake_results = []
        fmap_reals = []
        fmap_fakes = []
        for d in self.discriminators:
            real_result, fmap_real = d(real)
            fake_result, fmap_fake = d(fake)
            real_results.append(real_result)
            fake_results.append(fake_result)
            fmap_reals.append(fmap_real)
            fmap_fakes.append(fmap_fake)
        return real_results, fake_results, fmap_reals, fmap_fakes


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
        x = x.unsqueeze(dim=1)
        fmap = []
        for layer in self.convs:
            x = F.leaky_relu(layer(x), model_config.leaky_relu_slope)
            fmap.append(x)
        x = self.post_conv(x)
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

    def forward(self, real, fake):
        real_results = []
        fake_results = []
        fmap_reals = []
        fmap_fakes = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                real = self.poolings[i - 1](real)
                fake = self.poolings[i - 1](fake)
            real_result, fmap_real = d(real)
            fake_result, fmap_fake = d(fake)
            real_results.append(real_result)
            fmap_reals.append(fmap_real)
            fake_results.append(fake_result)
            fmap_fakes.append(fmap_fake)

        return real_results, fake_results, fmap_reals, fmap_fakes


def feature_loss(fmap_reals, fmap_fakes):
    loss = 0
    for f_real, f_fake in zip(fmap_reals, fmap_fakes):
        for r, f in zip(f_real, f_fake):
            loss += F.l1_loss(r, f, reduction='mean')
    return loss * 2 


def discriminator_loss(disc_real_outputs, disc_fake_outputs):
    loss = 0
    r_losses = []
    f_losses = []
    for disc_real, disc_fake in zip(disc_real_outputs, disc_fake_outputs):
        r_loss = F.mse_loss(disc_real, torch.ones_like(disc_real))
        f_loss = F.mse_loss(disc_fake, torch.zeros_like(disc_fake))
        loss += (r_loss + f_loss)
        r_losses.append(r_loss.item())
        f_losses.append(f_loss.item())

    return loss, r_losses, f_losses


def generator_loss(disc_outputs):
    loss = 0
    fake_losses = []
    for d in disc_outputs:
        lss = F.mse_loss(d, torch.ones_like(d))
        fake_losses.append(lss)
        loss += lss
    return loss, fake_losses