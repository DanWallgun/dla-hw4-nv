import torch
import torch.nn.functional as F


def feature_loss(fmap_reals, fmap_fakes):
    loss = 0
    for f_real, f_fake in zip(fmap_reals, fmap_fakes):
        for r, f in zip(f_real, f_fake):
            loss += F.l1_loss(r, f, reduction='mean')
    return loss


def discriminator_loss(disc_real_outputs, disc_fake_outputs):
    loss = 0
    for disc_real, disc_fake in zip(disc_real_outputs, disc_fake_outputs):
        r_loss = F.mse_loss(disc_real, torch.ones_like(disc_real))
        f_loss = F.mse_loss(disc_fake, torch.zeros_like(disc_fake))
        loss += r_loss + f_loss
    return loss


def generator_loss(disc_outputs):
    loss = 0
    fake_losses = []
    for d in disc_outputs:
        lss = F.mse_loss(d, torch.ones_like(d))
        fake_losses.append(lss)
        loss += lss
    return loss