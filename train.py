import os
import itertools

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from wandb_writer import WanDBWriter
from dataset import WavMelDataset
from models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss, discriminator_loss
from configs import model_config, train_config
from mels import melspec_config


def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)
set_random_seed(42)


def pad_difference(x, y, value=0):
    if x.size(-1) > y.size(-1):
        y = F.pad(y, (0, x.size(-1) - y.size(-1)), value=value)
    elif x.size(-1) < y.size(-1):
        x = F.pad(x, (0, y.size(-1) - x.size(-1)), value=value)
    return x, y


def set_requires_grad(model, value):
    for p in model.parameters():
        p.requires_grad = value


def main():
    device = train_config.device

    generator = Generator().to(device)
    mpdiscriminator = MultiPeriodDiscriminator().to(device)
    msdiscriminator = MultiScaleDiscriminator().to(device)

    goptimizer = torch.optim.AdamW(generator.parameters(), train_config.learning_rate, betas=train_config.betas)
    doptimizer = torch.optim.AdamW(
        itertools.chain(msdiscriminator.parameters(), mpdiscriminator.parameters()),
        train_config.learning_rate, betas=train_config.betas
    )

    if os.path.exists(train_config.full_frequent_checkpoint_name):
        full_ckpt = torch.load(train_config.full_frequent_checkpoint_name)
        generator.load_state_dict(full_ckpt['generator'].state_dict())
        mpdiscriminator.load_state_dict(full_ckpt['mpdiscriminator'].state_dict())
        msdiscriminator.load_state_dict(full_ckpt['msdiscriminator'].state_dict())
        goptimizer.load_state_dict(full_ckpt['goptimizer'].state_dict())
        doptimizer.load_state_dict(full_ckpt['doptimizer'].state_dict())

    last_epoch = -1

    gscheduler = torch.optim.lr_scheduler.ExponentialLR(goptimizer, gamma=train_config.lr_decay, last_epoch=last_epoch)
    dscheduler = torch.optim.lr_scheduler.ExponentialLR(doptimizer, gamma=train_config.lr_decay, last_epoch=last_epoch)

    remove_channel_collator = lambda items: (torch.cat([item[0] for item in items], dim=0), torch.cat([item[1] for item in items], dim=0))

    train_ds = WavMelDataset(train_config.wav_path)
    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        collate_fn=remove_channel_collator,
        batch_size=train_config.batch_size,
        # pin_memory=True,
        num_workers=train_config.num_workers,
        drop_last=True
    )

    test_ds = WavMelDataset(train_config.test_wav_path, crop_segment=False)

    current_step = 0
    logger = WanDBWriter(train_config)
    tqdm_bar = tqdm(total=train_config.epochs * len(train_loader))

    generator.train()
    mpdiscriminator.train()
    msdiscriminator.train()


    for epoch in range(train_config.epochs):
        for idx, batch in enumerate(train_loader):
            current_step += 1
            tqdm_bar.update(1)
            logger.set_step(current_step)

            # wav = batch[0].to(device)
            # mel = batch[1].to(device)
            wav, mel = batch  # already on device

            fake_wav = generator(mel)
            fake_mel = train_ds.pad_melspec(fake_wav)

            # print(wav.size(), mel.size())
            # print(fake_wav.size(), fake_mel.size())

            wav, fake_wav = pad_difference(wav, fake_wav)
            mel, fake_mel = pad_difference(mel, fake_mel, value=melspec_config.pad_value)

            # print(wav.size(), mel.size())
            # print(fake_wav.size(), fake_mel.size())

            doptimizer.zero_grad()
            # period
            y_df_hat_r, y_df_hat_g, _, _ = mpdiscriminator(wav, fake_wav.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)
            # scale
            y_ds_hat_r, y_ds_hat_g, _, _ = msdiscriminator(wav, fake_wav.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
            loss_disc_all = loss_disc_s + loss_disc_f
            loss_disc_all.backward()
            doptimizer.step()

            logger.add_scalar('loss_disc_period', loss_disc_f)
            logger.add_scalar('loss_disc_scale', loss_disc_s)
            logger.add_scalar('loss_disc_all', loss_disc_all)
            
            # Generator
            goptimizer.zero_grad()
            # L1 Mel-Spectrogram Loss
            loss_mel = F.l1_loss(mel, fake_mel) * 45
            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpdiscriminator(wav, fake_wav)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msdiscriminator(wav, fake_wav)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
            loss_gen_all.backward()
            goptimizer.step()
            
            logger.add_scalar('discloss_gen_scale', loss_gen_s)
            logger.add_scalar('discloss_gen_period', loss_gen_f)
            logger.add_scalar('featureloss_gen_scale', loss_fm_s)
            logger.add_scalar('featureloss_gen_period', loss_fm_f)
            logger.add_scalar('l1loss_gen_mel', loss_mel)

            # log something
            if current_step % train_config.log_step == 0 and current_step != 0:
                logger.set_step(current_step, 'test')
                generator.eval()
                torch.cuda.empty_cache()
                with torch.no_grad():
                    val_err_tot = 0.0
                    for j, (wav, mel) in enumerate(test_ds):
                        # не делаем unsqueeze для получения "батча",
                        # так как уже есть одиночный канал,
                        # который мы не убираем в силу отсутствия collator-а
                        
                        fake_wav = generator(mel)
                        fake_mel = test_ds.pad_melspec(fake_wav)

                        logger.add_audio(f'generated/{j}', fake_wav, melspec_config.sr)

                        mel, fake_mel = pad_difference(mel, fake_mel, value=melspec_config.pad_value)
                        val_err_tot += F.l1_loss(mel, fake_mel).item()

                    logger.add_scalar("mel_spec_error", val_err_tot)
                generator.train()
            
            if current_step % train_config.frequent_save_current_model == 0 and current_step != 0:
                torch.save(
                    {
                        'generator': generator,
                        'mpdiscriminator': mpdiscriminator,
                        'msdiscriminator': msdiscriminator,
                        'goptimizer': goptimizer,
                        'doptimizer': doptimizer,
                    },
                    train_config.full_frequent_checkpoint_name
                )
        
        gscheduler.step()
        dscheduler.step()

if __name__ == '__main__':
    main()