import os
import itertools

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from wandb_writer import WanDBWriter
from dataset import WavMelDataset
from models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss, discriminator_loss
from configs import train_config
from mels import MelSpectrogramConfig, MelSpectrogram


def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)
set_random_seed(42)


def main():
    device = train_config.device
    last_epoch = train_config.last_epoch

    generator = Generator().to(device)
    mpdiscriminator = MultiPeriodDiscriminator().to(device)
    msdiscriminator = MultiScaleDiscriminator().to(device)

    goptimizer = torch.optim.AdamW(
        [
            {
                'params': generator.parameters(),
                'initial_lr': train_config.learning_rate * train_config.lr_decay ** (last_epoch + 1)
            }
        ],
        train_config.learning_rate, betas=train_config.betas,
    )
    doptimizer = torch.optim.AdamW(
        [
            {
                'params': itertools.chain(msdiscriminator.parameters(), mpdiscriminator.parameters()),
                'initial_lr': train_config.learning_rate * train_config.lr_decay ** (last_epoch + 1)
            }
        ],
        train_config.learning_rate, betas=train_config.betas,
    )

    if os.path.exists(train_config.full_frequent_checkpoint_name):
        full_ckpt = torch.load(train_config.full_frequent_checkpoint_name)
        generator.load_state_dict(full_ckpt['generator'])
        mpdiscriminator.load_state_dict(full_ckpt['mpdiscriminator'])
        msdiscriminator.load_state_dict(full_ckpt['msdiscriminator'])
        goptimizer.load_state_dict(full_ckpt['goptimizer'])
        doptimizer.load_state_dict(full_ckpt['doptimizer'])

    gscheduler = torch.optim.lr_scheduler.ExponentialLR(
        goptimizer,
        gamma=train_config.lr_decay,
        last_epoch=last_epoch
    )
    dscheduler = torch.optim.lr_scheduler.ExponentialLR(
        doptimizer,
        gamma=train_config.lr_decay,
        last_epoch=last_epoch
    )

    train_ds = WavMelDataset(train_config.wav_path)
    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers,
    )
    test_ds = WavMelDataset(train_config.test_wav_path, crop_segment=False)
    
    current_step = (last_epoch + 1) * len(train_loader)
    logger = WanDBWriter(train_config)
    tqdm_bar = tqdm(total=(train_config.epochs - last_epoch - 1) * len(train_loader))

    melspec_transform = MelSpectrogram(MelSpectrogramConfig()).to(device)
    generator.train()
    mpdiscriminator.train()
    msdiscriminator.train()

    for epoch in range(last_epoch + 1, train_config.epochs):
        for idx, batch in enumerate(train_loader):
            current_step += 1
            tqdm_bar.update(1)
            logger.set_step(current_step)

            wav = batch.to(device)
            mel = melspec_transform(wav)
            
            fake_wav = generator(mel)
            fake_mel = melspec_transform(fake_wav)

            # Discriminators
            doptimizer.zero_grad()
            # period
            real_results, fake_results, _, _ = mpdiscriminator(wav, fake_wav.detach())
            loss_disc_p, _, _ = discriminator_loss(real_results, fake_results)
            # scale
            real_results, fake_results, _, _ = msdiscriminator(wav, fake_wav.detach())
            loss_disc_s, _, _ = discriminator_loss(real_results, fake_results)
            loss_disc_all = loss_disc_s + loss_disc_p
            loss_disc_all.backward()
            doptimizer.step()

            logger.add_scalar('loss_disc_period', loss_disc_p)
            logger.add_scalar('loss_disc_scale', loss_disc_s)
            logger.add_scalar('loss_disc_all', loss_disc_all)
            
            # Generator
            goptimizer.zero_grad()
            # period
            _, fake_results, fmap_reals, fmap_fakes = mpdiscriminator(wav, fake_wav)
            loss_fm_p = feature_loss(fmap_reals, fmap_fakes)
            loss_gen_p, _ = generator_loss(fake_results)
            # scale
            _, fake_results, fmap_reals, fmap_fakes = msdiscriminator(wav, fake_wav)
            loss_fm_s = feature_loss(fmap_reals, fmap_fakes)
            loss_gen_s, _ = generator_loss(fake_results)
            # reconstruction
            loss_mel = F.l1_loss(mel, fake_mel) * 45
            loss_gen_all = loss_gen_s + loss_gen_p + loss_fm_s + loss_fm_p + loss_mel
            loss_gen_all.backward()
            goptimizer.step()
            
            logger.add_scalar('discloss_gen_period', loss_gen_p)
            logger.add_scalar('featureloss_gen_period', loss_fm_p)
            logger.add_scalar('discloss_gen_scale', loss_gen_s)
            logger.add_scalar('featureloss_gen_scale', loss_fm_s)
            logger.add_scalar('l1loss_gen_mel', loss_mel)
            logger.add_scalar('loss_gen_all', loss_gen_all)

            if current_step % train_config.log_step == 0 and current_step != 0:
                logger.set_step(current_step, 'test')
                generator.eval()
                torch.cuda.empty_cache()
                with torch.no_grad():
                    test_error_total = 0.0
                    for j, wav in enumerate(test_ds):
                        wav = wav.unsqueeze(0).to(device)
                        
                        fake_wav = generator(mel)
                        fake_mel = melspec_transform(fake_wav)

                        logger.add_audio(f'generated/{j}', fake_wav, MelSpectrogramConfig.sr)

                        test_error_total += F.l1_loss(mel, fake_mel).item()

                    logger.add_scalar('mel_spec_error', test_error_total)
                generator.train()
            
            if current_step % train_config.frequent_save_current_model == 0 and current_step != 0:
                torch.save(
                    {
                        'generator': generator.state_dict(),
                        'mpdiscriminator': mpdiscriminator.state_dict(),
                        'msdiscriminator': msdiscriminator.state_dict(),
                        'goptimizer': goptimizer.state_dict(),
                        'doptimizer': doptimizer.state_dict(),
                    },
                    train_config.full_frequent_checkpoint_name
                )
        
        gscheduler.step()
        dscheduler.step()

if __name__ == '__main__':
    main()