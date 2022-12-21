import argparse

import torch
import torchaudio
from torch.nn import functional as F

from dataset import pad_melspec_transform
from models import Generator
from configs import train_config
from mels import MelSpectrogram, melspec_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav', required=True)
    parser.add_argument('--ckpt', required=False, default=train_config.full_frequent_checkpoint_name)
    args = parser.parse_args()

    wav = torchaudio.load(args.wav)[0].to(train_config.device)
    melspec_transform = MelSpectrogram(melspec_config).to(train_config.device)
    mel = pad_melspec_transform(melspec_transform, wav)

    generator = Generator().to(train_config.device)
    full_ckpt = torch.load(args.ckpt)
    generator.load_state_dict(full_ckpt['generator'].state_dict())
    
    fake_wav = generator(mel)
    torchaudio.save(
        args.wav + '-vocoder.wav',
        fake_wav.cpu(), melspec_config.sr
    )


if __name__ == '__main__':
    main()