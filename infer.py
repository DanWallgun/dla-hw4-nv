import argparse

import torch
import torchaudio
from torch.nn import functional as F

from models import Generator
from configs import train_config
from mels import MelSpectrogram, MelSpectrogramConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav', required=True)
    parser.add_argument('--ckpt', required=False, default='infer_checkpoint.dict')
    args = parser.parse_args()

    wav = torchaudio.load(args.wav)[0].to(train_config.device)
    melspec_transform = MelSpectrogram(MelSpectrogramConfig()).to(train_config.device)
    mel = melspec_transform(wav)

    generator = Generator().to(train_config.device).eval()
    full_ckpt = torch.load(args.ckpt)
    generator.load_state_dict(full_ckpt['generator'])
    
    fake_wav = generator(mel)
    torchaudio.save(
        args.wav + '-vocoder.wav',
        fake_wav.cpu(), MelSpectrogramConfig.sr
    )


if __name__ == '__main__':
    main()