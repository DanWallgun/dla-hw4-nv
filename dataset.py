import os
import random

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

from mels import MelSpectrogram, melspec_config
from configs import train_config


class WavMelDataset(Dataset):
    def __init__(self, wav_path):
        self.wav_path = wav_path
        self.melspec = MelSpectrogram(melspec_config, center=False).to(train_config.device)
        self.filenames = os.listdir(self.wav_path)
    def __len__(self):
        return len(self.filenames)
    def __getitem__(self, index):
        wav = torchaudio.load(os.path.join(self.wav_path, self.filenames[index])).to(train_config.device)
        if wav.size(1) < train_config.segment_size:
            wav = F.pad(wav, (0, train_config.segment_size - wav.size(1)))
        else:
            start = random.randint(0, wav.size(1) - train_config.segment_length)
            wav = wav[start:start + train_config.segment_length]
        return wav, self.pad_melspec(wav)
    def pad_melspec(self, wav):
        pad_length = (melspec_config.n_fft - melspec_config.hop_length) // 2
        padded_wav = F.pad(wav, (pad_length, pad_length), mode='reflect')
        return self.melspec(padded_wav)
    