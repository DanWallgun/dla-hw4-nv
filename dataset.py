import os
import random

import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

from configs import train_config


class WavMelDataset(Dataset):

    def __init__(self, wav_path, crop_segment=True):
        self.wav_path = wav_path
        self.crop_segment = crop_segment
        self.filenames = sorted(os.listdir(self.wav_path))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        wav = torchaudio.load(os.path.join(self.wav_path, self.filenames[index]))[0][0]
        if self.crop_segment:
            if wav.size(-1) < train_config.segment_size:
                wav = F.pad(wav, (0, train_config.segment_size - wav.size(-1)))
            else:
                start = random.randint(0, wav.size(-1) - train_config.segment_size)
                wav = wav[..., start:start + train_config.segment_size]
        return wav