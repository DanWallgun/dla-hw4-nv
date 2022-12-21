from dataclasses import dataclass

import torch


@dataclass
class HifiGANConfig:
    leaky_relu_slope = 0.1

    # generator
    upsample_rates = [8, 8, 2, 2]
    upsample_kernel_sizes = [16, 16, 4, 4]
    upsample_initial_channel = 512
    resblock_kernel_sizes = [3, 7, 11]
    resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]

    # discriminator
    discriminator_periods = [2, 3, 5, 7, 11]


@dataclass
class TrainConfig:
    wav_path = './data/LJSpeech-1.1/wavs'
    test_wav_path = './data/test_wavs'
    segment_size = 8192
    full_frequent_checkpoint_name = './ckpts/full_frequent_checkpoint.dict'

    checkpoint_path = "./ckpts"
    logger_path = "./logger"
    mel_ground_truth = "./mels"
    alignment_path = "./alignments"
    data_path = './data/train.txt'
    energy_path = "./energy"
    pitch_path = "./pitch"
    
    wandb_project = 'hifi-gan'
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    
    num_workers = 0
    batch_size = 16
    epochs = 2
    n_warm_up_step = 1

    learning_rate = 0.0002
    betas = (0.8, 0.99)
    lr_decay = 0.999

    log_step = 50
    frequent_save_current_model = 100
    save_step = 3000
    clear_Time = 20

    batch_expand_size = 32
    

# singletons
model_config = HifiGANConfig()
train_config = TrainConfig()