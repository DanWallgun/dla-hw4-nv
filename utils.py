import torch.nn as nn


def init_weights(module, mean=0.0, std=0.01):
    if isinstance(module, (nn.Conv1d, nn.Conv2d)):
        module.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return (kernel_size - 1) * dilation // 2
