import argparse
import torch
from configs import train_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=False, default=train_config.full_frequent_checkpoint_name)
    parser.add_argument('--names', required=True, nargs='+')
    args = parser.parse_args()

    full_ckpt = torch.load(args.ckpt)
    torch.save(
        {
            name: full_ckpt[name]
            for name in args.names
        },
        args.ckpt + '.converted',
    )


if __name__ == '__main__':
    main()