#!/bin/bash

if [ "$1" == "full" ]; then
    # full state https://drive.google.com/file/d/1gZTGZKnWgQVPaRD5wEmuD5igRSDhHOEH/view?usp=sharing
    gdown \
        "https://drive.google.com/u/0/uc?id=1gZTGZKnWgQVPaRD5wEmuD5igRSDhHOEH&confirm=yes" \
        -O ./ckpts/full_checkpoint.dict
else
    # converted (only generator) https://drive.google.com/file/d/1bI35lkfBiNIYN5Uf6DOJsjczNzlN7Vjh/view?usp=share_link
    gdown \
        "https://drive.google.com/u/0/uc?id=1bI35lkfBiNIYN5Uf6DOJsjczNzlN7Vjh&confirm=yes" \
        -O ./infer_checkpoint.dict
fi
