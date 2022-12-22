#!/bin/bash

if [ "$1" == "full" ]; then
    # full state https://drive.google.com/file/d/1E3tgnRApfl0RbAtbsNC7zaQxjVxy0szm/view?usp=sharing
    gdown \
        "https://drive.google.com/u/0/uc?id=1E3tgnRApfl0RbAtbsNC7zaQxjVxy0szm&confirm=yes" \
        -O ./ckpts/full_checkpoint.dict
else
    # converted (only generator) https://drive.google.com/file/d/1j6BdVOL5aQTpUb_reWInoUpfkvby3Njc/view?usp=sharing
    gdown \
        "https://drive.google.com/u/0/uc?id=1j6BdVOL5aQTpUb_reWInoUpfkvby3Njc&confirm=yes" \
        -O ./infer_checkpoint.dict
fi
