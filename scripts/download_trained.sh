#!/bin/bash

# https://drive.google.com/file/d/1p51DxEojwDHc4hUOt-u__ciiLJa4ljom/view?usp=sharing
gdown "https://drive.google.com/u/0/uc?id=1p51DxEojwDHc4hUOt-u__ciiLJa4ljom&confirm=yes" -O ./full_frequent_checkpoint.dict
mkdir ckpts
mv ./full_frequent_checkpoint.dict ./ckpts/full_frequent_checkpoint.dict