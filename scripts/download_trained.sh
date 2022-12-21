#!/bin/bash

# https://drive.google.com/file/d/104n-5M8QdchjKGF-07evNm-B11P9D6kt/view?usp=sharing
gdown https://drive.google.com/u/0/uc?id=104n-5M8QdchjKGF-07evNm-B11P9D6kt
mkdir ckpts
mv ./full_frequent_checkpoint.dict ./ckpts/full_frequent_checkpoint.dict