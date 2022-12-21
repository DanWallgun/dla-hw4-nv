# dla-hw4-nv

Чтобы учить, указываем параметры в configs.py и запускаем python train.py

## Inference
```
./scripts/download_trained.sh
python infer.py --wav ./data/test_wavs/audio_1.wav --ckpt ./ckpts/full_frequent_checkpoint.dict
```

## Report 
[WandB](https://wandb.ai/danwallgun/hifi-gan/reports/DLA-HW4-NV-Report--VmlldzozMTk3MTI5)

## Credits
- https://arxiv.org/abs/2010.05646
- https://github.com/jik876/hifi-gan