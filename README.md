# dla-hw4-nv

Чтобы учить, указываем параметры в `configs.py` и запускаем `python train.py`.
Перед этим можно скачать предобученный checkpoint (осторожно, ~1GB): `./scripts/download_trained.sh full`

## Inference
```bash
# python=3.9 is recommended
# for example conda create -n dla-test python=3.9

pip install -r requirements.txt
./scripts/download_trained.sh

python infer.py --wav ./data/test_wavs/audio_1.wav --ckpt ./infer_checkpoint.dict
python infer.py --wav ./data/test_wavs/audio_2.wav --ckpt ./infer_checkpoint.dict
python infer.py --wav ./data/test_wavs/audio_3.wav --ckpt ./infer_checkpoint.dict
```

## Results
В папке [data/test_results](https://github.com/DanWallgun/dla-hw4-nv/blob/main/data/test_results/) или в конце wandb отчёта  
[audio_1](https://github.com/DanWallgun/dla-hw4-nv/raw/main/data/test_results/vocoder-audio_1.wav)
[audio_2](https://github.com/DanWallgun/dla-hw4-nv/raw/main/data/test_results/vocoder-audio_2.wav)
[audio_3](https://github.com/DanWallgun/dla-hw4-nv/raw/main/data/test_results/vocoder-audio_2.wav)

## Report 
[WandB](https://wandb.ai/danwallgun/hifi-gan/reports/DLA-HW4-NV-Report--VmlldzozMTk3MTI5)

## Credits
- https://arxiv.org/abs/2010.05646
- https://github.com/jik876/hifi-gan
