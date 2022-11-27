# Confidence-aware Training of Smoothed Classifiers for Certified Robustness

This repository contains code for the paper
**"Confidence-aware Training of Smoothed Classifiers for Certified Robustness" (AAAI 2023)** 
by [Jongheon Jeong](https://jh-jeong.github.io), [Seojin Kim](https://seojin-kim.github.io) and [Jinwoo Shin](http://alinlab.kaist.ac.kr/shin.html). 


## Environmental Setup
```
conda create -n catrs python=3.7
conda activate catrs

# Below is for linux, with CUDA 10.2; see https://pytorch.org/ for the correct command for your system
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch 

conda install scipy pandas statsmodels matplotlib seaborn
pip install setGPU tensorboardX
```

## Preprocessing

Our method utilizes `smoothed prediction` from the model trained by `Gaussian (Cohen et al., 2019)` baseline with &sigma;=0.25. The script `code/smooth_prediction.py` loads pretrained model and smooth out its prediction. For CIFAR-10 dataset, following commands produce the smoothed predictions of Gaussian baseline. One may skip the Gaussian pre-training step by using our preprocessed results in `test/*` in [link](https://drive.google.com/drive/folders/1TcjIkgSzWPOigD9aJk37UK6BAR0nzgKK?usp=sharing). For a more detailed instruction, please check [`EXPERIMENTS.MD`](EXPERIMENTS.MD).
```
CUDA_VISIBLE_DEVICES=0 python code/train_cohen.py cifar10 cifar_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150 --noise 0.25 --id 0
CUDA_VISIBLE_DEVICES=0 python code/smooth_prediction.py cifar10 logs/cifar10/cohen/noise_0.25/cifar_resnet110/0/checkpoint.pth.tar 0.25 test/smooth_prediction/cifar10/cohen/0/noise_train_0.25.tsv --N=10000 --skip=1 --split=train
```


## Training

The main script `train_catrs.py` is largely based on the codebase from ([Cohen et al (2019)](https://github.com/locuslab/smoothing), [Salman et al (2019)](https://github.com/Hadisalman/smoothing-adversarial), [Jeong and Shin (2020)](https://github.com/jh-jeong/smoothing-consistency), and [Jeong et al (2021)](https://github.com/jh-jeong/smoothmix)); We also provide training scripts 
to reproduce other baseline methods in `train_*.py`, as listed in what follows:

| File | Description |
| ------ | ------ |
| [train_catrs.py](code/train_catrs.py) (ours) | The main script for CAT-RS (Confidence-Aware Training for Randomized Smoothing) |
| [train_cohen.py](code/train_cohen.py) | Gaussian augmentation (Cohen et al., 2019) |
| [train_stab.py](code/train_stab.py) | Stability training (Li et al., 2019) |
| [train_salman.py](code/train_salman.py) | SmoothAdv (Salman et al., 2019) |
| [train_macer.py](code/train_macer.py) | MACER (Zhai et al., 2020) |
| [train_consistency.py](code/train_consistency.py) |  Consistency (Jeong and Shin, 2020) |
| [train_smoothmix.py](code/train_smoothmix.py) |  SmoothMix (Jeong et al., 2021) |

Below, we provide a sample command line input to run `train_catrs.py`
```
# CAT-RS training for CIFAR-10 dataset with noise level sigma=0.25
CUDA_VISIBLE_DEVICES=0 python code/train_catrs.py cifar10 cifar_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150 \
--num-noise-vec 4 --noise 0.25 --id 0 --eps 256.0 --num-steps 4 --lbd 0.5
```

Checkpoints for our experiments can be found in `data/*` of [link](https://drive.google.com/drive/folders/1TcjIkgSzWPOigD9aJk37UK6BAR0nzgKK?usp=sharing). For a more detailed instruction to run experiments, please check [`EXPERIMENTS.MD`](EXPERIMENTS.MD).

## Cerifying

All the testing scripts are originally from https://github.com/locuslab/smoothing:

* The script [certify.py](code/certify.py) certifies the robustness of a smoothed classifier.  For example,

```
# CAT-RS certification for CIFAR-10 dataset with noise level sigma=0.25
python code/certify.py cifar10 model_output_dir/checkpoint.pth.tar 0.25 certification_output --alpha 0.001 --N0 100 --N 100000
```

will load the base classifier saved at `model_output_dir/checkpoint.pth.tar`, smooth it using noise level &sigma;=0.25,
and certify the CIFAR-10 test set with parameters `N0=100`, `N=100000`, and `alpha=0.001`.

Certification results for our experiments can be found in `data/*` of [link](https://drive.google.com/drive/folders/1TcjIkgSzWPOigD9aJk37UK6BAR0nzgKK?usp=sharing). For a more detailed instruction to run experiments, please check [`EXPERIMENTS.MD`](EXPERIMENTS.MD).

## Other functionalities
* The script [predict.py](code/predict.py) makes predictions using a smoothed classifier.  For example,

```python code/predict.py cifar10 model_output_dir/checkpoint.pth.tar 0.25 prediction_outupt --alpha 0.001 --N 1000```

will load the base classifier saved at `model_output_dir/checkpoint.pth.tar`, smooth it using noise level &sigma;=0.25,
and classify the CIFAR-10 test set with parameters `N=1000` and `alpha=0.001`.

* The script [analyze.py](code/analyze.py) contains some useful classes and functions to analyze the result data 
from [certify.py](code/certify.py) or [predict.py](code/predict.py).
