# Experiments

This document specifies how to replicate our results on CIFAR-100. 
Results on other datasets, e.g., MNIST, Fashion-MNIST, and CIFAR-10 can be reproduced in a similar way.

### Preprocessing
One may obtain the `smoothed prediction` by the following command:
```
# MNIST preprocessing
CUDA_VISIBLE_DEVICES=0 python code/train_cohen.py mnist lenet --lr 0.01 --lr_step_size 50 --epochs 150 --noise 0.25 --id 0
CUDA_VISIBLE_DEVICES=0 python code/smooth_prediction.py mnist logs/mnist/cohen/noise_0.25/lenet/0/checkpoint.pth.tar 0.25 test/smooth_prediction/mnist/cohen/0/noise_train_0.25.tsv --N=10000 --skip=1 --split=train

# Fashion-MNIST preprocessing
CUDA_VISIBLE_DEVICES=0 python code/train_cohen.py fmnist lenet --lr 0.01 --lr_step_size 50 --epochs 150 --noise 0.25 --id 0
CUDA_VISIBLE_DEVICES=0 python code/smooth_prediction.py fmnist logs/fmnist/cohen/noise_0.25/lenet/0/checkpoint.pth.tar 0.25 test/smooth_prediction/fmnist/cohen/0/noise_train_0.25.tsv --N=10000 --skip=1 --split=train

# CIFAR-10 preprocessing
CUDA_VISIBLE_DEVICES=0 python code/train_cohen.py cifar10 cifar_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150 --noise 0.25 --id 0
CUDA_VISIBLE_DEVICES=0 python code/smooth_prediction.py cifar10 logs/cifar10/cohen/noise_0.25/cifar_resnet110/0/checkpoint.pth.tar 0.25 test/smooth_prediction/cifar10/cohen/0/noise_train_0.25.tsv --N=10000 --skip=1 --split=train

# CIFAR-100 preprocessing
CUDA_VISIBLE_DEVICES=0 python code/train_cohen.py cifar100 cifar100_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150 --noise 0.25 --id 0
CUDA_VISIBLE_DEVICES=0 python code/smooth_prediction.py cifar100 logs/cifar100/cohen/noise_0.25/cifar100_resnet110/0/checkpoint.pth.tar \
0.25 test/smooth_prediction/cifar100/cohen/0/noise_train_0.25.tsv --N=10000 --skip=1 --split=train

# ImageNet preprocessing
python code/train_cohen.py imagenet resnet50 --lr 0.1 --lr_step_size 30 --epochs 90 --noise 0.25 --id 0
CUDA_VISIBLE_DEVICES=0 python code/smooth_prediction.py imagenet logs/imagenet/cohen/noise_0.25/resnet50/0/checkpoint.pth.tar \
0.25 test/smooth_prediction/resnet50/cohen/0/noise_train_0.25.tsv --N=10000 --skip=1 --split=train
```

### Training

Please specify GPU number(s) before run each line by modifying `CUDA_VISIBLE_DEVICES=[num]`.

```
# Baseline: Gaussian (Cohen et al., 2019)
CUDA_VISIBLE_DEVICES=[num] python code/train_cohen.py cifar100 cifar100_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150 --noise 0.25 --id 0
CUDA_VISIBLE_DEVICES=[num] python code/train_cohen.py cifar100 cifar100_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150 --noise 0.5 --id 0
CUDA_VISIBLE_DEVICES=[num] python code/train_cohen.py cifar100 cifar100_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150 --noise 1.0 --id 0

# Baseline: Stability training (Li et al., 2019)
CUDA_VISIBLE_DEVICES=[num] python code/train_stab.py cifar100 cifar100_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150 --noise 0.25 \
--lbd 2.0 --id 0
CUDA_VISIBLE_DEVICES=[num] python code/train_stab.py cifar100 cifar100_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150 --noise 0.5 \
--lbd 2.0 --id 0
CUDA_VISIBLE_DEVICES=[num] python code/train_stab.py cifar100 cifar100_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150 --noise 1.0 \
--lbd 1.0 --id 0

# Baseline: SmoothAdv (Salman et al., 2019)
CUDA_VISIBLE_DEVICES=[num] python code/train_salman.py cifar100 cifar100_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150  --noise 0.25 \
--attack PGD --epsilon 256.0 --num-steps 10 --warmup 10 --num-noise-vec 4 --id 0
CUDA_VISIBLE_DEVICES=[num] python code/train_salman.py cifar100 cifar100_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150  --noise 0.5 \
--attack PGD --epsilon 256.0 --num-steps 10 --warmup 10 --num-noise-vec 8 --id 0
CUDA_VISIBLE_DEVICES=[num] python code/train_salman.py cifar100 cifar100_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150  --noise 1.0 \
--attack PGD --epsilon 512.0 --num-steps 10 --warmup 10 --num-noise-vec 2 --id 0


# Baseline: MACER (Zhai et al., 2020)
CUDA_VISIBLE_DEVICES=[num] python code/train_macer.py cifar100 cifar100_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150  --noise 0.25 \
--num-noise-vec 16 --beta 16.0 --margin 8.0 --lbd 12.0 --id 0
CUDA_VISIBLE_DEVICES=[num] python code/train_macer.py cifar100 cifar100_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150  --noise 0.5 \
--num-noise-vec 16 --beta 16.0 --margin 8.0 --lbd 6.0 --id 0
CUDA_VISIBLE_DEVICES=[num] python code/train_macer.py cifar100 cifar100_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150  --noise 1.0 \
--num-noise-vec 16 --beta 16.0 --margin 8.0 --lbd 12.0 --deferred --id 0

# Baseline: Consistency (Jeong and Shin, 2020)
CUDA_VISIBLE_DEVICES=[num] python code/train_consistency.py cifar100 cifar100_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150  --noise 0.25 \
--num-noise-vec 2 --lbd 20 --id 0
CUDA_VISIBLE_DEVICES=[num] python code/train_consistency.py cifar100 cifar100_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150  --noise 0.5 \
--num-noise-vec 2 --lbd 10 --id 0
CUDA_VISIBLE_DEVICES=[num] python code/train_consistency.py cifar100 cifar100_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150  --noise 1.0 \
--num-noise-vec 2 --lbd 10 --id 0

# Baseline: SmoothMix (Jeong et al., 2021)

CUDA_VISIBLE_DEVICES=[num] python code/train_smoothmix.py cifar100 cifar100_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150 --noise 0.25 \
--num-noise-vec 2 --num-steps 4 --alpha 0.5 --mix_step 0 --eta 5.0 --id 0
CUDA_VISIBLE_DEVICES=[num] python code/train_smoothmix.py cifar100 cifar100_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150 --noise 0.5 \
--num-noise-vec 2 --num-steps 4 --alpha 1.0 --mix_step 1 --eta 5.0 --id 0
CUDA_VISIBLE_DEVICES=[num] python code/train_smoothmix.py cifar100 cifar100_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150 --noise 1.0 \
--num-noise-vec 2 --num-steps 4 --alpha 2.0 --mix_step 1 --eta 5.0 --id 0

# Ours: CAT-RS (Confidence-Aware Training for Randomized Smoothing)
CUDA_VISIBLE_DEVICES=[num] python code/train_catrs.py cifar100 cifar100_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150 \
--num-noise-vec 4 --noise 0.25 --id 0 --eps 256.0 --num-steps 4 --lbd 0.5
CUDA_VISIBLE_DEVICES=[num] python code/train_catrs.py cifar100 cifar100_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150 \
--num-noise-vec 4 --noise 0.5 --id 0 --eps 256.0 --num-steps 4 --lbd 1.0
CUDA_VISIBLE_DEVICES=[num] python code/train_catrs.py cifar100 cifar100_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150 \
--num-noise-vec 4 --noise 1.0 --id 0 --eps 256.0 --num-steps 4 --lbd 2.0

```
For training ImageNet, use the following command.

```
CUDA_VISIBLE_DEVICES=[num] python code/train_catrs.py imagenet resnet50 --lr 0.1 --lr_step_size 30 --epochs 90 --batch 256 \
--num_noise_vec 2 --noise 1.0 --id 0 --eps 256.0 --num-steps 1 --lbd 2.0 --lr_drop 85 --warmup 80 --confidence_mask
```

### Certification

```
# Baseline: Gaussian (Cohen et al., 2019)
python code/certify.py cifar100 logs/cifar100/cohen/noise_0.25/cifar100_resnet110/0/checkpoint.pth.tar 0.25 \
test/certify/cifar100/cohen/0/noise_0.25.tsv --N=100000 --skip=1
python code/certify.py cifar100 logs/cifar100/cohen/noise_0.5/cifar100_resnet110/0/checkpoint.pth.tar 0.5 \
test/certify/cifar100/cohen/0/noise_0.5.tsv --N=100000 --skip=1
python code/certify.py cifar100 logs/cifar100/cohen/noise_1.0/cifar100_resnet110/0/checkpoint.pth.tar 1.0 \
test/certify/cifar100/cohen/0/noise_1.0.tsv --N=100000 --skip=1

# Baseline: Stability training (Li et al., 2019)
python code/certify.py cifar100 logs/cifar100/stab/lbd_2.0/noise_0.25/cifar100_resnet110/0/checkpoint.pth.tar 0.25 \
test/certify/cifar100/stab/lbd_2.0/0/noise_0.25.tsv --N 100000 --skip 1
python code/certify.py cifar100 logs/cifar100/stab/lbd_2.0/noise_0.5/cifar100_resnet110/0/checkpoint.pth.tar 0.5 \
test/certify/cifar100/stab/lbd_2.0/0/noise_0.5.tsv --N 100000 --skip 1
python code/certify.py cifar100 logs/cifar100/stab/lbd_1.0/noise_1.0/cifar100_resnet110/0/checkpoint.pth.tar 1.0 \
test/certify/cifar100/stab/lbd_1.0/0/noise_1.0.tsv --N 100000 --skip 1

# Baseline: SmoothAdv (Salman et al., 2019)
python code/certify.py cifar100 logs/cifar100/salman/pgd_256.0_10_10/num_4/noise_0.25/cifar100_resnet110/0/checkpoint.pth.tar 0.25 \
test/certify/cifar100/salman/pgd_256.0_10_10/num_4/0/noise_0.25.tsv --N=100000 --skip=1
python code/certify.py cifar100 logs/cifar100/salman/pgd_256.0_10_10/num_8/noise_0.5/cifar100_resnet110/0/checkpoint.pth.tar 0.5 \
test/certify/cifar100/salman/pgd_256.0_10_10/num_8/0/noise_0.5.tsv --N=100000 --skip=1
python code/certify.py cifar100 logs/cifar100/salman/pgd_512.0_10_10/num_2/noise_1.0/cifar100_resnet110/0/checkpoint.pth.tar 1.0 \
test/certify/cifar100/salman/pgd_512.0_10_10/num_2/0/noise_1.0.tsv --N=100000 --skip=1

# Baseline: MACER (Zhai et al., 2020)
python code/certify.py cifar100 logs/cifar100/macer/num_16/lbd_12.0/gamma_8.0/beta_16.0/noise_0.25/cifar100_resnet110/0/checkpoint.pth.tar 0.25 \
test/certify/cifar100/macer/num_16/lbd_16.0/gamma_8.0/beta_16.0/0/noise_0.25.tsv --N=100000 --skip=1
python code/certify.py cifar100 logs/cifar100/macer/num_16/lbd_4.0/gamma_8.0/beta_16.0/noise_0.5/cifar100_resnet110/0/checkpoint.pth.tar 0.5 \
test/certify/cifar100/macer/num_16/lbd_4.0/gamma_8.0/beta_16.0/0/noise_0.5.tsv --N=100000 --skip=1
python code/certify.py cifar100 logs/cifar100/macer_deferred50/num_16/lbd_12.0/gamma_8.0/beta_16.0/noise_1.0/cifar100_resnet110/0/checkpoint.pth.tar 1.0 \
test/certify/cifar100/macer_deferred50/num_16/lbd_12.0/gamma_8.0/beta_16.0/noise_1.0.tsv --N=100000 --skip=1

# Baseline: Consistency (Jeong and Shin, 2020)
python code/certify.py cifar100 logs/cifar100/consistency/cohen/num_2/lbd_20.0/eta_0.5/noise_0.25/cifar100_resnet110/0/checkpoint.pth.tar 0.25 \
test/certify/cifar100/consistency/cohen/num_2/lbd_20.0/eta_0.5/noise_0.25.tsv --N=100000 --skip=1
python code/certify.py cifar100 logs/cifar100/consistency/cohen/num_2/lbd_10.0/eta_0.5/noise_0.5/cifar100_resnet110/0/checkpoint.pth.tar 0.5 \
test/certify/cifar100/consistency/cohen/num_2/lbd_10.0/eta_0.5/noise_0.5.tsv --N=100000 --skip=1
python code/certify.py cifar100 logs/cifar100/consistency/cohen/num_2/lbd_10.0/eta_0.5/noise_1.0/cifar100_resnet110/0/checkpoint.pth.tar 1.0 \
test/certify/cifar100/consistency/cohen/num_2/lbd_10.0/eta_0.5/noise_1.0.tsv --N=100000 --skip=1

# Baseline: SmoothMix (Jeong et al., 2021)
python code/certify.py cifar100 logs/cifar100/smix_0.5_4_m0/eta_5.0/num_2/noise_0.25/cifar100_resnet110/0/checkpoint.pth.tar 0.25 \
test/certify/cifar100/smix_0.5_4_m0/eta_5.0/num_2/0/noise_0.25.tsv --N=100000 --skip=1
python code/certify.py cifar100 logs/cifar100/smix_1.0_4_m1/eta_5.0/num_2/noise_0.5/cifar100_resnet110/0/checkpoint.pth.tar 0.5 \
test/certify/cifar100/smix_1.0_4_m1/eta_5.0/num_2/0/noise_0.5.tsv --N=100000 --skip=1
python code/certify.py cifar100 logs/cifar100/smix_2.0_4_m1/eta_5.0/num_2/noise_1.0/cifar100_resnet110/0/checkpoint.pth.tar 1.0 \
test/certify/cifar100/smix_2.0_4_m1/eta_5.0/num_2/0/noise_1.0.tsv --N=100000 --skip=1

# Ours: CAT-RS (Confidence-Aware Training for Randomized Smoothing)
python code/certify.py cifar100 logs/cifar100/catrs/adv_256.0_4/lbd_0.5/num_4/noise_0.25/cifar100_resnet110/0/checkpoint.pth.tar 0.25 \
test/certify/cifar100/catrs/adv_256.0_4/lbd_0.5/num_4/0/noise_0.25.tsv --N=100000 --skip=1
python code/certify.py cifar100 logs/cifar100/catrs/adv_256.0_4/lbd_1.0/num_4/noise_0.5/cifar100_resnet110/0/checkpoint.pth.tar 0.5 \
test/certify/cifar100/catrs/adv_256.0_4/lbd_1.0/num_4/0/noise_0.5.tsv --N=100000 --skip=1
python code/certify.py cifar100 logs/cifar100/catrs/adv_256.0_4/lbd_2.0/num_4/noise_1.0/cifar100_resnet110/0/checkpoint.pth.tar 1.0 \
test/certify/cifar100/catrs/adv_256.0_4/lbd_2.0/num_4/0/noise_1.0.tsv --N=100000 --skip=1
```

### Certification on corruption dataset

One may download corruption dataset CIFAR-10-C from [here](https://zenodo.org/record/2535967/#.Y4MA7y8RoUF) and MNIST-C from [here](https://zenodo.org/record/3239543\#.YisCti8RpQJ).

```
# Ours: CAT-RS (Confidence-Aware Training for Randomized Smoothing)
python code/certify_c.py logs/mnist/catrs/adv_256.0_4/lbd_1.0/num_4/noise_0.25/lenet/0/checkpoint.pth.tar test/certify/mnistc/0.25/catrs/ --N=100000 --skip=1 --base_c_path=PATH_TO_MNIST-C
```
