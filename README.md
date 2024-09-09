# Code for SEAL

This is the implementation of our ICML'24 paper [Aligned Objective for Soft-Pseudo-Label Generation in Supervised Learning](https://proceedings.mlr.press/v235/xu24k.html).

## Dependencies

This code requires the following:

- python 3.8.16 
- pytorch 2.0.0
- torchvision 0.15.0
- numpy 1.23.5
- higher 0.2.1
- difftopk 0.2.0 

## Training

**Supervised Learning in `seal_sl`**

1. Download CIFAR-10/CIFAR-100/Tiny-ImageNet dataset into `/path/to/data`.

2. Run the following demos:

```bash
python train.py --dataset cifar10 --data-dir /path/to/data --gpu 0 --seed 0 --warmup --init-step 5000 --inner-iter 1 --warmup-epochs 50 --temp 5 --alpha 1

python train.py --dataset cifar100 --data-dir /path/to/data --gpu 0 --seed 0 --warmup --init-step 5000 --inner-iter 1 --warmup-epochs 100 --temp 4 --alpha 1

python train.py --dataset tinyimagenet --data-dir /path/to/data --gpu 0 --seed 0 --warmup --init-step 5000 --inner-iter 2 --warmup-epochs 50 --temp 3 --alpha 0.5
```

**Partial Label Learning in `seal_pll`**

1. Download CIFAR-10/CIFAR-100 dataset into `/path/to/data`.

2. Run the following demos:

```bash
python train.py --dataset cifar10 --partial-rate 0.3 --data-dir /path/to/data --gpu 0 --seed 0 --meta-start-epoch 5 --ramp-up-epoch 200 --super-weight 0.01 --keep-value 0.1 --update-mom 0.8

python train.py --dataset cifar10 --partial-rate 0.5 --data-dir /path/to/data --gpu 0 --seed 0 --meta-start-epoch 1 --ramp-up-epoch 180 --super-weight 0.5 --keep-value 0.01 --update-mom 0.5

python train.py --dataset cifar10 --partial-rate 0.7 --data-dir /path/to/data --gpu 0 --seed 0 --meta-start-epoch 10 --ramp-up-epoch 160 --super-weight 0.01 --keep-value 0.5 --update-mom 0.3

python train.py --dataset cifar100 --partial-rate 0.03 --data-dir /path/to/data --gpu 0 --seed 0 --meta-start-epoch 10 --ramp-up-epoch 200 --super-weight 0.01 --keep-value 0.5 --update-mom 0.2

python train.py --dataset cifar100 --partial-rate 0.05 --data-dir /path/to/data --gpu 0 --seed 0 --meta-start-epoch 1 --ramp-up-epoch 200 --super-weight 0.01 --keep-value 0.01 --update-mom 0.4

python train.py --dataset cifar100 --partial-rate 0.1 --data-dir /path/to/data --gpu 0 --seed 0 --meta-start-epoch 10 --ramp-up-epoch 180 --super-weight 0.5 --keep-value 0.1 --update-mom 0.2
```