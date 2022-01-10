#!/bin/bash

# Change data-dir to refer to the path of training dataset on your machine
# Following datasets needs to be manually downloaded before training: melanoma, afhq, celeba, cars, flowers, gtsrb.
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8101 main.py \
    --arch UNet --dataset mnist --class-cond --epochs 100 --batch-size 256 --sampling-steps 100

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8102  main.py \
    --arch UNet --dataset mnist_m --class-cond --epochs 250 --batch-size 256 --sampling-steps 100 \
    --data-dir ~/datasets/all_mnist/mnist_m/train/

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8104 main.py \
    --arch UNet --dataset melanoma --class-cond --epochs 250 --batch-size 128 --sampling-steps 50 \
    --data-dir ~/datasets/medical/melanoma/org_balanced/train/

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8105  main.py \
    --arch UNet --dataset cifar10 --class-cond --epochs 500 --batch-size 256 --sampling-steps 100

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8106 main.py \
    --arch UNet --dataset afhq --class-cond --epochs 500 --batch-size 128 --sampling-steps 50 \
    --data-dir ~/datasets/misc/afhq256/train/

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8107 main.py \
    --arch UNet --dataset celeba --class-cond --epochs 100 --batch-size 128 --sampling-steps 50 \
    --data-dir ~/datasets/misc/celebA_male_smile_64_balanced/train/

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8108 main.py \
    --arch UNet --dataset cars --class-cond --epochs 500 --batch-size 128 --sampling-steps 50 \
    --data-dir ~/datasets/misc/stanford_cars/train/

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8109 main.py \
    --arch UNet --dataset flowers --class-cond --epochs 1000 --batch-size 128 --sampling-steps 50 \
    --data-dir ~/datasets/misc/oxford_102_flowers/

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8110 main.py \
    --arch UNet --dataset gtsrb --class-cond --epochs 500 --batch-size 256 --sampling-steps 100 \
    --data-dir ~/datasets/misc/gtsrb/GTSRB/Final_Training/Images/


