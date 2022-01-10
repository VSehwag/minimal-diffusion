#!/bin/bash

# First download pretrained diffusion models from https://drive.google.com/drive/folders/1BMTpNF-FSsGrWGZomcM4OS36CootbLRj?usp=sharing
# You can also find 50,000 pre-sampled synthetic images for each dataset at 
# https://drive.google.com/drive/folders/1KRWie7honV_mwPlmTgH8vrU0izQXm4UT?usp=sharing

sampling_args="--arch UNet --class-cond --sampling-steps 250 --sampling-only --save-dir ./sampled_images/"

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8101 main.py \
    --arch UNet --dataset mnist --batch-size 256 --num-sampled-images 50000 $sampling_args \
    --pretrained-ckpt ./trained_models/UNet_mnist-epoch_100-timesteps_1000-class_condn_True_ema_0.9995.pt

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8102 main.py \
    --arch UNet --dataset mnist_m --batch-size 256 --num-sampled-images 50000 $sampling_args \
    --pretrained-ckpt ./trained_models/UNet_mnist_m-epoch_250-timesteps_1000-class_condn_True_ema_0.9995.pt

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8103 main.py \
    --arch UNet --dataset cifar10 --batch-size 256 --num-sampled-images 50000 $sampling_args \
    --pretrained-ckpt ./trained_models/UNet_cifar10-epoch_500-timesteps_1000-class_condn_True_ema_0.9995.pt 

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8104 main.py \
    --arch UNet --dataset melanoma --batch-size 128 --num-sampled-images 50000 $sampling_args \
    --pretrained-ckpt ./trained_models/UNet_melanoma-epoch_250-timesteps_1000-class_condn_True_ema_0.9995.pt

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8205 main.py \
    --arch UNet --dataset afhq --batch-size 128 --num-sampled-images 50000 $sampling_args \
    --pretrained-ckpt ./trained_models/UNet_afhq-epoch_500-timesteps_1000-class_condn_True_ema_0.9995.pt

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8206 main.py \
    --arch UNet --dataset celeba --batch-size 128 --num-sampled-images 50000 $sampling_args \
    --pretrained-ckpt ./trained_models/UNet_celeba-epoch_100-timesteps_1000-class_condn_True_ema_0.9995.pt 

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8207 main.py \
    --arch UNet --dataset cars --batch-size 128 --num-sampled-images 50000 $sampling_args \
    --pretrained-ckpt ./trained_models/UNet_cars-epoch_500-timesteps_1000-class_condn_True_ema_0.9995.pt

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8208 main.py \
    --arch UNet --dataset flowers --batch-size 128 --num-sampled-images 50000 $sampling_args \
    --pretrained-ckpt ./trained_models/UNet_flowers-epoch_1000-timesteps_1000-class_condn_True_ema_0.9995.pt

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8209 main.py \
    --arch UNet --dataset gtsrb --batch-size 256 --num-sampled-images 50000 $sampling_args \
    --pretrained-ckpt ./trained_models/UNet_gtsrb-epoch_500-timesteps_1000-class_condn_True_ema_0.9995.pt


