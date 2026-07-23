#!/bin/bash

conda activate cyclegan_ulf
python train.py \
    --data_path ../data/CycleGAN/ULF/train/ \
    --label_path ../data/CycleGAN/HF/train/ \
    --batch_size 2 \
    --patch_size 128 128 128 \
    --netG resvit \
    --name cyclegan_model_1 \
    --checkpoints_dir ../models/ \
    --input_nc 2 \
    --output_nc 2 \
    --init_type kaiming \
    --data_nums 1 2 \
    --label_nums 1 2 \
    --split_train \
    --pretrain weights/pretrain \
    --pretrain_epoch 400