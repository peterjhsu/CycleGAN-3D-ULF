#!/bin/bash

conda activate cyclegan_ulf
python test.py \
    --image ../data/CycleGAN/ULF/test \
    --result ../output/cyclegan_output \
    --batch_size 1 \
    --patch_size 128 128 128 \
    --netG resvit \
    --name cyclegan_model_1 \
    --checkpoints_dir ../models/ \
    --input_nc 2 \
    --output_nc 2 \
    --data_nums 1 2 \
    --label_nums 1 2 \
    --which_epoch 1000
python test.py \
    --which_direction BtoA \
    --image ../data/CycleGAN/HF/test \
    --result ../output/cyclegan_output \
    --batch_size 1 \
    --patch_size 128 128 128 \
    --netG resvit \
    --name cyclegan_model_1 \
    --checkpoints_dir ../models/ \
    --input_nc 2 \
    --output_nc 2 \
    --data_nums 1 2 \
    --label_nums 1 2 \
    --which_epoch 1000