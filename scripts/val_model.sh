#!/bin/bash

conda activate cyclegan_ulf
python /gpfs/data/johnsonplab/data/hyperfine/cyclegan_model/test.py \
    --image ../data/CycleGAN/ULF_Tomo_KCL/val/ \
    --result ../output/CycleGAN_val/ \
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