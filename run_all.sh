#!/usr/bin/env bash

mkdir -p  output/vits16/0.1/
mkdir -p  output/vitb16/0.1/
mkdir -p  output/vits16/0.01/
mkdir -p  output/vitb16/0.01/


python run_with_submitit.py   --arch vit_small --patch_size 16 --out_dim 65536 --norm_last_layer false --momentum_teacher 0.996 --use_bn_in_head false --warmup_teacher_temp 0.04 --teacher_temp 0.07 --warmup_teacher_temp_epochs 2 --use_fp16 True --weight_decay 0.04 --weight_decay_end 0.4 --clip_grad 0 --batch_size_per_gpu 48 --epochs 100 --freeze_last_layer 1 --lr 0.0005 --min_lr 1e-05 --optimizer adamw --drop_path_rate 0.1 --local_crops_number 10  --global_crops_scale 0.25 1.0 --local_crops_scale 0.05 0.2 --data_path data/imagenet256/ --output_dir output/vits16/0.1/ --saveckp_freq 1 --seed 0 --num_workers 10 --beta 0.1 --warmup_epochs 1 --ngpus 1

python run_with_submitit.py  --arch vit_base --patch_size 16 --out_dim 65536 --norm_last_layer true --momentum_teacher 0.996 --use_bn_in_head false --warmup_teacher_temp 0.04 --teacher_temp 0.07 --warmup_teacher_temp_epochs 2 --use_fp16 True --weight_decay 0.04 --weight_decay_end 0.4 --clip_grad 0.3 --batch_size_per_gpu 24 --epochs 50 --freeze_last_layer 1 --lr 0.00075 --min_lr 2e-06 --optimizer adamw --drop_path_rate 0.1 --local_crops_number 10 --global_crops_scale 0.25 1.0 --local_crops_scale 0.05 0.25 --data_path data/imagenet256/ --output_dir output/vitb16/0.1/ --saveckp_freq 1 --seed 0 --num_workers 10 --warmup_epochs 1 --beta 0.1 --ngpus 1

python run_with_submitit.py   --arch vit_small --patch_size 16 --out_dim 65536 --norm_last_layer false --momentum_teacher 0.996 --use_bn_in_head false --warmup_teacher_temp 0.04 --teacher_temp 0.07 --warmup_teacher_temp_epochs 2 --use_fp16 True --weight_decay 0.04 --weight_decay_end 0.4 --clip_grad 0 --batch_size_per_gpu 48 --epochs 100 --freeze_last_layer 1 --lr 0.0005 --min_lr 1e-05 --optimizer adamw --drop_path_rate 0.1 --local_crops_number 10  --global_crops_scale 0.25 1.0 --local_crops_scale 0.05 0.2 --data_path data/imagenet256/ --output_dir output/vits16/0.01/ --saveckp_freq 1 --seed 0 --num_workers 10 --beta 0.01 --warmup_epochs 1 --ngpus 1

python run_with_submitit.py   --arch vit_base --patch_size 16 --out_dim 65536 --norm_last_layer true --momentum_teacher 0.996 --use_bn_in_head false --warmup_teacher_temp 0.04 --teacher_temp 0.07 --warmup_teacher_temp_epochs 2 --use_fp16 True --weight_decay 0.04 --weight_decay_end 0.4 --clip_grad 0.3 --batch_size_per_gpu 24 --epochs 50 --freeze_last_layer 1 --lr 0.00075 --min_lr 2e-06 --optimizer adamw --drop_path_rate 0.1 --local_crops_number 10 --global_crops_scale 0.25 1.0 --local_crops_scale 0.05 0.25 --data_path data/imagenet256/ --output_dir output/vitb16/0.01/ --saveckp_freq 1 --seed 0 --num_workers 10 --warmup_epochs 1 --beta 0.01 --ngpus 1
