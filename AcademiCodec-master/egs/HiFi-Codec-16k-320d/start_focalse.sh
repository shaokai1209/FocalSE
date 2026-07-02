#!/bin/bash
source path.sh
set -e

log_root="logs_focalse_test"
pretrained_root="logs_600bps"           

input_training_file="/home/shaokai/seCodec251209/train_clean.lst"
input_validation_file="/home/shaokai/seCodec251209/dev_clean.lst"
noise_filelist="/home/shaokai/seCodec251209/noisy_train.lst"

export CUDA_VISIBLE_DEVICES=6

python ${BIN_DIR}/train_focalse.py \
    --config config_16k_320d.json \
    --checkpoint_path ${log_root} \
    --pretrained_path ${pretrained_root} \
    --input_training_file ${input_training_file} \
    --input_validation_file ${input_validation_file} \
    --noise_filelist ${noise_filelist} \
    --checkpoint_interval 5000 \
    --summary_interval 100 \
    --validation_interval 5000