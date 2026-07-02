#!/bin/bash

EXP_NAME="CBR_16k"
DEVICE=0
CKPT_BASE_DIR="/home/shaokai/NoiseRobustVRVQ-main/stage1"
TAG="best"
#INPUT_AUDIO_DIR="/home/shaokai/seCodec251209/test_noisy_snr/16khz_csv/snr_-5"
#INPUT_AUDIO_DIR="/home/shaokai/TiCodec-main/egs/test-clean_16khz"
INPUT_AUDIO_DIR="/home/datasets/data_shaokai/ESC-50-master/audio_16khz"
OUTPUT_DIR="/home/shaokai/seCodec251209/target_dac_noise_gen/2500bps"
SAMPLE_RATE=16000

echo "======================================"
echo "开始执行第一阶段推理脚本"
echo "实验名称：${EXP_NAME}"
echo "GPU设备：${DEVICE}"
echo "输入音频目录：${INPUT_AUDIO_DIR}"
echo "重构音频保存目录：${OUTPUT_DIR}"
echo "======================================"

python infer_stage1_dac.py --ckpt-base-dir ${CKPT_BASE_DIR} --exp-name "${EXP_NAME}" --tag "${TAG}" --input-audio-dir "${INPUT_AUDIO_DIR}" --output-dir "${OUTPUT_DIR}" --device ${DEVICE} --sample-rate ${SAMPLE_RATE}

echo "======================================"
echo "第一阶段推理脚本执行结束，重构音频已保存至：${OUTPUT_DIR}"
echo "======================================"
