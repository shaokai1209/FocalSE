#!/bin/bash
set -e

EXP_NAME="CBR_feat_denoise_16k"
GPU_DEVICE="0"
CKPT_BASE_DIR="/home/shaokai/NoiseRobustVRVQ-main/focalSE_NANSC_625bps"
NOISY_AUDIO_DIR="/home/shaokai/seCodec251209/test_noisy_snr/16khz_csv/snr_-5"
OUTPUT_DIR="/home/shaokai/seCodec251209/4_NANSC_625bps/LibriTTS_clean_test_16khz/snr_-5"

CLASSIFICATION_RESULT_PATH="${OUTPUT_DIR}/noise_classification_results.csv"

echo "======================================"
echo "开始执行推理脚本"
echo "实验名称：${EXP_NAME}"
echo "GPU设备：${GPU_DEVICE}"
echo "带噪音频目录：${NOISY_AUDIO_DIR}"
echo "增强音频保存目录：${OUTPUT_DIR}"
echo "分类结果保存路径：${CLASSIFICATION_RESULT_PATH}"
echo "======================================"

mkdir -p "${OUTPUT_DIR}"

python infer_focalse_lightNR.py \
    --ckpt-base-dir "${CKPT_BASE_DIR}" \
    --exp-name "${EXP_NAME}" \
    --tag "latest" \
    --noisy-audio-dir "${NOISY_AUDIO_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --device "${GPU_DEVICE}" \
    --vrvq-level 1.0 \
    --sample-rate 16000 \
    --classification-result-path "${CLASSIFICATION_RESULT_PATH}"

# 执行结束
echo "======================================"
echo "推理脚本执行结束！"
echo "增强音频保存至：${OUTPUT_DIR}"
echo "噪声分类数字标签保存至：${CLASSIFICATION_RESULT_PATH}"
echo "======================================"
