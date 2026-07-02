#!/bin/bash
source path.sh 

# 训练好的 FocalSE 检查点
ckpt=/home/shaokai/AcademiCodec-master/egs/HiFi-Codec-16k-320d/logs_focalse/g_00300000
echo "Checkpoint: ${ckpt}"

# 输入带噪音频目录
wav_dir=/home/shaokai/seCodec251209/test_noisy_snr/16khz_csv/snr_-5

# 输出目录
outputdir=/home/shaokai/seCodec251209/4_hificodec_600bps/LibriTTS_clean_test_16khz/enhanced/snr_-5
mkdir -p ${outputdir}

# 推理
python3 ${BIN_DIR}/focalse_vqvae_inference.py \
    --config_path=config_16k_320d.json \
    --checkpoint_path=${ckpt} \
    --input_wavdir=${wav_dir} \
    --outputdir=${outputdir} \
    --device=cuda