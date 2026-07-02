#!/bin/bash
source path.sh

ckpt=/home/shaokai/AcademiCodec-master/egs/HiFi-Codec-16k-320d/logs_600bps/g_00290000
echo checkpoint path: ${ckpt}

# the path of test wave
wav_dir=/home/shaokai/seCodec251209/test_noisy_snr/16khz_csv/snr_-5

outputdir=/home/shaokai/seCodec251209/4_hificodec_600bps/LibriTTS_clean_test_16khz/snr_-5
mkdir -p ${outputdir}

python3 ${BIN_DIR}/vqvae_copy_syn.py \
    --model_path=${ckpt} \
    --config_path=config_16k_320d.json \
    --input_wavdir=${wav_dir} \
    --outputdir=${outputdir} \
    --num_gens=10000 \
    --sample_rate=16000
