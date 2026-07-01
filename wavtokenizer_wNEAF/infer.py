# --coding:utf-8--
import os
import glob
import torch
import torchaudio
from decoder.pretrained import WavTokenizer

device = torch.device('cuda:0')

# ---------- 配置 ----------
input_dir = "/home/shaokai/seCodec251209/test_noisy_snr/24khz_csv/snr_-5"
out_folder = '/home/shaokai/seCodec251209/4_wavtokenizer_900bps/LibriTTS_clean_test_24khz'
ll = "snr-5"

config_path = "/home/shaokai/WavTokenizer-github0611/configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
model_path = "/home/shaokai/WavTokenizer-github0611/model_save/24khz_900bps/lightning_logs/version_1/checkpoints/wavtokenizer_checkpoint_epoch=54_step=546370_val_loss=4.6170.ckpt"

# ---------- 准备输出目录 ----------
out_dir = os.path.join(out_folder, ll)
os.makedirs(out_dir, exist_ok=True)

# ---------- 扫描所有音频文件 ----------
audio_extensions = ('*.wav', '*.flac', '*.mp3')  # 按需添加
audio_paths = []
for ext in audio_extensions:
    audio_paths.extend(glob.glob(os.path.join(input_dir, ext)))
# 如果子目录也要扫描，可以用 recursive=True 的 glob
# audio_paths = [p for p in glob.glob(os.path.join(input_dir, '**', '*'), recursive=True)
#                if p.lower().endswith(('.wav', '.flac', '.mp3'))]

audio_paths.sort()  # 排序保证顺序可复现
print(f"Found {len(audio_paths)} audio files.")

# ---------- 加载模型 ----------
wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
wavtokenizer.to(device)
wavtokenizer.eval()

# ---------- 编码所有音频 ----------
features_all = []
bandwidth_id = torch.tensor([0], device=device)  # 带宽 ID 固定为 0，直接放到 GPU 上

for i, audio_path in enumerate(audio_paths):
    wav, sr = torchaudio.load(audio_path)
    # 如果需要重采样/转单声道，可以取消下面的注释
    # wav = convert_audio(wav, sr, 24000, 1)
    wav = wav.to(device)
    print(f"Encoding [{i}/{len(audio_paths)}]: {audio_path}")

    with torch.no_grad():
        features, discrete_code = wavtokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)
    features_all.append(features.cpu())  # 把特征移到 CPU 以节省 GPU 显存，解码时再移回

# ---------- 解码并保存 ----------
for i, (audio_path, features) in enumerate(zip(audio_paths, features_all)):
    print(f"Decoding [{i}/{len(audio_paths)}]: {audio_path}")
    features = features.to(device)

    with torch.no_grad():
        audio_out = wavtokenizer.decode(features, bandwidth_id=bandwidth_id)

    # 保存音频
    audio_out = audio_out.cpu()  # (1, T)
    save_path = os.path.join(out_dir, os.path.basename(audio_path))
    torchaudio.save(save_path, audio_out, sample_rate=24000, encoding='PCM_S', bits_per_sample=16)

print("All done!")