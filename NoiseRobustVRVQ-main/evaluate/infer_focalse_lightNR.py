import argparse
import os; opj = os.path.join
from glob import glob
import re
import csv

import numpy as np
import torch
from einops import rearrange

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from model.dac_focalse_lightNR import DAC_VRVQ_FeatureDenoise

from model.utils import generate_mask_hard
import warnings
from audiotools import AudioSignal
import math

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(description="带噪音频→增强音频 + 噪声分类标签")
    parser.add_argument("--ckpt-base-dir", type=str, required=True,
                        help="模型权重根目录")
    parser.add_argument("--exp-name", type=str, 
                        default="CBR_feat_denoise_16k",
                        help="实验名称")
    parser.add_argument("--tag", type=str,
                        default="best",
                        help="模型权重标签（latest/best）")
    parser.add_argument('--noisy-audio-dir', type=str, required=True,
                        help="带噪音频文件夹路径")
    parser.add_argument('--output-dir', type=str,
                         default='evaluate/enhanced_audio',
                        help="增强音频保存目录")
    parser.add_argument('--device', type=int, default=0,
                        help="GPU索引（-1=CPU）")
    parser.add_argument('--vrvq-level', type=float, default=1.0,
                        help="VBR模式增强等级")
    parser.add_argument('--sample-rate', type=int, default=16000,
                        help="模型输入采样率")
    parser.add_argument('--classification-result-path', type=str, 
                        default="noise_classification_results.csv",
                        help="噪声分类结果CSV路径")
    
    return parser.parse_args()


def load_noisy_audio_from_dir(audio_dir, sample_rate=16000):
    audio_extensions = ["*.wav", "*.flac", "*.mp3"]
    audio_paths = []
    for ext in audio_extensions:
        audio_paths.extend(glob(os.path.join(audio_dir, ext)))
    
    if not audio_paths:
        raise ValueError(f"在 {audio_dir} 中未找到任何支持的音频文件")
    
    audio_paths.sort()
    audio_list = []
    
    for audio_path in audio_paths:
        try:
            signal = AudioSignal(audio_path)
            if signal.sample_rate != sample_rate:
                signal.resample(sample_rate)
            if signal.num_channels > 1:
                signal = signal.to_mono()
            audio_tensor = signal.audio_data
            audio_name = os.path.splitext(os.path.basename(audio_path))[0]
            audio_list.append((audio_tensor, audio_name, signal.sample_rate))
            print(f"✅ 成功加载：{audio_path}")
        
        except Exception as e:
            print(f"❌ 加载失败：{audio_path}，错误信息：{str(e)}")
            continue
    
    return audio_list

def init_noise_classification_csv(csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "audio_name",          
            "snr_value",           
            "model_output_label",  
            "enhanced_audio_path"  
        ])
    print(f"✅ 噪声分类结果CSV已初始化：{csv_path}")

def write_noise_classification_result(csv_path, audio_name, snr, num_label, enhanced_audio_path):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([audio_name, snr, num_label, enhanced_audio_path])

def infer_audio_simple(
    *,
    model: DAC_VRVQ_FeatureDenoise, 
    audio_noisy, 
    audio_name,
    output_dir, 
    vrvq_level=1.0,
    device,
    classification_result_path
):
    torch.cuda.empty_cache()
    
    sr = model.sample_rate
    snr = -10
    snr_pattern = r"snr([+-]?\d+)"
    snr_match = re.search(snr_pattern, audio_name)
    if snr_match:
        snr = int(snr_match.group(1))
    print(f"🔍 音频[{audio_name}] - 提取到SNR值：{snr} dB")

    audio_noisy = audio_noisy.to(device)
    audio_noisy = model.preprocess(audio_noisy, sr)
    
    with torch.no_grad():
        model_output = model(
            audio_data_noisy=audio_noisy,
            audio_data_clean=None,
            audio_data_noise=None,
            sample_rate=sr,
            level=vrvq_level
        )
    
    enhanced_audio = model_output["audio"]
    enhanced_signal = AudioSignal(enhanced_audio, sample_rate=sr)
    
    try:
        noise_logits = model_output["noise_logits"]
        model_label = torch.argmax(noise_logits, dim=-1).cpu().item()
        model_label = max(0, min(49, model_label))
    except Exception as e:
        print(f"⚠️  噪声分类失败：{str(e)}")
        model_label = -1

    # 仅保存增强音频
    enhanced_audio_path = opj(output_dir, f"{audio_name}_enhanced.wav")
    save_audio_only(enhanced_signal, output_dir, f"{audio_name}_enhanced")
    print(f"✅ 增强音频已保存：{enhanced_audio_path}")

    # 写入分类结果
    write_noise_classification_result(
        classification_result_path,
        audio_name,
        snr,
        model_label,
        enhanced_audio_path
    )
    print(f"✅ 噪声分类完成：{audio_name} → 模型输出标签：{model_label}")
    
    torch.cuda.empty_cache()
    return enhanced_audio

def save_audio_only(signal: AudioSignal, output_dir, name):
    os.makedirs(output_dir, exist_ok=True)
    sig_cpu = signal[0].detach().cpu()
    audio_save_path = opj(output_dir, f"{name}.wav")
    sig_cpu.write(audio_save_path)

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    args = parse_args()
    
    if args.device >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
        print(f"📌 使用GPU设备：cuda:{args.device}")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        print(f"📌 GPU内存不足，自动切换到CPU推理")
    
    print("\n📌 开始加载模型...")
    ckpt_dir = opj(args.ckpt_base_dir, "stage2", 
                   args.exp_name, args.tag, "dac_vrvq_featuredenoise")
    ckpt_path = opj(ckpt_dir, "weights.pth")
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"模型权重文件不存在：{ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location=device)
    weights = ckpt["state_dict"]
    config = ckpt["metadata"]["kwargs"]
    
    for kw in list(config.keys()):
        if kw.startswith("noise_cls_") and kw not in ["noise_cls_num_classes", "noise_cls_lstm_hidden", "noise_cls_dropout"]:
            del config[kw]
    
    model = DAC_VRVQ_FeatureDenoise(**config)
    filtered_weights = {k: v for k, v in weights.items() if k in model.state_dict()}
    model.load_state_dict(filtered_weights, strict=True)
    
    model.eval()
    model.to(device)
    model.float()
    
    print(f"✅ 模型加载成功！实验名称：{args.exp_name}，模型类型：{model.model_type}")
    print(f"✅ 模型采样率：{model.sample_rate}，码本数量：{model.n_codebooks}")
    print(f"✅ 降噪块索引：{model.denoise_block_idx}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    init_noise_classification_csv(args.classification_result_path)
    
    print(f"\n📌 开始加载带噪音频（目录：{args.noisy_audio_dir}）...")
    audio_list = load_noisy_audio_from_dir(args.noisy_audio_dir, args.sample_rate)
    print(f"✅ 带噪音频加载完成，共加载 {len(audio_list)} 个有效音频")
    
    print(f"\n📌 开始批量推理...")
    batch_size = 10
    for idx in range(0, len(audio_list), batch_size):
        batch_audio = audio_list[idx:idx+batch_size]
        for sub_idx, (audio_tensor, audio_name, original_sr) in enumerate(batch_audio):
            global_idx = idx + sub_idx + 1
            print(f"\n[{global_idx}/{len(audio_list)}] 正在处理：{audio_name}")
            try:
                infer_audio_simple(
                    model=model,
                    audio_noisy=audio_tensor,
                    audio_name=audio_name,
                    output_dir=args.output_dir,
                    vrvq_level=args.vrvq_level,
                    device=device,
                    classification_result_path=args.classification_result_path
                )
            except Exception as e:
                print(f"❌ 处理失败：{audio_name}，错误信息：{str(e)}")
                enhanced_audio_path = opj(args.output_dir, f"{audio_name}_enhanced.wav")
                write_noise_classification_result(
                    args.classification_result_path,
                    audio_name,
                    snr if 'snr' in locals() else -10,
                    -3,
                    enhanced_audio_path
                )
        torch.cuda.empty_cache()
    
    print(f"\n🎉 所有音频推理完成！")
