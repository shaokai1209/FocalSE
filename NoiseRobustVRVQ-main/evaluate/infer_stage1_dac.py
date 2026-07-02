import argparse
import os; opj = os.path.join
from glob import glob

import numpy as np
import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

import warnings
from audiotools import AudioSignal
from model.dac_vrvq import DAC_VRVQ

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def parse_args():
    parser = argparse.ArgumentParser(description="第一阶段推理脚本：音频→重构音频（DAC_VRVQ模型）")
    parser.add_argument("--ckpt-base-dir", type=str, required=True,
                        help="模型权重根目录（对应训练脚本的save_path根目录）")
    parser.add_argument("--exp-name", type=str, 
                        default="CBR_16k",
                        help="实验名称（对应训练时的save_path名称）")
    parser.add_argument("--tag", type=str,
                        default="best",
                        help="模型权重标签")
    parser.add_argument('--input-audio-dir', type=str, required=True,
                        help="输入音频文件夹路径")
    parser.add_argument('--output-dir', type=str,
                         default='stage1_enhanced_audio',
                         help="重构音频保存目录")
    parser.add_argument('--device', type=int, default=0,
                        help="GPU索引（-1表示CPU，默认0）")
    parser.add_argument('--sample-rate', type=int, default=16000,
                        help="模型输入采样率")
    
    return parser.parse_args()


def load_audio_from_dir(audio_dir, sample_rate=16000):
    audio_extensions = ["*.wav", "*.flac", "*.mp3"]
    audio_paths = []
    for ext in audio_extensions:
        audio_paths.extend(glob(os.path.join(audio_dir, ext)))
    
    if not audio_paths:
        raise ValueError(f"在 {audio_dir} 中未找到任何支持的音频文件（.wav/.flac/.mp3）")
    
    audio_paths.sort()
    audio_list = []
    
    for audio_path in audio_paths:
        try:
            signal = AudioSignal(audio_path)
            original_sr = signal.sample_rate
            
            if signal.sample_rate != sample_rate:
                signal.resample(sample_rate)
            
            if signal.num_channels > 1:
                signal = signal.to_mono()
            
            audio_tensor = signal.audio_data
            audio_name = os.path.splitext(os.path.basename(audio_path))[0]
            
            audio_list.append((audio_tensor, audio_name, original_sr))
        
        except Exception as e:
            print(f"❌ 加载失败：{audio_path}，错误信息：{str(e)}")
            continue
    
    return audio_list


def infer_audio_stage1(
    *,
    model: DAC_VRVQ, 
    audio_input, 
    audio_name,
    output_dir, 
    device
):
    sr = model.sample_rate
    audio_input = audio_input.to(device)

    if hasattr(model, "preprocess"):
        audio_input = model.preprocess(audio_input, sr)
    
    with torch.no_grad():
        out = model(audio_input, sr)
        recon_audio = out["audio"]
    
    recon_signal = AudioSignal(recon_audio, sample_rate=sr)
    save_from_audio(recon_signal, output_dir, f"{audio_name}")
    
    print(f"✅ 重构音频已保存：{opj(output_dir, f'{audio_name}.wav')}")
    return recon_audio


def save_from_audio(signal: AudioSignal, output_dir, name):
    os.makedirs(output_dir, exist_ok=True)
    sig_cpu = signal[0].detach().cpu()
    audio_save_path = opj(output_dir, f"{name}.wav")
    sig_cpu.write(audio_save_path)


def load_training_metadata(ckpt_base_dir, exp_name, tag):
    ckpt_dir = opj(ckpt_base_dir, exp_name, tag, "dac_vrvq")
    ckpt_path = opj(ckpt_dir, "weights.pth")
    extra_path = opj(ckpt_dir, "extra.pth")
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"第一阶段模型权重文件不存在：{ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location="cpu")
    weights = ckpt["state_dict"]
    
    model_kwargs = {}
    if os.path.exists(extra_path):
        extra = torch.load(extra_path, map_location="cpu")
        if "metadata.pth" in extra and "kwargs" in extra["metadata.pth"]:
            model_kwargs = extra["metadata.pth"]["kwargs"]
            print(f"✅ 从extra.pth加载到训练配置：{list(model_kwargs.keys())}")
    elif "metadata.pth" in ckpt and "kwargs" in ckpt["metadata.pth"]:
        model_kwargs = ckpt["metadata.pth"]["kwargs"]
        print(f"✅ 从weights.pth加载到训练配置：{list(model_kwargs.keys())}")
    else:
        print("⚠️  未找到训练元数据，使用兜底配置（需确认与训练一致！）")
        model_kwargs = {
            "sample_rate": 16000,
            "n_codebooks": 8,
            "codebook_size": 1024,
            "codebook_dim": 8,
            "model_type":"CBR"
        }
    
    import inspect
    sig = inspect.signature(DAC_VRVQ.__init__)
    valid_params = list(sig.parameters.keys())
    model_kwargs = {k: v for k, v in model_kwargs.items() if k in valid_params}
    print(f"✅ 过滤后有效模型参数：{list(model_kwargs.keys())}")
    
    if "sample_rate" not in model_kwargs:
        model_kwargs["sample_rate"] = 16000
    
    return model_kwargs, weights


if __name__ == "__main__":
    args = parse_args()
    if args.device >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
        print(f"📌 使用GPU设备：cuda:{args.device}")
    else:
        device = torch.device("cpu")
        print(f"📌 使用CPU设备")
    print("\n📌 开始加载第一阶段DAC_VRVQ模型...")
    model_kwargs, weights = load_training_metadata(
        ckpt_base_dir=args.ckpt_base_dir,
        exp_name=args.exp_name,
        tag=args.tag
    )
    
    model = DAC_VRVQ(**model_kwargs)
    
    missing_keys, unexpected_keys = model.load_state_dict(weights, strict=False)
    if missing_keys:
        print(f"⚠️  权重中缺失的键（不影响核心推理）：{missing_keys[:5]}")
    if unexpected_keys:
        print(f"⚠️  模型中不存在的权重键（已跳过）：{unexpected_keys[:5]}")
    
    model.eval()
    model.to(device)
    
    print(f"✅ 第一阶段模型加载成功！")
    print(f"✅ 模型核心配置：")
    print(f"   - 采样率：{model.sample_rate}")
    print(f"   - 码本数量：{model.n_codebooks if hasattr(model, 'n_codebooks') else '未知'}")
    print(f"   - 是否包含imp_subnet：{hasattr(model.quantizer, 'imp_subnet') if hasattr(model, 'quantizer') else False}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n📌 重构音频将保存至：{args.output_dir}")
    
    print(f"\n📌 开始加载输入音频（目录：{args.input_audio_dir}）...")
    audio_list = load_audio_from_dir(args.input_audio_dir, model.sample_rate)
    print(f"✅ 输入音频加载完成，共加载 {len(audio_list)} 个有效音频")
    
    print(f"\n📌 开始第一阶段批量推理...")
    skip_count = 0
    process_count = 0
    
    for idx, (audio_tensor, audio_name, original_sr) in enumerate(audio_list, 1):
        target_wav_path = opj(args.output_dir, f"{audio_name}.wav")
        
        if os.path.exists(target_wav_path):
            print(f"[{idx}/{len(audio_list)}] ⏭️  跳过已存在：{audio_name}")
            skip_count += 1
            continue
        
        print(f"[{idx}/{len(audio_list)}] 🔄 正在处理：{audio_name}")
        try:
            infer_audio_stage1(
                model=model,
                audio_input=audio_tensor,
                audio_name=audio_name,
                output_dir=args.output_dir,
                device=device
            )
            process_count += 1
        except Exception as e:
            print(f"❌ 处理失败：{audio_name}，错误信息：{str(e)}")
            continue
    
    print(f"\n🎉 第一阶段推理完成！")
    print(f"   - 新处理文件数：{process_count}")
    print(f"   - 跳过已存在文件数：{skip_count}")
    print(f"   - 保存目录：{args.output_dir}")
