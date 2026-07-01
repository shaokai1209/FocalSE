import os
import sys
import glob
import torch
import torchaudio
import yaml
from typing import Any, Dict

sys.path.insert(0, '/home/shaokai/WavTokenizer-github0611')
sys.path.insert(0, '/home/shaokai/NoiseRobustVRVQ-main')

from decoder.pretrained import instantiate_class
from decoder.experiment_focalse import WavTokenizer_FocalSE, hubert_model


def load_model_from_config_checkpoint(
    config_path: str,
    ckpt_path: str,
    device: torch.device
) -> WavTokenizer_FocalSE:

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg['model']['init_args']

    feature_extractor = instantiate_class(
        args=(),
        init=model_cfg['feature_extractor']
    )
    backbone = instantiate_class(args=(), init=model_cfg['backbone'])
    head = instantiate_class(args=(), init=model_cfg['head'])

    model_init_args = {
        k: v for k, v in model_cfg.items()
        if k not in ('feature_extractor', 'backbone', 'head')
    }

    model = WavTokenizer_FocalSE(
        feature_extractor=feature_extractor,
        backbone=backbone,
        head=head,
        **model_init_args
    )

    state_dict_raw = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in state_dict_raw:
        state_dict_raw = state_dict_raw['state_dict']

    state_dict = {}
    for k, v in state_dict_raw.items():
        if k.startswith('module.'):
            k = k[7:]
        state_dict[k] = v

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")

    model.to(device)
    model.eval()

    model.hubert.to(device)

    return model


def main():
    config_path = "/home/shaokai/WavTokenizer-github0611/configs/wavtokenizer_focalse.yaml"
    ckpt_path = "/home/shaokai/WavTokenizer-github0611/model_save/24khz_900bps_focalse/lightning_logs/version_0/checkpoints/wavtokenizer_focalse_epoch=4_step=62090_val_loss=0.7430.ckpt"

    input_dir = "/home/shaokai/seCodec251209/test_noisy_snr/24khz_csv/snr_5"  
    output_dir = "/home/shaokai/seCodec251209/4_wavtokenizer_900bps/LibriTTS_clean_test_24khz/enhanced/snr_5" 

    sample_rate = 24000  

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = load_model_from_config_checkpoint(config_path, ckpt_path, device)

    os.makedirs(output_dir, exist_ok=True)

    audio_extensions = ('*.wav', '*.flac', '*.mp3')
    audio_paths = []
    for ext in audio_extensions:
        audio_paths.extend(glob.glob(os.path.join(input_dir, ext)))
    audio_paths.sort()
    print(f"Found {len(audio_paths)} audio files.")

    bandwidth_id = torch.tensor([0], device=device)

    with torch.no_grad():
        for i, audio_path in enumerate(audio_paths):
            print(f"Processing [{i+1}/{len(audio_paths)}]: {audio_path}")

            wav, sr = torchaudio.load(audio_path)
            if sr != sample_rate:
                wav = torchaudio.functional.resample(wav, sr, sample_rate)
            if wav.size(0) > 1:
                wav = wav.mean(dim=0, keepdim=True) 
            wav = wav.to(device)

            audio_hat, _, _ = model(wav, bandwidth_id=bandwidth_id)

            audio_out = audio_hat.cpu()
            save_name = os.path.basename(audio_path)
            torchaudio.save(
                os.path.join(output_dir, save_name),
                audio_out,
                sample_rate=sample_rate,
                encoding='PCM_S',
                bits_per_sample=16
            )

    print("All done!")


if __name__ == "__main__":
    main()
