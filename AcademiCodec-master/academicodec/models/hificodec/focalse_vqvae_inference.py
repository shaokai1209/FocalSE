#!/usr/bin/env python3
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import librosa
import torch
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, '/home/shaokai/NoiseRobustVRVQ-main')
from model.focal_se import FocalSpeechEnhancementModule

from academicodec.models.hificodec.env import AttrDict
from academicodec.models.hificodec.models import Encoder, Generator, Quantizer


class FocalSeVqvaeInference:

    def __init__(self, config_path, checkpoint_path, device='cuda'):
        self.device = torch.device(device)

        with open(config_path) as f:
            json_config = json.loads(f.read())
        self.h = AttrDict(json_config)
        self.sample_rate = self.h.sampling_rate

        self.encoder = Encoder(self.h)
        self.quantizer = Quantizer(self.h)
        self.generator = Generator(self.h)
        self.focal_se = FocalSpeechEnhancementModule(
            input_dim=512,
            hidden_dim=256,
            num_transformer_layers=4,
            num_heads=8,
            dropout=0.1,
            mask_beta=2.0,
            focal_window=3,
            window_size=7,
            causal=False
        )

        ckpt = torch.load(checkpoint_path, map_location='cpu')
        self.encoder.load_state_dict(ckpt['encoder'], strict=True)
        self.quantizer.load_state_dict(ckpt['quantizer'], strict=True)
        self.generator.load_state_dict(ckpt['generator'], strict=True)
        self.focal_se.load_state_dict(ckpt['focal_se'], strict=True)

        self.encoder.to(self.device)
        self.quantizer.to(self.device)
        self.generator.to(self.device)
        self.focal_se.to(self.device)

        self.encoder.eval()
        self.quantizer.eval()
        self.generator.eval()
        self.focal_se.eval()

        if hasattr(self.h, 'upsample_rates'):
            self.total_stride = int(np.prod(self.h.upsample_rates))
        else:
            self.total_stride = 320

    def _pad_to_stride(self, wav):
        length = wav.shape[-1]
        if length % self.total_stride == 0:
            return wav
        pad_len = self.total_stride - (length % self.total_stride)
        return torch.nn.functional.pad(wav, (0, pad_len))

    @torch.no_grad()
    def enhance(self, noisy_wav_tensor):
        if noisy_wav_tensor.dim() == 1:
            noisy_wav_tensor = noisy_wav_tensor.unsqueeze(0)
        noisy_wav_tensor = noisy_wav_tensor.to(self.device)

        orig_len = noisy_wav_tensor.shape[-1]
        noisy_wav_tensor = self._pad_to_stride(noisy_wav_tensor)

        x = noisy_wav_tensor.unsqueeze(1)

        c_noisy = self.encoder(x)                # (B, 512, T')
        enhanced_c, _ = self.focal_se(c_noisy)   # (B, 512, T')
        q, _, _ = self.quantizer(enhanced_c)     # (B, 512, T')
        syn = self.generator(q)                  # (B, 1, T_out)

        syn = syn.squeeze(1).cpu()               # (B, T_out)
        syn = syn[..., :orig_len]
        return syn.squeeze(0).numpy()

    @torch.no_grad()
    def enhance_from_path(self, wav_path):
        wav, _ = librosa.load(wav_path, sr=self.sample_rate, mono=True)
        fid = os.path.splitext(os.path.basename(wav_path))[0]
        enhanced = self.enhance(torch.tensor(wav).float())
        return fid, enhanced


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputdir', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='focalse 训练得到的 g_*.ckpt 文件')
    parser.add_argument('--input_wavdir', type=str, required=True,
                        help='带噪 wav 目录')
    parser.add_argument('--config_path', type=str, default='config_16k_320d.json')
    parser.add_argument('--num_gens', type=int, default=None,
                        help='最多处理文件数，默认全部')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])
    args = parser.parse_args()

    Path(args.outputdir).mkdir(parents=True, exist_ok=True)

    print("Loading FocalSE + VQ-VAE model ...")
    model = FocalSeVqvaeInference(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        device=args.device
    )
    print("Model ready.")

    wav_paths = sorted(glob.glob(os.path.join(args.input_wavdir, '*.wav')))
    if args.num_gens is not None:
        wav_paths = wav_paths[:args.num_gens]
    print(f"Found {len(wav_paths)} wav files to process.")

    for wav_path in tqdm(wav_paths):
        try:
            fid, enhanced_wav = model.enhance_from_path(wav_path)
            out_path = os.path.join(args.outputdir, f"{fid}.wav")
            sf.write(out_path, enhanced_wav, model.sample_rate)
        except Exception as e:
            print(f"Error on {wav_path}: {e}")

    print(f"Done. Enhanced files saved to {args.outputdir}")


if __name__ == "__main__":
    main()
