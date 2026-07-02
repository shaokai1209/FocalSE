# academicodec/models/hificodec/noisy_dataset.py
import os, re, random
import numpy as np
import torch
import torchaudio
import librosa
from meldataset import MelDataset, mel_spectrogram

def _extract_esc50_label(noise_path: str) -> int:
    pattern = r'[A-Z]-(\d+)\.wav$'
    match = re.search(pattern, os.path.basename(noise_path), re.IGNORECASE)
    if not match:
        return -1
    label = int(match.group(1))
    if 0 <= label <= 49:
        return label
    return -1

class NoisyMelDataset(MelDataset):
    def __init__(self, training_files, segment_size, n_fft, num_mels, hop_size,
                 win_size, sampling_rate, fmin, fmax, noise_filelist,
                 snr_range=(-10, 20), split=True, shuffle=True, n_cache_reuse=1,
                 device=None, fmax_loss=None, fine_tuning=False, base_mels_path=None):
        super().__init__(training_files, segment_size, n_fft, num_mels, hop_size,
                         win_size, sampling_rate, fmin, fmax, split, shuffle,
                         n_cache_reuse, device, fmax_loss, fine_tuning, base_mels_path)
        with open(noise_filelist) as f:
            self.noise_paths = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        self.noise_labels = [_extract_esc50_label(p) for p in self.noise_paths]
        self.snr_range = snr_range

    def _load_random_noise(self):
        idx = random.randint(0, len(self.noise_paths) - 1)
        path = self.noise_paths[idx]
        label = self.noise_labels[idx]
        noise, sr = torchaudio.load(path)
        if sr != self.sampling_rate:
            noise = torchaudio.functional.resample(noise, sr, self.sampling_rate)
        return noise, label

    def __getitem__(self, index):
        mel, audio, filename, mel_loss = super().__getitem__(index)

        noise, noise_label = self._load_random_noise()

        if noise.size(-1) < audio.size(-1):
            repeat = (audio.size(-1) // noise.size(-1)) + 1
            noise = noise.repeat(1, repeat)[:, :audio.size(-1)]
        else:
            start = random.randint(0, noise.size(-1) - audio.size(-1))
            noise = noise[:, start:start + audio.size(-1)]

        snr = random.uniform(*self.snr_range)
        audio_rms = torch.sqrt(torch.mean(audio ** 2) + 1e-8)
        noise_rms = torch.sqrt(torch.mean(noise ** 2) + 1e-8)
        scale = audio_rms / (noise_rms * 10 ** (snr / 20))
        scaled_noise = scale * noise
        noisy_audio = audio + scaled_noise
        noisy_audio = torch.clamp(noisy_audio, -1.0, 1.0)

        return mel.squeeze(), audio.squeeze(0), filename, mel_loss.squeeze(), noisy_audio.squeeze(0), scaled_noise.squeeze(0), noise_label
