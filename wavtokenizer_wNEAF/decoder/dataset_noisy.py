import os, re, random
import numpy as np
from torch.utils.data import DataLoader
import torch
import torchaudio
from decoder.dataset import VocosDataModule, VocosDataset, DataConfig

def _extract_esc50_label(noise_path: str) -> int:
    pattern = r'[A-Z]-(\d+)\.wav$'
    match = re.search(pattern, os.path.basename(noise_path), re.IGNORECASE)
    if match:
        label = int(match.group(1))
        if 0 <= label <= 49:
            return label
    return random.randint(0, 49)

class NoisyVocosDataset(VocosDataset):
    def __init__(self, cfg: DataConfig, noise_filelist: str, snr_range=(-10, 20), train=True):
        super().__init__(cfg, train)
        with open(noise_filelist) as f:
            self.noise_paths = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        self.noise_labels = [_extract_esc50_label(p) for p in self.noise_paths]
        self.snr_range = snr_range
        self.train = train

    def _load_noise(self):
        idx = random.randint(0, len(self.noise_paths)-1)
        noise_path = self.noise_paths[idx]
        noise_label = self.noise_labels[idx]
        noise, sr = torchaudio.load(noise_path)
        if sr != self.sampling_rate:
            noise = torchaudio.functional.resample(noise, sr, self.sampling_rate)
        return noise, noise_label

    def __getitem__(self, index):
        clean = super().__getitem__(index).unsqueeze(0)  # [1, T]
        noise, noise_label = self._load_noise()
        if noise.size(-1) < clean.size(-1):
            repeat = int(np.ceil(clean.size(-1) / noise.size(-1)))
            noise = noise.repeat(1, repeat)[:, :clean.size(-1)]
        else:
            start = random.randint(0, noise.size(-1) - clean.size(-1))
            noise = noise[:, start:start+clean.size(-1)]
        snr = random.uniform(*self.snr_range)
        clean_rms = torch.sqrt(torch.mean(clean**2) + 1e-8)
        noise_rms = torch.sqrt(torch.mean(noise**2) + 1e-8)
        scale = clean_rms / (noise_rms * 10**(snr/20))
        scaled_noise = scale * noise
        noisy = clean + scaled_noise
        noisy = torch.clamp(noisy, -1.0, 1.0)
        return clean.squeeze(0), noisy.squeeze(0), scaled_noise.squeeze(0), noise_label

class NoisyVocosDataModule(VocosDataModule):
    def __init__(self, train_params: dict, val_params: dict, noise_filelist: str):
        train_cfg = DataConfig(**train_params)
        val_cfg = DataConfig(**val_params)
        super().__init__(train_cfg, val_cfg)
        self.noise_filelist = noise_filelist

    def train_dataloader(self):
        dataset = NoisyVocosDataset(self.train_config, self.noise_filelist, train=True)
        return DataLoader(dataset, batch_size=self.train_config.batch_size,
                          num_workers=self.train_config.num_workers, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        dataset = NoisyVocosDataset(self.val_config, self.noise_filelist, train=False)
        return DataLoader(dataset, batch_size=self.val_config.batch_size,
                          num_workers=self.val_config.num_workers, shuffle=False, pin_memory=True)
