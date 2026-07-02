import glob
import random
import os

import torch
import torchaudio
from torch.utils.data import Dataset


class NSynthDataset(Dataset):
    """Dataset to load audio data (supports folder or .lst file input)."""

    def __init__(self, audio_path): 
        super().__init__()
        self.filenames = []
        self.max_len = 24000  

        if os.path.isdir(audio_path):
            self.filenames.extend(glob.glob(os.path.join(audio_path, "*.wav")))
        
        elif os.path.isfile(audio_path) and audio_path.endswith('.lst'):
            with open(audio_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if os.path.exists(line):
                            self.filenames.append(line)
                        else:
                            print(f"警告：.lst 文件中路径不存在，已跳过：{line}")
        
        else:
            raise ValueError(
                f"Invalid 'audio_path'! Expected a folder or .lst file, but got: {audio_path}"
            )


        if not self.filenames:
            raise RuntimeError(
                f"No valid .wav files found! Check your input: {audio_path}\n"
                "If using .lst file: ensure it contains valid audio paths (one per line).\n"
                "If using folder: ensure it has .wav files."
            )

        print(f"Found {len(self.filenames)} valid .wav files from {audio_path}")

        try:
            _, self.sr = torchaudio.load(self.filenames[0])
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file: {self.filenames[0]}. Error: {str(e)}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        ans = torch.zeros(1, self.max_len)
        try:
            audio, _ = torchaudio.load(self.filenames[index])
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file: {self.filenames[index]}. Error: {str(e)}")
        
        if audio.shape[1] > self.max_len:
            st = random.randint(0, audio.shape[1] - self.max_len - 1)
            ed = st + self.max_len
            return audio[:, st:ed]
        else:
            ans[:, :audio.shape[1]] = audio
            return ans
