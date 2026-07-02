import os; opj = os.path.join
import re
import pandas as pd
from typing import Callable, List, Union, Dict
from audiotools import AudioSignal
from audiotools.core import util

from torch.utils.data import Dataset
import random

import torch
import numpy as np

class AudioLoader_EARS_Piared:
    def __init__(
        self,
        srcs_clean: List[str],
        srcs_noisy: List[str],
        shuffle: bool = True,
        shuffle_state: int = 0,
    ):
        self.clean_list = []
        self.noisy_list = []
        
        for src in srcs_clean:
            if src.endswith('.lst'):
                if not os.path.exists(src):
                    raise ValueError(f"Clean .lst file does not exist: {src}")
                
                with open(src, 'r') as f:
                    lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                    
                    for line in lines:
                        if os.path.exists(line):
                            self.clean_list.append(line)
                        else:
                            print(f"  WARNING: Clean file does not exist: {line}")
            else:
                print(f"Processing clean directory: {src}")
                try:
                    clean_list = util.read_sources(
                        [src], relative_path="", ext=[".wav"]
                    )
                    for clist in clean_list:
                        self.clean_list.extend([c['path'] for c in clist])
                except Exception as e:
                    print(f"ERROR processing clean source {src}: {e}")
        
        for src in srcs_noisy:
            if src.endswith('.lst'):
                if not os.path.exists(src):
                    raise ValueError(f"Noisy .lst file does not exist: {src}")
                
                with open(src, 'r') as f:
                    lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                    
                    for line in lines:
                        if os.path.exists(line):
                            self.noisy_list.append(line)
                        else:
                            print(f"  WARNING: Noisy file does not exist: {line}")
            else:
                print(f"Processing noisy directory: {src}")
                try:
                    noisy_list = util.read_sources(
                        [src], relative_path="", ext=[".wav"]
                    )
                    for nlist in noisy_list:
                        self.noisy_list.extend([n['path'] for n in nlist])
                except Exception as e:
                    print(f"ERROR processing noisy source {src}: {e}")
        
        if len(self.clean_list) == 0:
            raise ValueError(f"clean_list is empty! Check paths: {srcs_clean}")
        if len(self.noisy_list) == 0:
            raise ValueError(f"noisy_list is empty! Check paths: {srcs_noisy}")
        
        min_len = min(len(self.clean_list), len(self.noisy_list))
        self.clean_list = self.clean_list[:min_len]
        self.noisy_list = self.noisy_list[:min_len]
        
        if shuffle:
            state = util.random_state(shuffle_state)
            shuffle_idx = list(range(len(self.clean_list)))
            state.shuffle(shuffle_idx)
            self.clean_list = [self.clean_list[ii] for ii in shuffle_idx]
            self.noisy_list = [self.noisy_list[ii] for ii in shuffle_idx]
    
    def _safe_load_audio(self, path, offset=None, duration=None, state=None, loudness_cutoff=-40):
        try:
            if offset is None and duration is not None:
                return AudioSignal.salient_excerpt(
                    path,
                    duration=duration,
                    state=state,
                    loudness_cutoff=loudness_cutoff,
                )
            elif offset is not None and duration is not None:
                return AudioSignal(
                    path,
                    offset=offset,
                    duration=duration,
                )
            else:
                return AudioSignal(path)
        except Exception as e:
            if "empty" in str(e).lower():
                try:
                    signal = AudioSignal(path)
                    if signal.audio_data is not None and signal.audio_data.shape[1] > 0:
                        if duration is not None and signal.duration > duration:
                            max_offset = max(0, signal.duration - duration - 0.1)
                            offset = state.uniform(0, max_offset) if state else 0.0
                            actual_duration = min(duration, signal.duration - offset)
                            signal = AudioSignal(path, offset=offset, duration=actual_duration)
                        return signal
                    else:
                        raise ValueError(f"Loaded empty signal from {path}")
                except Exception as e2:
                    raise RuntimeError(f"Cannot load audio file {path}: {e2}")
            else:
                raise e

    def __call__(
        self,
        state,
        sample_rate: int, 
        duration: float,
        loudness_cutoff: float = -40,
        num_channels: int = 1,
        offset: float = None,
        item_idx: int = None,
    ):
        if len(self.clean_list) == 0:
            raise ValueError("clean_list is empty!")
        if len(self.noisy_list) == 0:
            raise ValueError("noisy_list is empty!")
        
        if item_idx is not None:
            item_idx = item_idx % len(self.clean_list)
            clean_path = self.clean_list[item_idx]
            noisy_path = self.noisy_list[item_idx]
        else:
            item_idx = state.randint(len(self.clean_list))
            clean_path = self.clean_list[item_idx]
            noisy_path = self.noisy_list[item_idx]
        
        try:
            signal_clean = self._safe_load_audio(
                clean_path, 
                offset=offset, 
                duration=duration,
                state=state,
                loudness_cutoff=loudness_cutoff
            )
            
            if offset is None and duration is not None:
                offset = signal_clean.metadata.get("offset", 0.0)
            
            signal_noisy = self._safe_load_audio(
                noisy_path,
                offset=offset,
                duration=duration,
                state=state,
                loudness_cutoff=loudness_cutoff
            )
            
        except Exception as e:
            print(f"ERROR: Failed to load audio pair {item_idx}")
            print(f"  Clean: {clean_path}")
            print(f"  Noisy: {noisy_path}")
            print(f"  Offset: {offset}, Duration: {duration}")
            print(f"  Error: {e}")
            raise e
        
        if num_channels == 1:
            signal_clean = signal_clean.to_mono()
            signal_noisy = signal_noisy.to_mono()
        
        signal_clean = signal_clean.resample(sample_rate)
        signal_noisy = signal_noisy.resample(sample_rate)
        
        if duration is not None:
            if signal_clean.duration < duration:
                signal_clean = signal_clean.zero_pad_to(int(duration * sample_rate))
                signal_noisy = signal_noisy.zero_pad_to(int(duration * sample_rate))
        
        items = {
            "signal_clean": signal_clean,
            "signal_noisy": signal_noisy,
            "item_idx": item_idx,
            "path_clean": str(clean_path),
            "path_noisy": str(noisy_path),
        }
        
        return items
            
        

class AudioDataset_EARS_Paired:
    def __init__(
        self,
        loader: AudioLoader_EARS_Piared,
        sample_rate: int,
        n_examples: int = 1000,
        duration: float = 0.5,
        loudness_cutoff: float = -40,
        num_channels: int = 1,
        without_replacement: bool = True,
    ):
        self.loader = loader
        self.sample_rate = sample_rate
        self.n_examples = n_examples
        self.duration = duration
        self.loudness_cutoff = loudness_cutoff
        self.num_channels = num_channels
        self.without_replacement = without_replacement
        
    def __len__(self):
        return self.n_examples
        
    def __getitem__(self, idx):
        state = util.random_state(idx)
        loader_kwargs = {
            "state": state,
            "sample_rate": self.sample_rate,
            "duration": self.duration,
            "loudness_cutoff": self.loudness_cutoff,
            "num_channels": self.num_channels,
            "item_idx": idx if self.without_replacement else None,
        }
        item = self.loader(**loader_kwargs)
        item['idx'] = idx
        return item
    
    @staticmethod
    def collate(list_of_dicts: Union[list, dict], n_splits: int = None):
        return util.collate(list_of_dicts, n_splits=n_splits)

class AudioLoader_EARS_Clean:
    def __init__(
        self,
        srcs_clean: List[str],
        shuffle: bool = True,
        shuffle_state: int = 0,
    ):
        self.clean_list = []
        valid_ext = (".wav", ".flac", ".mp3", ".mp4")
        
        for src in srcs_clean:
            if not os.path.exists(src):
                raise ValueError(f"❌ .lst文件不存在：{src}")
            
            with open(src, "r", encoding="utf-8") as f:
                lines = [
                    l.strip() for l in f.readlines() 
                    if l.strip() and not l.startswith("#")
                ]
            
            for line in lines:
                if line.lower().endswith(valid_ext) and os.path.exists(line):
                    self.clean_list.append(line)
        
        if len(self.clean_list) == 0:
            raise ValueError(
                f"❌ clean_list为空！\n"
                f"srcs_clean: {srcs_clean}\n"
                f".lst文件总行数：{len(lines)}\n"
                f"有效音频路径数：0\n"
                f"请检查：1.音频路径是否存在 2.音频格式是否在{valid_ext}范围内"
            )
        
        self.clean_list = sorted(self.clean_list)
        if shuffle:
            state = util.random_state(shuffle_state)
            shuffle_idx = list(range(len(self.clean_list)))
            state.shuffle(shuffle_idx)
            self.clean_list = [self.clean_list[ii] for ii in shuffle_idx]

    def __call__(
        self,
        state,
        sample_rate: int, 
        duration: float,
        loudness_cutoff: float = -40,
        num_channels: int = 1,
        offset: float = None,
        item_idx: int = None,
    ):
        if len(self.clean_list) == 0:
            raise ValueError("clean_list is empty! Cannot sample audio.")
        
        if item_idx is not None:
            item_idx = item_idx % len(self.clean_list)
            clean_path = self.clean_list[item_idx]
        else:
            item_idx = state.randint(len(self.clean_list))
            clean_path = self.clean_list[item_idx]
        
        if offset is None:
            if duration is not None:
                signal_clean = AudioSignal.salient_excerpt(
                    clean_path,
                    duration=duration,
                    state=state,
                    loudness_cutoff=loudness_cutoff,
                )
                offset = signal_clean.metadata["offset"]
            else:
                signal_clean = AudioSignal(clean_path)
        else:
            signal_clean = AudioSignal(
                clean_path,
                offset=offset,
                duration=duration,
            )
        
        if num_channels == 1:
            signal_clean = signal_clean.to_mono()
        signal_clean = signal_clean.resample(sample_rate)
        
        if duration is not None:
            if signal_clean.duration < duration:
                signal_clean = signal_clean.zero_pad_to(int(duration * sample_rate))
                    
        items = {
            "signal_clean": signal_clean,
            "item_idx": item_idx,
            "path_clean": str(clean_path),
        }
        return items
    

class AudioDataset_EARS_Clean:
    def __init__(
        self,
        loader: AudioLoader_EARS_Clean,
        sample_rate: int,
        n_examples: int = 1000,
        duration: float = 0.5,
        loudness_cutoff: float = -40,
        num_channels: int = 1,
        without_replacement: bool = True,
    ):
        self.loader = loader
        self.sample_rate = sample_rate
        self.n_examples = n_examples
        self.duration = duration
        self.loudness_cutoff = loudness_cutoff
        self.num_channels = num_channels
        self.without_replacement = without_replacement
        
    def __len__(self):
        return self.n_examples
        
    def __getitem__(self, idx):
        state = util.random_state(idx)
        loader_kwargs = {
            "state": state,
            "sample_rate": self.sample_rate,
            "duration": self.duration,
            "loudness_cutoff": self.loudness_cutoff,
            "num_channels": self.num_channels,
            "item_idx": idx % len(self.loader.clean_list) if self.without_replacement else None,
        }
        item = self.loader(**loader_kwargs)
        item['idx'] = idx
        return item
    
    @staticmethod
    def collate(list_of_dicts: Union[list, dict], n_splits: int = None):
        return util.collate(list_of_dicts, n_splits=n_splits)

class AudioLoader_EARS_DynamicNoisy:
    def __init__(
        self,
        srcs_clean: List[str],
        srcs_noise: List[str],
        snr_list: List[int] = [-10, -5, 0, 5, 10, 15, 20],
        shuffle: bool = True,
        shuffle_state: int = 0,
    ):
        self.clean_list = []
        valid_ext = (".wav", ".flac", ".mp3", ".mp4")
        for src in srcs_clean:
            if src.endswith('.lst'):
                if not os.path.exists(src):
                    raise ValueError(f"Clean .lst file does not exist: {src}")
                with open(src, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                for line in lines:
                    if line.lower().endswith(valid_ext) and os.path.exists(line):
                        self.clean_list.append(line)
            else:
                print(f"Processing clean directory: {src}")
                try:
                    clean_list = util.read_sources([src], relative_path="", ext=[".wav"])
                    for clist in clean_list:
                        self.clean_list.extend([c['path'] for c in clist])
                except Exception as e:
                    print(f"ERROR processing clean source {src}: {e}")
        
        self.noise_list = []
        self.noise_labels = []
        for src in srcs_noise:
            if src.endswith('.lst'):
                if not os.path.exists(src):
                    raise ValueError(f"Noise .lst file does not exist: {src}")
                with open(src, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                
                for line in lines:
                    if line.lower().endswith(valid_ext) and os.path.exists(line):
                        self.noise_list.append(line)
                        noise_label = self._extract_esc50_label(line)
                        self.noise_labels.append(noise_label)
            else:
                print(f"Processing noise directory: {src}")
                try:
                    noise_list = util.read_sources([src], relative_path="", ext=[".wav"])
                    for nlist in noise_list:
                        for n_path in [n['path'] for n in nlist]:
                            self.noise_list.append(n_path)
                            self.noise_labels.append(self._extract_esc50_label(n_path))
                except Exception as e:
                    print(f"ERROR processing noise source {src}: {e}")
        
        if len(self.clean_list) == 0:
            raise ValueError("clean_list is empty! Check your clean LST/directory.")
        if len(self.noise_list) == 0:
            raise ValueError("noise_list is empty! Check your noise LST/directory.")
        assert len(self.noise_list) == len(self.noise_labels), \
            f"噪声路径数({len(self.noise_list)})与标签数({len(self.noise_labels)})不一致！"
        
        if shuffle:
            state = util.random_state(shuffle_state)
            shuffle_idx = list(range(len(self.clean_list)))
            state.shuffle(shuffle_idx)
            self.clean_list = [self.clean_list[ii] for ii in shuffle_idx]
        
        self.snr_list = snr_list
        self.num_clean = len(self.clean_list)
        self.num_noise = len(self.noise_list)

    def _extract_esc50_label(self, noise_path: str) -> int:
        pattern = r'[A-Z]-(\d+)\.wav$'
        match = re.search(pattern, noise_path, re.IGNORECASE)
        
        if not match:
            raise ValueError(
                f"❌ 噪声文件名格式不匹配ESC-50规则，无法提取标签！\n"
                f"路径：{noise_path}\n"
                f"规则：文件名末尾必须包含 [字母]-[数字].wav（如A-34.wav）"
            )
        
        label = int(match.group(1))
        
        if not (0 <= label <= 49):
            raise ValueError(
                f"❌ 提取的噪声标签超出0-49范围！\n"
                f"路径：{noise_path}\n"
                f"提取标签：{label}\n"
                f"要求：标签必须在0-49之间"
            )
        return label

    def _safe_load_audio(self, path, offset=None, duration=None, state=None, loudness_cutoff=-40):
        try:
            if offset is None and duration is not None:
                return AudioSignal.salient_excerpt(
                    path,
                    duration=duration,
                    state=state,
                    loudness_cutoff=loudness_cutoff,
                )
            elif offset is not None and duration is not None:
                return AudioSignal(
                    path,
                    offset=offset,
                    duration=duration,
                )
            else:
                return AudioSignal(path)
        except Exception as e:
            if "empty" in str(e).lower():
                try:
                    signal = AudioSignal(path)
                    if signal.audio_data is not None and signal.audio_data.shape[1] > 0:
                        if duration is not None and signal.duration > duration:
                            max_offset = max(0, signal.duration - duration - 0.1)
                            offset = state.uniform(0, max_offset) if state else 0.0
                            actual_duration = min(duration, signal.duration - offset)
                            signal = AudioSignal(path, offset=offset, duration=actual_duration)
                        return signal
                    else:
                        raise ValueError(f"Loaded empty signal from {path}")
                except Exception as e2:
                    raise RuntimeError(f"Cannot load audio file {path}: {e2}")
            else:
                raise e

    def _align_audio_length(self, clean_sig: AudioSignal, noise_sig: AudioSignal) -> AudioSignal:
        clean_len = clean_sig.signal_length
        noise_len = noise_sig.signal_length

        if noise_len >= clean_len:
            noise_sig = noise_sig[..., :clean_len]
        else:
            repeat_times = (clean_len // noise_len) + 1
            noise_sig_audio = torch.cat([noise_sig.audio_data] * repeat_times, dim=-1)
            noise_sig.audio_data = noise_sig_audio[..., :clean_len]
        return noise_sig

    def _add_noise_with_snr(self, clean_sig: AudioSignal, noise_sig: AudioSignal, snr: float) -> tuple:
        clean_audio = clean_sig.audio_data
        noise_audio = noise_sig.audio_data

        eps = 1e-8
        clean_energy = torch.sum(clean_audio ** 2) + eps
        noise_energy = torch.sum(noise_audio ** 2) + eps

        snr_linear = 10 ** (snr / 10)
        noise_gain = torch.sqrt(clean_energy / (noise_energy * snr_linear))
        scaled_noise_audio = noise_audio * noise_gain

        noisy_audio = clean_audio + scaled_noise_audio
        noisy_audio = torch.clamp(noisy_audio, -1.0, 1.0)

        noisy_sig = AudioSignal(noisy_audio, clean_sig.sample_rate)
        scaled_noise_sig = AudioSignal(scaled_noise_audio, clean_sig.sample_rate)

        return noisy_sig, scaled_noise_sig

    def __call__(
        self,
        state,
        sample_rate: int,
        duration: float,
        loudness_cutoff: float = -40,
        num_channels: int = 1,
        offset: float = None,
        item_idx: int = None,
    ):
        if item_idx is not None:
            item_idx = item_idx % self.num_clean
            clean_path = self.clean_list[item_idx]
        else:
            item_idx = state.randint(self.num_clean)
            clean_path = self.clean_list[item_idx]

        noise_idx = item_idx % self.num_noise
        noise_path = self.noise_list[noise_idx]
        noise_label = self.noise_labels[noise_idx]

        try:
            signal_clean = self._safe_load_audio(
                clean_path, offset=offset, duration=duration, state=state, loudness_cutoff=loudness_cutoff
            )
            if offset is None and duration is not None:
                offset = signal_clean.metadata.get("offset", 0.0)
            
            signal_noise = self._safe_load_audio(
                noise_path, offset=offset, duration=duration, state=state, loudness_cutoff=loudness_cutoff
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load audio pair: {e}")

        signal_noise = self._align_audio_length(signal_clean, signal_noise)
        target_snr = random.choice(self.snr_list)
        signal_noisy, signal_scaled_noise = self._add_noise_with_snr(signal_clean, signal_noise, target_snr)

        if num_channels == 1:
            signal_clean = signal_clean.to_mono()
            signal_noisy = signal_noisy.to_mono()
            signal_scaled_noise = signal_scaled_noise.to_mono()

        signal_clean = signal_clean.resample(sample_rate)
        signal_noisy = signal_noisy.resample(sample_rate)
        signal_scaled_noise = signal_scaled_noise.resample(sample_rate)

        if duration is not None:
            target_length = int(duration * sample_rate)
            if signal_clean.duration < duration:
                signal_clean = signal_clean.zero_pad_to(target_length)
                signal_noisy = signal_noisy.zero_pad_to(target_length)
                signal_scaled_noise = signal_scaled_noise.zero_pad_to(target_length)

        items = {
            "signal_clean": signal_clean,
            "signal_noisy": signal_noisy,
            "signal_scaled_noise": signal_scaled_noise,
            "item_idx": item_idx,
            "noise_idx": noise_idx,
            "snr_used": target_snr,
            "path_clean": str(clean_path),
            "path_noise": str(noise_path),
            "noise_label": noise_label,
        }
        return items

class AudioDataset_EARS_DynamicNoisy:
    def __init__(
        self,
        loader: AudioLoader_EARS_DynamicNoisy,
        sample_rate: int,
        n_examples: int = 1000,
        duration: float = 0.5,
        loudness_cutoff: float = -40,
        num_channels: int = 1,
        without_replacement: bool = True,
    ):
        self.loader = loader
        self.sample_rate = sample_rate
        self.n_examples = n_examples
        self.duration = duration
        self.loudness_cutoff = loudness_cutoff
        self.num_channels = num_channels
        self.without_replacement = without_replacement

    def __len__(self):
        return self.n_examples

    def __getitem__(self, idx):
        state = util.random_state(idx)
        loader_kwargs = {
            "state": state,
            "sample_rate": self.sample_rate,
            "duration": self.duration,
            "loudness_cutoff": self.loudness_cutoff,
            "num_channels": self.num_channels,
            "item_idx": idx % len(self.loader.clean_list) if self.without_replacement else None,
        }
        item = self.loader(**loader_kwargs)
        item['idx'] = idx
        return item

    @staticmethod
    def collate(list_of_dicts: Union[list, dict], n_splits: int = None):
        return util.collate(list_of_dicts, n_splits=n_splits)
