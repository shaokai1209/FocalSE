"""
This code is copied from the original dac repository.
source: https://github.com/descriptinc/descript-audio-codec/blob/main/dac/nn/loss.py
"""
import typing
from typing import List

import torch
import torch.nn.functional as F
from audiotools import AudioSignal
from audiotools import STFTParams
from torch import nn

import re


class L1Loss(nn.L1Loss):
    """L1 Loss between AudioSignals. Defaults
    to comparing ``audio_data``, but any
    attribute of an AudioSignal can be used.

    Parameters
    ----------
    attribute : str, optional
        Attribute of signal to compare, defaults to ``audio_data``.
    weight : float, optional
        Weight of this loss, defaults to 1.0.

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/distance.py
    """

    def __init__(self, attribute: str = "audio_data", weight: float = 1.0, **kwargs):
        self.attribute = attribute
        self.weight = weight
        super().__init__(**kwargs)

    def forward(self, x: AudioSignal, y: AudioSignal):
        """
        Parameters
        ----------
        x : AudioSignal
            Estimate AudioSignal
        y : AudioSignal
            Reference AudioSignal

        Returns
        -------
        torch.Tensor
            L1 loss between AudioSignal attributes.
        """
        if isinstance(x, AudioSignal):
            x = getattr(x, self.attribute)
            y = getattr(y, self.attribute)
        return super().forward(x, y)


class SISDRLoss(nn.Module):
    """
    Computes the Scale-Invariant Source-to-Distortion Ratio between a batch
    of estimated and reference audio signals or aligned features.

    Parameters
    ----------
    scaling : int, optional
        Whether to use scale-invariant (True) or
        signal-to-noise ratio (False), by default True
    reduction : str, optional
        How to reduce across the batch (either 'mean',
        'sum', or none).], by default ' mean'
    zero_mean : int, optional
        Zero mean the references and estimates before
        computing the loss, by default True
    clip_min : int, optional
        The minimum possible loss value. Helps network
        to not focus on making already good examples better, by default None
    weight : float, optional
        Weight of this loss, defaults to 1.0.

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/distance.py
    """

    def __init__(
        self,
        scaling: int = True,
        reduction: str = "mean",
        zero_mean: int = True,
        clip_min: int = None,
        weight: float = 1.0,
    ):
        self.scaling = scaling
        self.reduction = reduction
        self.zero_mean = zero_mean
        self.clip_min = clip_min
        self.weight = weight
        super().__init__()

    def forward(self, x: AudioSignal, y: AudioSignal):
        eps = 1e-8
        # nb, nc, nt
        if isinstance(x, AudioSignal):
            references = x.audio_data
            estimates = y.audio_data
        else:
            references = x
            estimates = y

        nb = references.shape[0]
        references = references.reshape(nb, 1, -1).permute(0, 2, 1)
        estimates = estimates.reshape(nb, 1, -1).permute(0, 2, 1)

        # samples now on axis 1
        if self.zero_mean:
            mean_reference = references.mean(dim=1, keepdim=True)
            mean_estimate = estimates.mean(dim=1, keepdim=True)
        else:
            mean_reference = 0
            mean_estimate = 0

        _references = references - mean_reference
        _estimates = estimates - mean_estimate

        references_projection = (_references**2).sum(dim=-2) + eps
        references_on_estimates = (_estimates * _references).sum(dim=-2) + eps

        scale = (
            (references_on_estimates / references_projection).unsqueeze(1)
            if self.scaling
            else 1
        )

        e_true = scale * _references
        e_res = _estimates - e_true

        signal = (e_true**2).sum(dim=1)
        noise = (e_res**2).sum(dim=1)
        sdr = -10 * torch.log10(signal / noise + eps)

        if self.clip_min is not None:
            sdr = torch.clamp(sdr, min=self.clip_min)

        if self.reduction == "mean":
            sdr = sdr.mean()
        elif self.reduction == "sum":
            sdr = sdr.sum()
        return sdr


class MultiScaleSTFTLoss(nn.Module):
    """Computes the multi-scale STFT loss from [1].

    Parameters
    ----------
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [2048, 512]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default nn.L1Loss()
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 1.0
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    pow : float, optional
        Power to raise magnitude to before taking log, by default 2.0
    weight : float, optional
        Weight of this loss, by default 1.0
    match_stride : bool, optional
        Whether to match the stride of convolutional layers, by default False

    References
    ----------

    1.  Engel, Jesse, Chenjie Gu, and Adam Roberts.
        "DDSP: Differentiable Digital Signal Processing."
        International Conference on Learning Representations. 2019.

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/spectral.py
    """

    def __init__(
        self,
        window_lengths: List[int] = [2048, 512],
        loss_fn: typing.Callable = nn.L1Loss(),
        clamp_eps: float = 1e-5,
        mag_weight: float = 1.0,
        log_weight: float = 1.0,
        pow: float = 2.0,
        weight: float = 1.0,
        match_stride: bool = False,
        window_type: str = None,
    ):
        super().__init__()
        self.stft_params = [
            STFTParams(
                window_length=w,
                hop_length=w // 4,
                match_stride=match_stride,
                window_type=window_type,
            )
            for w in window_lengths
        ]
        self.loss_fn = loss_fn
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.clamp_eps = clamp_eps
        self.weight = weight
        self.pow = pow

    def forward(self, x: AudioSignal, y: AudioSignal):
        """Computes multi-scale STFT between an estimate and a reference
        signal.

        Parameters
        ----------
        x : AudioSignal
            Estimate signal
        y : AudioSignal
            Reference signal

        Returns
        -------
        torch.Tensor
            Multi-scale STFT loss.
        """
        loss = 0.0
        for s in self.stft_params:
            x.stft(s.window_length, s.hop_length, s.window_type)
            y.stft(s.window_length, s.hop_length, s.window_type)
            loss += self.log_weight * self.loss_fn(
                x.magnitude.clamp(self.clamp_eps).pow(self.pow).log10(),
                y.magnitude.clamp(self.clamp_eps).pow(self.pow).log10(),
            )
            loss += self.mag_weight * self.loss_fn(x.magnitude, y.magnitude)
        return loss


class MelSpectrogramLoss(nn.Module):
    """Compute distance between mel spectrograms. Can be used
    in a multi-scale way.

    Parameters
    ----------
    n_mels : List[int]
        Number of mels per STFT, by default [150, 80],
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [2048, 512]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default nn.L1Loss()
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 1.0
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    pow : float, optional
        Power to raise magnitude to before taking log, by default 2.0
    weight : float, optional
        Weight of this loss, by default 1.0
    match_stride : bool, optional
        Whether to match the stride of convolutional layers, by default False

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/spectral.py
    """

    def __init__(
        self,
        n_mels: List[int] = [150, 80],
        window_lengths: List[int] = [2048, 512],
        loss_fn: typing.Callable = nn.L1Loss(),
        clamp_eps: float = 1e-5,
        mag_weight: float = 1.0,
        log_weight: float = 1.0,
        pow: float = 2.0,
        weight: float = 1.0,
        match_stride: bool = False,
        mel_fmin: List[float] = [0.0, 0.0],
        mel_fmax: List[float] = [None, None],
        window_type: str = None,
    ):
        super().__init__()
        self.stft_params = [
            STFTParams(
                window_length=w,
                hop_length=w // 4,
                match_stride=match_stride,
                window_type=window_type,
            )
            for w in window_lengths
        ]
        self.n_mels = n_mels
        self.loss_fn = loss_fn
        self.clamp_eps = clamp_eps
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.weight = weight
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.pow = pow

    def forward(self, x: AudioSignal, y: AudioSignal):
        """Computes mel loss between an estimate and a reference
        signal.

        Parameters
        ----------
        x : AudioSignal
            Estimate signal
        y : AudioSignal
            Reference signal

        Returns
        -------
        torch.Tensor
            Mel loss.
        """
        loss = 0.0
        for n_mels, fmin, fmax, s in zip(
            self.n_mels, self.mel_fmin, self.mel_fmax, self.stft_params
        ):
            kwargs = {
                "window_length": s.window_length,
                "hop_length": s.hop_length,
                "window_type": s.window_type,
            }
            x_mels = x.mel_spectrogram(n_mels, mel_fmin=fmin, mel_fmax=fmax,** kwargs)
            y_mels = y.mel_spectrogram(n_mels, mel_fmin=fmin, mel_fmax=fmax, **kwargs)

            loss += self.log_weight * self.loss_fn(
                x_mels.clamp(self.clamp_eps).pow(self.pow).log10(),
                y_mels.clamp(self.clamp_eps).pow(self.pow).log10(),
            )
            loss += self.mag_weight * self.loss_fn(x_mels, y_mels)
        return loss


class GANLoss(nn.Module):
    """
    Computes a discriminator loss, given a discriminator on
    generated waveforms/spectrograms compared to ground truth
    waveforms/spectrograms. Computes the loss for both the
    discriminator and the generator in separate functions.
    """

    def __init__(self, discriminator):
        super().__init__()
        self.discriminator = discriminator

    def forward(self, fake, real):
        d_fake = self.discriminator(fake.audio_data)
        d_real = self.discriminator(real.audio_data)
        return d_fake, d_real

    def discriminator_loss(self, fake, real):
        d_fake, d_real = self.forward(fake.clone().detach(), real)

        loss_d = 0
        for x_fake, x_real in zip(d_fake, d_real):
            loss_d += torch.mean(x_fake[-1] ** 2)
            loss_d += torch.mean((1 - x_real[-1]) ** 2)
        return loss_d

    def generator_loss(self, fake, real):
        d_fake, d_real = self.forward(fake, real)

        loss_g = 0
        for x_fake in d_fake:
            loss_g += torch.mean((1 - x_fake[-1]) ** 2)

        loss_feature = 0

        for i in range(len(d_fake)):
            for j in range(len(d_fake[i]) - 1):
                loss_feature += F.l1_loss(d_fake[i][j], d_real[i][j].detach())
        return loss_g, loss_feature


class EncFeatureLoss(nn.Module):
    """
    computes feature loss between gt and noisy in encoder feature space
    """
    def __init__(
        self,
        loss_fn: typing.Callable = nn.L1Loss(),
    ):
        super().__init__()
        self.loss_fn = loss_fn
    
    def forward(self, enc_fmaps: dict):
        """
        enc_fmaps: output of an encoder
        - gt:
            - gt_{i}
        - noisy
            - denoised_{i}
            - imp_map_input: Not used
        """
        gt_maps = enc_fmaps["gt"]
        noisy_maps = enc_fmaps["noisy"]
        
        # import pdb; pdb.set_trace()
        comp = re.compile(r'\d+')
        # idx_list = [re.search(r'\d+', key).group() for key in gt_maps.keys()]
        idx_list = [comp.search(key).group() for key in gt_maps.keys() if key != "imp_map_input"]
        loss = 0
        for idx in idx_list:
            # loss += self.loss_fn(gt_maps[f"gt_{idx}"], noisy_maps[f"denoised_{idx}"])
            loss += self.loss_fn(noisy_maps[f"denoised_{idx}"], gt_maps[f"gt_{idx}"])
        
        return loss

class NoiseFeatureLoss(nn.Module):
    def __init__(
        self,
        loss_fn: typing.Callable = nn.L1Loss(),  # 和原始损失默认一致
    ):
        super().__init__()
        self.loss_fn = loss_fn  # 复用原始损失函数配置
    
    def forward(self, enc_fmaps: dict):
        # 1. 取出特征map（严格对应encoder新增的forward_noise输出）
        noise_maps = enc_fmaps["noise"]  # 真实噪声特征：noise_1、noise_3
        noisy_maps = enc_fmaps["noisy"]  # 预测噪声特征：pred_noise_1、pred_noise_3
        
        # 2. 👉 完全复用原始正则提取逻辑，一字未改
        comp = re.compile(r'\d+')
        idx_list = [comp.search(key).group() for key in noise_maps.keys() if key != "imp_map_input"]
        
        # 3. 👉 完全复用原始遍历+累加逻辑，仅替换特征键名
        loss = 0
        for idx in idx_list:
            # ✅ 核心匹配关系：pred_noise_{idx} ↔ noise_{idx}（和原始gt↔denoised对齐）
            loss += self.loss_fn(noisy_maps[f"pred_noise_{idx}"], noise_maps[f"noise_{idx}"])
        
        return loss

class NoiseClassificationLoss(nn.Module):
    def __init__(self, weight: float = 1.0, loss_fn: typing.Callable = nn.CrossEntropyLoss()):
        super().__init__()
        self.weight = weight
        self.loss_fn = loss_fn  # 可配置损失函数，更灵活

    def forward(self, enc_fmaps: dict):
        # 1. 动态提取噪声分类logits的索引（参考NoiseFeatureLoss的写法）
        comp = re.compile(r'\d+')
        # 从 noisy_maps 中找到所有 noise_logits_* 的key，提取数字索引
        noisy_maps = enc_fmaps["noisy"]
        logits_keys = [k for k in noisy_maps.keys() if k.startswith("noise_logits_")]
        idx_list = [comp.search(key).group() for key in logits_keys if comp.search(key)]
        
        # 2. 查找标签和logits（动态遍历提取到的索引）
        noise_labels = noisy_maps.get("noise_labels", None)
        noise_logits_list = []
        for idx in idx_list:
            logits_key = f"noise_logits_{idx}"
            if logits_key in noisy_maps:
                logits = noisy_maps[logits_key]
                if logits is not None:
                    noise_logits_list.append(logits)
        
        # 4. 边界条件
        if noise_labels is None or len(noise_logits_list) == 0:
            ref_tensor = noisy_maps["imp_map_input"]
            return torch.tensor(0.0, device=ref_tensor.device, requires_grad=False)
        
        # 5. 动态适配logits维度（兼容2维/3维）
        loss = 0.0
        for logits in noise_logits_list:
            if len(logits.shape) == 2:
                # 2维：(B, C) → 直接计算交叉熵
                loss += self.loss_fn(logits, noise_labels)
            elif len(logits.shape) == 3:
                # 3维：(B, C, T) → 展平计算
                b, c, t = logits.shape
                logits_flat = logits.permute(0, 2, 1).reshape(-1, c)
                if len(noise_labels.shape) == 1:
                    labels_flat = noise_labels.unsqueeze(1).repeat(1, t).reshape(-1)
                else:
                    labels_flat = noise_labels.reshape(-1)
                loss += self.loss_fn(logits_flat, labels_flat)
        
        # 6. 返回加权平均损失
        return self.weight * (loss / len(noise_logits_list))