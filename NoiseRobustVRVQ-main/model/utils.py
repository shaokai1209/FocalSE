import torch
import math
from einops import rearrange
import numpy as np
from torchmetrics.audio import PerceptualEvaluationSpeechQuality as PESQ
from torchmetrics.audio import ShortTimeObjectiveIntelligibility as STOI
import torchmetrics
import audiotools
from torchmetrics.audio import DeepNoiseSuppressionMeanOpinionScore as DNSMOS
from torchmetrics.audio import NonIntrusiveSpeechQualityAssessment as NISQA
from copy import deepcopy
import time 

def logcosh(alpha, pmk):
    """
    For stable training, 
    we divide the calculation into two cases: pmk >= 0 and pmk < 0.
    """
    EPS = 1e-10

    mask1 = pmk >= 0
    pmk1 = pmk * mask1.detach()
    numer1 = math.exp(alpha) + torch.exp(-2*pmk1*alpha)
    denom1 = torch.exp(alpha*(-2*pmk1+1)) + 1
    mask_smooth1 = (torch.log(numer1 + EPS) - torch.log(denom1 + EPS)) / (2*alpha) + 0.5
    

    mask2 = pmk < 0
    pmk2 = pmk * mask2.detach()
    numer2 = torch.exp(alpha*(2*pmk2+1)) + 1
    denom2 = math.exp(alpha) + torch.exp(alpha*2*pmk2)
    mask_smooth2 = (torch.log(numer2 + EPS) - torch.log(denom2 + EPS)) / (2*alpha) + 0.5
    
    mask_smooth = mask_smooth1 * mask1 + mask_smooth2 * mask2
    return mask_smooth


def generate_mask_ste(x, nq, alpha=1, function="logcosh", shift:float=None):
    device = x.device
    nqs = torch.arange(nq, dtype=torch.float).to(device) # (nq, ), [0, 1, ..., nq-1]
    nqs = rearrange(nqs, 'n -> 1 n 1')
    xmnq = x - nqs # (B, nq, T)
    
    if function=='logcosh':
        mask_smooth = logcosh(alpha, xmnq)
    elif function=='square':
        mask_smooth = torch.clamp(xmnq, 0, 1)
    elif function=='sigmoid':
        mask_smooth = torch.sigmoid(xmnq * alpha)    
    else:
        raise ValueError(f"Invalid function: {function}")
    
    mask_quant = torch.where(xmnq>=0, torch.ones_like(xmnq), torch.zeros_like(xmnq)).float()
    final_mask = mask_smooth + (mask_quant - mask_smooth).detach()
    return final_mask

def generate_mask_hard(x, nq):
    device = x.device
    nqs = torch.arange(nq, dtype=torch.float).to(device) # (nq, ), [0, 1, ..., nq-1]
    nqs = rearrange(nqs, 'n -> 1 n 1')
    xmnq = x - nqs # (B, nq, T)
    mask_quant = torch.where(xmnq>=0, torch.ones_like(xmnq), torch.zeros_like(xmnq)).float()
    return mask_quant


def cal_bpf_from_mask(mask, bits_per_codebook):
    """
    mask: (B, Nq, Frames)
    bits_per_codebook: (Nq, )
    """
    bits_per_codebook = torch.tensor(bits_per_codebook, device=mask.device) ## (Nq, )
    bits_per_codebook = rearrange(bits_per_codebook, 'nq -> 1 nq 1')
    mask_bits = mask * bits_per_codebook
    bpf = torch.sum(mask_bits) / (mask.shape[0] * mask.shape[2])
    return bpf.item()

def apply_straight(x, a, scale):
    """
    if x <= 1/(a+1), y=scale * ax
    else: y=scale/a*x + scale*(a-1)/a
    """
    output = torch.where(x <= 1/(a+1), scale*a*x, scale/a*x + scale*(a-1)/a)
    return output

def cal_entropy(bincount_list):
    n_codebooks = len(bincount_list)
    entropy_list = []
    pct_list = []
    for i in range(n_codebooks):
        bit = math.ceil(math.log2(bincount_list[i].shape[0]))
        counts = bincount_list[i]
        counts = (counts / counts.sum()).clamp(1e-10) ## 각 index의 확률
        entropy_i = -(counts * counts.log()).sum().item() * np.log2(np.e) 
        pct_i = entropy_i / bit
        entropy_list.append(entropy_i)
        pct_list.append(pct_i)
    # print(f"Entropy for each codebook: {entropy_list}")
    # print(f"Effective percentage: {pct_list}")
    return entropy_list, pct_list


def cal_metrics(recons, signal, state, loss_fn="mel"):
    # assert loss_fn in ["mel", "stft", "waveform", "SDR", "SI-SDR", "L1", "DAC-SISDR", "ViSQOL", "ViSQOL-speech"]
    if loss_fn == "mel":
        return state.mel_loss(recons, signal).item()
    elif loss_fn == "stft":
        return state.stft_loss(recons, signal).item()
    elif loss_fn == "waveform":
        return state.waveform_loss(recons, signal).item()
    elif loss_fn == "SDR":
        recons = recons.audio_data
        signal = signal.audio_data
        if recons.abs().max() == 0 or signal.abs().max() == 0:
            return np.nan  
        # result = torchmetrics.functional.signal_to_distortion_ratio(recons, signal)
        result = torchmetrics.functional.signal_distortion_ratio(recons, signal)
        result = result.mean().item()
        return result
    elif loss_fn == "SI-SDR":
        recons = recons.audio_data
        signal = signal.audio_data
        # return torchmetrics.functional.si_sdr(recons, signal).item()
        result = torchmetrics.functional.scale_invariant_signal_distortion_ratio(recons, signal)
        result = result.mean().item()
        return result
    elif loss_fn == "L1":
        recons = recons.audio_data
        signal = signal.audio_data
        result = torchmetrics.functional.mean_absolute_error(recons, signal)
        result = result.mean().item()
        return result
    elif loss_fn == "SI-SNR":
        recons = recons.audio_data
        signal = signal.audio_data 
        result = torchmetrics.functional.scale_invariant_signal_noise_ratio(recons, signal)
        result = result.mean().item()
        return result
    elif loss_fn == "SNR":
        recons = recons.audio_data
        signal = signal.audio_data
        result = torchmetrics.functional.signal_noise_ratio(recons, signal)
        result = result.mean().item()
        return result
    elif loss_fn == "DAC-SISDR":
        return state.dac_sisdr_loss(signal, recons).item()
    elif loss_fn == "ViSQOL":
        ## resample to 48k
        result = audiotools.metrics.quality.visqol(recons, signal)
        if isinstance(result, torch.Tensor):
            result = result.mean()
            result = result.item()
        return result
    elif loss_fn == "ViSQOL-speech":
        ## resample to 16k
        result = audiotools.metrics.quality.visqol(recons, signal, "speech")
        if isinstance(result, torch.Tensor):
            result = result.mean()
            result = result.item()
        return result
    elif loss_fn == "PESQ":
        sr = signal.sample_rate
        if sr != 16000:
            # signal = deepcopy(signal).resample(16000)
            # recons = deepcopy(recons).resample(16000)
            signal = signal.clone().resample(16000)
            recons = recons.clone().resample(16000)
        recons = recons.audio_data
        signal = signal.audio_data
        pesq = PESQ(16000, 'wb')
        result = pesq(recons, signal)
        result = result.mean().item()
        return result
    elif loss_fn == "STOI":
        sr = signal.sample_rate
        recons = recons.audio_data
        signal = signal.audio_data
        stoi = STOI(sr, extended=False)
        result = stoi(recons, signal)
        result = result.mean().item()
        return result
    elif loss_fn == "ESTOI":
        sr = signal.sample_rate
        recons = recons.audio_data
        signal = signal.audio_data
        stoi = STOI(sr, extended=True)
        result = stoi(recons, signal)
        result = result.mean().item()
        return result
    else:
        raise ValueError(f"Unknown loss function: {loss_fn}")


def si_sdr_components(s_hat, s, n):
    # s_target
    alpha_s = np.dot(s_hat, s) / np.linalg.norm(s)**2
    s_target = alpha_s * s

    # e_noise
    alpha_n = np.dot(s_hat, n) / np.linalg.norm(n)**2
    e_noise = alpha_n * n

    # e_art
    e_art = s_hat - s_target - e_noise
    
    return s_target, e_noise, e_art

def energy_ratios(s_hat, s, n):
    s_target, e_noise, e_art = si_sdr_components(s_hat, s, n)

    si_sdr = 10*np.log10(np.linalg.norm(s_target)**2 / np.linalg.norm(e_noise + e_art)**2)
    si_sir = 10*np.log10(np.linalg.norm(s_target)**2 / np.linalg.norm(e_noise)**2)
    si_sar = 10*np.log10(np.linalg.norm(s_target)**2 / np.linalg.norm(e_art)**2)

    return si_sdr, si_sir, si_sar



def mean_std(data):
    data = data[~np.isnan(data)]
    mean = np.mean(data)
    std = np.std(data)
    return mean, std    


def cal_metrics_visqol(recons, signal):
    """
    compute the metrics. at once
    recons_t: (1, 1, T)
    """

    if signal.sample_rate != 16000:
        # assert False
        # signal_16k = signal.resample(16000)
        # recons_16k = recons.resample(16000)
        signal_16k = deepcopy(signal).resample(16000)
        recons_16k = deepcopy(recons).resample(16000)
    else:
        signal_16k = signal
        recons_16k = recons
        
    visqol_speech = audiotools.metrics.quality.visqol(recons_16k, signal_16k, "speech").mean().item()
    return_dict = {}
    return_dict["ViSQOL-speech"] = visqol_speech
    
    return return_dict

    
def cal_metrics_full(recons, signal, cal_visqol=True):
    """
    compute the metrics. at once
    recons_t: (1, 1, T)
    """
    st = time.time()
    recons_t = recons.audio_data
    signal_t = signal.audio_data
    sr_ori = signal.sample_rate
    assert recons_t.shape[0] == 1 and recons_t.shape[1]==1
    
    si_sdr_tm = torchmetrics.functional.scale_invariant_signal_distortion_ratio(recons_t, signal_t).mean().item()


    if signal.sample_rate != 16000:
        signal_16k = deepcopy(signal).resample(16000)
        recons_16k = deepcopy(recons).resample(16000)
    else:
        signal_16k = signal
        recons_16k = recons
    pesq_tm = PESQ(16000, 'wb')(recons_16k.audio_data, signal_16k.audio_data).mean().item()
    stoi_tm = STOI(16000, extended=False)(recons_16k.audio_data, signal_16k.audio_data).mean().item()
    estoi_tm = STOI(16000, extended=True)(recons_16k.audio_data, signal_16k.audio_data).mean().item()

    if cal_visqol:
        visqol_speech = audiotools.metrics.quality.visqol(recons_16k, signal_16k, "speech").mean().item()
    
    return_dict = {
        "SI-SDR": si_sdr_tm,
        "PESQ": pesq_tm,
        "STOI": stoi_tm,
        "ESTOI": estoi_tm,
        "ViSQOL-speech": visqol_speech if cal_visqol else None
    }
    
    ### NISQA
    st = time.time()
    nisqa = NISQA(sr_ori)
    nisqa_overall_mos, nisqa_noisiness, nisqa_discontinuity, nisqa_coloration, nisqa_loudness = \
        nisqa(recons_t.squeeze())
        
    nisqa_dict = {
        "NISQA_overall_MOS":nisqa_overall_mos.item()
    }
    return_dict.update(nisqa_dict)
   
    return return_dict