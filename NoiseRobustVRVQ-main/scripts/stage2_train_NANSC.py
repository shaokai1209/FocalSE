"""
This code is heavily adapted and modified from the original DAC training code.  
Original source: https://github.com/descriptinc/descript-audio-codec/blob/main/scripts/train.py
"""
import os
import sys
import warnings
import re

warnings.filterwarnings("ignore")

from dataclasses import dataclass
from pathlib import Path
import argbind
import torch
import torch.nn as nn
import torch.nn.functional as F
from audiotools import AudioSignal
from audiotools import ml
from audiotools.core import util
from audiotools.data.datasets import ConcatDataset
from audiotools.ml.decorators import timer
from audiotools.ml.decorators import Tracker
from audiotools.ml.decorators import when
from torch.utils.tensorboard import SummaryWriter
from time import time
from tqdm import tqdm

from transformers import HubertModel, Wav2Vec2FeatureExtractor
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("/home/shaokai/llsdr-main/hubert-base-ls960")
hubert_model = HubertModel.from_pretrained("/home/shaokai/llsdr-main/hubert-base-ls960").eval()
for param in hubert_model.parameters():
    param.requires_grad = False


class SemanticAdapter(nn.Module):
    def __init__(self, in_dim=768, out_dim=1024):
        super().__init__()
        self.adapter = nn.Conv1d(in_dim, out_dim, kernel_size=1)
    def forward(self, x):
        return self.adapter(x)

semantic_adapter = SemanticAdapter(in_dim=768, out_dim=1024)

sys.path.append(str(Path(__file__).resolve().parents[1]))
from model.utils import cal_bpf_from_mask, cal_entropy
from data.loaders import (
    AudioLoader_EARS_DynamicNoisy,  
    AudioDataset_EARS_DynamicNoisy  
)
from model.dac_focalse_lightNR import DAC_VRVQ_FeatureDenoise
from model.discriminator import Discriminator
from model import loss
import math


# Enable cudnn autotuner to speed up training
torch.backends.cudnn.benchmark = bool(int(os.getenv("CUDNN_BENCHMARK", 1)))

## Optimizers
AdamW = argbind.bind(torch.optim.AdamW, "generator", "discriminator")
Accelerator = argbind.bind(ml.Accelerator, without_prefix=True)

## Model
DAC_VRVQ_FeatureDenoise = argbind.bind(DAC_VRVQ_FeatureDenoise)
Discriminator = argbind.bind(Discriminator)

AudioDataset = argbind.bind(AudioDataset_EARS_DynamicNoisy, "train", "val") 
AudioLoader = argbind.bind(AudioLoader_EARS_DynamicNoisy, "train", "val")   

## Loss
filter_fn = lambda fn: hasattr(fn, "forward") and "Loss" in fn.__name__
losses = argbind.bind_module(loss, filter_fn=filter_fn)

@argbind.bind("generator", "discriminator")
def ExponentialLR(optimizer, gamma: float = 1.0, warmup: int=0):
    if warmup==0:
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    else:
        def lr_lambda(current_step):
            if current_step < warmup:
                return float(current_step) / float(max(1, warmup))
            return gamma ** (current_step - warmup)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def get_infinite_loader(dataloader):
    while True:
        for batch in dataloader:
            yield batch

@argbind.bind("train", "val")
def build_dataset(
    sample_rate: int,
    folders: dict = None,
    snr_list: str = "-10,-5,0,5,10,15,20" 
):
    snr_list = [int(snr) for snr in snr_list.split(",")]
    loader = AudioLoader(
        srcs_clean = folders["clean"],
        srcs_noise = folders["noisy"], 
        snr_list = snr_list  
    )
    dataset = AudioDataset(
        loader=loader,
        sample_rate=sample_rate
    )
    return dataset

@dataclass
class State:
    generator: DAC_VRVQ_FeatureDenoise
    optimizer_g: AdamW
    scheduler_g: ExponentialLR
    
    discriminator: Discriminator
    optimizer_d: AdamW
    scheduler_d: ExponentialLR
    
    stft_loss: losses.MultiScaleSTFTLoss
    mel_loss: losses.MelSpectrogramLoss
    gan_loss: losses.GANLoss
    waveform_loss: losses.L1Loss
    enc_feat_loss: losses.EncFeatureLoss
    noise_feat_loss: losses.NoiseFeatureLoss 
    ce_loss: torch.nn.CrossEntropyLoss

    train_data: AudioDataset
    val_data: AudioDataset

    tracker: Tracker
    train_with_clean: bool

def count_mamba_params(model: torch.nn.Module):
    records = []
    for name, module in tqdm(model.named_modules(), desc="Scanning modules"):
        if "Mamba" in module.__class__.__name__:
            num = sum(p.numel() for p in module.parameters())
            records.append((name, num))
    total = sum(num for _, num in records)
    return total

@argbind.bind(without_prefix=True)
def load(
    args,
    accel: ml.Accelerator,
    tracker: Tracker,
    save_path: str,
    resume: bool = False,
    tag: str = "latest",
    pretrained_path: str = None,
    load_discriminator: bool = False,
    train_with_clean: bool = False,
):    
    generator, g_extra = None, {}
    discriminator, d_extra = None, {}
    
    if resume:
        kwargs = {
            "folder": f"{save_path}/{tag}",
            "map_location": "cpu",
            "package": False,
        }
        tracker.print(f"Resuming from {str(Path('.').absolute())}/{kwargs['folder']}")
        if (Path(kwargs["folder"]) / "dac_vrvq_featuredenoise").exists():
            _, g_extra = DAC_VRVQ_FeatureDenoise.load_from_folder(**kwargs)
            generator = DAC_VRVQ_FeatureDenoise()
            ckpt_gen = Path(kwargs["folder"]) / "dac_vrvq_featuredenoise" / "weights.pth"
            ckpt_gen = torch.load(ckpt_gen, map_location="cpu")
            generator.load_state_dict(ckpt_gen["state_dict"], strict=True)
        else:
            raise ValueError("No Generator model found in the folder")
        if (Path(kwargs["folder"]) / "discriminator").exists():
            _, d_extra = Discriminator.load_from_folder(**kwargs)
            discriminator = Discriminator()
            ckpt_disc = Path(kwargs["folder"]) / "discriminator" / "weights.pth"
            ckpt_disc = torch.load(ckpt_disc, map_location="cpu")
            discriminator.load_state_dict(ckpt_disc["state_dict"], strict=True)
        else:
            raise ValueError("No Discriminator model found in the folder")
        
    elif not resume:
        print("### Start Training from Pretrained Model of Stage 1")
        assert pretrained_path is not None
        tracker.print(f"Loading pretrained model from {pretrained_path}")
        ckpt_gen = torch.load(
            os.path.join(pretrained_path, "dac_vrvq", "weights.pth"),
        )
        generator = DAC_VRVQ_FeatureDenoise()
        generator.load_state_dict(ckpt_gen["state_dict"], strict=False)
        if load_discriminator:
            ckpt_disc = torch.load(
                os.path.join(pretrained_path, "discriminator", "weights.pth"),
            )
            discriminator = Discriminator()
            discriminator.load_state_dict(ckpt_disc["state_dict"], strict=True)
        
    generator = DAC_VRVQ_FeatureDenoise() if generator is None else generator
    discriminator = Discriminator() if discriminator is None else discriminator
    
    tracker.print(generator)
    tracker.print(discriminator)
    
    generator = accel.prepare_model(generator)
    discriminator = accel.prepare_model(discriminator)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if accel.use_ddp:
        gen_original = generator.module if hasattr(generator, 'module') else generator
        dis_original = discriminator.module if hasattr(discriminator, 'module') else discriminator
    
        generator = nn.parallel.DistributedDataParallel(
            gen_original,
            device_ids=[local_rank],         
            output_device=local_rank,  
            find_unused_parameters=True, 
            check_reduction=False,
        )
        discriminator = nn.parallel.DistributedDataParallel(
            dis_original,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
            check_reduction=False,
        )

    with argbind.scope(args, "generator"):
        optimizer_g = AdamW(generator.parameters(), use_zero=accel.use_ddp)
        scheduler_g = ExponentialLR(optimizer_g)
    with argbind.scope(args, "discriminator"):
        optimizer_d = AdamW(discriminator.parameters(), use_zero=accel.use_ddp)
        scheduler_d = ExponentialLR(optimizer_d)
        
    if "optimizer.pth" in g_extra:
        optimizer_g.load_state_dict(g_extra["optimizer.pth"])
        print(f"Loaded optimizer_g from {save_path}/{tag}/dac_vrvq_featuredenoise/optimizer.pth")
    if "scheduler.pth" in g_extra:
        scheduler_g.load_state_dict(g_extra["scheduler.pth"])
        print(f"Loaded scheduler_g from {save_path}/{tag}/dac_vrvq_featuredenoise/scheduler.pth")
    if "tracker.pth" in g_extra:
        tracker.load_state_dict(g_extra["tracker.pth"])
        print(f"Loaded tracker from {save_path}/{tag}/dac_vrvq_featuredenoise/tracker.pth")

    if "optimizer.pth" in d_extra:
        optimizer_d.load_state_dict(d_extra["optimizer.pth"])
        print(f"Loaded optimizer_d from {save_path}/{tag}/discriminator/optimizer.pth")
    if "scheduler.pth" in d_extra:
        scheduler_d.load_state_dict(d_extra["scheduler.pth"])
        print(f"Loaded scheduler_d from {save_path}/{tag}/discriminator/scheduler.pth")

    sample_rate = accel.unwrap(generator).sample_rate
    with argbind.scope(args, "train"):
        train_data = build_dataset(sample_rate)  
    with argbind.scope(args, "val"):
        val_data = build_dataset(sample_rate)  
        
    waveform_loss = losses.L1Loss()
    stft_loss = losses.MultiScaleSTFTLoss()
    mel_loss = losses.MelSpectrogramLoss()
    gan_loss = losses.GANLoss(discriminator)
    enc_feat_loss = losses.EncFeatureLoss()
    noise_feat_loss = losses.NoiseFeatureLoss() 
    ce_loss = torch.nn.CrossEntropyLoss()

    return State(
        generator=generator,
        optimizer_g=optimizer_g,
        scheduler_g=scheduler_g,
        discriminator=discriminator,
        optimizer_d=optimizer_d,
        scheduler_d=scheduler_d,
        waveform_loss=waveform_loss,
        stft_loss=stft_loss,
        mel_loss=mel_loss,
        gan_loss=gan_loss,
        enc_feat_loss=enc_feat_loss,
        noise_feat_loss=noise_feat_loss,
        ce_loss=ce_loss,
        tracker=tracker,
        train_data=train_data,
        val_data=val_data,
        train_with_clean=train_with_clean,
    )
    
@timer()
@torch.no_grad()
def val_loop(batch, state, accel):
    twc = state.train_with_clean
    state.generator.eval()
    batch = util.prepare_batch(batch, accel.device)
    
    signal_clean = batch["signal_clean"].clone()
    signal_noisy = batch["signal_noisy"].clone()
    signal_scaled_noise = batch["signal_scaled_noise"].clone()
    
    assert signal_clean.shape == signal_noisy.shape
    assert signal_clean.sample_rate == signal_noisy.sample_rate
    
    sr = signal_clean.sample_rate
    out = state.generator(
        audio_data_noisy=signal_noisy.audio_data,
        audio_data_clean=signal_clean.audio_data if twc else None,
        audio_data_noise=signal_scaled_noise.audio_data if twc else None,
        sample_rate=sr
    )
    
    recons = AudioSignal(out["audio"], sr)
    imp_map = out["imp_map"]
    if imp_map is not None:
        rate_loss = imp_map.mean()
    else:
        rate_loss = None
    
    mel_loss = state.mel_loss(recons, signal_clean)
    
    ## Encoder feature loss
    if twc:
        enc_fmaps = out["enc_fmaps"]
        enc_loss = state.enc_feat_loss(enc_fmaps)
        noise_loss = state.noise_feat_loss(enc_fmaps) 
    else:
        enc_loss = 0.0
    
    ce_noise_loss = 0.0
    if "noise_logits" in out and "noise_label" in batch:
        try:
            noise_logits = out["noise_logits"]
            noise_labels = batch["noise_label"]
            
            if len(noise_labels.shape) > 1:
                noise_labels = torch.argmax(noise_labels, dim=-1)
            
            ce_noise_loss = state.ce_loss(noise_logits, noise_labels)
        except Exception as e:
            ce_noise_loss = 0.0
    
    return_dict = {
        "loss": state.mel_loss(recons, signal_clean),
        "mel/loss": mel_loss,
        "stft/loss": state.stft_loss(recons, signal_clean),
        "waveform/loss": state.waveform_loss(recons, signal_clean),
        "vq/rate_loss": rate_loss,
        "enc_feat/feat_loss": enc_loss,
        "noise_feat/feat_loss": noise_loss,
        "ce/noise_loss": ce_noise_loss,
    }
    
    return return_dict
    

@timer()
def train_loop_base(state, batch, accel, lambdas, feat_train_step=None):
    twc = state.train_with_clean

    state.generator.train()
    state.discriminator.train()
    output = {}

    hubert_model.to(accel.device)
    semantic_adapter.to(accel.device)
    semantic_adapter.train()

    try:
        n_codebooks = state.generator.n_codebooks
    except:
        n_codebooks = state.generator.module.n_codebooks

    batch = util.prepare_batch(batch, accel.device)
    with torch.no_grad():
        signal_clean = batch["signal_clean"].clone()
        signal_noisy = batch["signal_noisy"].clone()
        signal_scaled_noise = batch["signal_scaled_noise"].clone()
        assert signal_clean.shape == signal_noisy.shape
        assert signal_clean.sample_rate == signal_noisy.sample_rate
        sr = signal_clean.sample_rate

@timer()
def train_loop_base(state, batch, accel, lambdas, feat_train_step=None):
    twc = state.train_with_clean

    state.generator.train()
    state.discriminator.train()
    output = {}

    hubert_model.to(accel.device)
    semantic_adapter.to(accel.device)
    semantic_adapter.train()

    try:
        n_codebooks = state.generator.n_codebooks
    except:
        n_codebooks = state.generator.module.n_codebooks

    batch = util.prepare_batch(batch, accel.device)
    with torch.no_grad():
        signal_clean = batch["signal_clean"].clone()
        signal_noisy = batch["signal_noisy"].clone()
        signal_scaled_noise = batch["signal_scaled_noise"].clone()
        assert signal_clean.shape == signal_noisy.shape
        assert signal_clean.sample_rate == signal_noisy.sample_rate
        sr = signal_clean.sample_rate

    with torch.no_grad(), accel.autocast():
        clean_audio = signal_clean.audio_data.squeeze(1)
        
        input_values = feature_extractor(
            clean_audio.cpu().numpy(),
            sampling_rate=sr,
            return_tensors="pt"
        ).input_values.to(accel.device)

        hu_output = hubert_model(input_values, output_hidden_states=True)
        
        semantic_feat = torch.mean(torch.stack(hu_output.hidden_states), dim=0)
        semantic_feat = semantic_feat.permute(0, 2, 1)

    with accel.autocast():
        out = state.generator(
            audio_data_noisy=signal_noisy.audio_data,
            audio_data_clean=signal_clean.audio_data if twc else None,
            audio_data_noise=signal_scaled_noise.audio_data if twc else None,
            sample_rate=sr
        )
        recons = AudioSignal(out["audio"], sr)
        enhanced_z = out["z"]       
        clean_z = out["z_clean"]  
        
        commitment_loss = out["vq/commitment_loss"]
        codebook_loss = out["vq/codebook_loss"]
        imp_map = out["imp_map"]

    with accel.autocast():
        semantic_feat = semantic_adapter(semantic_feat)
        semantic_feat = F.interpolate(semantic_feat, size=enhanced_z.shape[-1], mode="linear", align_corners=False)
        residual_enhanced = enhanced_z - semantic_feat
        residual_clean = clean_z - semantic_feat
        output["semantic/res_loss"] = F.l1_loss(residual_enhanced, residual_clean.detach())

    with accel.autocast():
        output["adv/disc_loss"] = state.gan_loss.discriminator_loss(recons, signal_clean)

    state.optimizer_d.zero_grad()
    accel.backward(output["adv/disc_loss"])
    accel.scaler.unscale_(state.optimizer_d)
    output["other/grad_norm_d"] = torch.nn.utils.clip_grad_norm_(
        state.discriminator.parameters(), 10.0
    )
    accel.step(state.optimizer_d)
    state.scheduler_d.step()

    with accel.autocast():
        output["stft/loss"] = state.stft_loss(recons, signal_clean)
        output["mel/loss"] = state.mel_loss(recons, signal_clean)
        output["waveform/loss"] = state.waveform_loss(recons, signal_clean)
        (
            output["adv/gen_loss"],
            output["adv/feat_loss"],
        ) = state.gan_loss.generator_loss(recons, signal_clean)
        output["vq/commitment_loss"] = commitment_loss
        output["vq/codebook_loss"] = codebook_loss

        if imp_map is not None:
            rate_loss = imp_map.mean()
            output["vq/rate_loss"] = rate_loss
            output["vq/rate_loss_scaled"] = rate_loss * n_codebooks
        else:
            rate_loss = None
            
        loss_enc_feat = 0.0
        loss_noise_feat = 0.0
        if twc:
            enc_fmaps = out["enc_fmaps"]
            loss_enc_feat = state.enc_feat_loss(enc_fmaps)
            loss_noise_feat = state.noise_feat_loss(enc_fmaps)   
            output["enc_feat/feat_loss"] = loss_enc_feat
            output["noise_feat/feat_loss"] = loss_noise_feat 
        else:
            output["enc_feat/feat_loss"] = 0.0
        
        output["ce/noise_loss"] = 0.0
        if "noise_logits" in out and "noise_label" in batch:
            try:
                noise_logits = out["noise_logits"]
                noise_labels = batch["noise_label"]
                if len(noise_labels.shape) > 1:
                    noise_labels = torch.argmax(noise_labels, dim=-1)
                ce_loss_val = state.ce_loss(noise_logits, noise_labels)
                output["ce/noise_loss"] = ce_loss_val
            except Exception as e:
                output["ce/noise_loss"] = 0.0

        global_step = state.tracker.step
        if feat_train_step is not None:
            assert twc, "Please train with clean signal"
            if global_step < feat_train_step:
                output["loss"] = lambdas["enc_feat/feat_loss"] * loss_enc_feat + lambdas["noise_feat/feat_loss"] * loss_noise_feat
                if "ce/noise_loss" in lambdas:
                    output["loss"] += lambdas["ce/noise_loss"] * output["ce/noise_loss"]
            else:
                output["loss"] = sum([v * output[k] for k, v in lambdas.items()])
        else:
            output["loss"] = sum([v * output[k] for k, v in lambdas.items()])

    state.optimizer_g.zero_grad()
    accel.backward(output["loss"])
    accel.scaler.unscale_(state.optimizer_g)
    output["other/grad_norm_g"] = torch.nn.utils.clip_grad_norm_(
        state.generator.parameters(), 10.0
    )
    accel.step(state.optimizer_g)
    state.scheduler_g.step()
    accel.update()

    output["other/learning_rate_g"] = state.optimizer_g.param_groups[0]["lr"]
    output["other/batch_size"] = signal_clean.batch_size * accel.world_size
    
    return {k: v for k, v in sorted(output.items())}


def checkpoint_base(state, save_iters, save_path, package=True):
    metadata = {"logs": state.tracker.history}

    tags = ["latest"]
    state.tracker.print(f"Saving to {str(Path('.').absolute())}")
    if state.tracker.is_best("val", "mel/loss"):
        state.tracker.print(f"Best generator so far")
        tags.append("best")
    if state.tracker.step in save_iters:
        tags.append(f"{state.tracker.step // 1000}k")

    for tag in tags:
        generator_extra = {
            "optimizer.pth": state.optimizer_g.state_dict(),
            "scheduler.pth": state.scheduler_g.state_dict(),
            "tracker.pth": state.tracker.state_dict(),
            "metadata.pth": metadata,
        }
        accel.unwrap(state.generator).metadata = metadata
        accel.unwrap(state.generator).save_to_folder(
            f"{save_path}/{tag}", generator_extra, package=package
        )
        discriminator_extra = {
            "optimizer.pth": state.optimizer_d.state_dict(),
            "scheduler.pth": state.scheduler_d.state_dict(),
        }
        accel.unwrap(state.discriminator).save_to_folder(
            f"{save_path}/{tag}", discriminator_extra, package=package
        )

    
@torch.no_grad()
def save_samples_base(state, val_idx, writer):
    state.tracker.print("Saving audio samples to TensorBoard")
    state.generator.eval()

    samples = [state.val_data[idx] for idx in val_idx]
    batch = state.val_data.collate(samples)
    batch = util.prepare_batch(batch, accel.device)

    signal_clean = batch["signal_clean"].clone()
    signal_noisy = batch["signal_noisy"].clone()
    signal_scaled_noise = batch["signal_scaled_noise"].clone() 

    out = state.generator(
        audio_data_noisy=signal_noisy.audio_data,
        audio_data_clean=None,
        audio_data_noise=None,
        sample_rate=signal_noisy.sample_rate
    )
    recons = AudioSignal(out["audio"], signal_noisy.sample_rate)
    bs = signal_clean.shape[0]
    
    audio_dict = {}
    if state.tracker.step == 0:
        audio_dict["signal_clean"] = signal_clean
        audio_dict["signal_noisy"] = signal_noisy 
        audio_dict["signal_scaled_noise"] = signal_scaled_noise 
    audio_dict["signal_recons"] = recons

    for k, v in audio_dict.items():
        for nb in range(v.batch_size):
            v[nb].cpu().write_audio_to_tb(
                f"{k}/sample_{nb}.wav", writer, state.tracker.step
            )
            
    mask_imp = out["mask_imp"]
    if mask_imp is not None:
        for nb in range(bs):
            mask = mask_imp[nb]
            mask = mask * 0.7
            mask = mask.unsqueeze(0).unsqueeze(0)
            writer.add_images(f"imp_map/sample_{nb}", mask, state.tracker.step)


def validate_base(state, val_dataloader, accel):
    for idx, batch in enumerate(val_dataloader):
        output = val_loop(batch, state, accel)
        
    if hasattr(state.optimizer_g, "consolidate_state_dict"):
        state.optimizer_g.consolidate_state_dict()
        state.optimizer_d.consolidate_state_dict()
    return output


### TRAIN
@argbind.bind(without_prefix=True)
def train(
    args,
    accel: ml.Accelerator,
    seed: int = 0,
    save_path: str = "ckpt",
    num_iters: int = 250000,
    save_iters: list = [10000, 50000, 100000, 200000],
    sample_freq: int = 10000,
    valid_freq: int = 1000,
    batch_size: int = 12,
    val_batch_size: int = 10,
    num_workers: int = 8,
    val_idx: list = [0, 1, 2, 3, 4, 5, 6, 7],
    lambdas: dict = {
        "mel/loss": 15.0,
        "adv/feat_loss": 2.0,
        "adv/gen_loss": 1.0,
        "vq/commitment_loss": 0.25,
        "vq/codebook_loss": 1.0,
        "vq/rate_loss":1.0,
        "enc_feat/feat_loss":0.2, 
        "noise_feat/feat_loss":0.2,
        "ce/noise_loss": 0.1,
        "semantic/res_loss": 0.1,
    },
    save_package=False,
    feat_train_step=None,
):
    global train_loop, val_loop, validate, save_samples, checkpoint
    
    util.seed(seed)
    Path(save_path).mkdir(exist_ok=True, parents=True)
    writer = (
        SummaryWriter(log_dir=f"{save_path}/logs") if accel.local_rank == 0 else None
    )
    tracker = Tracker(
        writer=writer, log_file=f"{save_path}/log.txt", rank=accel.local_rank
    )
    
    state = load(args, accel, tracker, save_path)
    train_dataloader = accel.prepare_dataloader(
        state.train_data,
        start_idx=state.tracker.step * batch_size,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=state.train_data.collate,
    )
    train_dataloader = get_infinite_loader(train_dataloader)
    val_dataloader = accel.prepare_dataloader(
        state.val_data,
        start_idx=0,
        num_workers=num_workers,
        batch_size=val_batch_size,
        collate_fn=state.val_data.collate,
        persistent_workers=True if num_workers > 0 else False,
    )

    train_loop = tracker.log("train", "value", history=False)(
        tracker.track("train", num_iters, completed=state.tracker.step)(train_loop_base)
    )
    val_loop = tracker.track("val", len(val_dataloader))(val_loop)
    validate = tracker.log("val", "mean")(validate_base)
    save_samples = when(lambda: accel.local_rank == 0)(save_samples_base)
    checkpoint = when(lambda: accel.local_rank == 0)(checkpoint_base)

    TRAIN_GLOBAL_CLEAN_NUM = 149736
    total_batch_size = batch_size * accel.world_size
    steps_per_epoch = max(1, math.ceil(TRAIN_GLOBAL_CLEAN_NUM / total_batch_size))

    for tracker.step, batch in enumerate(train_dataloader, start=tracker.step):
        if tracker.step % 100 == 0:
            print(f"Config: {args['args.load']}")
            current_step = tracker.step
            current_epoch_float = current_step / steps_per_epoch  
            completed_epochs = current_step // steps_per_epoch 
            remainder = current_step % steps_per_epoch  
        
            if remainder == 0 and current_step > 0:
                current_epoch_progress = 100.0
            else:
                current_epoch_progress = (remainder / steps_per_epoch) * 100
        
            print(f"Epoch: {current_epoch_float:.2f} "
                  f"(Completed: {completed_epochs}, "
                  f"Progress: {current_epoch_progress:.1f}%), "
                  f"Step: {current_step}")
        
        output_loop = train_loop(state, batch, accel, lambdas, feat_train_step=feat_train_step)

        last_iter = (
            tracker.step == num_iters - 1 if num_iters is not None else False
        )
        if tracker.step % sample_freq == 0 or last_iter:
            save_samples(state, val_idx, writer)

        if tracker.step % valid_freq == 0 or last_iter:
            validate(state, val_dataloader, accel)
            checkpoint(state, save_iters, save_path, package=save_package)
            tracker.done("val", f"Iteration {tracker.step}")

        if last_iter:
            break 
            
    return save_path


if __name__ == "__main__":
    args = argbind.parse_args()
    args["args.debug"] = int(os.getenv("LOCAL_RANK", 0)) == 0
    with argbind.scope(args):
        with Accelerator() as accel:
            if accel.local_rank != 0:
                sys.tracebacklimit = 0
            save_path = train(args, accel)
