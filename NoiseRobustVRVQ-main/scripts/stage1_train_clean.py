"""
This code is heavily adapted and modified from the original DAC training code.  
Original source: https://github.com/descriptinc/descript-audio-codec/blob/main/scripts/train.py
"""
import os
if 'CUDA_VISIBLE_DEVICES' in os.environ and 'LOCAL_RANK' in os.environ:
    cuda_devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    local_rank = int(os.environ['LOCAL_RANK'])
    if local_rank < len(cuda_devices):
        os.environ['LOCAL_RANK'] = cuda_devices[local_rank]

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import argbind
import torch
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

sys.path.append(str(Path(__file__).resolve().parents[1]))
from model.utils import cal_bpf_from_mask, cal_entropy
from data.loaders import AudioLoader_EARS_Clean, AudioDataset_EARS_Clean
from model.dac_vrvq import DAC_VRVQ
from model.discriminator import Discriminator
from model import loss
import math

warnings.filterwarnings("ignore", category=UserWarning)


# Enable cudnn autotuner to speed up training
# (can be altered by the funcs.seed function)
torch.backends.cudnn.benchmark = bool(int(os.getenv("CUDNN_BENCHMARK", 1)))
# Uncomment to trade memory for speed.

## Optimizers
AdamW = argbind.bind(torch.optim.AdamW, "generator", "discriminator")
Accelerator = argbind.bind(ml.Accelerator, without_prefix=True)

## Model
DAC_VRVQ = argbind.bind(DAC_VRVQ)
Discriminator = argbind.bind(Discriminator)

AudioDataset = argbind.bind(AudioDataset_EARS_Clean, "train", "val")
AudioLoader = argbind.bind(AudioLoader_EARS_Clean, "train", "val")

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
                return float(current_step) / float(max(1, warmup))  # warmup 단계: 학습률을 선형으로 증가
            return gamma ** (current_step - warmup)  # warmup 후: Exponential decay

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def get_infinite_loader(dataloader):
    while True:
        for batch in dataloader:
            yield batch

@argbind.bind("train", "val")
def build_dataset(
    sample_rate: int,
    folders: dict = None,
):
    loader = AudioLoader(
        srcs_clean = folders["clean"],
    )
    dataset = AudioDataset(
        loader=loader,
        sample_rate=sample_rate
    )
    return dataset

@dataclass
class State:
    generator: DAC_VRVQ
    optimizer_g: AdamW
    scheduler_g: ExponentialLR
    
    discriminator: Discriminator
    optimizer_d: AdamW
    scheduler_d: ExponentialLR
    
    stft_loss: losses.MultiScaleSTFTLoss
    mel_loss: losses.MelSpectrogramLoss
    gan_loss: losses.GANLoss
    waveform_loss: losses.L1Loss
    
    train_data: AudioDataset
    val_data: AudioDataset

    tracker: Tracker
    
def count_trainable_params(model: torch.nn.Module):
    trainable_params = []
    non_trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append((name, param.numel()))
        else:
            non_trainable_params.append((name, param.numel()))
    
    total_trainable = sum(num for _, num in trainable_params)
    total_non_trainable = sum(num for _, num in non_trainable_params)
    total = total_trainable + total_non_trainable
    
    print(f"\n===== 参数统计 =====")
    print(f"总参数数量: {total / 1e6:.2f}M")
    print(f"可训练参数: {total_trainable / 1e6:.2f}M ({total_trainable/total*100:.2f}%)")
    print(f"冻结参数: {total_non_trainable / 1e6:.2f}M ({total_non_trainable/total*100:.2f}%)")
    print(f"\n可训练参数列表:")
    for name, num in trainable_params[:10]:
        print(f"  {name}: {num / 1e3:.2f}K")
    if len(trainable_params) > 10:
        print(f"  ... 共 {len(trainable_params)} 个可训练参数")
    return total_trainable

@argbind.bind(without_prefix=True)
def load(
    args,
    accel: ml.Accelerator,
    tracker: Tracker,
    save_path: str,
    resume: bool = False,
    tag: str = "latest", ## 'best', '100k' etc
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
        if (Path(kwargs["folder"]) / "dac_vrvq").exists():
            _, g_extra = DAC_VRVQ.load_from_folder(**kwargs)
            generator = DAC_VRVQ()
            ckpt_gen = Path(kwargs["folder"]) / "dac_vrvq" / "weights.pth"
            ckpt_gen = torch.load(ckpt_gen, map_location="cpu")
            generator.load_state_dict(ckpt_gen["state_dict"], strict=True)
        else:
            raise ValueError("No Generator model found in the folder")
        if (Path(kwargs["folder"]) / "discriminator").exists():
            # discriminator, d_extra = Discriminator.load_from_folder(**kwargs)
            _, d_extra = Discriminator.load_from_folder(**kwargs)
            discriminator = Discriminator()
            ckpt_disc = Path(kwargs["folder"]) / "discriminator" / "weights.pth"
            ckpt_disc = torch.load(ckpt_disc, map_location="cpu")
            discriminator.load_state_dict(ckpt_disc["state_dict"], strict=True)
        else:
            raise ValueError("No Discriminator model found in the folder")
    
    generator = DAC_VRVQ() if generator is None else generator
    print("=============生成器参数量===========")
    count_trainable_params(generator)
    discriminator = Discriminator() if discriminator is None else discriminator
    
    tracker.print(generator)
    tracker.print(discriminator)
    
    generator = accel.prepare_model(generator)
    discriminator = accel.prepare_model(discriminator)
    
    with argbind.scope(args, "generator"):
        optimizer_g = AdamW(generator.parameters(), use_zero=accel.use_ddp)
        scheduler_g = ExponentialLR(optimizer_g)
    with argbind.scope(args, "discriminator"):
        optimizer_d = AdamW(discriminator.parameters(), use_zero=accel.use_ddp)
        scheduler_d = ExponentialLR(optimizer_d)
    
    # import pdb; pdb.set_trace()
        
    if "optimizer.pth" in g_extra:
        optimizer_g.load_state_dict(g_extra["optimizer.pth"])
    if "scheduler.pth" in g_extra:
        scheduler_g.load_state_dict(g_extra["scheduler.pth"])
    if "tracker.pth" in g_extra:
        tracker.load_state_dict(g_extra["tracker.pth"])

    if "optimizer.pth" in d_extra:
        optimizer_d.load_state_dict(d_extra["optimizer.pth"])
    if "scheduler.pth" in d_extra:
        scheduler_d.load_state_dict(d_extra["scheduler.pth"])

    sample_rate = accel.unwrap(generator).sample_rate
    with argbind.scope(args, "train"):
        train_data = build_dataset(sample_rate)
    with argbind.scope(args, "val"):
        val_data = build_dataset(sample_rate)
        
    waveform_loss = losses.L1Loss()
    stft_loss = losses.MultiScaleSTFTLoss()
    mel_loss = losses.MelSpectrogramLoss()
    gan_loss = losses.GANLoss(discriminator)

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
        tracker=tracker,
        train_data=train_data,
        val_data=val_data,
    )
    
@timer()
@torch.no_grad()
def val_loop(batch, state, accel):
    state.generator.eval()
    batch = util.prepare_batch(batch, accel.device)
    
    signal_clean = batch["signal_clean"].clone()

    sr = signal_clean.sample_rate
    out = state.generator(signal_clean.audio_data, sr)
    recons = AudioSignal(out["audio"], sr)
    imp_map = out["imp_map"]
    if imp_map is not None:
        rate_loss = imp_map.mean()
    else:
        rate_loss = None
    
    mel_loss = state.mel_loss(recons, signal_clean)
    
    return_dict = {
        "loss": state.mel_loss(recons, signal_clean),
        "mel/loss": mel_loss,
        "stft/loss": state.stft_loss(recons, signal_clean),
        "waveform/loss": state.waveform_loss(recons, signal_clean),
        "vq/rate_loss": rate_loss,
    }
    
    return return_dict
    

@timer()
def train_loop(state, batch, accel, lambdas):
    state.generator.train()
    state.discriminator.train()
    output = {}

    try:
        n_codebooks = state.generator.n_codebooks
    except:
        n_codebooks = state.generator.module.n_codebooks

    batch = util.prepare_batch(batch, accel.device)
    with torch.no_grad():
        signal_clean = batch["signal_clean"].clone()
        sr = signal_clean.sample_rate

    with accel.autocast():
        out = state.generator(signal_clean.audio_data, sr)
        recons = AudioSignal(out["audio"], sr)
        commitment_loss = out["vq/commitment_loss"]
        codebook_loss = out["vq/codebook_loss"]
        imp_map = out["imp_map"]

    ### Discriminator
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

    ### Generator
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
        
        output["loss"] = sum([v * output[k] for k, v in lambdas.items()])

    ### Generator: backward
    state.optimizer_g.zero_grad()
    accel.backward(output["loss"])
    accel.scaler.unscale_(state.optimizer_g)
    output["other/grad_norm_g"] = torch.nn.utils.clip_grad_norm_(
        state.generator.parameters(), 1e3
    )
    accel.step(state.optimizer_g)
    state.scheduler_g.step()
    accel.update()

    output["other/learning_rate_g"] = state.optimizer_g.param_groups[0]["lr"]
    output["other/batch_size"] = signal_clean.batch_size * accel.world_size
    
    output["other/signal_mean"] = signal_clean.audio_data.mean().item()
    output["other/signal_std"] = signal_clean.audio_data.std().item()
    
    return {k: v for k, v in sorted(output.items())}


def checkpoint(state, save_iters, save_path, package=True):
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
def save_samples(state, val_idx, writer):
    state.tracker.print("Saving audio samples to TensorBoard")
    state.generator.eval()

    samples = [state.val_data[idx] for idx in val_idx]
    batch = state.val_data.collate(samples)
    batch = util.prepare_batch(batch, accel.device)
    signal_clean = batch["signal_clean"].clone()

    out = state.generator(signal_clean.audio_data, signal_clean.sample_rate)
    recons = AudioSignal(out["audio"], signal_clean.sample_rate)
    bs = signal_clean.shape[0]
    
    # audio_dict = {"recons": recons}
    audio_dict = {}
    if state.tracker.step == 0:
        audio_dict["signal_clean"] = signal_clean
    audio_dict["signal_recons"] = recons

    for k, v in audio_dict.items():
        for nb in range(v.batch_size):
            # import pdb; pdb.set_trace()
            v[nb].cpu().write_audio_to_tb(
                f"{k}/sample_{nb}.wav", writer, state.tracker.step
            )
            
    """ Plot the importance map """
    # mask_imp = out["imp_map"] # (B, nq, T)
    mask_imp = out["mask_imp"]
    if mask_imp is not None:
        # for nb in range(v.batch_size):
        for nb in range(bs):
            mask = mask_imp[nb]
            mask = mask * 0.7
            mask = mask.unsqueeze(0).unsqueeze(0)
            writer.add_images(f"imp_map/sample_{nb}", mask, state.tracker.step)


def validate(state, val_dataloader, accel):
    for idx, batch in enumerate(val_dataloader):
        output = val_loop(batch, state, accel)
        
    # Consolidate state dicts if using ZeroRedundancyOptimizer
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
        "mel/loss": 100.0,
        "adv/feat_loss": 2.0,
        "adv/gen_loss": 1.0,
        "vq/commitment_loss": 0.25,
        "vq/codebook_loss": 1.0,
        "vq/rate_loss":1.0
    },
    save_package=False
):
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

    # Wrap the functions so that they neatly track in TensorBoard + progress bars
    # and only run when specific conditions are met.
    global train_loop, val_loop, validate, save_samples, checkpoint
    train_loop = tracker.log("train", "value", history=False)(
        tracker.track("train", num_iters, completed=state.tracker.step)(train_loop)
    )
    val_loop = tracker.track("val", len(val_dataloader))(val_loop)
    validate = tracker.log("val", "mean")(validate)

    # These functions run only on the 0-rank process
    save_samples = when(lambda: accel.local_rank == 0)(save_samples)
    checkpoint = when(lambda: accel.local_rank == 0)(checkpoint)


    for tracker.step, batch in enumerate(train_dataloader, start=tracker.step): ## 이 라인이 오래 걸림.
        if tracker.step % 50 == 0:
            print(f"Config: {args['args.load']}")
            print("Step:", tracker.step)
        
        output_loop = train_loop(state, batch, accel, lambdas)
        # import pdb; pdb.set_trace()

        last_iter = (
            tracker.step == num_iters - 1 if num_iters is not None else False
        )
        if tracker.step % sample_freq == 0 or last_iter:
            save_samples(state, val_idx, writer)

        if tracker.step % valid_freq == 0 or last_iter:
            validate(state, val_dataloader, accel)
            checkpoint(state, save_iters, save_path, package=save_package)
            # Reset validation progress bar, print summary since last validation.
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
            
