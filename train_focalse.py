# academicodec/models/hificodec/train_focalse.py
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse
import json
import sys
import math
import random
import torchaudio
import torch
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel

sys.path.insert(0, '/home/shaokai/NoiseRobustVRVQ-main')
from model.focal_se import FocalSpeechEnhancementModule
from model.dac_focalse_lightNR import NoiseClassifier

from transformers import HubertModel, Wav2Vec2FeatureExtractor

hubert_model = HubertModel.from_pretrained("/home/shaokai/llsdr-main/hubert-base-ls960").eval()
feature_extractor_hubert = Wav2Vec2FeatureExtractor.from_pretrained("/home/shaokai/llsdr-main/hubert-base-ls960")
for p in hubert_model.parameters():
    p.requires_grad = False

class SemanticAdapter(torch.nn.Module):
    def __init__(self, in_dim=768, out_dim=512):
        super().__init__()
        self.adapter = torch.nn.Conv1d(in_dim, out_dim, kernel_size=1)
    def forward(self, x):
        return self.adapter(x)

semantic_adapter = SemanticAdapter()

from academicodec.models.hificodec.env import AttrDict, build_env
from academicodec.models.hificodec.meldataset import mel_spectrogram, get_dataset_filelist
from academicodec.models.encodec.msstftd import MultiScaleSTFTDiscriminator
from academicodec.models.hificodec.models import (
    Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator,
    feature_loss, generator_loss, discriminator_loss, Encoder, Quantizer
)
from academicodec.utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint

from noisy_dataset import NoisyMelDataset
from losses import EncFeatureLoss, NoiseFeatureLoss, SemanticResidualLoss

torch.backends.cudnn.benchmark = True

def train(rank, a, h):
    torch.cuda.set_device(rank)
    if h.num_gpus > 1:
        init_process_group(
            backend=h.dist_config['dist_backend'],
            init_method=h.dist_config['dist_url'],
            world_size=h.dist_config['world_size'] * h.num_gpus,
            rank=rank
        )

    torch.cuda.manual_seed(h.seed)
    device = torch.device(f'cuda:{rank}')

    encoder = Encoder(h).to(device)
    generator = Generator(h).to(device)
    quantizer = Quantizer(h).to(device)

    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)
    mstftd = MultiScaleSTFTDiscriminator(32).to(device)

    focal_se = FocalSpeechEnhancementModule(
        input_dim=512,
        hidden_dim=256,
        num_transformer_layers=4,
        num_heads=8,
        dropout=0.1,
        mask_beta=2.0,
        focal_window=3,
        window_size=7,
        causal=False
    ).to(device)

    noise_classifier = NoiseClassifier(
        in_channels=512,
        num_classes=50,
        dropout=0.1
    ).to(device)

    semantic_adapter.to(device)
    hubert_model.to(device)

    pretrained_path = a.pretrained_path
    cp_g = scan_checkpoint(pretrained_path, 'g_')

    if cp_g is None:
        raise FileNotFoundError(f"No generator checkpoint found in {pretrained_path}")

    if rank == 0:
        print(f"Loading generator from: {cp_g}")

    state_dict_g = load_checkpoint(cp_g, device)

    encoder.load_state_dict(state_dict_g['encoder'], strict=True)
    quantizer.load_state_dict(state_dict_g['quantizer'], strict=True)
    generator.load_state_dict(state_dict_g['generator'], strict=True)

    encoder_clean = Encoder(h).to(device)
    encoder_clean.load_state_dict(state_dict_g['encoder'], strict=True)
    for p in encoder_clean.parameters():
        p.requires_grad = False
    if rank == 0:
        print("Clean encoder is frozen and will NOT be updated.")

    steps = 0
    last_epoch = -1

    if h.num_gpus > 1:
        encoder = DistributedDataParallel(encoder, device_ids=[rank])
        quantizer = DistributedDataParallel(quantizer, device_ids=[rank])
        generator = DistributedDataParallel(generator, device_ids=[rank])
        focal_se = DistributedDataParallel(focal_se, device_ids=[rank])
        noise_classifier = DistributedDataParallel(noise_classifier, device_ids=[rank])
        mpd = DistributedDataParallel(mpd, device_ids=[rank])
        msd = DistributedDataParallel(msd, device_ids=[rank])
        mstftd = DistributedDataParallel(mstftd, device_ids=[rank])

    optim_g = torch.optim.Adam(
        itertools.chain(
            encoder.parameters(),
            quantizer.parameters(),
            generator.parameters(),
            focal_se.parameters(),
            noise_classifier.parameters(),
            semantic_adapter.parameters()
        ),
        h.learning_rate,
        betas=[h.adam_b1, h.adam_b2]
    )
    optim_d = torch.optim.Adam(
        itertools.chain(msd.parameters(), mpd.parameters(), mstftd.parameters()),
        h.learning_rate,
        betas=[h.adam_b1, h.adam_b2]
    )

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay)

    train_files, val_files = get_dataset_filelist(a)
    trainset = NoisyMelDataset(
        train_files,
        h.segment_size, h.n_fft, h.num_mels, h.hop_size, h.win_size,
        h.sampling_rate, h.fmin, h.fmax,
        noise_filelist=a.noise_filelist,
        snr_range=(-10, 20),
        shuffle=False if h.num_gpus > 1 else True,
        device=device,
        fmax_loss=h.fmax_for_loss
    )
    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None
    train_loader = DataLoader(
        trainset, num_workers=h.num_workers, shuffle=False,
        sampler=train_sampler, batch_size=h.batch_size, pin_memory=True, drop_last=True
    )

    if rank == 0:
        validset = NoisyMelDataset(
            val_files,
            h.segment_size, h.n_fft, h.num_mels, h.hop_size, h.win_size,
            h.sampling_rate, h.fmin, h.fmax,
            noise_filelist=a.noise_filelist,
            snr_range=(10, 10),
            shuffle=False, device=device, fmax_loss=h.fmax_for_loss
        )
        val_loader = DataLoader(validset, num_workers=1, shuffle=False, batch_size=1, pin_memory=True, drop_last=True)
        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    enc_feat_loss_fn = EncFeatureLoss()
    noise_feat_loss_fn = NoiseFeatureLoss()
    semantic_loss_fn = SemanticResidualLoss(hubert_model, semantic_adapter)
    ce_loss_fn = torch.nn.CrossEntropyLoss()

    plot_gt_once = False

    for epoch in range(max(0, last_epoch), a.training_epochs):
        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for batch in train_loader:
            x, y, fname, y_mel, noisy, scaled_noise, noise_label = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).unsqueeze(1)
            y_mel = y_mel.to(device, non_blocking=True)
            noisy = noisy.to(device, non_blocking=True).unsqueeze(1)
            scaled_noise = scaled_noise.to(device, non_blocking=True).unsqueeze(1)
            noise_label = noise_label.to(device, non_blocking=True).long()

            with torch.no_grad():
                c_clean = encoder_clean(y)   
                c_noise = encoder_clean(scaled_noise) 
            c_noisy = encoder(noisy)
            enhanced_c, pred_noise_feat = focal_se(c_noisy)

            q, loss_q, _ = quantizer(enhanced_c)
            y_g_hat = generator(q)

            optim_d.zero_grad()

            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
            y_disc_r, _ = mstftd(y)
            y_disc_gen, _ = mstftd(y_g_hat.detach())
            loss_disc_stft, _, _ = discriminator_loss(y_disc_r, y_disc_gen)

            loss_disc_all = loss_disc_s + loss_disc_f + loss_disc_stft
            loss_disc_all.backward()
            optim_d.step()

            optim_g.zero_grad()

            y_g_hat_mel = mel_spectrogram(
                y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                h.hop_size, h.win_size, h.fmin, h.fmax_for_loss
            )
            y_r_mel_1 = mel_spectrogram(y.squeeze(1), 512, h.num_mels, h.sampling_rate, 120, 512, h.fmin, h.fmax_for_loss)
            y_g_mel_1 = mel_spectrogram(y_g_hat.squeeze(1), 512, h.num_mels, h.sampling_rate, 120, 512, h.fmin, h.fmax_for_loss)
            y_r_mel_2 = mel_spectrogram(y.squeeze(1), 256, h.num_mels, h.sampling_rate, 60, 256, h.fmin, h.fmax_for_loss)
            y_g_mel_2 = mel_spectrogram(y_g_hat.squeeze(1), 256, h.num_mels, h.sampling_rate, 60, 256, h.fmin, h.fmax_for_loss)
            y_r_mel_3 = mel_spectrogram(y.squeeze(1), 128, h.num_mels, h.sampling_rate, 30, 128, h.fmin, h.fmax_for_loss)
            y_g_mel_3 = mel_spectrogram(y_g_hat.squeeze(1), 128, h.num_mels, h.sampling_rate, 30, 128, h.fmin, h.fmax_for_loss)

            loss_mel1 = F.l1_loss(y_r_mel_1, y_g_mel_1)
            loss_mel2 = F.l1_loss(y_r_mel_2, y_g_mel_2)
            loss_mel3 = F.l1_loss(y_r_mel_3, y_g_mel_3)
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45 + loss_mel1 + loss_mel2

            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            y_stftd_r, fmap_stftd_r = mstftd(y)
            y_stftd_g, fmap_stftd_g = mstftd(y_g_hat)

            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_fm_stft = feature_loss(fmap_stftd_r, fmap_stftd_g)
            loss_gen_f, _ = generator_loss(y_df_hat_g)
            loss_gen_s, _ = generator_loss(y_ds_hat_g)
            loss_gen_stft, _ = generator_loss(y_stftd_g)

            loss_gan = loss_gen_f + loss_gen_s + loss_gen_stft + loss_fm_f + loss_fm_s + loss_fm_stft

            loss_enc_feat = enc_feat_loss_fn({'feat': c_clean}, {'feat': enhanced_c})
            loss_noise_feat = noise_feat_loss_fn({'feat': pred_noise_feat}, {'feat': c_noise})
            noise_logits = noise_classifier(pred_noise_feat)
            loss_ce = ce_loss_fn(noise_logits, noise_label)

            clean_16k = torchaudio.functional.resample(
                y.squeeze(1), orig_freq=h.sampling_rate, new_freq=16000
            )
            loss_sem = semantic_loss_fn(clean_16k, y.squeeze(1), c_clean, enhanced_c)

            loss_gen_all = (
                loss_mel
                + loss_q * 10
                + loss_gan
                + loss_enc_feat * 0.5
                + loss_noise_feat * 0.5
                + loss_ce * 0.2
                + loss_sem * 0.2
            )

            loss_gen_all.backward()
            optim_g.step()

            if rank == 0:
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()
                    print(
                        f'Steps: {steps:d}, Gen Loss: {loss_gen_all:.3f}, '
                        f'Q: {loss_q:.3f}, Mel: {mel_error:.3f}, '
                        f'EncFeat: {loss_enc_feat:.3f}, NoiseFeat: {loss_noise_feat:.3f}, '
                        f'CE: {loss_ce:.3f}, Sem: {loss_sem:.3f}'
                    )

                if steps % a.checkpoint_interval == 0 and steps != 0:
                    ckpt_path = f"{a.checkpoint_path}/g_{steps:08d}"
                    save_checkpoint(
                        ckpt_path,
                        {
                            'generator': (generator.module if h.num_gpus > 1 else generator).state_dict(),
                            'encoder': (encoder.module if h.num_gpus > 1 else encoder).state_dict(),
                            'quantizer': (quantizer.module if h.num_gpus > 1 else quantizer).state_dict(),
                            'focal_se': (focal_se.module if h.num_gpus > 1 else focal_se).state_dict(),
                            'noise_classifier': (noise_classifier.module if h.num_gpus > 1 else noise_classifier).state_dict(),
                            'semantic_adapter': semantic_adapter.state_dict(),
                        },
                        num_ckpt_keep=a.num_ckpt_keep
                    )
                    ckpt_path_do = f"{a.checkpoint_path}/do_{steps:08d}"
                    save_checkpoint(
                        ckpt_path_do,
                        {
                            'mpd': (mpd.module if h.num_gpus > 1 else mpd).state_dict(),
                            'msd': (msd.module if h.num_gpus > 1 else msd).state_dict(),
                            'mstftd': (mstftd.module if h.num_gpus > 1 else mstftd).state_dict(),
                            'optim_g': optim_g.state_dict(),
                            'optim_d': optim_d.state_dict(),
                            'steps': steps,
                            'epoch': epoch
                        },
                        num_ckpt_keep=a.num_ckpt_keep
                    )

                if steps % a.summary_interval == 0:
                    sw.add_scalar("train/gen_loss_total", loss_gen_all, steps)
                    sw.add_scalar("train/mel_error", mel_error, steps)
                    sw.add_scalar("train/enc_feat_loss", loss_enc_feat, steps)
                    sw.add_scalar("train/noise_feat_loss", loss_noise_feat, steps)
                    sw.add_scalar("train/ce_loss", loss_ce, steps)
                    sw.add_scalar("train/sem_loss", loss_sem, steps)

                if steps % a.validation_interval == 0 and steps != 0:
                    generator.eval()
                    encoder.eval()
                    quantizer.eval()
                    focal_se.eval()
                    torch.cuda.empty_cache()
                    val_err_tot = 0
                    with torch.no_grad():
                        for j, val_batch in enumerate(val_loader):
                            x_val, y_val, _, y_mel_val, noisy_val, _, _ = val_batch
                            y_val = y_val.to(device).unsqueeze(1)
                            y_mel_val = y_mel_val.to(device)
                            noisy_val = noisy_val.to(device).unsqueeze(1)

                            c_noisy_val = encoder(noisy_val)
                            enhanced_val, _ = focal_se(c_noisy_val)
                            q_val, _, _ = quantizer(enhanced_val)
                            y_g_hat_val = generator(q_val)

                            y_g_hat_mel_val = mel_spectrogram(
                                y_g_hat_val.squeeze(1),
                                h.n_fft, h.num_mels, h.sampling_rate,
                                h.hop_size, h.win_size, h.fmin, h.fmax_for_loss
                            )
                            min_len = min(y_mel_val.size(2), y_g_hat_mel_val.size(2))
                            val_err_tot += F.l1_loss(
                                y_mel_val[:, :, :min_len],
                                y_g_hat_mel_val[:, :, :min_len]
                            ).item()

                            if j <= 4 and rank == 0:
                                if not plot_gt_once:
                                    sw.add_audio('gt/y_{}'.format(j), y_val[0], steps, h.sampling_rate)
                                    sw.add_figure('gt/y_spec_{}'.format(j), plot_spectrogram(x_val[0]), steps)
                                sw.add_audio('generated/y_hat_{}'.format(j), y_g_hat_val[0], steps, h.sampling_rate)
                                y_hat_spec = mel_spectrogram(
                                    y_g_hat_val.squeeze(1),
                                    h.n_fft, h.num_mels, h.sampling_rate,
                                    h.hop_size, h.win_size, h.fmin, h.fmax
                                )
                                sw.add_figure(
                                    'generated/y_hat_spec_{}'.format(j),
                                    plot_spectrogram(y_hat_spec.squeeze(0).cpu().numpy()),
                                    steps
                                )
                        val_err = val_err_tot / (j + 1)
                        sw.add_scalar("validation/mel_spec_error", val_err, steps)
                        if not plot_gt_once:
                            plot_gt_once = True

                    generator.train()
                    encoder.train()
                    quantizer.train()
                    focal_se.train()

            steps += 1

        scheduler_g.step()
        scheduler_d.step()

        if rank == 0:
            print(f'Time taken for epoch {epoch + 1} is {int(time.time() - time.time())} sec\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_training_file', required=True)
    parser.add_argument('--input_validation_file', required=True)
    parser.add_argument('--noise_filelist', required=True)
    parser.add_argument('--checkpoint_path', default='logs_focalse')
    parser.add_argument('--pretrained_path', required=True)
    parser.add_argument('--config', default='config_16k_320d.json')
    parser.add_argument('--training_epochs', default=2000, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=5000, type=int)
    parser.add_argument('--num_ckpt_keep', default=50, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)
    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        h.num_gpus = 0

    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h))
    else:
        train(0, a, h)


if __name__ == '__main__':
    main()