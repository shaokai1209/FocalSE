import sys, math, numpy as np
import torch, torchaudio, torch.nn.functional as F
import pytorch_lightning as pl
import transformers

torch.set_float32_matmul_precision('high')

sys.path.insert(0, '/home/shaokai/NoiseRobustVRVQ-main')
from model.dac_focalse_lightNR import NoiseClassifier
from transformers import HubertModel, Wav2Vec2FeatureExtractor

from decoder.experiment import WavTokenizer
from decoder.feature_extractors_focalse import EncodecFeaturesWithFocalSE
from decoder.modules import safe_log
from decoder.helpers import plot_spectrogram_to_numpy


hubert_model = HubertModel.from_pretrained("/home/shaokai/llsdr-main/hubert-base-ls960").eval()
for p in hubert_model.parameters():
    p.requires_grad = False

class SemanticAdapter(torch.nn.Module):
    def __init__(self, in_dim=768, out_dim=512):
        super().__init__()
        self.adapter = torch.nn.Conv1d(in_dim, out_dim, kernel_size=1)
    def forward(self, x):
        return self.adapter(x)

class WavTokenizer_FocalSE(WavTokenizer):
    def __init__(
        self,
        *args,
        noise_classes: int = 50,
        noise_cls_dropout: float = 0.1,
        focal_loss_coeff: float = 0.5,
        noise_ce_coeff: float = 0.2,
        sem_loss_coeff: float = 0.5,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.automatic_optimization = False

        self.noise_classifier = NoiseClassifier(
            in_channels=self.feature_extractor.focal_se.input_dim,
            num_classes=noise_classes,
            dropout=noise_cls_dropout
        )

        self.semantic_adapter = SemanticAdapter(
            in_dim=768,
            out_dim=self.feature_extractor.focal_se.input_dim
        )

        self.hubert = hubert_model

        self.focal_loss_coeff = focal_loss_coeff
        self.noise_ce_coeff = noise_ce_coeff
        self.sem_loss_coeff = sem_loss_coeff

        self.validation_outputs = []

    def configure_optimizers(self):
                disc_params = [
            {"params": self.multiperioddisc.parameters()},
            {"params": self.multiresddisc.parameters()},
            {"params": self.dac.parameters()},
        ]
        gen_params = [
            {"params": self.feature_extractor.parameters()},
            {"params": self.backbone.parameters()},
            {"params": self.head.parameters()},
            {"params": self.noise_classifier.parameters()},
            {"params": self.semantic_adapter.parameters()},
        ]

        opt_disc = torch.optim.AdamW(disc_params, lr=self.hparams.initial_learning_rate)
        opt_gen = torch.optim.AdamW(gen_params, lr=self.hparams.initial_learning_rate)

        max_steps = self.trainer.max_steps // 2
        scheduler_disc = transformers.get_cosine_schedule_with_warmup(
            opt_disc, num_warmup_steps=self.hparams.num_warmup_steps, num_training_steps=max_steps,
        )
        scheduler_gen = transformers.get_cosine_schedule_with_warmup(
            opt_gen, num_warmup_steps=self.hparams.num_warmup_steps, num_training_steps=max_steps,
        )

        return (
            [opt_disc, opt_gen],
            [{"scheduler": scheduler_disc, "interval": "step"}, {"scheduler": scheduler_gen, "interval": "step"}],
        )

    def on_fit_start(self):
        super().on_fit_start()
        self.hubert.to(self.device)

        encoder = self.feature_extractor.encodec.encoder
        self._clean_encoder_state = {
            name: param.data.clone()
            for name, param in encoder.named_parameters()
        }

    @torch.no_grad()
    def _extract_clean_feat(self, x):
        encoder = self.feature_extractor.encodec.encoder
        current_state = {}
        for name, param in encoder.named_parameters():
            current_state[name] = param.data
            param.data = self._clean_encoder_state[name]
        feat = encoder(x)
        for name, param in encoder.named_parameters():
            param.data = current_state[name]
        return feat

    def forward(self, noisy_audio, bandwidth_id=None):
        quantized, codes, commit_loss, (e_noisy, enhanced_e, pred_noise_feat) = \
            self.feature_extractor(noisy_audio, bandwidth_id)
        x = self.backbone(quantized, bandwidth_id=bandwidth_id)
        audio_hat = self.head(x)
        return audio_hat, commit_loss, (e_noisy, enhanced_e, pred_noise_feat)

    def training_step(self, batch, batch_idx):
        opt_disc, opt_gen = self.optimizers()
        sch_disc, sch_gen = self.lr_schedulers()

        clean_audio, noisy_audio, scaled_noise, noise_label = batch
        bandwidth_id = torch.randint(0, len(self.feature_extractor.bandwidths), (1,), device=self.device)

        if self.train_discriminator:
            with torch.no_grad():
                audio_hat, _, _ = self(noisy_audio, bandwidth_id)

            loss_dac = self.dacdiscriminator.discriminator_loss(audio_hat.unsqueeze(1), clean_audio.unsqueeze(1))
            real_mp, gen_mp, _, _ = self.multiperioddisc(y=clean_audio, y_hat=audio_hat, bandwidth_id=bandwidth_id)
            real_mrd, gen_mrd, _, _ = self.multiresddisc(y=clean_audio, y_hat=audio_hat, bandwidth_id=bandwidth_id)
            loss_mp, _, _ = self.disc_loss(real_mp, gen_mp)
            loss_mrd, _, _ = self.disc_loss(real_mrd, gen_mrd)
            loss_mp /= len(real_mp)
            loss_mrd /= len(real_mrd)
            loss_d = loss_mp + self.hparams.mrd_loss_coeff * loss_mrd + loss_dac

            opt_disc.zero_grad()
            self.manual_backward(loss_d)
            opt_disc.step()
            sch_disc.step()

            self.log("discriminator/total", loss_d, prog_bar=True)
            self.log("discriminator/multi_period_loss", loss_mp)
            self.log("discriminator/multi_res_loss", loss_mrd)
            self.log("discriminator/dac", loss_dac)

        audio_hat, commit_loss, (e_noisy, enhanced_e, pred_noise_feat) = self(noisy_audio, bandwidth_id)

        loss_dac_1 = loss_dac_2 = 0.0
        loss_gen_mp = loss_gen_mrd = loss_fm_mp = loss_fm_mrd = 0.0

        if self.train_discriminator:
            loss_dac_1, loss_dac_2 = self.dacdiscriminator.generator_loss(audio_hat.unsqueeze(1), clean_audio.unsqueeze(1))
            _, gen_mp, fmap_rs_mp, fmap_gs_mp = self.multiperioddisc(y=clean_audio, y_hat=audio_hat, bandwidth_id=bandwidth_id)
            _, gen_mrd, fmap_rs_mrd, fmap_gs_mrd = self.multiresddisc(y=clean_audio, y_hat=audio_hat, bandwidth_id=bandwidth_id)
            loss_gen_mp, _ = self.gen_loss(gen_mp)
            loss_gen_mrd, _ = self.gen_loss(gen_mrd)
            loss_gen_mp = loss_gen_mp / len(gen_mp) if gen_mp else 0
            loss_gen_mrd = loss_gen_mrd / len(gen_mrd) if gen_mrd else 0
            loss_fm_mp = self.feat_matching_loss(fmap_r=fmap_rs_mp, fmap_g=fmap_gs_mp) / len(fmap_rs_mp)
            loss_fm_mrd = self.feat_matching_loss(fmap_r=fmap_rs_mrd, fmap_g=fmap_gs_mrd) / len(fmap_rs_mrd)

            self.log("generator/multi_period_loss", loss_gen_mp)
            self.log("generator/multi_res_loss", loss_gen_mrd)
            self.log("generator/feature_matching_mp", loss_fm_mp)
            self.log("generator/feature_matching_mrd", loss_fm_mrd)
            self.log("generator/loss_dac_1", loss_dac_1)
            self.log("generator/loss_dac_2", loss_dac_2)

        mel_loss = self.melspec_loss(audio_hat.unsqueeze(1), clean_audio.unsqueeze(1))

        base_loss = (
            loss_gen_mp
            + self.hparams.mrd_loss_coeff * loss_gen_mrd
            + loss_fm_mp
            + self.hparams.mrd_loss_coeff * loss_fm_mrd
            + self.mel_loss_coeff * mel_loss
            + 1000 * commit_loss
            + loss_dac_1
            + loss_dac_2
        )

        with torch.no_grad():
            c_clean = self._extract_clean_feat(clean_audio.unsqueeze(1))
            c_noise = self._extract_clean_feat(scaled_noise.unsqueeze(1))

        enc_feat_loss = F.l1_loss(enhanced_e, c_clean)
        noise_feat_loss = F.l1_loss(pred_noise_feat, c_noise)
        noise_logits = self.noise_classifier(pred_noise_feat)
        ce_loss = F.cross_entropy(noise_logits, noise_label.long())

        clean_16k = torchaudio.functional.resample(clean_audio, orig_freq=self.hparams.sample_rate, new_freq=16000)
        sem_loss = self._compute_semantic_loss(clean_16k, c_clean, enhanced_e)

        focal_loss_total = (
            self.focal_loss_coeff * enc_feat_loss
            + self.focal_loss_coeff * noise_feat_loss
            + self.noise_ce_coeff * ce_loss
            + self.sem_loss_coeff * sem_loss
        )

        total_loss = base_loss + focal_loss_total

        opt_gen.zero_grad()
        self.manual_backward(total_loss)
        opt_gen.step()
        sch_gen.step()

        self.log("generator/total_loss", total_loss, prog_bar=True)
        self.log("generator/mel_loss", mel_loss, prog_bar=True)
        self.log("generator/commit_loss", commit_loss)
        self.log("generator/enc_feat_loss", enc_feat_loss, prog_bar=True)
        self.log("generator/noise_feat_loss", noise_feat_loss, prog_bar=True)
        self.log("generator/ce_loss", ce_loss, prog_bar=True)
        self.log("generator/sem_loss", sem_loss, prog_bar=True)
        self.log("mel_loss_coeff", self.mel_loss_coeff)

        if self.global_step % 1000 == 0 and self.global_rank == 0:
            self.logger.experiment.add_audio("train/clean", clean_audio[0].data.cpu(), self.global_step, self.hparams.sample_rate)
            self.logger.experiment.add_audio("train/noisy", noisy_audio[0].data.cpu(), self.global_step, self.hparams.sample_rate)
            self.logger.experiment.add_audio("train/pred", audio_hat[0].data.cpu(), self.global_step, self.hparams.sample_rate)

    def _compute_semantic_loss(self, clean_16k, c_clean, enhanced_c):
        with torch.no_grad():
            hu_out = self.hubert(input_values=clean_16k, output_hidden_states=True)
            sem_feat = torch.stack(hu_out.hidden_states, dim=0).mean(dim=0)
        sem_feat = sem_feat.permute(0, 2, 1)
        sem_feat = self.semantic_adapter(sem_feat)

        min_len = min(sem_feat.size(-1), c_clean.size(-1), enhanced_c.size(-1))
        sem_feat = sem_feat[..., :min_len]
        c_clean = c_clean[..., :min_len]
        enhanced_c = enhanced_c[..., :min_len]

        residual_clean = c_clean - sem_feat
        residual_enhanced = enhanced_c - sem_feat
        return F.l1_loss(residual_enhanced, residual_clean.detach())

    def validation_step(self, batch, batch_idx):
        clean_audio, noisy_audio, scaled_noise, noise_label = batch
        bandwidth_id = torch.tensor([0], device=self.device)
        audio_hat, commit_loss, _ = self(noisy_audio, bandwidth_id)

        mel_loss = self.melspec_loss(audio_hat.unsqueeze(1), clean_audio.unsqueeze(1))

        if self.hparams.evaluate_utmos and not hasattr(self, 'utmos_model'):
            from metrics.UTMOS import UTMOSScore
            self.utmos_model = UTMOSScore(device=self.device)

        val_loss = mel_loss + 1000 * commit_loss
        self.validation_outputs.append({
            "val_loss": val_loss,
            "mel_loss": mel_loss,
            "commit_loss": commit_loss,
            "audio_input": clean_audio[0],
            "audio_pred": audio_hat[0],
        })

    def on_validation_epoch_end(self):
        if not self.validation_outputs:
            return
        
        outputs = self.validation_outputs
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_mel_loss = torch.stack([x["mel_loss"] for x in outputs]).mean()
        avg_commit_loss = torch.stack([x["commit_loss"] for x in outputs]).mean()

        self.log("val_loss", avg_loss, sync_dist=True)
        self.log("val/mel_loss", avg_mel_loss, sync_dist=True)
        self.log("val/commit_loss", avg_commit_loss, sync_dist=True)

        if self.global_rank == 0:
            audio_in = outputs[0]["audio_input"]
            audio_pred = outputs[0]["audio_pred"]
            self.logger.experiment.add_audio("val/clean", audio_in.data.cpu().numpy(), self.global_step, self.hparams.sample_rate)
            self.logger.experiment.add_audio("val/pred", audio_pred.data.cpu().numpy(), self.global_step, self.hparams.sample_rate)

        self.validation_outputs.clear()

    @property
    def global_step(self):
        return self.trainer.fit_loop.epoch_loop.total_batch_idx
