# decoder/experiment.py
import math, numpy as np, pytorch_lightning as pl, torch, torchaudio, transformers, yaml
from decoder.discriminator_dac import DACDiscriminator
from decoder.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from decoder.feature_extractors import FeatureExtractor
from decoder.heads import FourierHead
from decoder.helpers import plot_spectrogram_to_numpy
from decoder.loss import DiscriminatorLoss, GeneratorLoss, FeatureMatchingLoss, MelSpecReconstructionLoss, DACGANLoss
from decoder.models import Backbone
from decoder.modules import safe_log

class VocosExp(pl.LightningModule):
    def __init__(self, feature_extractor, backbone, head, resume_config, resume_model,
                 sample_rate=24000, initial_learning_rate=2e-4, num_warmup_steps=0,
                 mel_loss_coeff=45, mrd_loss_coeff=1.0, pretrain_mel_steps=0,
                 decay_mel_coeff=False, evaluate_utmos=False, evaluate_pesq=False,
                 evaluate_periodicty=False, resume=False):
        super().__init__()
        self.save_hyperparameters(ignore=["feature_extractor", "backbone", "head"])
        self.automatic_optimization = False         # 启用手动优化

        self.feature_extractor = feature_extractor
        self.backbone = backbone
        self.head = head
        self.resume_config = resume_config
        self.resume_model = resume_model
        self.resume = resume

        self.multiperioddisc = MultiPeriodDiscriminator()
        self.multiresddisc = MultiResolutionDiscriminator()
        self.dac = DACDiscriminator(sample_rate=sample_rate)
        self.dacdiscriminator = DACGANLoss(self.dac)

        self.disc_loss = DiscriminatorLoss()
        self.gen_loss = GeneratorLoss()
        self.feat_matching_loss = FeatureMatchingLoss()
        self.melspec_loss = MelSpecReconstructionLoss(sample_rate=sample_rate)

        self.train_discriminator = False
        self.base_mel_coeff = self.mel_loss_coeff = mel_loss_coeff

        self.validation_outputs = []    # 用于 on_validation_epoch_end

    def configure_optimizers(self):
        disc_params = [{"params": self.multiperioddisc.parameters()},
                       {"params": self.multiresddisc.parameters()},
                       {"params": self.dac.parameters()}]
        gen_params = [{"params": self.feature_extractor.parameters()},
                      {"params": self.backbone.parameters()},
                      {"params": self.head.parameters()}]
        opt_disc = torch.optim.AdamW(disc_params, lr=self.hparams.initial_learning_rate)
        opt_gen = torch.optim.AdamW(gen_params, lr=self.hparams.initial_learning_rate)
        max_steps = self.trainer.max_steps // 2
        scheduler_disc = transformers.get_cosine_schedule_with_warmup(opt_disc, self.hparams.num_warmup_steps, max_steps)
        scheduler_gen = transformers.get_cosine_schedule_with_warmup(opt_gen, self.hparams.num_warmup_steps, max_steps)
        return [opt_disc, opt_gen], [scheduler_disc, scheduler_gen]

    def forward(self, audio_input, **kwargs):
        features, _, commit_loss = self.feature_extractor(audio_input, **kwargs)
        x = self.backbone(features, **kwargs)
        audio_output = self.head(x)
        return audio_output, commit_loss

    def training_step(self, batch, batch_idx):
        opt_disc, opt_gen = self.optimizers()
        sch_disc, sch_gen = self.lr_schedulers()
        audio_input = batch                     # 原始干净语音

        # 判别器训练
        if self.train_discriminator:
            with torch.no_grad():
                audio_hat, _ = self(audio_input)
            loss_dac = self.dacdiscriminator.discriminator_loss(audio_hat.unsqueeze(1), audio_input.unsqueeze(1))
            real_mp, gen_mp, _, _ = self.multiperioddisc(y=audio_input, y_hat=audio_hat)
            real_mrd, gen_mrd, _, _ = self.multiresddisc(y=audio_input, y_hat=audio_hat)
            loss_mp, loss_mp_real, _ = self.disc_loss(real_mp, gen_mp)
            loss_mrd, loss_mrd_real, _ = self.disc_loss(real_mrd, gen_mrd)
            loss_mp /= len(loss_mp_real)
            loss_mrd /= len(loss_mrd_real)
            loss_d = loss_mp + self.hparams.mrd_loss_coeff * loss_mrd + loss_dac
            opt_disc.zero_grad()
            self.manual_backward(loss_d)
            opt_disc.step()
            sch_disc.step()
            self.log("discriminator/total", loss_d, prog_bar=True)
            return

        # 生成器训练
        audio_hat, commit_loss = self(audio_input)
        gan_loss = 0.0
        if self.train_discriminator:
            loss_dac_1, loss_dac_2 = self.dacdiscriminator.generator_loss(audio_hat.unsqueeze(1), audio_input.unsqueeze(1))
            _, gen_mp, fmap_rs_mp, fmap_gs_mp = self.multiperioddisc(y=audio_input, y_hat=audio_hat)
            _, gen_mrd, fmap_rs_mrd, fmap_gs_mrd = self.multiresddisc(y=audio_input, y_hat=audio_hat)
            loss_gen_mp, list_gen_mp = self.gen_loss(gen_mp)
            loss_gen_mrd, list_gen_mrd = self.gen_loss(gen_mrd)
            loss_gen_mp = loss_gen_mp / len(list_gen_mp) if len(list_gen_mp)>0 else 0
            loss_gen_mrd = loss_gen_mrd / len(list_gen_mrd) if len(list_gen_mrd)>0 else 0
            loss_fm_mp = self.feat_matching_loss(fmap_r=fmap_rs_mp, fmap_g=fmap_gs_mp) / len(fmap_rs_mp)
            loss_fm_mrd = self.feat_matching_loss(fmap_r=fmap_rs_mrd, fmap_g=fmap_gs_mrd) / len(fmap_rs_mrd)
            gan_loss = loss_gen_mp + self.hparams.mrd_loss_coeff * loss_gen_mrd + loss_fm_mp + self.hparams.mrd_loss_coeff * loss_fm_mrd + loss_dac_1 + loss_dac_2

        mel_loss = self.melspec_loss(audio_hat, audio_input)
        loss_g = mel_loss + 1000 * commit_loss + gan_loss
        opt_gen.zero_grad()
        self.manual_backward(loss_g)
        opt_gen.step()
        sch_gen.step()

        self.log("generator/total_loss", loss_g, prog_bar=True)
        self.log("generator/mel_loss", mel_loss)
        self.log("commit_loss", commit_loss)
        if self.global_step % 1000 == 0 and self.global_rank == 0:
            self.logger.experiment.add_audio("train/audio_in", audio_input[0], self.global_step, self.hparams.sample_rate)
            self.logger.experiment.add_audio("train/audio_pred", audio_hat[0], self.global_step, self.hparams.sample_rate)
            mel = safe_log(self.melspec_loss.mel_spec(audio_input[0]))
            mel_hat = safe_log(self.melspec_loss.mel_spec(audio_hat[0]))
            self.logger.experiment.add_image("train/mel_target", plot_spectrogram_to_numpy(mel.cpu().numpy()), self.global_step, dataformats="HWC")
            self.logger.experiment.add_image("train/mel_pred", plot_spectrogram_to_numpy(mel_hat.cpu().numpy()), self.global_step, dataformats="HWC")

    def on_validation_epoch_start(self):
        self.validation_outputs.clear()
        if self.hparams.evaluate_utmos:
            from metrics.UTMOS import UTMOSScore
            if not hasattr(self, "utmos_model"):
                self.utmos_model = UTMOSScore(device=self.device)

    def validation_step(self, batch, batch_idx):
        audio_input = batch
        audio_hat, commit_loss = self(audio_input)
        # 评估指标（与原始保持一致）
        audio_16k = torchaudio.functional.resample(audio_input, orig_freq=self.hparams.sample_rate, new_freq=16000)
        audio_hat_16k = torchaudio.functional.resample(audio_hat, orig_freq=self.hparams.sample_rate, new_freq=16000)
        periodicity_loss = pitch_loss = f1_score = 0.0
        if self.hparams.evaluate_periodicty:
            from metrics.periodicity import calculate_periodicity_metrics
            periodicity_loss, pitch_loss, f1_score = calculate_periodicity_metrics(audio_16k, audio_hat_16k)
        utmos_score = self.utmos_model.score(audio_hat_16k.unsqueeze(1)).mean() if self.hparams.evaluate_utmos else torch.zeros(1, device=self.device)
        pesq_score = torch.zeros(1, device=self.device)
        if self.hparams.evaluate_pesq:
            from pesq import pesq
            pesq_val = 0
            for ref, deg in zip(audio_16k.cpu().numpy(), audio_hat_16k.cpu().numpy()):
                pesq_val += pesq(16000, ref, deg, "wb", on_error=1)
            pesq_score = torch.tensor(pesq_val / len(audio_16k))
        mel_loss = self.melspec_loss(audio_hat.unsqueeze(1), audio_input.unsqueeze(1))
        total_loss = mel_loss + (5 - utmos_score) + (5 - pesq_score) + 1000 * commit_loss
        self.validation_outputs.append({
            "val_loss": total_loss, "mel_loss": mel_loss, "utmos_score": utmos_score,
            "pesq_score": pesq_score, "periodicity_loss": periodicity_loss,
            "pitch_loss": pitch_loss, "f1_score": f1_score,
            "audio_input": audio_input[0], "audio_pred": audio_hat[0]
        })

    def on_validation_epoch_end(self):
        if not self.validation_outputs:
            return
        outputs = self.validation_outputs
        if self.global_rank == 0:
            audio_in = outputs[0]["audio_input"]
            audio_pred = outputs[0]["audio_pred"]
            self.logger.experiment.add_audio("val_in", audio_in.data.cpu().numpy(), self.global_step, self.hparams.sample_rate)
            self.logger.experiment.add_audio("val_pred", audio_pred.data.cpu().numpy(), self.global_step, self.hparams.sample_rate)
            mel_target = safe_log(self.melspec_loss.mel_spec(audio_in))
            mel_hat = safe_log(self.melspec_loss.mel_spec(audio_pred))
            self.logger.experiment.add_image("val_mel_target", plot_spectrogram_to_numpy(mel_target.data.cpu().numpy()), self.global_step, dataformats="HWC")
            self.logger.experiment.add_image("val_mel_hat", plot_spectrogram_to_numpy(mel_hat.data.cpu().numpy()), self.global_step, dataformats="HWC")
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        mel_loss = torch.stack([x["mel_loss"] for x in outputs]).mean()
        utmos_score = torch.stack([x["utmos_score"] for x in outputs]).mean()
        pesq_score = torch.stack([x["pesq_score"] for x in outputs]).mean()
        periodicity_loss = np.array([x["periodicity_loss"] for x in outputs]).mean()
        pitch_loss = np.array([x["pitch_loss"] for x in outputs]).mean()
        f1_score = np.array([x["f1_score"] for x in outputs]).mean()
        self.log("val_loss", avg_loss, sync_dist=True)
        self.log("val/mel_loss", mel_loss, sync_dist=True)
        self.log("val/utmos_score", utmos_score, sync_dist=True)
        self.log("val/pesq_score", pesq_score, sync_dist=True)
        self.log("val/periodicity_loss", periodicity_loss, sync_dist=True)
        self.log("val/pitch_loss", pitch_loss, sync_dist=True)
        self.log("val/f1_score", f1_score, sync_dist=True)

    @property
    def global_step(self):
        return self.trainer.fit_loop.epoch_loop.total_batch_idx

    def on_train_batch_start(self, *args):
        self.train_discriminator = self.global_step >= self.hparams.pretrain_mel_steps

    def on_train_batch_end(self, *args):
        if not self.hparams.decay_mel_coeff:
            return
        max_steps = self.trainer.max_steps // 2
        if self.global_step < self.hparams.num_warmup_steps:
            self.mel_loss_coeff = self.base_mel_coeff
        else:
            progress = (self.global_step - self.hparams.num_warmup_steps) / (max_steps - self.hparams.num_warmup_steps)
            self.mel_loss_coeff = self.base_mel_coeff * max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


class WavTokenizer(VocosExp):
    def __init__(self, feature_extractor, backbone, head, resume_config, resume_model,
                 sample_rate=24000, initial_learning_rate=2e-4, num_warmup_steps=0,
                 mel_loss_coeff=45, mrd_loss_coeff=1.0, pretrain_mel_steps=0,
                 decay_mel_coeff=False, evaluate_utmos=False, evaluate_pesq=False,
                 evaluate_periodicty=False, resume=False):
        super().__init__(feature_extractor, backbone, head, resume_config, resume_model,
                         sample_rate, initial_learning_rate, num_warmup_steps,
                         mel_loss_coeff, mrd_loss_coeff, pretrain_mel_steps,
                         decay_mel_coeff, evaluate_utmos, evaluate_pesq,
                         evaluate_periodicty, resume)
        self.multiperioddisc = MultiPeriodDiscriminator(num_embeddings=len(self.feature_extractor.bandwidths))
        self.multiresddisc = MultiResolutionDiscriminator(num_embeddings=len(self.feature_extractor.bandwidths))
        self.dac = DACDiscriminator()
        if self.resume:
            self._load_pretrained()

    def _load_pretrained(self):
        print('加载预训练模型:', self.resume_model)
        state_dict_raw = torch.load(self.resume_model, map_location=self.device)['state_dict']
        # 拆分权重
        enc, dec, quant, bb, hd, mp, mr, dac = dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict()
        for k, v in state_dict_raw.items():
            if k.startswith('feature_extractor.encodec.quantizer'):
                num = int(k[46]) if k[46].isdigit() else 0
                if num <= 7: quant[k[36:]] = v
            elif k.startswith('feature_extractor.encodec.encoder'): enc[k[34:]] = v
            elif k.startswith('feature_extractor.encodec.decoder'): dec[k[34:]] = v
            elif k.startswith('backbone.'): bb[k[9:]] = v
            elif k.startswith('head.'): hd[k[5:]] = v
            elif k.startswith('multiperioddisc.'): mp[k[16:]] = v
            elif k.startswith('multiresddisc.'): mr[k[14:]] = v
            elif k.startswith('dac.'): dac[k[4:]] = v
        self.feature_extractor.encodec.encoder.load_state_dict(enc, strict=True)
        self.feature_extractor.encodec.decoder.load_state_dict(dec, strict=True)
        self.feature_extractor.encodec.quantizer.load_state_dict(quant, strict=True)
        self.backbone.load_state_dict(bb, strict=True)
        self.head.load_state_dict(hd, strict=True)
        self.multiperioddisc.load_state_dict(mp, strict=True)
        self.multiresddisc.load_state_dict(mr, strict=True)
        self.dac.load_state_dict(dac, strict=True)

    def training_step(self, batch, batch_idx):
        bandwidth_id = torch.randint(low=0, high=len(self.feature_extractor.bandwidths), size=(1,), device=self.device)
        return super().training_step(batch, batch_idx, bandwidth_id=bandwidth_id)

    def validation_step(self, batch, batch_idx):
        bandwidth_id = torch.tensor([0], device=self.device)

        return super().validation_step(batch, batch_idx, bandwidth_id=bandwidth_id)

    def on_validation_epoch_end(self):
        if self.global_rank == 0:
            *_, audio_in, _ = self.validation_outputs[0].values() if self.validation_outputs else (None, None)
            if audio_in is not None:
                self.feature_extractor.encodec.set_target_bandwidth(self.feature_extractor.bandwidths[0])
                encodec_audio = self.feature_extractor.encodec(audio_in[None, None, :])
                self.logger.experiment.add_audio("encodec", encodec_audio[0, 0].data.cpu().numpy(), self.global_step, self.hparams.sample_rate)
        super().on_validation_epoch_end()