import torch
import torch.nn as nn
import torch.nn.functional as F

class EncFeatureLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, fmap_gt, fmap_noisy):
        loss = 0.0
        for k in fmap_gt:
            if k in fmap_noisy:
                loss += self.loss(fmap_gt[k], fmap_noisy[k])
        return loss

class NoiseFeatureLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, fmap_noise_pred, fmap_noise_gt):
        loss = 0.0
        for k in fmap_noise_gt:
            if k in fmap_noise_pred:
                loss += self.loss(fmap_noise_pred[k], fmap_noise_gt[k])
        return loss

class SemanticResidualLoss(nn.Module):
    def __init__(self, hubert_model, semantic_adapter):
        super().__init__()
        self.hubert = hubert_model
        self.adapter = semantic_adapter
        for p in self.hubert.parameters():
            p.requires_grad = False

    def get_semantic_feat(self, audio_16k):
        with torch.no_grad():
            hu_out = self.hubert(input_values=audio_16k, output_hidden_states=True)
            feat = torch.stack(hu_out.hidden_states, dim=0).mean(dim=0)  # [B, T, 768]
        feat = feat.permute(0, 2, 1)  # [B, 768, T]
        feat = self.adapter(feat)      # [B, 512, T]
        return feat

    def forward(self, clean_audio, c_clean, enhanced_c):
        sem_clean = self.get_semantic_feat(clean_audio)
        min_len = min(sem_clean.size(-1), c_clean.size(-1), enhanced_c.size(-1))
        sem_clean = sem_clean[..., :min_len]
        c_clean_trim = c_clean[..., :min_len]
        enhanced_c_trim = enhanced_c[..., :min_len]

        residual_clean = c_clean_trim - sem_clean
        residual_enhanced = enhanced_c_trim - sem_clean
        return F.l1_loss(residual_enhanced, residual_clean.detach())
