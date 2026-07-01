import numpy as np
import torch
from decoder.feature_extractors import EncodecFeatures
from model.focal_se import FocalSpeechEnhancementModule

class EncodecFeaturesWithFocalSE(EncodecFeatures):
    def __init__(self, *args, focal_input_dim=512, focal_hidden_dim=256,
                 focal_num_layers=4, focal_heads=8, focal_dropout=0.1, **kwargs):
        super().__init__(*args, **kwargs)

        self.focal_se = FocalSpeechEnhancementModule(
            input_dim=focal_input_dim,
            hidden_dim=focal_hidden_dim,
            num_transformer_layers=focal_num_layers,
            num_heads=focal_heads,
            dropout=focal_dropout,
            mask_beta=2.0,
            focal_window=3,
            window_size=7,
            causal=False
        )

    def forward(self, audio_data, bandwidth_id=None):
        if audio_data.dim() == 2:
            audio_data = audio_data.unsqueeze(1)   
        e = self.encodec.encoder(audio_data)
        enhanced_e, pred_noise_feat = self.focal_se(e)

        if bandwidth_id is not None:
            target_bw = self.bandwidths[bandwidth_id.item()]
        else:
            target_bw = self.bandwidths[0]
        q_res = self.encodec.quantizer(enhanced_e, self.frame_rate, bandwidth=target_bw)
        quantized = q_res.quantized
        codes = q_res.codes
        commit_loss = q_res.penalty
        return quantized, codes, commit_loss, (e, enhanced_e, pred_noise_feat)
