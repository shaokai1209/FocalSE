"""
This code is a modified version of the `dac` module from the DAC GitHub repository.  
Original sources: 
    - https://github.com/descriptinc/descript-audio-codec/blob/main/dac/model/dac.py
    - https://github.com/SonyResearch/VRVQ/blob/main/models/dac_vrvq.py
"""
import math
from typing import List, Union
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from audiotools import AudioSignal
from audiotools.ml import BaseModel

from .layers import Snake1d, WNConv1d
from .layers import init_weights, ResidualUnit,  EncoderBlock, DecoderBlock

from .focal_se import (
    FocalSpeechEnhancementModule,
    LearnableSigmoid1D
)
from .dac_base import CodecMixin
from .quantize import ResidualVectorQuantize, VBRResidualVectorQuantize

class BasicBlock1D(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class LightResNet1D(nn.Module):
    def __init__(self, in_channels=1024, base_channels=128, dropout=0.1):
        super().__init__()
        self.in_channels = base_channels
        self.dropout = dropout
        self.transition = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(BasicBlock1D, base_channels, 1, stride=1, dropout=dropout)
        self.layer2 = self._make_layer(BasicBlock1D, base_channels*2, 1, stride=2, dropout=dropout)
        self.layer3 = self._make_layer(BasicBlock1D, base_channels*4, 1, stride=2, dropout=dropout)
        self.layer4 = self._make_layer(BasicBlock1D, base_channels*8, 1, stride=2, dropout=dropout)
        
        self.channel_align = nn.Sequential(
            nn.Conv1d(base_channels*8, 2048, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        self.avgpool = nn.AdaptiveAvgPool1d(8)

    def _make_layer(self, block, out_channels, blocks, stride, dropout):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, dropout))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, dropout=dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.transition(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.channel_align(x) 
        x = self.avgpool(x)
        x = torch.mean(x, dim=-1)
        
        return x


class NoiseClassifier(nn.Module):
    def __init__(self, 
                 in_channels=1024,
                 num_classes=50,
                 dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = LightResNet1D(in_channels=in_channels, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(2048, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_feat = self.backbone(x)
        logits = self.classifier(res_feat)
        return logits


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int=64,
        strides: List[int]=[2, 4, 8, 8],
        latent_dim: int=512,
    ):
        super().__init__()
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]
        
        for stride in strides:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride)]
        self.block += [
            Snake1d(d_model),
            WNConv1d(d_model, latent_dim, kernel_size=3, padding=1),
        ]
        
        self.block = nn.Sequential(*self.block)
    
    def forward(self, x, return_feat=False):
        num_blocks = len(self.block)
        for i, layer in enumerate(self.block):
            x = layer(x)
            if i == num_blocks - 3 and return_feat:
                feat = x
        out = x
        if return_feat:
            return out, feat
        return out

class Decoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        d_out: int = 1,
    ):
        super().__init__()
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride)]
        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
class DAC_VRVQ(BaseModel, CodecMixin):
    def __init__(
        self,
        encoder_dim: int = 64, 
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        n_codebooks: int = 9,
        codebook_size: Union[int, list] = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: float = 0.0,
        sample_rate: int = 44100,
        model_type: str="VBR",
        full_codebook_rate: float=0.0,
        use_framewise_dropout: bool=False,
        level_min: float=None,
        level_max: float=None,
        level_dist: str="uniform",
        operator_mode: str = "scaling",
        imp_map_input: str = "feature",
        detach_imp_map_input: bool = False,
        imp2mask_alpha: float = 1.0,
        imp2mask_func: str="logcosh",
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))
        self.latent_dim = latent_dim
        self.hop_length = np.prod(encoder_rates)
        self.encoder = Encoder(encoder_dim, encoder_rates, latent_dim)
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.imp_map_input = imp_map_input
        self.model_type = model_type
        
        if model_type == "CBR":
            self.quantizer = ResidualVectorQuantize(
                input_dim=latent_dim,
                n_codebooks=n_codebooks,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
                quantizer_dropout=quantizer_dropout,
            )
        elif model_type == "VBR":
            self.quantizer = VBRResidualVectorQuantize(
                input_dim=latent_dim,
                n_codebooks=n_codebooks,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
                quantizer_dropout=quantizer_dropout,
                full_codebook_rate=full_codebook_rate,
                use_framewise_masking=use_framewise_dropout,
                level_min=level_min,
                level_max=level_max,
                level_dist=level_dist,
                operator_mode=operator_mode,
                imp_map_input=imp_map_input,
                detach_imp_map_input=detach_imp_map_input,
                imp2mask_alpha=imp2mask_alpha,
                imp2mask_func=imp2mask_func,
            )
        else:
            raise ValueError(f"Invalid RVQ model_type: {model_type}")
        
        self.decoder = Decoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
        )
        self.sample_rate = sample_rate
        self.apply(init_weights)
        self.delay = self.get_delay()
        
    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate
        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))
        return audio_data
    
    def encode(
        self,
        audio_data: torch.Tensor,
        n_quantizers: int = None,
        level: int = 1,
    ):
        z, feat = self.encoder(audio_data, return_feat=True)
        if self.model_type == "CBR":
            quant_inp = {"z": z, "n_quantizers": n_quantizers}
        elif self.model_type == "VBR":
            quant_inp = {"z": z, "n_quantizers": n_quantizers,
                         "feat_enc": feat, "level": level}
        
        out_quant_dict = self.quantizer(**quant_inp)
        return out_quant_dict
    
    def decode(self, z: torch.Tensor):
        return self.decoder(z)
    
    def forward(
        self,
        audio_data: torch.Tensor,
        sample_rate: int = None,
        n_quantizers: int = None,
        level: int = 1,
    ):
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data, sample_rate)
        if self.model_type == "CBR":
            enc_inp = {"audio_data": audio_data, "n_quantizers": n_quantizers}
        elif self.model_type == "VBR":
            enc_inp = {"audio_data": audio_data, "n_quantizers": n_quantizers, "level": level}
        out_enc_dict = self.encode(**enc_inp)
        z_q = out_enc_dict["z_q"]
        x = self.decode(z_q)
        
        out_forward_dict = {
            "audio": x[..., :length],
            "z": z_q,
            "codes": out_enc_dict["codes"],
            "latents": out_enc_dict["latents"],
            "vq/commitment_loss": out_enc_dict["commitment_loss"],
            "vq/codebook_loss": out_enc_dict["codebook_loss"],
            "imp_map": out_enc_dict.get("imp_map", None),
            "mask_imp": out_enc_dict.get("mask_imp", None),
        }
        return out_forward_dict
        

class EncoderWithFeatureDenoiser(nn.Module):
    def __init__(
        self,
        d_model: int=64,
        strides: List[int]=[2, 4, 8, 8],
        latent_dim: int=512,
        denoise_block_idx = [1, 3],
        clean_train: bool = False,
        se_hidden_dim: int = 256,
        se_num_transformer_layers: int = 4,
        se_num_heads: int = 8,
        se_dropout: float = 0.1,
        se_mask_beta: float = 2.0,
        se_focal_window: int = 3,
        se_window_size: int = 7,
        se_causal: bool = False,
        use_modulation: bool = False,
    ):
        super().__init__()
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]
        d_model_list = []
        for idx, stride in enumerate(strides):
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride)]
            d_model_list.append(d_model)
            
        self.block += [
            Snake1d(d_model),
            WNConv1d(d_model, latent_dim, kernel_size=3, padding=1),
        ]
        
        self.block = nn.Sequential(*self.block)
        self.strides = strides
        self.dn_block_idx = denoise_block_idx
        self.block_denoise_dict = nn.ModuleDict(
            {f"block_denoise_{idx}": FocalSpeechEnhancementModule(
                input_dim=d_model_list[idx-1],
                hidden_dim=se_hidden_dim,
                num_transformer_layers=se_num_transformer_layers,
                num_heads=se_num_heads,
                dropout=se_dropout,
                mask_beta=se_mask_beta,
                focal_window=se_focal_window,
                window_size=se_window_size,
                causal=se_causal
            ) for idx in denoise_block_idx}
        )
        self.clean_train = clean_train
        if not clean_train:
            self.freeze_non_denoising_blocks()
        
        self.n_blocks = len(self.block)
        self.n_strides = len(self.strides)
        self.denoise_block_idx = denoise_block_idx
        self.use_modulation = use_modulation
        
    def freeze_non_denoising_blocks(self):
        for name, param in self.named_parameters():
            if "block_denoise" not in name:
                param.requires_grad = False

    def check_grads(self):
        for name, param in self.named_parameters():
            print(name, param.requires_grad)
        
    def forward_gt(self, x_gt):
        with torch.no_grad():
            fmap_gt = {}
            x = self.block[0](x_gt).detach()
            for i in range(1, self.n_blocks):
                x = self.block[i](x).detach()
                if i in range(1, self.n_strides+1):
                    if i in self.denoise_block_idx:
                        fmap_gt[f"gt_{i}"] = x.detach()
                if i == self.n_strides:
                    fmap_gt["imp_map_input"] = x.detach()
                if i > self.n_strides:
                    break
            return x.detach(), fmap_gt
    
    def forward_noisy(self, x_noisy):
        fmap_noisy = {}
        x = self.block[0](x_noisy)
        assert self.n_strides == 4
        for ii in range(1, self.n_strides+1):
            x = self.block[ii](x)
            if ii in self.denoise_block_idx:
               
                enhanced_features, pred_noise_feat = self.block_denoise_dict[f"block_denoise_{ii}"](x)
                x = enhanced_features
                fmap_noisy[f"denoised_{ii}"] = x
                fmap_noisy[f"pred_noise_{ii}"] = pred_noise_feat
        
        fmap_noisy["imp_map_input"] = x
        for i in range(self.n_strides+1, self.n_blocks):
            x = self.block[i](x)
        return x, fmap_noisy

    def forward_noise(self, x_noise):
        with torch.no_grad():
            fmap_noise = {}
            x = self.block[0](x_noise).detach()
            for i in range(1, self.n_blocks):
                x = self.block[i](x).detach()
                if i in range(1, self.n_strides+1):
                    if i in self.denoise_block_idx:
                        fmap_noise[f"noise_{i}"] = x.detach()
                if i == self.n_strides:
                    fmap_noise["imp_map_input"] = x.detach()
                if i > self.n_strides:
                    break
            return x.detach(), fmap_noise

    def forward(self, x_noisy, x_gt):
        assert x_noisy is not None
        outs = {}
        fmaps = {}
        if x_gt is not None:
            x_gt, fmap_gt = self.forward_gt(x_gt)
        else:
            x_gt, fmap_gt = None, None
        
        x_n, fmap_n = self.forward_noisy(x_noisy)
        outs["z"] = x_n
        outs["z_clean"] = x_gt
        fmaps["gt"] = fmap_gt
        fmaps["noisy"] = fmap_n
        return outs, fmaps    

class DAC_VRVQ_FeatureDenoise(BaseModel, CodecMixin):
    def __init__(
        self,
        ## Original DAC Configs
        encoder_dim: int = 64, 
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        n_codebooks: int = 9,
        codebook_size: Union[int, list] = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: float = 0.0,
        sample_rate: int = 44100,
        
        ## Feature Denoiser
        denoise_block_idx: List[int] = [1, 3],
        use_modulation: bool=False,
        se_hidden_dim: int = 256,
        se_num_transformer_layers: int = 4,
        se_num_heads: int = 8,
        se_dropout: float = 0.1,
        se_mask_beta: float = 2.0,
        se_focal_window: int = 3,
        se_window_size: int = 7,
        se_causal: bool = False,
        
        ## VBR Configs
        model_type: str="VBR",
        full_codebook_rate: float=0.0,
        use_framewise_dropout: bool=False,
        level_min: float=None,
        level_max: float=None,
        level_dist: str="uniform",
        operator_mode: str = "scaling",
        imp_map_input: str = "feature",
        detach_imp_map_input: bool = False,
        imp2mask_alpha: float = 1.0,
        imp2mask_func: str="logcosh",
        clean_train: bool = False,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate
        self.denoise_block_idx = denoise_block_idx 
        self.se_hidden_dim = se_hidden_dim 

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))
        self.latent_dim = latent_dim
        self.hop_length = np.prod(encoder_rates)  
        
        self.encoder = EncoderWithFeatureDenoiser(
            d_model=encoder_dim,
            strides=encoder_rates,
            latent_dim=latent_dim,
            denoise_block_idx=denoise_block_idx,
            clean_train=clean_train,
            se_hidden_dim=se_hidden_dim,
            se_num_transformer_layers=se_num_transformer_layers,
            se_num_heads=se_num_heads,
            se_dropout=se_dropout,
            se_mask_beta=se_mask_beta,
            se_focal_window=se_focal_window,
            se_window_size=se_window_size,
            se_causal=se_causal,
            use_modulation= use_modulation,
        )

        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.imp_map_input = imp_map_input
        self.model_type = model_type
        if model_type == "CBR":
            self.quantizer = ResidualVectorQuantize(
                input_dim=latent_dim,
                n_codebooks=n_codebooks,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
                quantizer_dropout=quantizer_dropout,
            )
        elif model_type == "VBR":
            self.quantizer = VBRResidualVectorQuantize(
                input_dim=latent_dim,
                n_codebooks=n_codebooks,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
                quantizer_dropout=quantizer_dropout,
                full_codebook_rate=full_codebook_rate,
                use_framewise_masking=use_framewise_dropout,
                level_min=level_min,
                level_max=level_max,
                level_dist=level_dist,
                operator_mode=operator_mode,
                imp_map_input=imp_map_input,
                detach_imp_map_input=detach_imp_map_input,
                imp2mask_alpha=imp2mask_alpha,
                imp2mask_func=imp2mask_func,
            )
        else:
            raise ValueError(f"Invalid RVQ model_type: {model_type}")
        
        self.decoder = Decoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
        )
        
        self.noise_classifier = NoiseClassifier(
            in_channels=1024, 
            num_classes=50,
            dropout=0.1  
        )
        
        self.sample_rate = sample_rate
        self.apply(init_weights)
        self.delay = self.get_delay()
    
    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate, f"Sample rate mismatch: {sample_rate} vs {self.sample_rate}"
        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))
        return audio_data
    
    def encode(
        self,
        audio_data_noisy: torch.Tensor,
        audio_data_gt: torch.Tensor,
        audio_data_noise: torch.Tensor,
        n_quantizers: int = None,
        level: int = 1,
        infer_clean_without_denoising: bool = False,
    ):
        outs, fmaps = self.encoder(x_noisy=audio_data_noisy, x_gt=audio_data_gt)
        z = outs["z"]
        z_clean = outs["z_clean"]
        feat_enc = fmaps["noisy"]["imp_map_input"]

        if audio_data_noise is not None:
            _, fmap_noise = self.encoder.forward_noise(audio_data_noise)
            fmaps["noise"] = fmap_noise 
        
        if infer_clean_without_denoising:
            assert z_clean is not None
            z = z_clean
            feat_enc = fmaps["gt"]["imp_map_input"]
            
        if self.model_type == "CBR":
            quant_inp = {"z": z, "n_quantizers": n_quantizers}
        elif self.model_type == "VBR":
            quant_inp = {"z": z, "n_quantizers": n_quantizers,
                         "feat_enc": feat_enc, "level": level}
        
        out_quant_dict = self.quantizer(**quant_inp)
        out_quant_dict["enc_fmaps"] = fmaps
        out_quant_dict["z_clean"] = z_clean
        return out_quant_dict
    
    def decode(self, z: torch.Tensor):
        return self.decoder(z)
    
    def forward(
        self,
        audio_data_noisy: torch.Tensor,
        audio_data_clean: torch.Tensor,
        audio_data_noise: torch.Tensor,
        sample_rate: int = None,
        n_quantizers: int = None,
        level: int = 1,
        infer_clean_without_denoising: bool = False,
    ):
        if infer_clean_without_denoising:
            assert audio_data_clean is not None
            
        length = audio_data_noisy.shape[-1]
        audio_data_noisy = self.preprocess(audio_data_noisy, sample_rate)
        if audio_data_clean is not None:
            audio_data_clean = self.preprocess(audio_data_clean, sample_rate)
        if audio_data_noise is not None:
            audio_data_noise = self.preprocess(audio_data_noise, sample_rate)
        
        enc_inp = {
            "audio_data_noisy": audio_data_noisy,
            "audio_data_gt": audio_data_clean,
            "audio_data_noise": audio_data_noise,
            "n_quantizers": n_quantizers,
            "infer_clean_without_denoising": infer_clean_without_denoising,
        }
        if self.model_type == "VBR":
            enc_inp["level"] = level
            
        out_enc_dict = self.encode(**enc_inp)
        z_clean = out_enc_dict["z_clean"]
        z_q = out_enc_dict["z_q"]
        x = self.decode(z_q)
        
        fmaps = out_enc_dict["enc_fmaps"]
        pred_noise_feat_list = []
        for idx in self.denoise_block_idx:
            pred_noise_feat = fmaps["noisy"][f"pred_noise_{idx}"]
            pred_noise_feat_list.append(pred_noise_feat) 
        pred_noise_feat = torch.cat(pred_noise_feat_list, dim=1)  # [B, 1024, T]
        noise_logits = self.noise_classifier(pred_noise_feat)     # [B, 50]
        
        out_forward_dict = {
            "audio": x[..., :length],
            "z": z_q,
            "z_clean": z_clean,
            "codes": out_enc_dict["codes"],
            "latents": out_enc_dict["latents"],
            "vq/commitment_loss": out_enc_dict["commitment_loss"],
            "vq/codebook_loss": out_enc_dict["codebook_loss"],
            "imp_map": out_enc_dict.get("imp_map", None),
            "mask_imp": out_enc_dict.get("mask_imp", None),
            "enc_fmaps": out_enc_dict["enc_fmaps"],
            "noise_logits": noise_logits,
        }
        return out_forward_dict
