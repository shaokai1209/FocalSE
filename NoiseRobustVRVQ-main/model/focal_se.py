import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange 
from .layers_mamba import DenoisingMambaBlock
LRELU_SLOPE = 0.1

class FocalModulation(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: int = 7,
        focal_window: int = 3,
        dropout: float = 0.1,
        causal: bool = False,
        focal_level: int = 2,
        focal_factor: int = 2,
        normalize_modulator: bool = False
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.causal = causal
        self.normalize_modulator = normalize_modulator
        self.in_proj = nn.Linear(dim, 2 * dim + num_heads * (focal_level + 1))
        self.focal_layers = nn.ModuleList()
        self.causal_pads = []
        for k in range(focal_level):
            kernel_size = focal_factor * k + focal_window
            self.causal_pads.append(kernel_size - 1)
            conv_padding = 0 if causal else "same"
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv1d(dim, dim, kernel_size, padding=conv_padding, groups=dim, bias=False),
                    nn.GELU()
                )
            )

        if causal:
            self.global_conv = nn.Conv1d(
                dim, dim, kernel_size=window_size, padding=0, groups=dim, bias=False
            )
            self.global_pad = window_size - 1

        self.context_proj = nn.Conv1d(dim, dim, kernel_size=1)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        x_proj = self.in_proj(x)
        q, ctx, gates = torch.split(
            x_proj,
            [self.dim, self.dim, self.num_heads * (self.focal_level + 1)],
            dim=-1
        )
        ctx = ctx.permute(0, 2, 1).contiguous()
        gates = gates.reshape(B, T, self.num_heads, self.focal_level + 1).permute(0, 2, 3, 1).contiguous()
        q = q.reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 3, 1).contiguous()

        context_all = 0.0
        for level in range(self.focal_level):
            if self.causal:
                ctx_pad = F.pad(ctx, [self.causal_pads[level], 0], mode="replicate")
                ctx_layer = self.focal_layers[level](ctx_pad)
            else:
                ctx_layer = self.focal_layers[level](ctx)
                
            ctx_layer = ctx_layer.reshape(B, self.num_heads, self.head_dim, T)
            context_all += ctx_layer * gates[:, :, level:level+1, :]

        if not self.causal:
            ctx_global = ctx.mean(dim=-1, keepdim=True)  # (B, C, 1)
            ctx_global = self.act(ctx_global)
            ctx_global = ctx_global.reshape(B, self.num_heads, self.head_dim, 1)
            context_all += ctx_global * gates[:, :, -1:, :]
        else:
            ctx_pad = F.pad(ctx, [self.global_pad, 0], mode="replicate")
            ctx_global = self.global_conv(ctx_pad)
            ctx_global = self.act(ctx_global)
            ctx_global = ctx_global.reshape(B, self.num_heads, self.head_dim, T)
            context_all += ctx_global * gates[:, :, -1:, :]

        if self.normalize_modulator:
            context_all = context_all / (self.focal_level + 1)
            
        context_all = context_all.reshape(B, C, T)
        modulator = self.context_proj(context_all)
        modulator = modulator.permute(0, 2, 1).contiguous()
        q = q.permute(0, 3, 1, 2).reshape(B, T, C)
        out = q * modulator
        out = self.out_proj(out)
        out = self.dropout(out)

        return out

class FocalEncoderBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        stride: int = 2,
        kernel_size: int = 3,
        num_heads: int = 8,
        window_size: int = 7,
        focal_window: int = 3,
        dropout: float = 0.1,
        causal: bool = False,
    ):
        super().__init__()
        self.stride = stride
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.downsample_conv = nn.Conv1d(
            in_dim, out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            bias=False
        )
        
        self.norm1 = nn.LayerNorm(out_dim)
        self.focal_mod = FocalModulation(out_dim, num_heads, window_size, focal_window, dropout, causal)
        self.norm2 = nn.LayerNorm(out_dim)
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * 4),
            nn.LeakyReLU(negative_slope=LRELU_SLOPE),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 4, out_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        B, C, T_in = x.shape
        x = rearrange(x, 'b c t -> b t c')
        
        x = rearrange(x, 'b t c -> b c t')
        x = self.downsample_conv(x)
        B, C_out, T_out = x.shape
        x = rearrange(x, 'b c t -> b t c')
        x = F.leaky_relu(x, negative_slope=LRELU_SLOPE)

        residual = x
        x = self.norm1(x)
        focal_out = self.focal_mod(x)
        x = residual + self.dropout(focal_out)

        residual = x
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        x = residual + self.dropout(ffn_out)

        x = rearrange(x, 'b t c -> b c t')
        downsample_info = {
            "T_in": T_in,
            "T_out": T_out,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding
        }
        return x, downsample_info

class FocalDecoderBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int = 3,
        num_heads: int = 8,
        window_size: int = 7,
        focal_window: int = 3,
        dropout: float = 0.1,
        causal: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.upsample_conv = nn.ConvTranspose1d(
            in_dim, out_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            output_padding=0,
            bias=False
        )
        
        self.norm1 = nn.LayerNorm(out_dim)
        self.focal_mod = FocalModulation(out_dim, num_heads, window_size, focal_window, dropout, causal)
        self.norm2 = nn.LayerNorm(out_dim)
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * 4),
            nn.LeakyReLU(negative_slope=LRELU_SLOPE),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 4, out_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, downsample_info: dict) -> torch.Tensor:
        T_in_down = downsample_info["T_in"]
        T_out_down = downsample_info["T_out"]
        kernel_size = downsample_info["kernel_size"]
        stride = downsample_info["stride"]
        padding_down = downsample_info["padding"]
        self.upsample_conv.stride = stride
        self.upsample_conv.padding = (padding_down,)
        output_padding = T_in_down - ( (T_out_down - 1) * stride + kernel_size - 2 * padding_down )
        output_padding = max(0, min(output_padding, stride - 1))
        self.upsample_conv.output_padding = output_padding

        x = rearrange(x, 'b c t -> b t c')
        x = rearrange(x, 'b t c -> b c t')
        x = F.leaky_relu(x, negative_slope=LRELU_SLOPE)
        x = self.upsample_conv(x)
        x = rearrange(x, 'b c t -> b t c')

        residual = x
        x = self.norm1(x)
        focal_out = self.focal_mod(x)
        x = residual + self.dropout(focal_out)

        residual = x
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        x = residual + self.dropout(ffn_out)

        x = rearrange(x, 'b t c -> b c t')
        return x

class LearnableSigmoid1D(nn.Module):
    def __init__(self, in_features, beta=2):
        super(LearnableSigmoid1D, self).__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requires_grad = True

    def forward(self, x):
        x = rearrange(x, 'b c t -> b t c')
        out = self.beta * torch.sigmoid(self.slope * x)
        out = rearrange(out, 'b t c -> b c t')
        return out

class SEBlock(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            dim, num_heads, batch_first=True, dropout=dropout
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.LeakyReLU(negative_slope=LRELU_SLOPE),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attention(x, x, x)
        x = residual + self.dropout(attn_out)
        
        residual = x
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        x = residual + self.dropout(ffn_out)
        
        return x

class FocalSpeechEnhancementModule(nn.Module):
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        num_transformer_layers: int = 4,
        downsample_strides: list = [2, 2],
        num_heads: int = 8,
        dropout: float = 0.1,
        mask_beta: float = 2.5,
        focal_window: int = 3,
        window_size: int = 7,
        causal: bool = False
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.downsample_strides = downsample_strides
        self.num_downsample_layers = len(downsample_strides)
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.downscale_layers = nn.ModuleList()
        in_dims = [hidden_dim] + [hidden_dim * (2 ** i) for i in range(1, self.num_downsample_layers)]
        out_dims = [hidden_dim * (2 ** i) for i in range(1, self.num_downsample_layers + 1)]
        for i in range(self.num_downsample_layers):
            self.downscale_layers.append(
                FocalEncoderBlock(
                    in_dim=in_dims[i],
                    out_dim=out_dims[i],
                    stride=downsample_strides[i],
                    kernel_size=3,
                    num_heads=num_heads,
                    window_size=window_size,
                    focal_window=focal_window,
                    dropout=dropout,
                    causal=causal
                )
            )

        self.transformer_dim = out_dims[-1]
        self.transformer_layers = nn.ModuleList([
            SEBlock(
                dim=self.transformer_dim,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_transformer_layers)
        ])

        self.upscale_layers = nn.ModuleList()
        upscale_in_dims = out_dims[::-1]
        upscale_out_dims = in_dims[::-1]
        for i in range(self.num_downsample_layers):
            self.upscale_layers.append(
                FocalDecoderBlock(
                    in_dim=upscale_in_dims[i],
                    out_dim=upscale_out_dims[i],
                    kernel_size=3,
                    num_heads=num_heads,
                    window_size=window_size,
                    focal_window=focal_window,
                    dropout=dropout,
                    causal=causal
                )
            )

        self.output_proj = nn.Conv1d(hidden_dim, input_dim, kernel_size=1)
        self.mask_activation = LearnableSigmoid1D(in_features=input_dim, beta=mask_beta)
        self.denoising_mamba = DenoisingMambaBlock(
            n_layer=3,
            in_channels=input_dim,
            proj_channels=256,
            d_state=16,
            d_conv=4,
            expand=4,
            activation='lsigmoid'
        )

        self.noise_proj = nn.Conv1d(input_dim, input_dim, kernel_size=1)
        self.noise_activation = LearnableSigmoid1D(in_features=input_dim, beta=mask_beta)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear, nn.ConvTranspose1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, noisy_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, C, T_original = noisy_features.shape
        if C != self.input_dim:
            raise ValueError(f"check channel:{self.input_dim}，{C}")

        x = self.input_proj(noisy_features)
        x = self.dropout(x)
        
        downsample_info_list = []
        for down_layer in self.downscale_layers:
            x, down_info = down_layer(x)
            downsample_info_list.append(down_info)

        x = x.permute(0, 2, 1)
        for layer in self.transformer_layers:
            x = layer(x)
        x = x.permute(0, 2, 1)

        downsample_info_list_rev = downsample_info_list[::-1]
        for i, up_layer in enumerate(self.upscale_layers):
            x = up_layer(x, downsample_info_list_rev[i])

        raw_mask_feature = self.output_proj(x)
        enhancement_mask = self.mask_activation(raw_mask_feature)

        enhanced_features = noisy_features * enhancement_mask

        x_noise_manba = self.denoising_mamba(noisy_features)
        raw_noise_mask = self.noise_proj(x_noise_manba)
        noise_mask = self.noise_activation(raw_noise_mask)
        pred_noise_feat = noisy_features * noise_mask - enhanced_features

        return enhanced_features, pred_noise_feat
