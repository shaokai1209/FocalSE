"""
source: https://github.com/SonyResearch/VRVQ/blob/main/models/importance_subnet.py
"""
from torch import nn
import torch
from .layers import Snake1d, WNConv1d


class ImportanceSubnet(nn.Module):
    def __init__(
        self,
        d_input, # 1024*8
        d_feat, # 1024
        intermediate_channels: list = [512, 128, 32, 8],
        out_channels=1,
        detach_input: bool = False,
    ):
        super().__init__()
        self._init_weights_zero()
        
        self.in_block = nn.Sequential(
            Snake1d(d_input),
            WNConv1d(d_input, d_feat, kernel_size=3, padding=1),
        )
        
        in_channels = [d_feat] + intermediate_channels
        out_channels = intermediate_channels + [out_channels]
        
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    Snake1d(in_channels[i]),
                    WNConv1d(in_channels[i], out_channels[i], kernel_size=3, padding=1),
                )
                for i in range(len(in_channels))
            ]
        )
        self.act_fn = nn.Sigmoid()
        self.detach_input = detach_input
    
    def forward(self, x_in):
        if self.detach_input:
            x_in = x_in.detach()
        x = self.in_block(x_in)
        for block in self.blocks:
            x = block(x)
        out = self.act_fn(x)
        return out # (B, 1, T)
        
    def _init_weights_zero(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.zeros_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)