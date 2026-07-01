"""
source: https://github.com/RoyChao19477/SEMamba
"""
from .layers import *

from functools import partial

from mamba_ssm.modules.mamba_simple import Mamba, Block
from mamba_ssm.models.mixer_seq_simple import _init_weights
from mamba_ssm.ops.triton.layernorm import RMSNorm

def create_mamba_block(d_model,
                       d_state,
                       d_conv,
                       expand,
                       norm_epsilon=0.00001,
                       layer_idx=0):
                       
    mixer_cls = partial(Mamba, layer_idx=layer_idx, d_state=d_state, d_conv=d_conv, expand=expand)
    norm_cls = partial(RMSNorm, eps=norm_epsilon)
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=False,
        residual_in_fp32=False
    )
    block.layer_idx = layer_idx
    return block


class MambaBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 d_state, ##16
                 d_conv, ## 4
                 expand, ## 4
                 ):
        super().__init__()
        n_layer=1
        self.forward_blocks = nn.ModuleList(create_mamba_block(in_channels, d_state, d_conv, expand) for i in range(n_layer))
        self.backward_blocks = nn.ModuleList(create_mamba_block(in_channels, d_state, d_conv, expand) for i in range(n_layer))
        self.apply(partial(_init_weights, n_layer=n_layer))

        self.covn1d_proj = nn.Conv1d(in_channels*2, in_channels, 1, 1)

    def forward(self, x):
        ## x: (B, C, T))
        x = rearrange(x, 'b c t -> b t c')
        x_forward, x_backward = x.clone(), torch.flip(x, [1])
        resi_forward, resi_backward = None, None

        ## Forward
        for layer in self.forward_blocks:
            x_forward, resi_forward = layer(x_forward, resi_forward)
        y_forward = (x_forward + resi_forward) if resi_forward is not None else x_forward

        ## Backward
        for layer in self.backward_blocks:
            x_backward, resi_backward = layer(x_backward, resi_backward)
        y_backward = torch.flip((x_backward + resi_backward), [1]) if resi_backward is not None else torch.flip(x_backward, [1])

        out = torch.cat([y_forward, y_backward], -1)
        out = rearrange(out, 'b t c -> b c t')
        out = self.covn1d_proj(out)
        return out
    

class LearnableSigmoid1D(nn.Module):
    """
    Learnable Sigmoid Activation Function for 1D inputs.
    
    This module applies a learnable slope parameter to the sigmoid activation function.
    """
    def __init__(self, in_features, beta=2):
        """
        Initialize the LearnableSigmoid1D module.
        
        Args:
        - in_features (int): Number of input features.
        - beta (float, optional): Scaling factor for the sigmoid function. Defaults to 1.
        """
        super(LearnableSigmoid1D, self).__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requires_grad = True

    def forward(self, x):
        """
        Forward pass for the LearnableSigmoid1D module.
        
        Args:
        - x (torch.Tensor): Input tensor.
        
        Returns:
        - torch.Tensor: Output tensor after applying the learnable sigmoid activation.
        """
        ## x: (b, c, t)
        
        x = rearrange(x, 'b c t -> b t c')
        out = self.beta * torch.sigmoid(self.slope * x)
        out = rearrange(out, 'b t c -> b c t')
        return out
        # return self.beta * torch.sigmoid(self.slope * x)
        

class DenoisingMambaBlock(nn.Module):
    def __init__(self,
                 n_layer,
                 in_channels,
                 proj_channels,
                 d_state,
                 d_conv,
                 expand,
                 activation='lsigmoid'
                 ):
        super(DenoisingMambaBlock, self).__init__()

        # self.blocks_mamba = nn.ModuleList(create_mamba_block(in_channels, d_state, d_conv, expand) for i in range(n_layer))
        self.proj_in = WNConv1d(in_channels, proj_channels, kernel_size=1, bias=False)
        self.proj_out = WNConv1d(proj_channels, in_channels, kernel_size=1, bias=False)
        
        self.blocks_mamba = nn.ModuleList([MambaBlock(proj_channels, d_state, d_conv, expand) for i in range(n_layer)])
        # self.blocks_mamba =MambaBlock(in_channels, d_state, d_conv, expand)
        if activation == 'lsigmoid':
            self.lsigmoid = LearnableSigmoid1D(in_channels)
        elif activation == 'gelu':
            self.lsigmoid = nn.GELU()
        elif activation == 'none':
            self.lsigmoid = nn.Identity()
        else:
            raise ValueError(f"Invalid activation function: {activation}")

    def forward(self, x):
        x = self.proj_in(x)  # (B, C, T)
        for layer in self.blocks_mamba:
            x = layer(x)
        x = self.proj_out(x)  # (B, C, T)
        x = self.lsigmoid(x)
        return x

def get_padding_1d(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)
    

if __name__=='__main__':
    device = torch.device('cuda:4')
    model = DenoisingMambaBlock(1, 64, 16, 4, 4)
    model.to(device)
    x = torch.randn(2, 64, 100).to(device)
    out = model(x)
    print(out.size())

    """
    Error on CPU:
    https://github.com/Dao-AILab/flash-attention/issues/523
    """