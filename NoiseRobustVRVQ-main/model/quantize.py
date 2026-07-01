"""
This code is heavily adapted and modified from the original DAC GitHub repository.  
Original source: 
    - https://github.com/descriptinc/descript-audio-codec/blob/main/dac/nn/quantize.py
    - https://github.com/SonyResearch/VRVQ/blob/main/models/quantize.py
"""
from typing import Union
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import weight_norm

from .layers import WNConv1d
from .importance_subnet import ImportanceSubnet
from .utils import generate_mask_ste, apply_straight, generate_mask_hard

EPS = 1e-10


class VectorQuantize(nn.Module):
    """
    Implementation of VQ similar to Karpathy's repo:
    https://github.com/karpathy/deep-vector-quantization
    Additionally uses following tricks from Improved VQGAN
    (https://arxiv.org/pdf/2110.04627.pdf):
        1. Factorized codes: Perform nearest neighbor lookup in low-dimensional space
            for improved codebook usage
        2. l2-normalized codes: Converts euclidean distance to cosine similarity which
            improves training stability
    """

    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def forward(self, z, loss_per_frame=False):
        """Quantized the input tensor using a fixed codebook and returns
        the corresponding codebook vectors

        Parameters
        ----------
        z : Tensor[B x D x T]

        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        Tensor[1]
            Commitment loss to train encoder to predict vectors closer to codebook
            entries
        Tensor[1]
            Codebook loss to update the codebook
        Tensor[B x T]
            Codebook indices (quantized discrete representation of input)
        Tensor[B x D x T]
            Projected latents (continuous representation of input before quantization)
        """

        # Factorized codes (ViT-VQGAN) Project input into low-dimensional space
        z_e = self.in_proj(z)  # z_e : (B x D x T)
        z_q, indices = self.decode_latents(z_e)

        reduce_dim = [1] if loss_per_frame else [1, 2]
        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean(reduce_dim)
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean(reduce_dim)

        z_q = (
            z_e + (z_q - z_e).detach()
        )  # noop in forward pass, straight-through gradient estimator in backward pass

        z_q = self.out_proj(z_q)

        return z_q, commitment_loss, codebook_loss, indices, z_e

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents):
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight  # codebook: (N x D)

        # L2 normalize encodings and codebook (ViT-VQGAN)
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        # Compute euclidean distance with codebook
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        z_q = self.decode_code(indices)
        return z_q, indices


class ResidualVectorQuantize(nn.Module):
    """
    Introduced in SoundStream: An end2end neural audio codec
    https://arxiv.org/abs/2107.03312
    """

    def __init__(
        self,
        input_dim: int = 512,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: float = 0.0,
    ):
        super().__init__()
        if isinstance(codebook_dim, int):
            codebook_dim = [codebook_dim for _ in range(n_codebooks)]

        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size

        self.quantizers = nn.ModuleList(
            [
                VectorQuantize(input_dim, codebook_size, codebook_dim[i])
                for i in range(n_codebooks)
            ]
        )
        self.quantizer_dropout = quantizer_dropout

    def forward(self, z, n_quantizers: int = None):
        """Quantized the input tensor using a fixed set of `n` codebooks and returns
        the corresponding codebook vectors
        Parameters
        ----------
        z : Tensor[B x D x T]
        n_quantizers : int, optional
            No. of quantizers to use
            (n_quantizers < self.n_codebooks ex: for quantizer dropout)
            Note: if `self.quantizer_dropout` is True, this argument is ignored
                when in training mode, and a random number of quantizers is used.
        Returns
        -------
        dict
            A dictionary with the following keys:

            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
        """
        z_q = 0
        residual = z
        commitment_loss = 0
        codebook_loss = 0

        codebook_indices = []
        latents = []

        if n_quantizers is None:
            n_quantizers = self.n_codebooks
        if self.training:
            n_quantizers = torch.ones((z.shape[0],)) * self.n_codebooks + 1
            dropout = torch.randint(1, self.n_codebooks + 1, (z.shape[0],))
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            n_quantizers[:n_dropout] = dropout[:n_dropout]
            n_quantizers = n_quantizers.to(z.device)

        for i, quantizer in enumerate(self.quantizers):
            if self.training is False and i >= n_quantizers:
                break

            z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(
                residual
            )

            # Create mask to apply quantizer dropout
            mask = (
                torch.full((z.shape[0],), fill_value=i, device=z.device) < n_quantizers
            )
            z_q = z_q + z_q_i * mask[:, None, None]
            residual = residual - z_q_i

            # Sum losses
            commitment_loss += (commitment_loss_i * mask).mean()
            codebook_loss += (codebook_loss_i * mask).mean()

            codebook_indices.append(indices_i)
            latents.append(z_e_i)

        codes = torch.stack(codebook_indices, dim=1)
        latents = torch.cat(latents, dim=1)
        
        out_dict = {
            "z_q": z_q,
            "codes": codes,
            "latents": latents,
            "commitment_loss": commitment_loss,
            "codebook_loss": codebook_loss,
        }
        return out_dict
        # return z_q, codes, latents, commitment_loss, codebook_loss

    def from_codes(self, codes: torch.Tensor, return_z_q_is=False):
        """Given the quantized codes, reconstruct the continuous representation
        Parameters
        ----------
        codes : Tensor[B x N x T]
            Quantized discrete representation of input
        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        """
        z_q = 0.0
        z_p = []
        n_codebooks = codes.shape[1]
        if return_z_q_is:
            z_q_is = []
            
        for i in range(n_codebooks):
            z_p_i = self.quantizers[i].decode_code(codes[:, i, :])
            z_p.append(z_p_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i
            if return_z_q_is:
                z_q_is.append(z_q_i)
        if return_z_q_is:
            z_q_is = torch.stack(z_q_is, dim=1)
            return z_q, torch.cat(z_p, dim=1), codes, z_q_is
        else:
            return z_q, torch.cat(z_p, dim=1), codes

    def from_latents(self, latents: torch.Tensor):
        """Given the unquantized latents, reconstruct the
        continuous representation after quantization.

        Parameters
        ----------
        latents : Tensor[B x N x T]
            Continuous representation of input after projection

        Returns
        -------
        Tensor[B x D x T]
            Quantized representation of full-projected space
        Tensor[B x D x T]
            Quantized representation of latent space
        """
        z_q = 0
        z_p = []
        codes = []
        dims = np.cumsum([0] + [q.codebook_dim for q in self.quantizers])

        n_codebooks = np.where(dims <= latents.shape[1])[0].max(axis=0, keepdims=True)[
            0
        ]
        for i in range(n_codebooks):
            j, k = dims[i], dims[i + 1]
            z_p_i, codes_i = self.quantizers[i].decode_latents(latents[:, j:k, :])
            z_p.append(z_p_i)
            codes.append(codes_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i

        return z_q, torch.cat(z_p, dim=1), torch.stack(codes, dim=1)


class VBRResidualVectorQuantize(ResidualVectorQuantize):
    def __init__(
        self,
        *,
        input_dim: int = 512,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: float = 0.0,
        ### VBR specific parameters
        full_codebook_rate: float = 0.5,
        use_framewise_masking: bool = False,
        level_min: float,
        level_max: float,
        level_dist: str = "uniform", ## in ["uniform", "log_uniform"]
        operator_mode: str = "scaling", ## in ["scaling", "exponential", "transformed_scaling"] ## Paper: scaling
        imp_map_input: str = "feature", ## in ["feature", "zqis"]
        detach_imp_map_input: bool = False, 
        imp2mask_alpha: float = 1.0,
        imp2mask_func: str="logcosh", ## logcosh, square, sigmoid
    ):
        super().__init__(
            input_dim=input_dim,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
        )

        self.full_codebook_rate = full_codebook_rate
        self.use_framewise_masking = use_framewise_masking
        self.level_min = level_min
        self.level_max = level_max
        self.level_dist = level_dist
        self.operator_mode = operator_mode
        self.imp_map_input = imp_map_input
        self.detach_imp_map_input = detach_imp_map_input
        self.imp2mask_alpha = imp2mask_alpha
        self.imp2mask_func = imp2mask_func
        
        if imp_map_input == "feature":
            imp_map_inp_channels = input_dim
            imp_map_feat_channels = input_dim
            intermediate_channels = [512, 128, 32, 8]
        elif imp_map_input == "zqis":
            imp_map_inp_channels = input_dim * n_codebooks
            imp_map_feat_channels = input_dim
            intermediate_channels = [512, 128, 32, 8]
        else:
            raise ValueError(f"Invalid imp_map_input: {imp_map_input}")
        
        self.imp_subnet = ImportanceSubnet(
            d_input=imp_map_inp_channels,
            d_feat=imp_map_feat_channels,
            intermediate_channels=intermediate_channels,
            out_channels=1, 
            detach_input=detach_imp_map_input
        )
    
    def forward(
        self,
        z: torch.Tensor,
        n_quantizers: int = None,
        feat_enc: torch.Tensor = None,
        level: float = None, ## only used in VBR inference.
    ):
        z_q = 0
        residual = z
        bs, ch, frames = z.shape # (B, D, T)
        
        commitment_loss = torch.zeros(bs, self.n_codebooks, frames).to(z.device)
        codebook_loss = torch.zeros(bs, self.n_codebooks, frames).to(z.device)
        
        codebook_indices = []
        latents = []
        z_q_is = []
        
        if n_quantizers is None:
            mode = "VBR"
            assert level is not None, "level must be specified in VBR mode"
        else:
            mode = "CBR"
            # assert level is None, "level must be None in CBR mode"
        
        for i, quantizer in enumerate(self.quantizers):
            if mode == "CBR" and n_quantizers is not None:
                if i >= n_quantizers:
                    break
                
            z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(residual, loss_per_frame=True)
            z_q_is.append(z_q_i)
            residual = residual - z_q_i ## We do not have to consider the effect of the masking for dropouts: 1. its frame-wise-based, and 2. once z_q_i is masked, then we don't use this residual anymore.
            commitment_loss[:, i, :] = commitment_loss_i
            codebook_loss[:, i, :] = codebook_loss_i
            
            codebook_indices.append(indices_i)
            latents.append(z_e_i)
        
        ## Importance Map
        # z_q_is: [(B, D, T), (B, D, T), ...]
        if mode=="VBR":
            z_q_is_cat = torch.cat(z_q_is, dim=1) # (B, D*N, T)
            if self.imp_map_input == "feature":
                imp_map_inp = feat_enc
            elif self.imp_map_input == "zqis":
                imp_map_inp = z_q_is_cat
            else:
                raise ValueError(f"Invalid imp_map_input: {self.imp_map_input}") 
            imp_map = self.imp_subnet(imp_map_inp)
            
            if self.training:
                assert self.level_min < self.level_max
                assert self.level_max < 20, "Level_max is too high, we also multiply n_codebooks when we use Simple Scaling function"
                if self.level_dist == "uniform":
                    random_levels = torch.rand((bs, 1, 1)) * (self.level_max - self.level_min) + self.level_min
                    random_levels = random_levels.to(z)
                elif self.level_dist == "log_uniform":
                    random_levels = torch.rand((bs, 1, 1)) * (math.log(self.level_max) - math.log(self.level_min)) + math.log(self.level_min) ## log uniform
                    random_levels = torch.exp(random_levels).to(z)
                else:
                    raise ValueError("Invalid level_dist")

                if self.operator_mode == "scaling":
                    # random_levels = random_levels.to(z)
                    imp_map_scaled = imp_map * random_levels * self.n_codebooks
                elif self.operator_mode == "exponential":
                    imp_map = torch.clamp(imp_map, EPS, 1.0)
                    imp_map_scaled = self.n_codebooks * torch.pow(imp_map, 1/random_levels)
                elif self.operator_mode == "transformed_scaling":
                    imp_map_scaled = apply_straight(imp_map, random_levels, self.n_codebooks)
                else:
                    raise NotImplementedError("Other operator modes are not implemented yet")
            else: ## Inference
                # if level is None: ## "Just for visualization"
                #     imp_map_scaled = imp_map * self.n_codebooks
                if self.operator_mode == "scaling":
                    imp_map_scaled = imp_map * level * self.n_codebooks
                elif self.operator_mode == "exponential":
                    imp_map = torch.clamp(imp_map, EPS, 1.0)
                    imp_map_scaled = self.n_codebooks * torch.pow(imp_map, 1/level)
                elif self.operator_mode == "transformed_scaling":
                    imp_map_scaled = apply_straight(imp_map, level, self.n_codebooks)
                else:
                    raise NotImplementedError("Other operator modes are not implemented yet")
                    
            mask_imp = generate_mask_ste(
                imp_map_scaled,
                self.n_codebooks,
                alpha=self.imp2mask_alpha,
                function=self.imp2mask_func,
            ) ## mask_imp: (B, Nq, T)
        
        elif mode == "CBR":
            imp_map_scaled = torch.ones((bs, 1, frames)).to(z) * n_quantizers
            imp_map = None
            mask_imp = torch.ones((bs, self.n_codebooks, frames)).to(z)
        else:
            raise ValueError("Invalid mode")
        
        ## Dropout / Full Codebook
        if self.training:
            if self.use_framewise_masking:
                dropout = torch.randint(1, self.n_codebooks + 1, (bs, 1, frames))
            else:
                dropout = torch.randint(1, self.n_codebooks + 1, (bs, 1, 1))
                dropout = dropout.expand(bs, 1, frames) ## (B, 1, T)
            n_full = int(bs * self.full_codebook_rate)
            n_dropout = int(bs * self.quantizer_dropout)
            n_imps = int(bs) - n_full - n_dropout
            
            dropout_mask = generate_mask_hard(dropout[:n_dropout], self.n_codebooks) ## (B, Nq, T)
            
            mask_imp[n_imps:n_imps+n_dropout] = dropout_mask.detach()
            mask_imp[n_imps+n_dropout:] = 1.0
        else:
            n_imps = bs
        
        ### Apply mask
        # mask_imp: (B, Nq, T)
        z_q_is_stack = torch.stack(z_q_is, dim=1) # (B, Nq, D, T)
        z_q = torch.sum(z_q_is_stack * mask_imp[:, :, None, :], dim=1, keepdim=False) 
        # (B, D, T)
        # commitment_loss: (B, Nq, T)
        commitment_loss = (commitment_loss * mask_imp.detach()).sum(dim=1).mean()
        codebook_loss = (codebook_loss * mask_imp.detach()).sum(dim=1).mean()
        
        codes = torch.stack(codebook_indices, dim=1)
        latents = torch.cat(latents, dim=1)
        if imp_map is not None:
            imp_map_out = imp_map[:n_imps]
        else:
            imp_map_out = None
            
        out_dict = {
            "z_q": z_q,
            "codes": codes,
            "latents": latents,
            "commitment_loss": commitment_loss,
            "codebook_loss": codebook_loss,
            "imp_map": imp_map_out,
            "mask_imp": mask_imp,
        }
        # import pdb; pdb.set_trace()
        
        return out_dict
                
    def from_latents(self, latents: torch.Tensor):
        raise NotImplementedError("Not implemented yet for VBRResidualVectorQuantize")
        
        


if __name__ == "__main__":
    rvq = ResidualVectorQuantize(quantizer_dropout=True)
    x = torch.randn(16, 512, 80)
    y = rvq(x)
    print(y["latents"].shape)