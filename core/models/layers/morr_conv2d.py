"""
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-01-27 01:08:44
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-07-18 00:40:18
"""
import numpy as np
import torch
import torch.fft
import torch.nn.functional as F
from pyutils.compute import im2col_2d, toeplitz
from pyutils.general import logger
from pyutils.initializer import morr_uniform_
from pyutils.quantize import input_quantize_fn, weight_quantize_fn
from pyutils.torch_train import set_torch_deterministic
from torch import nn
from torch.nn import Parameter, init
from torch.types import Device
from torchonn.devices.mrr import MORRConfig_20um_MQ
from torchonn.op.mrr_op import mrr_roundtrip_phase_to_tr_func, mrr_roundtrip_phase_to_tr_fused

__all__ = ["AllPassMORRCirculantConv2d"]


class AllPassMORRCirculantConv2d(nn.Module):
    """
    description: All-pass MORR Conv2d layer, assumes (1) block-circulant matrix (2) differential rails (3) learnable balancing factors.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        bias: bool = False,
        miniblock: int = 4,
        stride=1,
        padding=0,
        mode: str = "weight",
        v_max: float = 10.8,
        v_pi: float = 4.36,
        w_bit: int = 16,
        in_bit: int = 16,
        ### mrr parameter
        MORRConfig=MORRConfig_20um_MQ,
        ### trainable MORR nonlinearity
        trainable_morr_bias: bool = False,
        trainable_morr_scale: bool = False,
        device: Device = torch.device("cuda"),
    ):
        super(AllPassMORRCirculantConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mode = mode
        self.kernel_size = kernel_size
        self.miniblock = miniblock
        self.stride = stride
        self.padding = padding
        assert mode in {"weight", "phase", "voltage"}, logger.error(
            f"Mode not supported. Expected one from (weight, phase, voltage) but got {mode}."
        )

        self.v_max = v_max
        self.v_pi = v_pi
        self.gamma = np.pi / self.v_pi ** 2
        self.w_bit = w_bit
        self.in_bit = in_bit
        self.MORRConfig = MORRConfig
        self.mrr_a = MORRConfig.attenuation_factor
        self.mrr_r = MORRConfig.coupling_factor
        self.device = device
        self.trainable_morr_bias = trainable_morr_bias
        self.trainable_morr_scale = trainable_morr_scale
        ### calculate FWHM (rad)
        self.morr_fwhm = (
            -4
            * np.pi ** 2
            * MORRConfig.radius
            * MORRConfig.effective_index
            * (
                1 / MORRConfig.resonance_wavelength
                - 1 / (MORRConfig.resonance_wavelength - MORRConfig.bandwidth / 2)
            )
        )

        ### allocate parameters
        self.weight = None
        self.x_zero_pad = None
        self.morr_output_scale = None  ## learnable balancing factors implelemt by MRRs
        self.morr_input_bias = None  ## round-trip phase shift bias within MORR
        self.morr_input_scale = None  ## scaling factor for the round-trip phase shift within MORR
        self.morr_gain = (
            100 / (self.in_channels * self.kernel_size ** 2 // self.miniblock)
        ) ** 0.5  ## TIA gain, calculated such that output variance is around 1
        ### build trainable parameters
        self.build_parameters(mode)

        ### quantization tool
        self.input_quantizer = input_quantize_fn(self.in_bit, device=self.device)
        self.weight_quantizer = weight_quantize_fn(
            self.w_bit, alg="dorefa_pos"
        )  ## [0-1] positive only, maintain the original scale
        self.morr_output_scale_quantizer = weight_quantize_fn(
            self.w_bit, alg="dorefa_sym"
        )  ## [-1,1] full-range

        self.mrr_roundtrip_phase_to_tr = mrr_roundtrip_phase_to_tr_func(
            a=self.mrr_a, r=self.mrr_r, intensity=True
        )

        ### default set to slow forward
        self.disable_fast_forward()
        ### default set no gamma noise
        self.set_gamma_noise(0)
        ### default set no crosstalk
        self.disable_crosstalk()
        ### default set no phase variation
        self.disable_phase_variation()

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels).to(self.device))
        else:
            self.register_parameter("bias", None)

        self.finegrain_drop_mask = None

    def build_parameters(self, mode="weight"):
        ## weight mode
        self.out_channels_pad = int(np.ceil(self.out_channels / self.miniblock).item() * self.miniblock)
        self.weight_in_channels = self.kernel_size * self.kernel_size * self.in_channels
        self.weight_in_channels_pad = (
            int(np.ceil(self.weight_in_channels / self.miniblock).item()) * self.miniblock
        )
        self.grid_dim_y = self.out_channels_pad // self.miniblock
        self.grid_dim_x = self.weight_in_channels_pad // self.miniblock

        if mode in {"weight"}:
            self.weight = Parameter(
                torch.ones(
                    self.grid_dim_y, self.grid_dim_x, self.miniblock, device=self.device, dtype=torch.float
                )
            )
            self.morr_output_scale = Parameter(
                torch.zeros(max(1, self.grid_dim_x // 2) + 1, device=self.device)
            )
            if self.trainable_morr_bias:
                ### initialize with the finest-granularity, i.e., per mini-block
                self.morr_input_bias = Parameter(
                    torch.zeros(self.grid_dim_y, self.grid_dim_x, device=self.device, dtype=torch.float)
                )
            if self.trainable_morr_scale:
                ### initialize with the finest-granularity, i.e., per mini-block
                self.morr_input_scale = Parameter(
                    torch.zeros(self.grid_dim_y, self.grid_dim_x, device=self.device, dtype=torch.float)
                )
        elif mode == "phase":
            raise NotImplementedError
            self.phase = Parameter(self.phase)
        elif mode == "voltage":
            raise NotImplementedError
            self.voltage = Parameter(self.voltage)
        else:
            raise NotImplementedError

    def reset_parameters(self, morr_init: bool = False):
        if morr_init:
            ### nonlinear curve aware initialization
            morr_uniform_(
                self.weight,
                MORRConfig=self.MORRConfig,
                n_op=self.miniblock,
                biased=self.w_bit >= 16,
                gain=2 if self.in_bit < 16 else 1,
            )
            self.sigma_weight = self.weight.data.std().item()
            self.weight_quant_gain = None
            ### output distribution aware initialization to output scaling factor
            t1 = mrr_roundtrip_phase_to_tr_fused(
                torch.tensor([0]).float(), a=self.mrr_a, r=self.mrr_r, intensity=True
            )
            t2 = mrr_roundtrip_phase_to_tr_fused(
                torch.tensor([self.morr_fwhm * 2.4]).float(), a=self.mrr_a, r=self.mrr_r, intensity=True
            )
            g = ((t2 - t1) / (2.4 * self.morr_fwhm)).item()  ## 0~2.4 FWHM slope as a linear approximation

            self.sigma_out_scale = 4 / (3 * self.grid_dim_x ** 0.5 * g * self.morr_fwhm)
            self.out_scale_quant_gain = None
            init.normal_(self.morr_output_scale, 0, self.sigma_out_scale)

        else:
            nn.init.kaiming_normal_(self.weight)
            nn.init.normal_(self.morr_output_scale)
            self.sigma_weight = self.weight.data.std().item()
            self.weight_quant_gain = None
            self.sigma_out_scale = self.morr_output_scale.data.std().item()
            self.out_scale_quant_gain = None

        if self.morr_input_bias is not None:
            self.morr_input_bias.data.zero_()
        if self.morr_input_scale is not None:
            ### after sigmoid, it cooresponds to 1 scale
            init.zeros_(self.morr_input_scale.data)

        if self.bias is not None:
            init.uniform_(self.bias, 0, 0)

    def sync_parameters(self, src="weight"):
        """
        description: synchronize all parameters from the source parameters
        """

        raise NotImplementedError

    def build_weight(self):
        if self.w_bit < 16:
            ### differentiable quantizer based on STE to enable QAT (Dorefa-Net, arXiv 2016)
            weight = self.weight_quantizer(self.weight)
        else:
            weight = self.weight.abs()  ## have to be all positive
        if self.finegrain_drop_mask is not None:
            weight = weight.mul(self.finegrain_drop_mask.float())

        return weight

    def enable_fast_forward(self):
        self.fast_forward_flag = True

    def disable_fast_forward(self):
        self.fast_forward_flag = False

    def set_gamma_noise(self, noise_std, random_state=None):
        self.gamma_noise_std = noise_std
        # self.phase_quantizer.set_gamma_noise(noise_std, random_state)

    def load_parameters(self, param_dict):
        """
        description: update parameters based on this parameter dictionary\\
        param param_dict {dict of dict} {layer_name: {param_name: param_tensor, ...}, ...}
        """
        for name, param in param_dict.items():
            getattr(self, name).data.copy_(param)
        # if(self.mode == "phase"):
        #     self.build_weight(update_list=param_dict)

    def switch_mode_to(self, mode):
        self.mode = mode

    def get_power(self, mixtraining_mask=None):
        raise NotImplementedError
        masks = (
            mixtraining_mask
            if mixtraining_mask is not None
            else (self.mixedtraining_mask if self.mixedtraining_mask is not None else None)
        )
        if masks is not None:
            power = ((self.phase_U.data * masks["phase_U"]) % (2 * np.pi)).sum()
            power += ((self.phase_S.data * masks["phase_S"]) % (2 * np.pi)).sum()
            power += ((self.phase_V.data * masks["phase_V"]) % (2 * np.pi)).sum()
        else:
            power = ((self.phase_U.data) % (2 * np.pi)).sum()
            power += ((self.phase_S.data) % (2 * np.pi)).sum()
            power += ((self.phase_V.data) % (2 * np.pi)).sum()
        return power.item()

    def get_num_params(self, fullrank=False):
        if (self.dynamic_weight_flag == True) and (fullrank == False):
            total = self.basis.numel()
            if self.coeff_in is not None:
                total += self.coeff_in.numel()
            if self.coeff_out is not None:
                total += self.coeff_out.numel()
        else:
            total = self.out_channels * self.in_channels
        if self.bias is not None:
            total += self.bias.numel()

        return total

    def get_param_size(self, fullrank=False, fullprec=False):
        if (self.dynamic_weight_flag == True) and (fullrank == False):
            total = self.basis.numel() * self.w_bit / 8
            if self.coeff_in is not None:
                total += self.coeff_in.numel() * self.w_bit / 8
            if self.coeff_out is not None:
                total += self.coeff_out.numel() * self.w_bit / 8
        else:
            if fullprec:
                total = (self.out_channels * self.in_channels) * 4
            else:
                total = (self.out_channels * self.in_channels) * self.w_bit / 8
        if self.bias is not None:
            total += self.bias.numel() * 4
        return total

    def input_modulator(self, x):
        ### voltage to power, which is proportional to the phase shift
        return x * x
        # return x

    def set_crosstalk_coupling_matrix(self, coupling_factor, drop_perc=0):
        ### crosstalk coupling matrix is a symmetric matrix, but the intra-MORR crosstalk can be taken as a round-trip phase shift scaling factor, which is proportional to the number of segments after pruned.
        ### drop-perc is the pruning percentage.
        assert 0 <= coupling_factor <= 1, logger.error(
            f"Coupling factor must in [0,1], but got {coupling_factor}"
        )

        self.crosstalk_factor = 1 + max(3, (self.miniblock * (1 - drop_perc) - 1)) * coupling_factor

    def enable_crosstalk(self):
        self.enable_thermal_crosstalk = True

    def disable_crosstalk(self):
        self.enable_thermal_crosstalk = False

    def set_phase_variation(self, phase_noise_std=0):
        self.phase_noise_std = phase_noise_std

    def enable_phase_variation(self):
        self.enable_phase_noise = True

    def disable_phase_variation(self):
        self.enable_phase_noise = False

    def enable_trainable_morr_scale(self):
        self.trainable_morr_scale = True

    def disable_trainable_morr_scale(self):
        self.trainable_morr_scale = False

    def enable_trainable_morr_bias(self):
        self.trainable_morr_bias = True

    def disable_trainable_morr_bias(self):
        self.trainable_morr_bias = False

    @property
    def morr_scale(self):
        return torch.sigmoid(self.morr_input_scale.unsqueeze(0).unsqueeze(-1)) + 0.2

    @property
    def morr_bias(self):
        # return 2 * self.morr_fwhm * torch.sigmoid(
        #         self.morr_input_bias.unsqueeze(0).unsqueeze(-1)
        #     )
        return self.morr_fwhm * torch.tanh(self.morr_input_bias.unsqueeze(0).unsqueeze(-1))

    def propagate_morr(self, weight, x):
        """
        @description: propagate through the analytically calculated transfer matrix of molg. We implement circulant matrix multiplication using circulant matrix mul
        @param weight {torch.Tensor} two phase shifters in the MZI-based attenuators
        @param x {torch.Tensor} complex-valued input
        @return: y {torch.Tensor} output of attenuators
        """
        ### weights: [p, q, k]
        ### x: [ks*ks*inc, h_out*w_out*bs]

        x = x.t()  # [h_out*w_out*bs, ks*ks*inc]
        x = x.view(x.size(0), self.grid_dim_x, self.miniblock)  # [h_out*w_out*bs, q, k]

        if self.enable_thermal_crosstalk and self.crosstalk_factor > 1:
            weight = weight * self.crosstalk_factor
        weight = toeplitz(weight).unsqueeze(0)  # [1, p, q, k, k]
        x = x.unsqueeze(1).unsqueeze(-1)  # [h*w*bs, 1, q, k, 1]
        x = weight.matmul(x).squeeze(-1)  # [h*w*bs, p, q, k]

        if self.enable_phase_noise and self.phase_noise_std > 1e-5:
            x = x + torch.zeros_like(x).normal_(0, self.phase_noise_std)  # [h*w*bs, p, q, k]

        ### Use theoretical transmission function for trainable MORR nonlinearity
        ### x is the phase detuning, x=0 means on-resonance
        ### phase: [h_out*w_out*bs, p, q, k]
        x = self.mrr_roundtrip_phase_to_tr(x)

        ### output scaling
        if self.w_bit < 16:
            morr_output_scale = self.morr_output_scale_quantizer(self.morr_output_scale)
            if self.out_scale_quant_gain is None:
                self.out_scale_quant_gain = self.sigma_out_scale / morr_output_scale.data.std().item()
            morr_output_scale = morr_output_scale.mul(
                self.out_scale_quant_gain
            )  ### gain factor from Tanh used in quantization
        else:
            morr_output_scale = self.morr_output_scale

        scale = morr_output_scale[:-1]
        scale_pad = morr_output_scale[-1:]
        if self.grid_dim_x % 2 == 0:
            # even blocks
            scale = torch.cat([scale, -scale], dim=0)
        else:
            # odd blocks
            if self.grid_dim_x > 1:
                scale = torch.cat([morr_output_scale, -scale], dim=0)
            else:
                scale = scale_pad
        scale = scale.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, q]

        x = scale.matmul(x)  # [1,1,1,q]x[h_out*w_out*bs, p, q, k]=[h_out*w_out*bs, p, 1, k]
        x = x.view(x.size(0), -1).t()  # [p*k, h_out*w_out*bs]
        if self.out_channels_pad > self.out_channels:
            x = x[: self.out_channels, :]  # [outc, h_out*w_out*bs]
        return x

    def morr_conv2d(self, X, W, stride=1, padding=0):
        ### W : [p, q, k]
        n_x = X.size(0)

        _, X_col, h_out, w_out = im2col_2d(
            None,
            X,
            stride,
            padding,
            w_size=(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size),
        )
        ## zero-padding X_col
        if self.weight_in_channels_pad > self.weight_in_channels:
            if self.x_zero_pad is None or self.x_zero_pad.size(1) != X_col.size(1):
                self.x_zero_pad = torch.zeros(
                    self.weight_in_channels_pad - self.weight_in_channels,
                    X_col.size(1),
                    dtype=torch.float32,
                    device=self.device,
                )

            X_col = torch.cat([X_col, self.x_zero_pad], dim=0)
        # matmul
        out = self.propagate_morr(W, X_col)  # [outc, w_out]
        out = out.view(self.out_channels, h_out, w_out, n_x)
        out = out.permute(3, 0, 1, 2).contiguous()

        return out

    def get_finegrain_drop_mask(self, topk):
        if self.w_bit < 16:
            weight = self.weight_quantizer(self.weight.data)  # [p, q, k]
        else:
            weight = self.weight.data.abs()
        indices = weight.argsort(dim=-1)
        mask = torch.ones_like(weight, dtype=torch.bool, device=weight.device)
        # drop_idx = int(drop_perc * weight.size(2))
        # drop_idx = weight.size(2) - max(4, weight.size(2) - drop_idx)
        drop_indices = indices[:, :, 0:-topk]
        mask.scatter_(2, drop_indices, 0)
        self.finegrain_drop_mask = mask
        return mask

    def apply_finegrain_drop_mask(self, mask):
        if self.w_bit < 16:
            self.weight.data.masked_fill_(~mask.view_as(self.weight.data), -1000)
        else:
            self.weight.data.masked_fill_(~mask.view_as(self.weight.data), 0)

    def forward(self, x):
        if self.in_bit < 16:
            x = self.input_quantizer(x)
        weight = self.build_weight()
        x = self.input_modulator(x)
        x = self.morr_conv2d(x, weight, stride=self.stride, padding=self.padding)

        if self.bias is not None:
            x = x + self.b.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        return x
