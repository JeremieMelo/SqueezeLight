import numpy as np
import torch
import torch.fft
from pyutils.compute import toeplitz
from pyutils.general import logger
from pyutils.initializer import morr_uniform_
from pyutils.quantize import input_quantize_fn, weight_quantize_fn
from torch import nn
from torch.nn import Parameter, init
from torchonn.devices.mrr import MORRConfig_20um_MQ
from torchonn.op.mrr_op import mrr_roundtrip_phase_to_tr_func, mrr_roundtrip_phase_to_tr_fused
from torchonn.op.mzi_op import phase_quantize_fn, voltage_quantize_fn

__all__ = ["AllPassMORRCirculantLinear"]


class AllPassMORRCirculantLinear(nn.Module):
    """
    description: All-pass MORR Linear layer, assumes (1) block-circulant matrix (2) differential rails (3) learnable balancing factors.
    """

    def __init__(
        self,
        in_channel,
        out_channel,
        bias=False,
        miniblock=4,
        mode="weight",
        v_max=10.8,
        v_pi=4.36,
        w_bit=16,
        in_bit=16,
        ### mrr parameter
        MORRConfig=MORRConfig_20um_MQ,
        ### trainable MORR nonlinearity
        trainable_morr_bias=False,
        trainable_morr_scale=False,
        device=torch.device("cuda"),
    ):
        super(AllPassMORRCirculantLinear, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.mode = mode
        self.miniblock = miniblock
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
            100 / (self.in_channel // self.miniblock)
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

        self.voltage_quantizer = voltage_quantize_fn(self.w_bit, self.v_pi, self.v_max)
        self.phase_quantizer = phase_quantize_fn(self.w_bit, self.v_pi, self.v_max, gamma_noise_std=0)
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
            self.bias = Parameter(torch.Tensor(out_channel).to(self.device))
        else:
            self.register_parameter("bias", None)

        self.finegrain_drop_mask = None

    def build_parameters(self, mode="weight"):
        ## weight mode
        self.in_channel_pad = int(np.ceil(self.in_channel / self.miniblock).item() * self.miniblock)
        self.out_channel_pad = int(np.ceil(self.out_channel / self.miniblock).item() * self.miniblock)
        self.grid_dim_y = self.out_channel_pad // self.miniblock
        self.grid_dim_x = self.in_channel_pad // self.miniblock

        if mode in {"weight"}:
            self.weight = Parameter(
                torch.ones(
                    self.grid_dim_y, self.grid_dim_x, self.miniblock, device=self.device, dtype=torch.float
                )
            )

            self.morr_output_scale = Parameter(
                torch.randn(1, 1, max(1, self.grid_dim_x // 2) + 1, 1, device=self.device)
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

    def reset_parameters(self, morr_init: bool = False) -> None:
        ### nonlinear curve aware initialization
        if morr_init:
            ## initialize weight
            morr_uniform_(
                self.weight,
                MORRConfig=self.MORRConfig,
                n_op=self.miniblock,
                biased=self.w_bit >= 16,
                gain=2 if self.in_bit < 16 else 1,
            )  # quantization needs zero-center
            self.sigma_weight = self.weight.data.std().item()
            self.weight_quant_gain = None

            ## output distribution aware initialization to output scaling factor
            # init.uniform_(self.morr_output_scale, -1, 1) ## scaling need to performed after quantization
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
            init.kaiming_normal_(self.weight.data)
            init.normal_(self.morr_output_scale.data)
            self.sigma_weight = self.weight.data.std().item()
            self.weight_quant_gain = None
            self.sigma_out_scale = self.morr_output_scale.data.std().item()
            self.out_scale_quant_gain = None

        if self.morr_input_bias is not None:
            self.morr_input_bias.data.zero_()
        if self.morr_input_scale is not None:
            init.normal_(self.morr_input_scale.data, 2, 0.1)

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

            ## rescale weights after quantization can maintain the initialization distribution
            if self.weight_quant_gain is None:
                self.weight_quant_gain = self.sigma_weight / weight.data.std()
            if self.trainable_morr_scale:
                morr_scale = self.morr_scale * self.weight_quant_gain
            else:
                morr_scale = self.weight_quant_gain
            weight = weight.mul(morr_scale)  ### gain factor from Tanh used in quantization
            # if(self.trainable_morr_scale):
            #     weight = weight.mul(self.morr_scale)

            ### quantize learnable balancing factor
            morr_output_scale = self.morr_output_scale_quantizer(self.morr_output_scale)
            ## rescale after quantization is harmful
            # if(self.out_scale_quant_gain is None):
            #     self.sigma_out_scale_quant_gain = self.sigma_out_scale / morr_output_scale.data.std().item()
            # morr_output_scale = morr_output_scale.mul(self.sigma_out_scale_quant_gain)### gain factor from Tanh used in quantization
        else:
            weight = self.weight.abs()  # positive only
            morr_output_scale = self.morr_output_scale - self.morr_output_scale.data.mean()

        if self.finegrain_drop_mask is not None:
            weight = weight.mul(self.finegrain_drop_mask.float())

        ## differential balancing factor concatenation
        scale = morr_output_scale[..., :-1, :]
        scale_pad = morr_output_scale[..., -1:, :]
        if self.grid_dim_x % 2 == 0:
            # even blocks
            scale = torch.cat([scale, -scale], dim=2)  # [1, 1, q, 1]
        else:
            # odd blocks
            if self.grid_dim_x > 1:
                scale = torch.cat([morr_output_scale, -scale], dim=2)  # [1, 1, q, 1]
            else:
                scale = scale_pad  # [1, 1, q, 1]
        morr_output_scale = scale.squeeze(-1).unsqueeze(0)  # [1 ,1, 1, q]

        return weight, morr_output_scale

    def enable_fast_forward(self):
        self.fast_forward_flag = True

    def disable_fast_forward(self):
        self.fast_forward_flag = False

    def set_gamma_noise(self, noise_std, random_state=None):
        self.gamma_noise_std = noise_std
        # self.phase_quantizer.set_gamma_noise(noise_std, random_state)

    def set_crosstalk_factor(self, crosstalk_factor):
        self.crosstalk_factor = crosstalk_factor
        self.phase_quantizer.set_crosstalk_factor(crosstalk_factor)

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
            total = self.out_channel * self.in_channel
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
                total = (self.out_channel * self.in_channel) * 4
            else:
                total = (self.out_channel * self.in_channel) * self.w_bit / 8
        if self.bias is not None:
            total += self.bias.numel() * 4
        return total

    def input_modulator(self, x):
        ### voltage to power, which is proportional to the phase shift
        return x * x

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
    def morr_bias(self):
        if self.morr_input_bias is None:
            return None
        # return 2 * self.morr_fwhm * torch.sigmoid(self.morr_input_bias.unsqueeze(0).unsqueeze(-1))
        return self.morr_fwhm * torch.tanh(self.morr_input_bias.unsqueeze(0).unsqueeze(-1))

    @property
    def morr_scale(self):
        if self.morr_input_scale is None:
            return None
        return torch.sigmoid(self.morr_input_scale.unsqueeze(-1)) + 0.2  # [p, q, 1]

    def propagate_morr(self, weight, x, morr_output_scale):
        """
        @description: propagate through the analytically calculated transfer matrix of molg. We implement circulant matrix multiplication using fast circ matmul
        @param weight {torch.Tensor} two phase shifters in the MZI-based attenuators
        @param x {torch.Tensor} complex-valued input
        @param morr_output_scale {torch.Tensor} learnable balancing factors
        @return: y {torch.Tensor} output of attenuators
        """
        ### x : [bs, q, k]
        ### weights: [p, q, k]
        ### morr_output_scale: [1, 1, 1, q]

        ## build circulant weight matrix
        # crosstalk on the weights are much cheaper to compute than on the phase shift
        if self.enable_thermal_crosstalk and self.crosstalk_factor > 1:
            weight = weight * self.crosstalk_factor
        weight = toeplitz(weight).unsqueeze(0)  # [1,  p, q, k, k]
        x = x.unsqueeze(1).unsqueeze(-1)  # [bs, 1, q, k, 1]
        x = weight.matmul(x).squeeze(-1)  # [bs, p, q, k]

        if self.enable_phase_noise and self.phase_noise_std > 1e-5:
            x = x + torch.zeros_like(x).normal_(0, self.phase_noise_std)

        ### Use theoretical transmission function
        ### x is the phase detuning, x=0 means on-resonance
        ### phase: [bs, p, q, k]
        x = self.mrr_roundtrip_phase_to_tr(x)

        x = morr_output_scale.matmul(x)  # [1, 1, 1, q] x [bs, p, q, k] = [bs, p, 1, k]
        x = x.flatten(1)  # [bs, p*k]
        return x

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

    def forward_slow(self, x):
        assert (
            x.size(-1) == self.in_channel
        ), f"[E] Input dimension does not match the weight size {self.out_channel, self.in_channel}, but got input size ({tuple(x.size())}))"
        if self.in_bit < 16:
            x = self.input_quantizer(x)
        # if(not self.fast_forward_flag or self.weight is None):
        #     weight = self.build_weight()
        # else:
        #     weight = self.weight #.view(self.out_channel, -1)[:, :self.in_channel]

        weight = self.build_weight()
        if self.in_channel_pad > self.in_channel:
            if self.x_zero_pad is None or self.x_zero_pad.size(0) != x.size(0):
                self.x_zero_pad = torch.zeros(
                    x.size(0), self.in_channel_pad - self.in_channel, device=x.device, dtype=x.dtype
                )
            x = torch.cat([x, self.x_zero_pad], dim=1)

        x = x.view(-1, self.grid_dim_x, self.miniblock)
        # print(x.size())
        ### modulation
        ### assume the real input is the magnitude of the modulator output with fixed phase response
        ### x: [bs, q, k] -> [bs, q, k, 2]
        x = self.input_modulator(x)

        ### propagate through attenuator (weight)
        ### x: [bs, q, k, 2] -> [bs, p, q, k, 2]
        x = self.propagate_morr(weight, x)
        # print(x.size())

        ### propagate through photodetection, from optics to voltages
        ### x: [bs, p, q, k, 2] -> [bs, p, q, k]
        x = self.propagate_photodetection(x)
        # print(x.size())

        ### postprocessing before activation
        ### x: [bs, outc] -> [bs, outc]
        # out = self.postprocessing(x)

        if self.out_channel < self.out_channel_pad:
            x = x[..., : self.out_channel]
        if self.bias is not None:
            x = x + self.bias.unsqueeze(0)

        return x

    def forward(self, x):
        assert (
            x.size(-1) == self.in_channel
        ), f"[E] Input dimension does not match the weight size {self.out_channel, self.in_channel}, but got input size ({tuple(x.size())}))"
        if self.in_bit < 16:
            x = self.input_quantizer(x)

        weight, morr_output_scale = self.build_weight()
        if self.in_channel_pad > self.in_channel:
            if self.x_zero_pad is None or self.x_zero_pad.size(0) != x.size(0):
                self.x_zero_pad = torch.zeros(
                    x.size(0), self.in_channel_pad - self.in_channel, device=x.device, dtype=x.dtype
                )
            x = torch.cat([x, self.x_zero_pad], dim=1)

        x = x.view(-1, self.grid_dim_x, self.miniblock)

        ### modulation
        ### assume the real input is the magnitude of the modulator output with fixed phase response
        ### x: [bs, q, k] -> [bs, q, k]
        x = self.input_modulator(x)

        ### propagate through morr array (weight)
        ### x: [bs, q, k] -> [bs, p*k]
        x = self.propagate_morr(weight, x, morr_output_scale)

        if self.out_channel < self.out_channel_pad:
            x = x[..., : self.out_channel]
        if self.bias is not None:
            x = x + self.bias.unsqueeze(0)

        return x
