from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.fft
import torch.nn.functional as F
from pyutils.general import logger
from pyutils.torch_train import set_torch_deterministic
from torch import Tensor, nn
from torchonn.op.mrr_op import mrr_roundtrip_phase_to_tr_grad_fused

from .layers import AllPassMORRCirculantConv2d, AllPassMORRCirculantLinear

__all__ = ["MORR_CLASS_BASE"]


class MORR_CLASS_BASE(nn.Module):
    """MORR CNN for classification (MORR-ONN). MORR array-based convolution with learnable nonlinearity [SqueezeLight, DATE'21]"""

    _conv_linear = (AllPassMORRCirculantConv2d, AllPassMORRCirculantLinear)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset_parameters(self, random_state: int = None, morr_init: bool = False) -> None:
        for name, m in self.named_modules():
            if isinstance(m, self._conv_linear):
                if random_state is not None:
                    # deterministic seed, but different for different layer, and controllable by random_state
                    set_torch_deterministic(random_state + sum(map(ord, name)))
                m.reset_parameters(morr_init)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def set_gamma_noise(self, noise_std: float = 0.0, random_state: Optional[int] = None) -> None:
        self.gamma_noise_std = noise_std
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer.set_gamma_noise(noise_std, random_state=random_state)

    def set_crosstalk_factor(self, crosstalk_factor: float = 0.0) -> None:
        self.crosstalk_factor = crosstalk_factor
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer.set_crosstalk_factor(crosstalk_factor)

    def set_weight_bitwidth(self, w_bit: int) -> None:
        self.w_bit = w_bit
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer.set_weight_bitwidth(w_bit)

    def load_parameters(self, param_dict: Dict[str, Dict[str, Tensor]]) -> None:
        """
        description: update parameters based on this parameter dictionary\\
        param param_dict {dict of dict} {layer_name: {param_name: param_tensor, ...}, ...}
        """
        for layer_name, layer_param_dict in param_dict.items():
            self.layers[layer_name].load_parameters(layer_param_dict)

    def build_obj_fn(self, X: Tensor, y: Tensor, criterion: Callable) -> Callable:
        def obj_fn(X_cur=None, y_cur=None, param_dict=None):
            if param_dict is not None:
                self.load_parameters(param_dict)
            if X_cur is None or y_cur is None:
                data, target = X, y
            else:
                data, target = X_cur, y_cur
            pred = self.forward(data)
            return criterion(pred, target)

        return obj_fn

    def enable_fast_forward(self) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer.enable_fast_forward()

    def disable_fast_forward(self) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer.disable_fast_forward()

    def sync_parameters(self, src: str = "weight") -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer.sync_parameters(src=src)

    def switch_mode_to(self, mode: str) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer.switch_mode_to(mode)

    def enable_morr_phase_loss(self):
        self.morr_phase_loss_flag = True

    def disable_morr_phase_loss(self):
        self.morr_phase_loss_flag = False

    def calc_morr_phase_loss(self, phase, threshold=1):
        return torch.relu(phase - threshold).mean()

    def register_morr_phase_loss(self, loss):
        self.morr_phase_loss = loss

    def get_morr_phase_loss(self):
        return self.morr_phase_loss

    def enable_morr_gradient_loss(self):
        self.morr_gradient_loss_flag = True

    def disable_morr_gradient_loss(self):
        self.morr_gradient_loss_flag = False

    def calc_morr_gradient_loss(self, layer, phase):
        # return polynomial(phase, layer.morr_lambda_to_mag_curve_coeff_half_grad).abs().mean()
        return mrr_roundtrip_phase_to_tr_grad_fused(
            phase, layer.MORRConfig.attenuation_factor, layer.MORRConfig.coupling_factor, intensity=True
        )

    def register_morr_gradient_loss(self, loss):
        self.morr_gradient_loss = loss

    def get_morr_gradient_loss(self):
        return self.morr_gradient_loss

    def requires_morr_grad(self, mode=True):
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                if layer.morr_input_bias is not None:
                    layer.morr_input_bias.requires_grad_(mode)
                    layer.morr_input_scale.requires_grad_(mode)

    def get_finegrain_drop_mask(self, topk):
        ## each module stores a local pruning mask and uses the mask during forward without changing the weight tensor values.
        self.finegrain_drop_mask = {}
        for layer_name, layer in self.named_modules():
            if isinstance(layer, self._conv_linear):
                mask = layer.get_finegrain_drop_mask(topk=topk)
                self.finegrain_drop_mask[layer_name] = mask
        return self.finegrain_drop_mask

    def apply_finegrain_drop_mask(self):
        ## permanently apply pruning mask to the weight tensor
        if self.finegrain_drop_mask is None:
            print("[W] No finegrained drop mask is available.")
            return
        for layer_name, layer in self.named_modules():
            if isinstance(layer, self._conv_linear):
                mask = self.finegrain_drop_mask[layer_name]
                layer.apply_finegrain_drop_mask(mask=mask)

    def enable_crosstalk(self):
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer.enable_crosstalk()

    def disable_crosstalk(self):
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer.disable_crosstalk()

    def set_crosstalk_coupling_matrix(self, coupling_factor, drop_perc=0):
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer.set_crosstalk_coupling_matrix(coupling_factor, drop_perc)

    def enable_phase_variation(self):
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer.enable_phase_variation()

    def disable_phase_variation(self):
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer.disable_phase_variation()

    def set_phase_variation(self, phase_noise_std=0):
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer.set_phase_variation(phase_noise_std)

    def get_num_MORR(self):
        n_morr = {}
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                k = layer.miniblock
                n_morr[k] = n_morr.get(k, 0) + layer.grid_dim_x * layer.grid_dim_y
                n_morr[1] = n_morr.get(1, 0) + layer.grid_dim_x
        return n_morr, sum(i for i in n_morr.values())

    def forward(self, x):
        raise NotImplementedError
