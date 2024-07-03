import torch
from typing import Tuple
from torch import nn
from torchprofile import profile_macs

Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB

class ModelInfo():
    def __init__(self, model: nn.Module, input_shape: Tuple, data_width=32, device=torch.device('cpu')) -> None:
        self.model = model
        self.data_width = data_width
        self.device = device
        self.x = torch.rand(input_shape, device=self.device)

    def get_model_param_count(self, trainable:bool =True) -> int:
        cnt: int = 0
        if trainable:
            return sum(param.numel() for _, param in self.model.named_parameters() if param.requires_grad)
        else:
            return sum(param.numel() for _, param in self.model.named_parameters())
        
    def get_model_size(self, trainable:bool =True) -> int:
        return self.get_model_param_count(True) * self.data_width

    def get_model_macs(self) -> int:
        return profile_macs(self.model, self.x)

    def get_model_flops(self) -> int:
        return 2 * self.get_model_macs()

    def __str__(self) -> str:
        return f"Param: {self.get_model_param_count() / MiB: .4f} MiB, Flops: {self.get_model_flops() / 1e6: .4f} M"
