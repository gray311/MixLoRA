import math
import warnings
from typing import Any, List, Optional, Union

import dataclasses

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer
from peft.tuners.lora import LoraLayer, LoraModel
from peft.utils.other import transpose


class MixLoraLayer(LoraLayer):

    def __init__(self, base_layer: nn.Module, num_experts: int, num_experts_per_tok: int, expert_capacity: int, num_tasks: int, **kwargs):
        super().__init__(base_layer, **kwargs)
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.expert_capacity = expert_capacity
        self.num_tasks = num_tasks

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        self.lora_A[adapter_name] = MixLoraProjLayer(in_features=self.in_features, out_features=r, num_experts=self.num_experts, bias=False, proj_mode="down")
        self.lora_B[adapter_name] = MixLoraProjLayer(in_features=r, out_features=self.out_features, num_experts=self.num_experts, bias=False, proj_mode="up")

        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            for i in range(self.num_experts):
                nn.init.normal_(self.lora_A[adapter_name].mixlora_A[i].weight, mean=0.0, std=0.01)
                nn.init.zeros_(self.lora_B[adapter_name].mixlora_B[i].weight)



class MixLoraProjLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_experts: int, bias=False, proj_mode="down"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.proj_mode = proj_mode
        if proj_mode == "down":
            self.mixlora_A = nn.ModuleList([])
            for _ in range(self.num_experts):
                self.mixlora_A.append(nn.Linear(self.in_features, self.out_features, bias=bias))
        else:
            self.mixlora_B = nn.ModuleList([])
            for _ in range(self.num_experts):
                self.mixlora_B.append(nn.Linear(self.in_features, self.out_features, bias=bias))

    def forward(self, x: torch.Tensor):
        outputs = []
        for i in range(self.num_experts):
            if proj_mode == "down":
                outputs.append(self.mixlora_A[i](x))
            else:
                outputs.append(self.mixlora_B[i](x))
        return outputs


class MixLoraGate(nn.Module):
    def __init__(self, num_experts: int, num_experts_per_tok: int, dim: int):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.dim = dim
        self.gate = nn.Linear(dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor):
        gate_logits = self.gate(x)
        weights, selected_experts = torch.topk(gate_logits, self.num_experts_per_tok)
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(x.dtype)
        return weights, selected_experts


class MixLoraLinear(nn.Module, MixLoraLayer):
    # Lora implemented in a dense layer
    # nn.Linear is the pretrained weights in LLM, MMOELoraLayer is the designed trainable Lora
    def __init__(
            self,
            base_layer,
            adapter_name: str,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            is_target_conv_1d_layer: bool = False,
            init_lora_weights: Union[bool, str] = True,
            use_rslora: bool = False,
            **kwargs,
    ):
        super().__init__()
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        self.num_experts = kwargs.pop("num_experts", True)
        self.num_experts_per_tok = kwargs.pop("num_experts_per_tok", True)
        self.expert_capacity = kwargs.pop("expert_capacity", True)
        self.num_tasks = kwargs.pop("num_tasks", True)

        MixLoraLayer.__init__(self,
                              base_layer,
                              self.num_experts,
                              self.num_experts_per_tok,
                              self.expert_capacity,
                              self.num_tasks,
                              **kwargs)


        # init the Gate network
        self.lora_gate = nn.ModuleDict({})
        self.lora_gate.update(nn.ModuleDict({adapter_name: MixLoraGate(self.num_experts, self.num_experts_per_tok, self.in_features)}))

        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora)
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                weights, selected_experts = self.lora_gate[active_adapter](x)
                for i in range(self.num_experts):
                    lora_A = self.lora_A[active_adapter].mixlora_A[i]
                    lora_B = self.lora_B[active_adapter].mixlora_B[i]
                    dropout = self.lora_dropout[active_adapter]
                    scaling = self.scaling[active_adapter]
                    x = x.to(lora_A.weight.dtype)
                    expert_mask = (selected_experts == i)
                    expert_weights = (weights * expert_mask).sum(dim=-1, keepdim=True)
                    result += lora_B(lora_A(dropout(x))) * scaling * expert_weights

        result = result.to(previous_dtype)
        return result

