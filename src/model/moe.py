import dataclasses
from typing import List

import torch
import torch.nn.functional as F
from torch import nn
from .config import T5MoEConfig


class MoELayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module, config: T5MoEConfig):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.config = config

    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        gate_logits = self.gate(hidden_states)
        weights, selected_experts = torch.topk(gate_logits, self.config.num_experts_per_tok)
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(hidden_states.dtype)

        final_hidden_states = None
        for i, expert in enumerate(self.experts):
            expert_mask = (selected_experts == i)
            expert_weights = (weights * expert_mask).sum(dim=-1, keepdim=True)
            current_hidden_states = expert(hidden_states).mul_(expert_weights)
            if final_hidden_states is None:
                final_hidden_states = current_hidden_states
            else:
                final_hidden_states.add_(current_hidden_states)

        return final_hidden_states.view(batch_size, sequence_length, hidden_dim)