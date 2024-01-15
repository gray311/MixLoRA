import os
import sys
import logging
import pathlib
import typing
import warnings

project_path = pathlib.Path(__file__).parent.parent
sys.path.append(str(project_path))

import torch
import transformers
from transformers import AutoConfig, AutoTokenizer, AutoConfig

from config.lora_config import MixLoRA_Config
from pipeline.utils import load_checkpoint
from model.moe_t5 import T5ForConditionalGeneration
from model.config import T5MoEConfig

from peft import (
    MixLoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

model_name_or_path = "/home/myz/models/t5-v11-base"
model = load_checkpoint(model_name_or_path)

# print(MixLoRA_Config)
config = MixLoraConfig(**MixLoRA_Config)
model = get_peft_model(model, config)

print(model)
# for n, p in model.named_parameters():
#     if p.requires_grad:
#         print(n, p.size())


model.print_trainable_parameters()





