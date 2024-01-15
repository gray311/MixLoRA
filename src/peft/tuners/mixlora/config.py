# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum
from dataclasses import dataclass, field
from typing import Optional, Union

from peft.tuners.lora import LoraConfig
from peft.utils import PeftType




@dataclass
class MixLoraConfig(LoraConfig):
    num_experts: Optional[int] = field(default=8, metadata={"help": "number of experts"})
    num_experts_per_tok: Optional[int] = field(default=2, metadata={"help": "number of experts per token"})
    expert_capacity: Optional[int] = field(default=64, metadata={"help": "capacity of each expert"})
    num_tasks: Optional[int] = field(default=1, metadata={"help": "number of tasks"})

    def __post_init__(self):
            self.peft_type = PeftType.MIXLORA


