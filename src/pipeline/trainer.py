import os
import torch

import torch.nn as nn
from torch.utils.data import Sampler
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, default_collate
from packaging import version
from transformers import Trainer, Seq2SeqTrainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    ShardedDDPOption,
    logger,
)
from typing import List, Optional, Dict, Union, Any



class TextToTextTrainer(Seq2SeqTrainer):

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        if self.sharded_ddp == ShardedDDPOption.SIMPLE:
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            clf_head_parameters = [name for name in decay_parameters if "classification_" in name]
            params = opt_model.named_parameters()
            
            # if "glue" not in self.args.data_paths[0]:
            #     optimizer_grouped_parameters = [
            #             {
            #                 "params": [
            #                     p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in clf_head_parameters and p.requires_grad)
            #                 ],
            #                 "weight_decay": self.args.weight_decay,
            #             },
            #             {
            #                 "params": [
            #                     p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in clf_head_parameters and p.requires_grad)
            #                 ],
            #                 "weight_decay": 0.0,
            #             },
            #             {
            #                 "params": [
            #                     p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in clf_head_parameters and p.requires_grad)
            #                 ],
            #                 "weight_decay": self.args.weight_decay,
            #                 "lr": 0,
            #             },
            #             {
            #                 "params": [
            #                     p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in clf_head_parameters and p.requires_grad)
            #                 ],
            #                 "weight_decay": 0.0,
            #                 "lr": 0,
            #             },
            #         ]
            # else:
            #     optimizer_grouped_parameters = [
            #         {
            #             "params": [
            #                 p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
            #             ],
            #             "weight_decay": self.args.weight_decay,
            #         },
            #         {
            #             "params": [
            #                 p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            #             ],
            #             "weight_decay": 0.0,
            #         },
            #     ]
                
            optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            logger.info(f"updated parameters:{optimizer_grouped_parameters}")

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    skipped = 0
                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                            logger.info(f"skipped {module}: {skipped/2**20}M params")
                            manager.register_module_override(module, "weight", {"optim_bits": 32})
                            logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                    logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer