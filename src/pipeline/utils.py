import time
from contextlib import suppress

import os
import re
import torch
from tqdm import tqdm
from model.moe_t5 import T5ForConditionalGeneration
from model.config import T5MoEConfig

def save_state_dict(state_dict, output_dir, filename):
    for k in state_dict:
        state_dict[k] = state_dict[k].to(torch.float16).cpu()
    torch.save(state_dict, os.path.join(output_dir, filename))


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    return cast_dtype


def get_autocast(precision):
    if precision == "amp":
        return torch.cuda.amp.autocast
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    elif precision == "fp16":
        return lambda: torch.cuda.amp.autocast(dtype=torch.float16)
    else:
        return suppress


def get_checkpoint(model):
    state_dict = model.state_dict()

    for name, p in model.named_parameters():
        if not p.requires_grad:
            del state_dict[name]

    return state_dict


def load_checkpoint(model_name_or_path, from_scratch=True, from_hub=False, **kwargs):
    config = T5MoEConfig.from_pretrained(model_name_or_path)
    if config.is_moe and from_scratch:
        model = T5ForConditionalGeneration(config)
        checkpoint = torch.load(os.path.join(model_name_or_path, "pytorch_model.bin"))
        pattern = re.compile(r'\.feed_forward.experts\.\d+')
        for name, param in tqdm(model.state_dict().items(), desc="Initializing weigths of expert modules"):
            temp_name = name
            if "feed_forward.experts" in name:
                name = pattern.sub('', name)
            if name in checkpoint.keys():
                model.state_dict()[temp_name].data.copy_(checkpoint[name].data)
        if 'torch_dtype' in kwargs.keys():
            model.to(dtype=kwargs['torch_dtype'])
        if 'device_map' in kwargs.keys():
            model.to(device=kwargs['device_map'])
    else:
        model = T5ForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=kwargs['torch_dtype'] if 'torch_dtype' in kwargs.keys() else None,
            cache_dir=kwargs['cache_dir'] if 'cache_dir' in kwargs.keys() else None,
            device_map=kwargs['device_map'] if 'device_map' in kwargs.keys() else None,

        )
    return model


def filter_state_dict_to_trainable(model, state_dict):
    """
    Remove non-trainable parameters from model state dict.
    Exception: Embeddings will not be removed, even if frozen.
    This is because we need the new <image> <|endofchunk|> tokens to
    be consistent across initializations.
    """
    for (name, p,) in model.named_parameters():  # won't work for fsdp + use_orig_params=False
        if "fsdp" in name:
            continue
        if "embed" in name or isinstance(p, torch.nn.Embedding):
            continue
        if not p.requires_grad:
            name = name.replace("._checkpoint_wrapped_module", "")
            if name in state_dict:
                del state_dict[name]
            else:
                print(f"WARNING: filtering but {name} not in state_dict")

    # also remove the keys in state_dict generated from
    # lang_encoder.old_decoder_blocks and lang_encoder.gated_cross_attn_layers
    # because these are already saved in lang_encoder.model...
    to_delete = [
        n
        for n in state_dict.keys()
        if ("lang_encoder.base_model.model.old_decoder_blocks" in n)
        or ("lang_encoder.base_model.model.gated_cross_attn_layers" in n)
        or ("lang_encoder.base_model.old_decoder_blocks" in n)
        or ("lang_encoder.base_model.gated_cross_attn_layers" in n)
        or ("lang_encoder.old_decoder_blocks" in n)
        or ("lang_encoder.gated_cross_attn_layers" in n)
        or ("vision_encoder" in n)
        or ("word_embeddings" in n)
    ]
    for name in to_delete:
        del state_dict[name]
    return state_dict


def save_checkpoint(model, step, output_dir):
    """
    Save training checkpoint with model, optimizer, and lr_scheduler state.
    """

    model_state = model.state_dict()
    model_state = filter_state_dict_to_trainable(model, model_state)

    for k in model_state:
        model_state[k] = model_state[k].to(torch.float16).cpu()

    print(f"Saving checkpoint to {output_dir}/checkpoint.pt")
    torch.save(model_state, f"{output_dir}/checkpoint.pt")