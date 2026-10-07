##### Following design of LLaVA-NEXT
import os
import torch
import transformers
from transformers import (
    AutoConfig
)

from .qwen2vl import Qwen2VLModule
from .base import BaseModule
from ..utils.logging import logging

logger = logging.getLogger(__name__)



AVAILABLE_MODELS = {
    "none": BaseModule(),
    "qwen2vl": Qwen2VLModule(),
    #"llava_llama": "LlavaLlamaForCausalLM, LlavaConfig",
    #"llava_qwen": "LlavaQwenForCausalLM, LlavaQwenConfig",
    #"llava_mistral": "LlavaMistralForCausalLM, LlavaMistralConfig",
    #"llava_mixtral": "LlavaMixtralForCausalLM, LlavaMixtralConfig",
    # "llava_qwen_moe": "LlavaQwenMoeForCausalLM, LlavaQwenMoeConfig",    
    # Add other models as needed
}
#for model_name, model_classes in AVAILABLE_MODELS.items():
#    try:
#        exec(f"from .{model_name} import {model_classes}")
#    except Exception as e:
#        print(f"Failed to import {model_name} from .{model_name}. Error: {e}")


def get_model_processor(model_args, training_args):
    assert model_args.attn_implementation
    if model_args.attn_implementation == "sdpa" and torch.__version__ < "2.1.2":
        raise ValueError("The 'sdpa' attention implementation requires torch version 2.1.2 or higher.")

    customized_kwargs = dict()
    #customized_kwargs.update(bnb_model_from_pretrained_args)
    cfg_pretrained = None

    overwrite_config = {}
    if any(
        [
            model_args.rope_scaling_factor is not None,
            model_args.rope_scaling_type is not None,
            model_args.mm_spatial_pool_stride is not None,
            model_args.mm_spatial_pool_out_channels is not None,
            model_args.mm_spatial_pool_mode is not None,
            model_args.mm_resampler_type is not None,
        ]
    ):
        cfg_pretrained = AutoConfig.from_pretrained(model_args.model_name_or_path)

    if model_args.use_pos_skipping is not None and model_args.pos_skipping_range is not None:
        overwrite_config["use_pos_skipping"] = model_args.use_pos_skipping
        overwrite_config["pos_skipping_range"] = model_args.pos_skipping_range

    if model_args.rope_scaling_factor is not None and model_args.rope_scaling_type is not None:
        overwrite_config["rope_scaling"] = {
            "factor": model_args.rope_scaling_factor,
            "type": model_args.rope_scaling_type,
        }
        if training_args.model_max_length is None:
            training_args.model_max_length = cfg_pretrained.max_position_embeddings * model_args.rope_scaling_factor
            overwrite_config["max_sequence_length"] = training_args.model_max_length
        assert training_args.model_max_length == int(cfg_pretrained.max_position_embeddings * model_args.rope_scaling_factor), print(
            f"model_max_length: {training_args.model_max_length}, max_position_embeddings: {cfg_pretrained.max_position_embeddings}, rope_scaling_factor: {model_args.rope_scaling_factor}"
        )
        # overwrite_config["max_sequence_length"] = model_args.max_sequence_length
        # overwrite_config["tokenizer_model_max_length"] = model_args.tokenizer_model_max_length

    if model_args.mm_spatial_pool_stride is not None and model_args.mm_spatial_pool_out_channels is not None and model_args.mm_spatial_pool_mode is not None and model_args.mm_resampler_type is not None:
        overwrite_config["mm_resampler_type"] = model_args.mm_resampler_type
        overwrite_config["mm_spatial_pool_stride"] = model_args.mm_spatial_pool_stride
        overwrite_config["mm_spatial_pool_out_channels"] = model_args.mm_spatial_pool_out_channels
        overwrite_config["mm_spatial_pool_mode"] = model_args.mm_spatial_pool_mode

    if model_args.mm_spatial_pool_mode is not None:
        overwrite_config["mm_spatial_pool_mode"] = model_args.mm_spatial_pool_mode
    
    overwrite_config["attn_implementation"] = model_args.attn_implementation
    # Disable caching if gradient checkpointing is enabled (not supported)        
    overwrite_config["use_cache"] = (
        #False if training_args.gradient_checkpointing else True # this is set to True following open-r1
        False if training_args.gradient_checkpointing else model_args.use_cache
    )
    overwrite_config["torch_dtype"] = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    overwrite_config["low_cpu_mem_usage"] = False

    if overwrite_config:
        assert cfg_pretrained is not None, "cfg_pretrained is None"

        logger.info_rank0(f"Overwriting config with {overwrite_config}")
        for k, v in overwrite_config.items():
            setattr(cfg_pretrained, k, v)

        customized_kwargs["config"] = cfg_pretrained
    
    customized_kwargs["revision"] = model_args.model_revision
    customized_kwargs["trust_remote_code"] = model_args.trust_remote_code
    
    customized_kwargs["cache_dir"] = training_args.cache_dir
   
    
    if model_args.model_class_name in AVAILABLE_MODELS:
        # warning: for now we are not doing a check
        # it's possible that model_args.model_class_name and model_args.model_name_or_path does not match.
        model_attributes = AVAILABLE_MODELS[model_args.model_class_name]
        model = model_attributes.get_model(
            model_name_or_path=model_args.model_name_or_path,
            customized_kwargs=customized_kwargs
        )
        processor, pad_token_id = model_attributes.get_processor(
            model_args=model_args,
            training_args=training_args,
        )
            
    else:
        raise ValueError(f"Unknown model class {model_args}")
    
    # from now on is what originally outside of get_model function
    #if model_args.freeze_backbone:
    #   model.model.requires_grad_(False)
    
    # not doing enable_input_gradient stuff here,
    # as this would be handled later with GRPOTrainer
    
    
   
    
    return model_attributes, model, processor, pad_token_id, customized_kwargs


