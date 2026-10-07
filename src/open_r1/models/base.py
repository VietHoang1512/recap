from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase
)
import torch
from typing import Union, Any

from ..utils.logging import logging
from ..data_utils import maybe_apply_chat_template

logger = logging.getLogger(__name__)



DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"



class BaseModule:
    def get_model(self, model_name_or_path, customized_kwargs):
        logger.info_rank0(f"Using model class AutoModelForCausalLM from {model_name_or_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **customized_kwargs,
        )
        return model

    def get_processor(self, model_args, training_args):
        logger.info_rank0(f"Using processor class AutoTokenizer for {model_args.model_name_or_path}")        
        
        processor = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            revision=model_args.model_revision,
            trust_remote_code=model_args.trust_remote_code,
        )
        
        if training_args.chat_template is not None:
            processor.chat_template = training_args.chat_template
        elif processor.get_chat_template() is None:
            processor.chat_template = DEFAULT_CHAT_TEMPLATE
        
        if getattr(processor, "tokenizer",  None) is not None:
            pad_token_id = processor.tokenizer.pad_token_id
            processor.pad_token_id = pad_token_id
            processor.bos_token_id = processor.tokenizer.bos_token_id
            processor.eos_token_id = processor.tokenizer.eos_token_id
        else:
            assert isinstance(processor, PreTrainedTokenizerBase), "processor must be an instance of PreTrainedTokenizerBase if it has no tokenizer attribute"
            pad_token_id = processor.pad_token_id
            
        return processor, pad_token_id
    
    def get_vision_modules_keywords(self):  
        return []
    
    def get_custom_multimodal_keywords(self):
        return []

    def get_non_generate_params(self):
        return []
    
    def get_custom_processing_keywords(self):
        return []
    
     
    def prepare_prompt(self, processing_class, inputs: dict[str, Union[torch.Tensor, Any]]):
        prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"] for example in inputs]
        return prompts_text
    
    def prepare_model_inputs(self, processing_class, prompts_text, images, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False, return_additional_output=False):
        # This could only process pure-multimodal or pure-text inputs
        additional_output = None
        if len(images) > 0:
            prompt_inputs = processing_class(
                text=prompts_text,
                images=images,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
            additional_output = [{'image_grid_thw': image_grid_thw} for image_grid_thw in prompt_inputs['image_grid_thw']]
        else: # open-r1 used this
            prompt_inputs = processing_class(
                text=prompts_text,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        if return_additional_output:
            return prompt_inputs, additional_output
        return prompt_inputs
    def is_embeds_input(self):
        return False
        
