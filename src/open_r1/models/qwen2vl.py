from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    PreTrainedTokenizerBase
)
import torch
from typing import Union, Any

from .base import BaseModule, logger
from ..data_utils import maybe_apply_chat_template

def listinstr(lst, s):
    assert isinstance(lst, list)
    for item in lst:
        if item in s:
            return True
    return False


class Qwen2VLModule(BaseModule):
        
        
    def get_model(self, model_name_or_path, customized_kwargs):
        logger.info_rank0(f"Using model class Qwen2VLForConditionalGeneration for {model_name_or_path}")
        
        '''import inspect
        signature = inspect.signature(Qwen2VLForConditionalGeneration.from_pretrained)
        for name, param in signature.parameters.items():
            print(f"{name}: {param.annotation} (default={param.default})")
        print(customized_kwargs)
        assert False, "Pause"
        '''
        if listinstr(['2.5', '2_5', 'qwen25', 'mimo'], model_name_or_path.lower()):
            MODEL_CLS = Qwen2_5_VLForConditionalGeneration
        else:
            MODEL_CLS = Qwen2VLForConditionalGeneration
        model = MODEL_CLS.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path,
            **customized_kwargs,
        )
        return model

    def get_processor(self, model_args, training_args):
        logger.info_rank0(f"Using processor class AutoProcessor for {model_args.model_name_or_path}")        
        processing_class = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            revision=model_args.model_revision,
            trust_remote_code=model_args.trust_remote_code,
        )
        for processing_keyword in self.get_custom_processing_keywords():
            setattr(processing_class, processing_keyword, getattr(model_args, processing_keyword))
        if getattr(processing_class, "tokenizer",  None) is not None:
            pad_token_id = processing_class.tokenizer.pad_token_id
            processing_class.pad_token_id = pad_token_id
            processing_class.bos_token_id = processing_class.tokenizer.bos_token_id
            processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
        else:
            assert isinstance(processing_class, PreTrainedTokenizerBase), "processing_class must be an instance of PreTrainedTokenizerBase if it has no tokenizer attribute"
            pad_token_id = processing_class.pad_token_id
        
            
        return processing_class, pad_token_id
    
    def post_model_init(self, model, processing_class):
        # InternVL is different
        pass
    
    def get_vision_modules_keywords(self):  
        return ['visual']
    
    def get_custom_multimodal_keywords(self):
        return ['pixel_values', 'image_grid_thw']

    def get_non_generate_params(self):
        return []
    
    def get_custom_processing_keywords(self):
        return ['max_pixels', 'min_pixels'] #, 'do_rescale']
    

    # After prepare_prompt, each item looks like
    """
    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    <|vision_start|><|image_pad|><|vision_end|>Detect all objects belonging to the category 'keyboard' in the image, and provide the bounding boxes (between 0 and 1000, integer) and confidence (between 0 and 1, with two decimal places).
    If no object belonging to the category 'keyboard' in the image, return 'No Objects'.
    Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.The output answer format should be as follows:
    <think> ... </think> <answer>[{'Position': [x1, y1, x2, y2], 'Confidence': number}, ...]</answer>
    Please strictly follow the format.<|im_end|>
    <|im_start|>assistant""" 
