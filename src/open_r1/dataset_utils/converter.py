import os
from abc import abstractmethod
from dataclasses import dataclass, Field
from typing import TYPE_CHECKING, Literal, Optional, Union, Any,Dict, List
from enum import Enum, unique
import re
from PIL import Image
import numpy as np
from ..configs import GRPOScriptArguments
from ..utils.logging import logging

# ViRL39K ships images as paths relative to the dataset root; resolve them under $DATA_ROOT.
VIRL39K_IMAGE_ROOT = os.path.join(
    os.environ.get("DATA_ROOT", os.path.abspath("./share_data")), "ViRL39K"
)

@dataclass
class DatasetAttr:
    r"""Dataset attributes."""

    # basic configs
    load_from: Literal["hf_hub", "ms_hub", "om_hub", "script", "file", "disk"]
    dataset_name: str
    file_ext: Literal["arrow", "parquet"]
    formatting: Literal["alpaca", "sharegpt", "ViRFT_COCO", "SAT", "GEOQA", "LISA", "SCIENCEQA", "GEOQAV", "RefCOCO",  "RefCOCORaw"] = "alpaca"
    ranking: bool = False
    # extra configs
    subset: Optional[str] = None
    split: str = "train"
    folder: Optional[str] = None
    num_samples: Optional[int] = None
    # common columns
    system: Optional[str] = None
    tools: Optional[str] = None
    images: Optional[str] = None
    videos: Optional[str] = None
    audios: Optional[str] = None
    # dpo columns
    chosen: Optional[str] = None
    rejected: Optional[str] = None
    kto_tag: Optional[str] = None
    # alpaca columns
    prompt: Optional[str] = "instruction"
    query: Optional[str] = "input"
    response: Optional[str] = "output"
    history: Optional[str] = None
    # sharegpt columns
    messages: Optional[str] = "conversations"
    # sharegpt tags
    role_tag: Optional[str] = "from"
    content_tag: Optional[str] = "value"
    user_tag: Optional[str] = "human"
    assistant_tag: Optional[str] = "gpt"
    observation_tag: Optional[str] = "observation"
    function_tag: Optional[str] = "function_call"
    system_tag: Optional[str] = "system"

    def __repr__(self) -> str:
        return self.dataset_name

    def set_attr(self, key: str, obj: dict[str, Any], default: Optional[Any] = None) -> None:
        setattr(self, key, obj.get(key, default))

    def join(self, attr: dict[str, Any]) -> None:
        self.set_attr("formatting", attr, default="alpaca")
        self.set_attr("ranking", attr, default=False)
        self.set_attr("subset", attr)
        self.set_attr("split", attr, default="train")
        self.set_attr("folder", attr)
        self.set_attr("num_samples", attr)

        if "columns" in attr:
            column_names = ["prompt", "query", "response", "history", "prompt", "system", "tools"]
            column_names += ["images", "videos", "audios", "chosen", "rejected", "kto_tag"]
            for column_name in column_names:
                self.set_attr(column_name, attr["columns"])

        if "tags" in attr:
            tag_names = ["role_tag", "content_tag"]
            tag_names += ["user_tag", "assistant_tag", "observation_tag", "function_tag", "system_tag"]
            for tag in tag_names:
                self.set_attr(tag, attr["tags"])
    




logger = logging.getLogger(__name__)


template = ""
'''
You are a helpful AI Assistant, designed to provided well-reasoned and detailed responses. 
You FIRST think about the reasoning process as an internal monologue and then provide the user with the answer. 
The reasoning process MUST BE enclosed within <think> and </think> tags."

A conversation between User and Assistant. 
The user asks a question about the image, and the Assistant solves it. 
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
''\nUser: {question} \nAssistant: Let me solve this step by step.\n<think>''

A conversation between User and Assistant. 
The user asks a question, and the Assistant solves it. 
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. 
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>",
'''


SYSTEM_PROMPT = {
    "ViRFT_COCO": template,  
    "ViRFT_COCO_SFT":  template,  
    "LISA_SFT":  template,  
    "SAT": template,
    "GEOQA": template,
    "LISA": template,
    "LISA_BBOX": template,
    "SCIENCEQA": template,
    "GEOQAV": template,
    "LLaVA_OneVision_OCR": template,
    "LLaVA_OneVision_OCR_10k": template,
    "LLaVA_OneVision_OCR_10k_128": template,
    
    "LLaVA_OneVision_OCR_5k": template,
    
    "RefCOCO": template,
    "RLAIF_V": template,
    "ViRL39K": template,
    "ThinkLite70k": template,
    "ThinkLite11k": template,
    
}

@dataclass
class DatasetConverter:
    dataset_attr: "DatasetAttr"
    data_args: "GRPOScriptArguments" #: "Field"
    system_prompt: tuple
    data_source: str # which dataset is this from

    def _find_medias(self, medias: Union[Any, list[Any]]) -> Optional[list[Any]]:
        r"""Optionally concatenate media path to media dir when loading from local disk."""
        if not isinstance(medias, list):
            medias = [medias] if medias is not None else []
        elif len(medias) == 0:
            return None
        else:
            medias = medias[:]

        if self.dataset_attr.load_from in ["script", "file"] and isinstance(medias[0], str):
            for i in range(len(medias)):
                if os.path.isfile(os.path.join(self.data_args.media_dir, medias[i])):
                    medias[i] = os.path.join(self.data_args.media_dir, medias[i])
                else:
                    logger.warning_rank0_once(f"Media {medias[i]} does not exist in `media_dir`. Use original path.")

        return medias

    @abstractmethod
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        r"""Convert a single example in the dataset to the standard format."""
        ...



def get_dataset_converter(name: str, dataset_attr: "DatasetAttr", data_args,#: "Field"
    ) -> "DatasetConverter":
    r"""Get a dataset converter."""
    if name not in DATASET_CONVERTERS:
        raise ValueError(f"Dataset converter {name} not found.")
    if name not in SYSTEM_PROMPT:
        raise ValueError(f"System Prompt {name} not found.")
    
    return DATASET_CONVERTERS[name](dataset_attr, data_args, SYSTEM_PROMPT[name], name)


@dataclass
class VirftcocoBboxDatasetConverter(DatasetConverter):
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        # Format into conversation
        def make_conversation(example):
            if not self.data_args.use_system_prompt:
                return {
                    "prompt": [
                        {"role": "user", "content": example["problem"]},
                    ],
                }

            return {
                "prompt": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": example["problem"]},
                ],
            }

        def make_conversation_image(example):
            if not self.data_args.use_system_prompt:
                return {
                    "prompt": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": example["problem"]},
                            ],
                        },
                    ],
                }
            else:
                return {
                    "prompt": [
                        {"role": "system", "content": self.system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": example["problem"]},
                            ],
                        },
                    ],
                }
        
        example["problem"] = example["problem"].replace("[{'Position': [x1, y1, x2, y2], 'Confidence': number}, ...]", "[x1, y1, x2, y2]")
        if "image" in example:
            logger.info_rank0("has image in dataset")
            output = make_conversation_image(example)
            output['image'] = example['image']
        else:
            logger.info_rank0("no image in dataset")
            output = make_conversation(example)
        #output['problem'] = example['problem']
        image_width, image_height = output['image'].size
        output['solution'] = eval(example['solution'].replace("<answer>", "").replace("</answer>", "").strip())
        output['solution'] = f"<answer>{output['solution'][0]['Position']}</answer><image_width>{image_width}</image_width><image_height>{image_height}</image_height>"
        output['data_source'] = self.data_source
        output["chosen"]=None
        output["rejected"]=None
        output["messages"]=None
        output["task"] = "rl"
        output['image_path'] = None
        print(output)
        return output

@dataclass
class VirftcocoDatasetConverter(DatasetConverter):
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        # Format into conversation
        def make_conversation(example):
            if not self.data_args.use_system_prompt:
                return {
                    "prompt": [
                        {"role": "user", "content": example["problem"]},
                    ],
                }

            return {
                "prompt": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": example["problem"]},
                ],
            }

        def make_conversation_image(example):
            if not self.data_args.use_system_prompt:
                return {
                    "prompt": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": example["problem"]},
                            ],
                        },
                    ],
                }
            else:
                return {
                    "prompt": [
                        {"role": "system", "content": self.system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": example["problem"]},
                            ],
                        },
                    ],
                }
        

        if "image" in example:
            logger.info_rank0("has image in dataset")
            output = make_conversation_image(example)
            output['image'] = example['image']
        else:
            logger.info_rank0("no image in dataset")
            output = make_conversation(example)
        #output['problem'] = example['problem']
        output['solution'] = example['solution']
        output['data_source'] = self.data_source
        output["chosen"]=None
        output["rejected"]=None
        output["messages"]=None
        output["task"] = "rl"
        output['image_path'] = None
        return output


@dataclass
class VirftcocoSFTDatasetConverter(DatasetConverter):

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        # Build message list
        messages: list[dict[str, Any]] = []
        if self.data_args.use_system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text":self.system_prompt}]})
        problem = example["problem"].strip().replace("thinking process in <think> </think> and", "").replace("<think> ... </think> ","")

 
        
        if "image" in example:
            logger.info_rank0("has image in dataset")
            # img_array = np.array(example['image'])  # shape is (H, W) or (H, W, C)
            # img_scaled = (img_array * 255).astype(np.uint8)
            # # Convert back to a PIL Image
            # img_255 = Image.fromarray(img_scaled)       
            user_content = [
                {"type": "image"},
                {"type": "text", "text": problem},
            ]
        else:
            user_content = problem
        messages.append({"role": "user", "content": user_content})
        solution = example.get("solution", "").strip()
        messages.append({"role": "assistant", "content": [{"type": "text", "text":solution}]})
        # Assemble output
        output = {
            "messages":      messages,
            "data_source": self.data_source,
        }
        if "image" in example:
            output["image"] = example["image"]
        output['solution'] = None
        output["chosen"]=None
        output["rejected"]=None
        output["task"] = "sft"
        output["prompt"] = None
        output['image_path'] = None
        return output

@dataclass
class RefCOCODatasetConverter(DatasetConverter):
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        # Format into conversation
        def make_conversation(example):
            if not self.data_args.use_system_prompt:
                return {
                    "prompt": [
                        {"role": "user", "content": example["problem"]},
                    ],
                }

            return {
                "prompt": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": example["problem"]},
                ],
            }

        def make_conversation_image(example):
            if not self.data_args.use_system_prompt:
                return {
                    "prompt": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": example["problem"]},
                            ],
                        },
                    ],
                }
            else:
                return {
                    "prompt": [
                        {"role": "system", "content": self.system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": example["problem"]},
                            ],
                        },
                    ],
                }
        # FIMXE: to reason or not to reason?
        # example["problem"] = example["problem"].replace('<image>', '').strip() + "\nOutput the final answer in <answer> </answer> tags." 
        example["problem"] = example["problem"].replace('<image>', '').strip() + "\nOutput the final answer in <answer> </answer> tags. The output answer format should be as follows:\n<answer>[x1, y1, x2, y2]</answer>\nPlease strictly follow the format." 
        # example["problem"] = example["problem"].replace('<image>', '').strip() + "\nOutput the thinking process in <think> </think> and final answer in <answer> </answer> tags. The output answer format should be as follows:\n<answer>[x1, y1, x2, y2]</answer>" 
        
        
        if "image_path" in example:
            logger.info_rank0(f"has image in dataset {example['image_path']}")
            output = make_conversation_image(example)
            output['image_path'] = example['image_path']
        else:
            logger.info_rank0("no image in dataset")
            output = make_conversation(example)
        # output['problem'] = example["problem"]
        output['solution'] = f"<answer>{example['solution'].strip()}</answer>"
        output['data_source'] = self.data_source
        output["chosen"]=None
        output["rejected"]=None
        output["messages"]=None
        output['image'] = None
        output["task"] = "rl"
        return output



@dataclass
class OCRDatasetConverter(DatasetConverter):
    """
    Converts your OCR printing OCR dataset into chat‐style messages for SFT.
    """
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        # build the message sequence
        messages: list[dict[str, Any]] = []
        if self.data_args.use_system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        for conv in example.get("conversations", []):
            # map "from" → role
            role = "user" if conv.get("from") == self.dataset_attr.user_tag else "assistant"
            if role=="user":
                content = [{"type": "image"}, {"type": "text", "text":conv["value"].replace('<image>', '').strip() + "\nOutput the final answer in <answer> </answer> tags."}]
            else:
                content = [{"type": "text", "text":f"<answer>{conv['value'].replace('<image>', '').strip()}</answer>"}]
            messages.append({"role": role, "content": content})
        assert len(messages) == 2, f"{messages}"
        assert messages[0]["role"] == "user", f"{messages}"
        assert messages[1]["role"] == "assistant", f"{messages}"
            
            

        output: dict[str, Any] = {
            "messages":     messages,
            "data_source":  self.data_source,
            "image": example["image"],
            "chosen":       None,
            "rejected":     None,
            "solution":     None,
            "task":         "sft",
            "prompt":       None,
            # 'problem': None,
            'image_path':None
        }



        return output


@dataclass
class SPAVLDatasetConverter(DatasetConverter):
    """
    Converts a SPA-VL DPO example into a chat conversation prompt plus chosen/rejected messagess:
      {
        "prompt":   List[Dict("role","content")],
        "chosen":   "<best answer>",
        "rejected": "<worst answer>",
        "image":    <image_path>   # if present
        "data_source": "SPA-VL"
      }
    """
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        # 1) Build the conversation list
        messages: list[dict[str, str]] = []
        if self.data_args.use_system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        # assume the SPA-VL question field is named "question"
        question = example.get("question") or example.get("prompt") or example.get("instruction")
        if question is None:
            raise KeyError("SPA-VL example missing a 'question' (or 'prompt'/'instruction') field")
        messages.append({"role": "user", "content": question.strip()})

        # 2) Extract the chosen and rejected texts
        if "chosen" not in example or "rejected" not in example:
            raise KeyError("SPA-VL DPO examples must have both 'chosen' and 'rejected' fields")
        chosen = example["chosen"].strip()
        rejected = example["rejected"].strip()

        # 3) Assemble output
        output: dict[str, Any] = {
            "prompt":      messages,
            "chosen":       [{"role": "assistant", "content": chosen}],
            "rejected":    [{"role": "assistant", "content": rejected}],
            "data_source": self.data_source,
        }
        # 4) Pass through image if available
        if "image" in example:
            output["image"] = example["image"]
        output['solution'] = None
        output["task"] = "dpo"
        output["messages"]=None
        output['image_path'] = None
        
        return output

@dataclass
class RLAIFVDatasetConverter(DatasetConverter):

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        # 1) Build the conversation list
        messages: list[dict[str, str]] = []
        if self.data_args.use_system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        question = example["question"]  + "\nOutput the final answer in <answer> </answer> tags."
        messages.append({"role": "user", "content": [
                                {"type": "image"},
                                {"type": "text", "text": question.replace('<image>', '').strip()},
                            ]})
        # 2) Extract the chosen and rejected texts
        chosen =  [{"type": "text", "text": f"<answer>{example['chosen'].strip()}</answer>"}]
        rejected = [{"type": "text", "text": f"<answer>{example['rejected'].strip()}</answer>"}] 

        # 3) Assemble output
        output: dict[str, Any] = {
            "prompt":      messages,
            "chosen":       [{"role": "assistant", "content": chosen}],
            "rejected":    [{"role": "assistant", "content": rejected}],
            "data_source": self.data_source,
            "image_path": example["image_path"]
        }
        # 4) Pass through image if available
        if "image" in example:
            output["image"] = example["image"]
        output['solution'] = None
        output["task"] = "dpo"
        output["messages"]=None
        
        return output

# main difference from above: manually add additonal GRPO-specific prompts to tail of each problem
@dataclass
class SATDatasetConverter(DatasetConverter):
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        QUESTION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer (option) in <answer> </answer> tags."        
        # Format into conversation
        def make_conversation(example):
            if not self.data_args.use_system_prompt:
                return {
                    "prompt": [
                        {"role": "user", "content": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                }            
            
            return {
                "prompt": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": QUESTION_TEMPLATE.format(Question=example["problem"])},
                ],
            }

        def make_conversation_image(example):
            if not self.data_args.use_system_prompt:
                return {
                    "prompt": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                            ],
                        },
                    ],
                }
            else:
                return {
                    "prompt": [
                        {"role": "system", "content": self.system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                            ],
                        },
                    ],
                }
            
        

        if "image" in example:
            logger.info_rank0("has image in dataset")
            output = make_conversation_image(example)
            output['image'] = example['image']
        else:
            logger.info_rank0("no image in dataset")
            output = make_conversation(example)
        #output['problem'] = example['problem']
        output['solution'] = example['solution']
        output['data_source'] = self.data_source
        output["chosen"]=None
        output["rejected"]=None
        output["task"] = "rl"
        output["messages"]=None
        output['image_path'] = None
        
        return output

@dataclass
class ViRL39KDatasetConverter(DatasetConverter):
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        QUESTION_TEMPLATE = "{Question}\nOutput the thinking process in <think> </think> and final answer in <answer> </answer> tags."        
        # Format into conversation
        def make_conversation(example):
            if not self.data_args.use_system_prompt:
                return {
                    "prompt": [
                        {"role": "user", "content": QUESTION_TEMPLATE.format(Question=example["question"])},
                    ],
                }            
            
            return {
                "prompt": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": QUESTION_TEMPLATE.format(Question=example["question"])},
                ],
            }

        def make_conversation_image(example):
            if not self.data_args.use_system_prompt:
                return {
                    "prompt": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["question"])},
                            ],
                        },
                    ],
                }
            else:
                return {
                    "prompt": [
                        {"role": "system", "content": self.system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["question"])},
                            ],
                        },
                    ],
                }
            
        example["question"] = example["question"].replace('<image>', '').strip()

        if "image_list" in example:            
            # logger.info_rank0(f"has image in dataset {example['image']}")
            output = make_conversation_image(example)
            output['image_path'] = os.path.join(VIRL39K_IMAGE_ROOT, example['image_list'][0])
        else:
            logger.info_rank0("no image in dataset")
            output = make_conversation(example)
        output['solution'] = example['answer'].strip()[7:-1]
        output['solution'] = f"<answer>{output['solution'].strip()}</answer>"
        print("Converted", example['answer'], "to", output['solution'])
        output['data_source'] = self.data_source
        output["chosen"]=None
        output["rejected"]=None
        output["task"] = "rl"
        output["messages"]=None
        output['image'] = None
        # print(output)
        return output


@dataclass
class ThinkLiteDatasetConverter(DatasetConverter):
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        QUESTION_TEMPLATE = "{Question}\nOutput the thinking process in <think> </think> and final answer in <answer> </answer> tags."        
        # Format into conversation
        def make_conversation(example):
            if not self.data_args.use_system_prompt:
                return {
                    "prompt": [
                        {"role": "user", "content": QUESTION_TEMPLATE.format(Question=example["question"])},
                    ],
                }            
            
            return {
                "prompt": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": QUESTION_TEMPLATE.format(Question=example["question"])},
                ],
            }

        def make_conversation_image(example):
            if not self.data_args.use_system_prompt:
                return {
                    "prompt": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["question"])},
                            ],
                        },
                    ],
                }
            else:
                return {
                    "prompt": [
                        {"role": "system", "content": self.system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["question"])},
                            ],
                        },
                    ],
                }
            
        example["question"] = example["problem"].replace('<image>', '').strip()

        if "image" in example:            
            logger.info_rank0(f"has image in dataset {example['image']}")
            output = make_conversation_image(example)
            output['image'] = example['image']
        else:
            logger.info_rank0("no image in dataset")
            output = make_conversation(example)
        output['solution'] = f"<answer>{str(example['ground_truth']).strip()}</answer>"
        # print("Converted", example['answer'], "to", output['solution'])
        output['data_source'] = self.data_source
        output["chosen"]=None
        output["rejected"]=None
        output["task"] = "rl"
        output["messages"]=None
        # print(output)
        return output
        
DATASET_CONVERTERS = {
    "ViRFT_COCO_SFT": VirftcocoSFTDatasetConverter,
    "ViRFT_COCO": VirftcocoDatasetConverter,
    "SAT": SATDatasetConverter,
    "GEOQA": SATDatasetConverter,
    "LISA_SFT": VirftcocoSFTDatasetConverter,
    "LISA": VirftcocoDatasetConverter, 
     "LISA_BBOX": VirftcocoBboxDatasetConverter, 
    "SCIENCEQA": SATDatasetConverter,
    "GEOQAV": SATDatasetConverter,
    "SPAVL": SPAVLDatasetConverter, 
    "RefCOCO": RefCOCODatasetConverter,
    "RLAIF_V": RLAIFVDatasetConverter,
    "LLaVA_OneVision_OCR":OCRDatasetConverter,
    "LLaVA_OneVision_OCR_10k":OCRDatasetConverter,
    "LLaVA_OneVision_OCR_10k_128":OCRDatasetConverter,
    
    "LLaVA_OneVision_OCR_5k":OCRDatasetConverter,
    "ThinkLite70k": ThinkLiteDatasetConverter, 
    "ThinkLite11k": ThinkLiteDatasetConverter, 
    
    "ViRL39K":ViRL39KDatasetConverter
}