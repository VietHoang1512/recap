import os
import warnings
from unittest.mock import patch
from collections import defaultdict, deque
from contextlib import nullcontext
from typing import Any, Callable, Optional, Sized, Union, Dict, List, Literal
from packaging import version
import math
import torch
from torch import autocast
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler, DataLoader
import torch.distributed as dist
import PIL
import numpy as np
from transformers import (
    Trainer,
    TrainerCallback,
    PreTrainedTokenizerBase,
    GenerationConfig,
    is_wandb_available,
    PreTrainedModel
)
import transformers
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.data_utils import (
    apply_chat_template,
    is_conversational,
    maybe_apply_chat_template,
)
from trl.trainer.dpo_config import FDivergenceConstants, FDivergenceType
from trl.trainer.utils import     add_bos_token_if_needed, add_eos_token_if_needed
import io
import base64
import random
from qwen_vl_utils import process_vision_info, fetch_image
if is_peft_available():
    from peft import PeftConfig, get_peft_model

from datasets import Dataset, IterableDataset

from ..models import BaseModule
from ..configs import GRPOConfig
from ..vllm_serve import vLLMScriptArguments, WeightSyncWorker
from ..utils.logging import logging
from ..utils.import_utils import is_deepspeed_available, is_rich_available, is_vllm_available, is_liger_kernel_available
#from ..utils.vllm_client import VLLMClient
from ..utils.profiling import profiling_context, profiling_decorator
from ..utils import selective_log_softmax, pad, pil_image_to_base64, print_prompt_completions_sample
from ..data_utils import is_conversational
from ..dataset_utils.processor import prepare_images
from ..rewards import data_source_reward_func_checker

if is_liger_kernel_available():
    from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
    

if is_deepspeed_available():
    import deepspeed

if is_wandb_available():
    import wandb
    import pandas as pd

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


logger = logging.getLogger(__name__)


def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    variance = torch.nanmean((tensor - torch.nanmean(tensor, keepdim=True)) ** 2)  # Compute variance ignoring NaNs
    count = torch.sum(~torch.isnan(tensor))  # Count of non-NaN values
    variance *= count / (count - 1)  # Bessel's correction
    return torch.sqrt(variance)

IGNORE_INDEX = -100
class MixedRepeatRandomSampler(Sampler):
    #  
    """
    Splits the dataset into 'task' groups, then at each step emits a micro-batch
    (of size batch_size//mini_repeat_count) from **one** task-group, cycling
    through all groups in random or round‑robin order.  RL indices within each
    micro-batch can optionally be repeated mini_repeat_count times.
    """
    def __init__(
        self,
        data_source: Sized,
        batch_size: int,
        mini_repeat_count: int,
        repeat_count: int = 1,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.batch_size = batch_size
        self.mini_repeat_count = mini_repeat_count
        self.repeat_count = repeat_count
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)

        # 1) Group indices by task
        self.groups: Dict[str, List[int]] = {}
        for idx, ex in enumerate(data_source):
            logger.debug_rank0("ex", ex)
            task = ex["task"]
            self.groups.setdefault(task, []).append(idx)

        # 2) Precompute micro-batch size
        if batch_size % mini_repeat_count != 0:
            raise ValueError(f"batch_size must be divisible by mini_repeat_count, got batch_size={batch_size}, mini_repeat_count={mini_repeat_count}")

    def __iter__(self):
        for _epoch in range(self.repeat_count):
            # 3) Shuffle each group separately
            permuted: Dict[str, List[int]] = {}
            for task, idxs in self.groups.items():
                permuted_idxs = torch.tensor(idxs)[
                    torch.randperm(len(idxs), generator=self.rng)
                ].tolist()
                permuted[task] = permuted_idxs

            # 4) Determine chunk boundaries for each group
            chunks: Dict[str, List[List[int]]] = {}
            for task, idxs in permuted.items():
                if task != "rl":
                    chunks[task] = [
                        idxs[i : i + self.mini_repeat_count]
                        for i in range(0, len(idxs), self.mini_repeat_count)
                        if len(idxs[i : i + self.mini_repeat_count]) == self.mini_repeat_count
                    ]
                else:
                    chunks[task] = [ [idxs[i]] for i in range(0, len(idxs))]                    

            # 5) Interleave one micro‑batch per task in round‑robin
            task_cycle = list(chunks.keys())
            # you could also shuffle task_cycle here if you want random order
            # torch.random.manual_seed(self.rng.initial_seed()); random.shuffle(task_cycle)
            # random.shuffle(task_cycle)
            

            pointers = {t: 0 for t in task_cycle}
            exhausted = False
            while not exhausted:
                exhausted = True
                random.shuffle(task_cycle)
                for task in task_cycle:
                    ptr = pointers[task]
                    if ptr < len(chunks[task]):
                        exhausted = False
                        batch_idxs = chunks[task][ptr]
                        pointers[task] += 1

                        # 6) Yield indices:
                        #    - for RL tasks: repeat each index mini_repeat_count times
                        #    - for other tasks: yield once each
                        if task == "rl":
                            for idx in batch_idxs:
                                for _ in range(self.mini_repeat_count):
                                    yield idx
                        else:
                            for idx in batch_idxs:
                                yield idx

    def __len__(self):
        total = 0
        for task, idxs in self.groups.items():
            if task == "rl":
                total += len(idxs) * self.mini_repeat_count
            else:
                num_full = (len(idxs) // self.mini_repeat_count)*self.mini_repeat_count                
                total += num_full 
        return total * self.repeat_count

class UniformHomogeneousMixedRepeatRandomSampler(Sampler):
    """
    Sampler that, on each iteration, picks one task:
      - draws world_size*batch_size examples from that task,
      - for 'rl' tasks repeats each index mini_repeat_count times,
      - then shards the global batch so each GPU sees batch_size examples.
    """

    def __init__(
        self,
        data_source: Sized,
        per_device_batch_size: int,
        mini_repeat_count: int,
        repeat_count: int = 1,
        seed: Optional[int] = None,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.data_source = data_source
        self.per_bs = per_device_batch_size
        self.mini_repeat = mini_repeat_count
        self.repeat_count = repeat_count
        self.rank = rank
        self.world_size = world_size
        self.global_bs = self.per_bs * self.world_size

        if self.global_bs % self.mini_repeat != 0:
            raise ValueError(
                f"global batch size ({self.global_bs}) must be divisible by "
                f"mini_repeat_count ({self.mini_repeat})"
            )

        # single, torch‑based RNG to stay in sync
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)

        # group indices by task
        self.groups: dict[str, List[int]] = defaultdict(list)
        self.source2task: dict[str, List[int]] = defaultdict(list)
        for i, ex in enumerate(data_source):
            self.groups[ex["data_source"]].append(i)
            assert self.source2task.get(ex["data_source"]) is None or self.source2task.get(ex["data_source"]) == ex["task"], f'ex["task"]={ex["task"]}, self.source2task[ex["data_source"]]={self.source2task[ex["data_source"]]}'
            self.source2task[ex["data_source"]] = ex["task"]
        logger.debug_rank0("Using uniform sampler")
        
        logger.debug_rank0("self.source2tasks", self.source2task)
        logger.debug_rank0("self.groups", {k:len(v) for k,v in self.groups.items()})
    def __iter__(self):
        # how many *unique* RL indices per global-batch
        rl_unique = self.global_bs // self.mini_repeat

        for _ in range(self.repeat_count):
            # 1) shuffle each group's indices
            permuted = {
                source: torch.tensor(idxs)[
                    torch.randperm(len(idxs), generator=self.rng)
                ].tolist()
                for source, idxs in self.groups.items()
            }

            # 2) build per‑source lists of global‑chunks
            chunks_by_source: dict[str, List[List[int]]] = {}
            for source, idxs in permuted.items():
                if self.source2task[source] == "rl":
                    # chop into uniq blocks, then repeat each idx mini_repeat times
                    uniq_blocks = [
                        idxs[i : i + rl_unique]
                        for i in range(0, len(idxs), rl_unique)
                        if len(idxs[i : i + rl_unique]) == rl_unique
                    ]
                    chunks_by_source[source] = [
                        [u for u in block for _ in range(self.mini_repeat)]
                        for block in uniq_blocks
                    ]
                else:
                    # chop directly into global_bs blocks
                    chunks_by_source[source] = [
                        idxs[i : i + self.global_bs]
                        for i in range(0, len(idxs), self.global_bs)
                        if len(idxs[i : i + self.global_bs]) == self.global_bs
                    ]

            # 3) determine how many full rounds we can do (until one task exhausts)
            num_rounds = min(len(chunks) for chunks in chunks_by_source.values())
            sources = list(chunks_by_source.keys())

            # 4) for each round, shuffle task order, then yield that task's i-th chunk
            for round_idx in range(num_rounds):
                # NEW: torch-based shuffle, same on all ranks
                # order = torch.randperm(len(tasks), generator=self.rng).tolist()
                for source  in sources :
                    # task = tasks[t_i]
                    global_chunk = chunks_by_source[source][round_idx]

                    start = self.rank * self.per_bs
                    end   = start + self.per_bs
                    local_chunk = global_chunk[start:end]

                    # yield per-device_batch_size indices (all same task)
                    for idx in local_chunk:
                        yield idx

    def __len__(self):
        # per‑task number of full global‑chunks
        rl_unique = self.global_bs // self.mini_repeat
        counts = []
        for source, idxs in self.groups.items():
            if self.source2task[source] == "rl":
                counts.append(len(idxs) // rl_unique)
            else:
                counts.append(len(idxs) // self.global_bs)
            logger.debug_rank0(source, counts[-1], "rounds")
        num_rounds = min(counts)
        # total samples per rank = rounds × tasks × per_device_batch_size
        return num_rounds * len(self.groups) * self.per_bs * self.repeat_count

class DefaultHomogeneousMixedRepeatRandomSampler(Sampler):
    """
    Sampler that, on each iteration, picks one task:
      - draws world_size*batch_size examples from that task,
      - for 'rl' tasks repeats each index mini_repeat_count times,
      - then shards the global batch so each GPU sees batch_size examples.
    """

    def __init__(
        self,
        data_source: Sized,
        per_device_batch_size: int,
        mini_repeat_count: int,
        repeat_count: int = 1,
        seed: Optional[int] = None,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.data_source = data_source
        self.per_bs = per_device_batch_size
        self.mini_repeat = mini_repeat_count
        self.repeat_count = repeat_count
        self.rank = rank
        self.world_size = world_size
        self.global_bs = self.per_bs * self.world_size
        self.seed = seed
        if self.global_bs % self.mini_repeat != 0:
            raise ValueError(
                f"global batch size ({self.global_bs}) must be divisible by "
                f"mini_repeat_count ({self.mini_repeat})"
            )

        # single, torch‑based RNG to stay in sync
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)
            random.seed(self.seed)

        # group indices by task
        self.groups: dict[str, List[int]] = defaultdict(list)
        self.source2task: dict[str, List[int]] = defaultdict(list)
        for i, ex in enumerate(data_source):
            self.groups[ex["data_source"]].append(i)
            assert self.source2task.get(ex["data_source"]) is None or self.source2task.get(ex["data_source"]) == ex["task"], f'ex["task"]={ex["task"]}, self.source2task[ex["data_source"]]={self.source2task[ex["data_source"]]}'
            self.source2task[ex["data_source"]] = ex["task"]
        logger.debug_rank0("Using default sampler")
        logger.debug_rank0("self.source2tasks", self.source2task)
        logger.debug_rank0("self.groups", {k:len(v) for k,v in self.groups.items()})
    def __iter__(self):
        # how many *unique* RL indices per global-batch
        rl_unique = self.global_bs // self.mini_repeat

        for _ in range(self.repeat_count):
            # 1) shuffle each group's indices
            permuted = {
                source: torch.tensor(idxs)[
                    torch.randperm(len(idxs), generator=self.rng)
                ].tolist()
                for source, idxs in self.groups.items()
            }
            all_chunks = []
            for source, idxs in permuted.items():
                if self.source2task[source] == "rl":
                    # chop into uniq blocks, then repeat each idx mini_repeat times
                    uniq_blocks = [
                        idxs[i : i + rl_unique]
                        for i in range(0, len(idxs), rl_unique)
                        if len(idxs[i : i + rl_unique]) == rl_unique
                    ]
                    all_chunks += [
                        [u for u in block for _ in range(self.mini_repeat)]
                        for block in uniq_blocks
                    ]
                else:
                    # chop directly into global_bs blocks
                    all_chunks += [
                        idxs[i : i + self.global_bs]
                        for i in range(0, len(idxs), self.global_bs)
                        if len(idxs[i : i + self.global_bs]) == self.global_bs
                    ]

            # 3) determine how many full rounds we can do (until one task exhausts)
            random.shuffle(all_chunks)
            self.all_chunks = all_chunks
            # 4) for each round, shuffle task order, then yield that task's i-th chunk
            for global_chunk in all_chunks:

                start = self.rank * self.per_bs
                end   = start + self.per_bs
                local_chunk = global_chunk[start:end]

                # yield per-device_batch_size indices (all same task)
                for idx in local_chunk:
                    yield idx

    def __len__(self):
        return len(self.all_chunks)

class MixedTrainer(Trainer):
    #_tag_names = ["trl", "grpo"]
    def __init__(
        self,
        model_name_or_path: str,
        customized_kwargs: dict,
        model_attributes: BaseModule,
        model: PreTrainedModel,
        processing_class: PreTrainedTokenizerBase, 
        pad_token_id,
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,        
        #processing_class: Optional[PreTrainedTokenizerBase] = None,
        #reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        vllm_args: Optional[vLLMScriptArguments] = None,
    ):
        self.model_attributes = model_attributes
        # logger.info_rank0(f"self.model_attributes, {self.model_attributes}")
        # logger.info_rank0(f"processing_class, {processing_class}")
        
        # this is different from trl's grpo_trainer
        assert args is not None, "must pass in training_args!"
        
        
        # if peft is called, wrap (lora?)
        if peft_config is not None:
            if not is_peft_available():
                raise ImportError("PEFT is required to use `peft_config`. Run `pip install peft`.")
            # originally, open-r1 directly use get_peft_model function
            # however, this does not exclude vision tower parameters during lora training
            def find_all_linear_names(model, multimodal_keywords):
                cls = torch.nn.Linear
                lora_module_names = set()
                for name, module in model.named_modules():
                    # LoRA is not applied to the vision modules
                    if any(mm_keyword in name for mm_keyword in multimodal_keywords):
                        continue
                    if isinstance(module, cls):
                        lora_module_names.add(name)
                for m in lora_module_names:  # needed for 16-bit
                    if "embed_tokens" in m:
                        lora_module_names.remove(m)
                return list(lora_module_names)
            target_modules = find_all_linear_names(model, self.model_attributes.vision_modules_keywords)
            peft_config.target_modules = target_modules            
            # then followed by get_peft_model after target_modules is set
            model = get_peft_model(model, peft_config)
        
        
        # other than peft, also possible to freeze a part of model
        if args.freeze_vision_modules:
            logger.info_rank0("Freezing vision modules...")
            for n, p in model.named_parameters():
                if any(keyword in n for keyword in self.model_attributes.get_vision_modules_keywords()):
                    p.requires_grad = False
        self.is_encoder_decoder = model.config.is_encoder_decoder       
        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)
            
        # Reference model
        self.beta = args.beta
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_deepspeed_zero3_enabled():
            #self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
            self.ref_model = self.model_attributes.get_model(model_name_or_path, customized_kwargs)
        elif is_peft_model(model):
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        else:
            #raise NotImplementedError("Not sure it's safe to use open_r1's create_reference_model function on VLMs!")
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)

        # moved vlm-r1 operations to be inside module for cleaner usage 
        assert processing_class is not None, "processing_class is not initialized!"
        
        # this would align model and processing_class
        # if there's some additional hyperparameter need to be set for self.model_attributes, this is where it happens
        self.model_attributes.post_model_init(model, processing_class)
        if self.ref_model is not None:
            # this would align ref model and processing_class
            # however the self.model_attributes hyperparameter is already set above, so would not set again
            self.model_attributes.post_model_init(self.ref_model, processing_class)
        
        # DPO
        self.dpo_max_prompt_length = args.dpo_max_prompt_length
        self.dpo_max_completion_length = args.dpo_max_completion_length
        # self.max_length = args.max_length
        # self.generate_during_eval = args.generate_during_eval
        self.label_pad_token_id = IGNORE_INDEX
        self.padding_value = args.padding_value if args.padding_value is not None else processing_class.pad_token_id
        self.pad_token_id = processing_class.pad_token_id
        self.dpo_max_prompt_length = args.dpo_max_prompt_length
        self.truncation_mode = args.truncation_mode
        # self.max_completion_length = max_completion_length
        # self.processing_class = processing_class        
        self.pref_beta = args.pref_beta
        self.loss_type = args.loss_type
        self.bco_gemma = args.pref_bco_weight
        self.ftx_gamma = args.pref_ftx
        if args.loss_type == "simpo":
            self.simpo_gamma = args.simpo_gamma
        self.ld_alpha = args.ld_alpha 
        self.softmax_temp = args.softmax_temp
        if model is not None and hasattr(model, "get_rope_index"):  # for qwen2vl mrope
            self.get_rope_func = model.get_rope_index  # transformers < 4.52.0 or qwen2.5 omni
        elif model is not None and hasattr(model, "model") and hasattr(model.model, "get_rope_index"):
            self.get_rope_func = model.model.get_rope_index  # transformers >= 4.52.0
        else:
            self.get_rope_func = None  
        logger.debug_rank0("self.get_rope_func",self.get_rope_func)
        # EMA
        self.loss_ema = defaultdict(
            lambda: torch.tensor(0.0, device=self.accelerator.device)
        )
        self.ema_alpha = args.ema_alpha
        # DWA
        self.iteration_window = args.iteration_window
        self.dwa_hist = defaultdict(
            lambda: deque(maxlen=2*self.iteration_window)
        )
        self.preference = args.preference
        self.adaptive_convergence_instablity_tradeoff = True
        self.convergence_instablity_tradeoff = -1
        if args.convergence_instablity_tradeoff>=0 and args.convergence_instablity_tradeoff<=1:
            self.convergence_instablity_tradeoff = args.convergence_instablity_tradeoff
            self.adaptive_convergence_instablity_tradeoff = False
        self.max_steps = args.max_steps
        self.mixed_sampler = args.mixed_sampler
        # Reward functions
        # reward_funcs is guaranteed to be a list of functions, not strings
        # reward_processing_classes is guaranteed to be unset
        self.reward_funcs = reward_funcs
        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)
        self.reward_processing_classes = [None] * len(reward_funcs)
        
        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features
        if args.max_length is None:
            warnings.warn(
                "`max_length` is not set in the CPOConfig's init"
                " it will default to `512` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_length = 512
        else:
            max_length = args.max_length        
        # Training arguments
        self.max_prompt_length = args.max_prompt_length # default is 512
        # VLM-R1 set self.max_prompt_length=None. why?
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.use_vllm = args.use_vllm
        self.use_liger_loss = args.use_liger_loss
        if self.use_liger_loss:
            raise NotImplementedError("Liger Loss not debugged! Would fail now")
        
        # Datasets
        if (
            isinstance(train_dataset, IterableDataset)
            or isinstance(eval_dataset, IterableDataset)
            or (
                isinstance(eval_dataset, dict) and any(isinstance(ds, IterableDataset) for ds in eval_dataset.values())
            )
        ):
            # See https://github.com/huggingface/trl/issues/3213
            raise NotImplementedError(
                "Iterable datasets are not yet supported in GRPOTrainer. Please use a standard dataset instead."
            )
 
        
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        
        # self.generation_config is not set as vllm condition hanld it later
        
        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        # Tracks the number of iterations (forward + backward passes), including those within a grad accum cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = [None] * args.gradient_accumulation_steps

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True
        
        if self.use_liger_loss:
             if not is_liger_kernel_available():
                 raise ImportError(
                     "Liger is required to use `liger_loss` as the GRPO loss. Run `pip install liger-kernel`."
                 )
             if is_peft_model(model):
                 raise ValueError("Liger loss is not supported with a PEFT model.")
 
             self.liger_grpo_loss = LigerFusedLinearGRPOLoss(
                 beta=self.beta,
                 epsilon_low=self.epsilon_low,
                 epsilon_high=self.epsilon_high,
                 temperature=self.temperature,
                 use_ref_model=self.ref_model is not None,
             )

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        # below are additional metrics than VLM-R1
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        self.num_completions_to_print = args.num_completions_to_print
        
        
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )
        
        # Check if the per_device_train/eval_batch_size * num processes can be divided by the number of generations
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        if self.num_generations < 2:
            raise ValueError(
                f"GRPO requires at least 2 generations per prompt to calculate the advantages. "
                f"You provided {self.num_generations}, which is less than the minimum required."
            )
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
                f"batch size, the valid values for the number of generations are: {possible_values}."
            )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
                    f"eval batch size, the valid values for the number of generations are: {possible_values}."
                )
        
        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)
        
        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )

            if self.accelerator.is_main_process:
                #self.vllm_client = VLLMClient(
                #    args.vllm_server_host, args.vllm_server_port, connection_timeout=args.vllm_server_timeout
                #)
                world_size_patch = patch(
                    "torch.distributed.get_world_size", return_value=1
                )
                profiling_patch = patch(
                    "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
                    return_value=None,
                )
                with world_size_patch, profiling_patch:
                    logger.debug_rank0(f"Rank: {self.accelerator.process_index}  VLLM ", model_name_or_path)
                    self.llm = LLM(
                        model=model_name_or_path, # use the same model to initialize
                        # revision=vllm_args.revision,
                        # tensor_parallel_size=vllm_args.tensor_parallel_size,
                        device=f"cuda:{self.accelerator.num_processes}",  # take the next GPU idx
                        gpu_memory_utilization=vllm_args.gpu_memory_utilization,
                        dtype=self.model.dtype,
                        # Automatic Prefix Caching caches the KV cache of existing queries, so that a new query can
                        # directly reuse the KV cache if it shares the same prefix with one of the existing queries.
                        # This is particularly useful here because we generate completions from the same prompts.
                        # enable_prefix_caching=vllm_args.enable_prefix_caching,
                        # enforce_eager=vllm_args.enforce_eager,
                        # max_model_len=vllm_args.max_model_len,
                        # max_num_seqs=vllm_args.max_num_seqs,
                        # max_num_batched_tokens=vllm_args.max_num_batched_tokens,
                        # max_num_batched_tokens=4096, # default was 512; too small for VLM?
                        # worker_cls=WeightSyncWorker,
                        #mm_processor_kwargs = ({
                        #    processing_keyword: getattr(self.processing_class, processing_keyword)
                        #    for processing_keyword in self.model_attributes.get_custom_processing_keywords()
                        #}),
                        # hf_overrides={processing_keyword: getattr(self.processing_class, processing_keyword) for processing_keyword in self.model_attributes.get_custom_processing_keywords()}
                    )
                # initialize communicator between processes
                # self.llm.collective_rpc("init_communicator", args=(vllm_args.host, vllm_args.port, vllm_args.tensor_parallel_size))
                # Set up the communication group for weight broadcasting
                    
                # Guided decoding, if enabled
                # vLLM specific sampling arguments
                # self.guided_decoding_regex = args.vllm_guided_decoding_regex
                # if self.guided_decoding_regex is not None:
                #     guided_decoding = GuidedDecodingParams(backend="outlines", regex=self.guided_decoding_regex)
                # else:
                #     guided_decoding = None

                # Sampling parameters
                # temperature=self.temperature,
                self.sampling_params = SamplingParams(
                    n=self.num_generations,
                    repetition_penalty=1. if self.repetition_penalty is None else self.repetition_penalty,
                    temperature=1. if self.temperature is None else self.temperature,
                    top_p=1. if self.top_p is None else self.top_p,
                    top_k=-1 if self.top_k is None else self.top_k,
                    min_p=0. if self.min_p is None else self.min_p,
                    max_tokens=self.max_completion_length,
                    # guided_decoding=guided_decoding,
                )


            

            self._last_loaded_step = -1 # not using 0 as the first model is not necessarily the model we initialized  # tag to avoid useless loading during grad accumulation

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                pad_token_id=processing_class.pad_token_id,
                # below are advanced controls not in VLM-R1
                bos_token_id=processing_class.bos_token_id,
                eos_token_id=processing_class.eos_token_id, # this used to be a special condition in VLM-R1
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
                repetition_penalty=self.repetition_penalty,
                cache_implementation=args.cache_implementation,
                # synced_gpus=True,
            )
        
        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        # this is not in VLM-R1
        # disable for now AttributeError: 'GRPOTrainer' object has no attribute '_tag_names'
        #self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                # this is defined in trl package, and would set model to eval() in the end
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        # this is not in VLM-R1
        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        # don't need to set these as reward_funcs are not PreTrainedModel
        #for i, reward_func in enumerate(self.reward_funcs):
        #    if isinstance(reward_func, PreTrainedModel):
        #        self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

        

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: GRPOConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        # Enable gradient checkpointing on the base model for PEFT
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()
        # Enable gradient checkpointing for non-PEFT models
        else:
            # this try-except is referred from VLM-R1
            try:
                model.gradient_checkpointing_enable()
            except:
                # For InternVL; these operations are copied from the original training script of InternVL
                model.language_model.config.use_cache = False
                model.vision_model.gradient_checkpointing = True
                model.vision_model.encoder.gradient_checkpointing = True
                model.language_model._set_gradient_checkpointing()
                # This line is necessary, otherwise the `model.gradient_checkpointing_enable()` will be executed during the training process, leading to an error since InternVL does not support this operation.
                args.gradient_checkpointing = False

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            model.enable_input_require_grads()

        return model
    
    
    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]
            
    # def _get_train_sampler(self, train_dataset=None) -> Sampler:
    #     eff_bs = (
    #         self.args.per_device_train_batch_size
    #         * self.accelerator.num_processes
    #         * self.args.gradient_accumulation_steps
    #     )
    #     return HomogeneousMixedRepeatRandomSampler(
    #         data_source=self.train_dataset,
    #         mini_repeat_count=self.num_generations,               # only RL uses this
    #         batch_size=eff_bs,            # same as before
    #         repeat_count=self.num_iterations,
    #         seed=self.args.seed,
    #     )
    
    def get_train_dataloader(self):
        # 1) grab the base dataset & batch_size
        dataset    = self.train_dataset
        batch_size = self.args.per_device_train_batch_size  * self.args.gradient_accumulation_steps

        # 2) define a sampler_fn that constructs your distributed, per‑epoch sampler
        if self.mixed_sampler == "uniform":
            sampler =  UniformHomogeneousMixedRepeatRandomSampler(
                    data_source=self.train_dataset,
                    per_device_batch_size=self.args.per_device_train_batch_size,
                    mini_repeat_count=self.num_iterations,
                    seed=self.args.seed,
                    rank=self.accelerator.process_index,
                    world_size=self.accelerator.num_processes
                )
        elif self.mixed_sampler == "default":
            sampler =  DefaultHomogeneousMixedRepeatRandomSampler(
                    data_source=self.train_dataset,
                    per_device_batch_size=self.args.per_device_train_batch_size,
                    mini_repeat_count=self.num_iterations,
                    seed=self.args.seed,
                    rank=self.accelerator.process_index,
                    world_size=self.accelerator.num_processes
                )
        else:
            raise NotImplementedError(f"mixed_sampler {self.mixed_sampler} is not supported")
        # 3) delegate to Trainer’s _get_dataloader, passing is_training=True
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=True,                        # keep all your global chunks aligned
            num_workers=self.args.dataloader_num_workers,
        )
    @profiling_decorator
    def _get_last_hidden_state(self, model, input_ids, attention_mask, logits_to_keep, custom_multimodal_inputs):
        # unwrap the model to access the model.model
        unwrapped_model = self.accelerator.unwrap_model(model)
        last_hidden_state = unwrapped_model.model(input_ids=input_ids, attention_mask=attention_mask, #**custom_multimodal_inputs
                                                  ).last_hidden_state
        last_hidden_state = last_hidden_state[:, :-1, :]  # (B, L-1, H)
        if logits_to_keep is not None:
            last_hidden_state = last_hidden_state[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
        return last_hidden_state
    
    # Get the per-token log probabilities for the completions for the model and the reference model
    @profiling_decorator
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep, custom_multimodal_inputs):
        # custom_multimodal_inputs is a kwargs
        # trl/grpo_trainer: basemodule wants logits_to_keep only
        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
        # Warning!!!! logits_to_keep is not supported yet
        logits = model(input_ids=input_ids, attention_mask=attention_mask, **custom_multimodal_inputs).logits
        #assert False, [logits.shape, input_ids.shape, attention_mask.shape, logits_to_keep]
        # AssertionError: [torch.Size([8, 735, 151936]), torch.Size([8, 735]), torch.Size([8, 735]), 91]
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        #input_ids = input_ids[:, -logits_to_keep:] # VLM-R1: fixed to be [:, 1:]
        #input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
        # See https://github.com/huggingface/trl/issues/2770
        assert logits_to_keep < input_ids.shape[1] # at least exclude the first input ID
        input_ids = input_ids[:, -logits_to_keep:]
        logits = logits[:, -logits_to_keep:]
        # Divide logits by sampling temperature.
        # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
        logits = logits / self.temperature
        return selective_log_softmax(logits, input_ids)  # compute logprobs for the input tokens
    
    @profiling_decorator
    def _move_model_to_vllm(self):
        # For DeepSpeed ZeRO-3, we need to gather all parameters before operations
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        gather_if_zero3 = deepspeed.zero.GatheredParameters if zero_stage_3 else nullcontext

        if is_peft_model(self.model):
            # With PEFT and DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging, as merging
            # adapters in a sharded manner is not supported.
            with gather_if_zero3(list(self.model.parameters())):
                self.model.merge_adapter()
                state_dict = {}
                # Update vLLM weights while parameters are gathered
                for name, param in self.model.named_parameters():
                    # When using PEFT, we need to recover the original parameter name and discard some parameters
                    name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                    if self.model.prefix in name:
                        continue
                    # When module to save, remove its prefix and discard the original module
                    if "original_module" in name:
                        continue
                    name = name.replace("modules_to_save.default.", "")
                    state_dict[name] = param.data

                    #if self.accelerator.is_main_process:
                        #self.vllm_client.update_named_param(name, param.data)
                        #self.llm.collective_rpc("update_named_param_ours", args=(name, param.data))
                if self.accelerator.is_main_process:
                    self.llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(
                        state_dict.items())
                
                

                # Unmerge adapters while parameters are still gathered
                self.model.unmerge_adapter()
                # Parameters will automatically be repartitioned when exiting the context
        else:
            with gather_if_zero3(list(self.model.parameters())):
                state_dict = {}
                for name, param in self.model.named_parameters():
                    state_dict[name] = param.data
                if self.accelerator.is_main_process:
                    self.llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(
                        state_dict.items())
            '''
            # For non-PEFT models, simply gather and update each parameter individually.
            for name, param in self.model.named_parameters():
                with gather_if_zero3([param]):
                    if self.accelerator.is_main_process:
                        #self.vllm_client.update_named_param(name, param.data)
                        self.llm.collective_rpc("update_named_param_ours", args=(name, param.data))
            '''

        # Reset cache on main process
        if self.accelerator.is_main_process:
            self.llm.llm_engine.reset_prefix_cache()
            #self.vllm_client.reset_prefix_cache()

    @profiling_decorator
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        # this part of generating score completion is originally in compute_loss; 
        # moved here to further support evaluation (other than training)
        mode = "eval" if not self.is_in_train else "train"
        
        
        inputs = [{k:v for k,v in example.items() if v is not None} for example in inputs]  
        # print(f"Rank: {self.accelerator.process_index}  "
        # f"(local rank: {self.accelerator.local_process_index}, "
        # f"world size: {self.accelerator.num_processes})", "inputs", inputs)
        tasks = [ex["task"] for ex in inputs]
        assert len(set(tasks)) == 1, "The trainer can not support mixed inputs per device for now"
        sources = [ex["data_source"] for ex in inputs]
        assert len(set(sources)) == 1, "The trainer can not support mixed inputs per device for now"        
        task = tasks[0]        
        source = sources[0]
       
            # self.accelerator.wait_for_everyone()
        if task=="rl":

            ctx = (
                unwrap_model_for_generation(
                    self.model_wrapped,
                    self.accelerator,
                    gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                )
                
            )            
            with ctx as unwrapped_model:
                logger.debug_rank0(f"Rank: {self.accelerator.process_index}  PREPARING RL")
                if mode == "train":            
                    buffer_index = self._step % self.args.gradient_accumulation_steps
                    buffered_inputs = self._buffered_inputs[buffer_index]
                    if self.state.global_step % self.num_iterations == 0 or buffered_inputs is None:
                        # buffered_inputs=None can occur when resuming from a checkpoint
                        inputs = self._prepare_inputs_grpo(inputs, unwrapped_model)
                        self._buffered_inputs[buffer_index] = inputs
                    else:
                        inputs = buffered_inputs
                    self._step += 1
                else:
                    # In evaluation, we don't reuse completions across multiple updates, so we don't need to buffer inputs.
                    inputs = self._prepare_inputs_grpo(inputs, unwrapped_model)
                # self.accelerator.wait_for_everyone() 
                logger.debug_rank0(f"Rank: {self.accelerator.process_index}  PREPARED RL")
        
        if task=="sft":
            logger.debug_rank0(f"Rank: {self.accelerator.process_index}  PREPARING SFT")
            inputs = self._prepare_inputs_sft(inputs)
            logger.debug_rank0(f"Rank: {self.accelerator.process_index}  PREPARED SFT")
        if task=="dpo":
            logger.debug_rank0(f"Rank: {self.accelerator.process_index}  PREPARING DPO")
            inputs = self._prepare_inputs_dpo(inputs)
            logger.debug_rank0(f"Rank: {self.accelerator.process_index}  PREPARED DPO")
            # else:
            #     raise NotImplementedError(f"Task {task} is not supported")
        # torch.distributed.barrier()
        # self.accelerator.wait_for_everyone()
        # torch.cuda.synchronize()
        logger.debug_rank0(f"Rank: {self.accelerator.process_index}  FINISHED PREPARING")
        # print("Final inputs", inputs)
        # for k, v in inputs.items():
        #     try:
        #         print(k, v.shape)
        #     except:
        #         print(k, v)
        #         pass
        inputs["data_source"] = source
        inputs["task"] = task
        self.accelerator.wait_for_everyone()
        return inputs




    def _ensure_assistant_msgs(self, x):
        # chosen/rejected may be str, dict, or list of messages
        if isinstance(x, str):
            return [{"role": "assistant", "content": [{"type": "text", "text": x}]}]
        if isinstance(x, dict):
            return [x]
        if isinstance(x, list):
            return x
        raise ValueError(f"Unsupported chosen/rejected format: {type(x)}")
    def build_tokenized_answer(self, prompt, answer, images):
        """
        Llama tokenizer does satisfy `enc(a + b) = enc(a) + enc(b)`. It does ensure `enc(a + b) = enc(a) + enc(a +
        b)[len(enc(a)):]`. Reference:
            https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
        """
        # print(len(images), "prompt", prompt)

        prompt_input_ids = self.processing_class(text=prompt, images=images, add_special_tokens=False)["input_ids"][0]
        full_tokenized = self.processing_class(text=answer, images=images,  add_special_tokens=False)
        # FIXME: Hack
        full_tokenized = {k:v[0] for k,v in full_tokenized.items()}
        logger.debug_rank0("full_tokenized", full_tokenized)
        answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids) :]
        answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids) :]
        # print('full_tokenized["input_ids"]', full_tokenized["input_ids"])
        # print("answer_input_ids", answer_input_ids)
        # print("prompt_input_ids", prompt_input_ids)
        # Concat tokens to form `enc(a) + enc(a + b)[len(enc(a)):]`
        full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])

        # Prepare input tokens for token by token comparison
        full_input_ids = np.array(full_tokenized["input_ids"])

        if len(full_input_ids) != len(full_concat_input_ids):
            raise ValueError("Prompt input ids and answer input ids should have the same length.")

        # On some tokenizers, like Llama-2 tokenizer, there are occasions where tokens
        # can be merged together when tokenizing prompt+answer. This could result
        # on the last token from the prompt being different when tokenized on its own
        # vs when done as prompt+answer.
        response_token_ids_start_idx = len(prompt_input_ids)

        # If tokenized prompt is different than both prompt+answer, then it means the
        # last token has changed due to merging.
        if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
            response_token_ids_start_idx -= 1

        prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
        prompt_attention_mask = full_tokenized["attention_mask"][:response_token_ids_start_idx]

        if len(prompt_input_ids) != len(prompt_attention_mask):
            raise ValueError("Prompt input ids and attention mask should have the same length.")

        answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
        answer_attention_mask = full_tokenized["attention_mask"][response_token_ids_start_idx:]

        return dict(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            input_ids=answer_input_ids,
            attention_mask=answer_attention_mask,
            **{k:full_tokenized[k] for k in self.model_attributes.get_custom_multimodal_keywords()}
        )

    @profiling_decorator
    def _prepare_inputs_dpo(self, inputs):
        processor, tokenizer = self.processing_class, self.processing_class.tokenizer
        examples = []
        prepared_images = []
    
        # prepared_images = prepare_images(inputs)

        # mm_inputs = processor.image_processor(images=prepared_images+prepared_images,             
        #                                                 return_tensors="pt")

        # 1) Build texts via chat template
        prepared_images = prepare_images(inputs)
        for idx, input in enumerate(inputs):

            output = {}
            logger.debug_rank0("input", input)
            
            input.update(maybe_apply_chat_template(input, self.processing_class))
            logger.debug_rank0("process input", input)
            
            # processed_features = processor(images=ex["image"], text=ex["prompt"], add_special_tokens=False)
            processed_features = processor(images=[prepared_images[idx]], text=input["prompt"], add_special_tokens=False)
        
            
            # FIXME: Check other MM inputs
            logger.debug_rank0("processed_features", processed_features)
            prompt_ids = processed_features["input_ids"][0]
            # print('processed_features["pixel_values"]', processed_features["pixel_values"])
            # pixel_values = processed_features["pixel_values"][0]
            chosen_ids = tokenizer(input["chosen"], add_special_tokens=False)["input_ids"]
            rejected_ids = tokenizer(input["rejected"], add_special_tokens=False)["input_ids"]

            # FIXME: Check this
            chosen_ids = chosen_ids + [tokenizer.eos_token_id]
            rejected_ids = rejected_ids + [tokenizer.eos_token_id]

            # Truncate prompt and completion sequences
            # if self.dpo_max_prompt_length is not None:
            #     prompt_input_ids = prompt_input_ids[:self.dpo_max_prompt_length]
            # if self.dpo_max_completion_length is not None:
                # chosen_input_ids = chosen_input_ids[:self.dpo_max_completion_length]
                # rejected_input_ids = rejected_input_ids[:self.dpo_max_completion_length]
            source_len, target_len = infer_seqlen(
                len(prompt_ids), max(len(chosen_ids), len(rejected_ids)), self.dpo_max_prompt_length
            )
            prompt_ids = prompt_ids[:source_len]
            chosen_ids = chosen_ids[:target_len]
            rejected_ids = rejected_ids[:target_len]
            chosen_input_ids = prompt_ids + chosen_ids
            chosen_labels = [IGNORE_INDEX] * source_len + chosen_ids
            rejected_input_ids = prompt_ids + rejected_ids
            rejected_labels = [IGNORE_INDEX] * source_len + rejected_ids
            
            output = {
                # "pixel_values": processed_images["pixel_values"][idx],
                # "image_grid_thw": processed_images["image_grid_thw"][idx],
                
                "chosen_input_ids": chosen_input_ids,
                "rejected_input_ids": rejected_input_ids,
                "chosen_labels": chosen_labels,
                "rejected_labels": rejected_labels,
            }
            # output["chosen_attention_mask"] = [1] * len(output["chosen_input_ids"])
            # output["rejected_attention_mask"] = [1] * len(output["rejected_input_ids"])
            
            # if "pixel_attention_mask" in processed_features:
            #     output["pixel_attention_mask"] = processed_features["pixel_attention_mask"][0]
            # if "image_sizes" in processed_features:
            #     output["image_sizes"] = processed_features["image_sizes"][0]
            # if 'image_grid_thw'in processed_features:
            #     output["image_grid_thw"] = processed_features["image_grid_thw"][0]
            examples.append(output)
        
        concatenated_features = []
        for key in ("chosen","rejected"):
            for feature in examples:
                target_feature = {
                    "input_ids": feature[f"{key}_input_ids"],
                    # "attention_mask": feature[f"{key}_attention_mask"],
                    "labels": feature[f"{key}_labels"],
                    # "images": feature["images"],
                    # "pixel_values": feature["pixel_values"],
                    # "image_grid_thw": feature["image_grid_thw"],
                    
                }
                # if "pixel_attention_mask" in feature:
                #     target_feature["pixel_attention_mask"] = feature["pixel_attention_mask"]
                # if "image_sizes" in feature:
                #     target_feature["image_sizes"] = feature["image_sizes"]      
                         
                concatenated_features.append(target_feature)

        # Convert to tensor
        input_ids = [torch.tensor(concatenated_feature["input_ids"]) for concatenated_feature in concatenated_features]
        attention_mask = [torch.ones_like(input_ids) for input_ids in input_ids]

        # FIXME: Check padding side!!!
        # Pad
        padding_side = tokenizer.padding_side
        # padding_side = "left"
        batch = {}
        # batch["input_ids"] = pad(input_ids, padding_value=self.pad_token_id, padding_side=padding_side)
        # batch["attention_mask"] = pad(attention_mask, padding_value=0, padding_side=padding_side)
        batch["input_ids"] = input_ids
        batch["attention_mask"] = attention_mask
        # labels=batch["labels"]
        batch = pad_without_fast_tokenizer_warning(tokenizer, batch, padding=True, max_length=None, return_tensors="pt")
        # prepare decoder_input_ids
        # if (
        #     labels is not None
        #     and self.model is not None
        #     and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        # ):
        #     decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=batch["labels"])
        #     batch["decoder_input_ids"] = decoder_input_ids
        # if "pixel_values" in concatenated_features[0]:
        #     # batch["pixel_values"] = pad(pixel_values, padding_value=0.0)
        #     batch["pixel_values"] = torch.tensor([concatenated_feature["pixel_values"] for concatenated_feature in concatenated_features])
        # if "pixel_attention_mask" in concatenated_features[0]:
        #     # batch["pixel_attention_mask"] = pad(pixel_attention_mask, padding_value=0)
        #     batch["pixel_attention_mask"] = [concatenated_feature["pixel_attention_mask"] for concatenated_feature in concatenated_features]
            
        # if "image_sizes" in concatenated_features[0]:
        #     batch["image_sizes"] = torch.tensor([concatenated_feature["image_sizes"] for concatenated_feature in concatenated_features])
        # if "image_grid_thw" in concatenated_features[0]:
        #     batch["image_grid_thw"] = torch.tensor([concatenated_feature["image_grid_thw"] for concatenated_feature in concatenated_features])            
        mm_inputs = processor.image_processor(images=prepared_images+prepared_images, return_tensors="pt")
        # FIXME:
        self.get_rope_func = None
        if self.get_rope_func is not None:
            
            rope_index_kwargs = {
                "input_ids": batch["input_ids"],
                "image_grid_thw": mm_inputs.get("image_grid_thw"),
                "video_grid_thw": mm_inputs.get("video_grid_thw"),
                "attention_mask": (batch["attention_mask"] >= 1).float(),
            }
            if "second_per_grid_ts" in mm_inputs:  # for qwen2vl
                rope_index_kwargs["second_per_grid_ts"] = mm_inputs.get("second_per_grid_ts")
            elif "video_second_per_grid" in mm_inputs:  # for qwen2.5 omni
                rope_index_kwargs["second_per_grids"] = mm_inputs.get("video_second_per_grid")

            if getattr(self.model.config, "model_type", None) == "qwen2_5_omni_thinker":  # for qwen2.5 omni
                rope_index_kwargs["use_audio_in_video"] = getattr(self.processor, "use_audio_in_video", False)
                feature_attention_mask = mm_inputs.get("feature_attention_mask", None)
                if feature_attention_mask is not None:  # FIXME: need to get video image lengths
                    audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
                    rope_index_kwargs["audio_seqlens"] = audio_feature_lengths  # prepare for input

                batch["position_ids"], rope_deltas = self.get_rope_func(**rope_index_kwargs)
                batch["rope_deltas"] = rope_deltas - (1 - rope_index_kwargs["attention_mask"]).sum(
                    dim=-1
                ).unsqueeze(-1)
            else:  # for qwen2vl
                batch["position_ids"], batch["rope_deltas"] = self.get_rope_func(**rope_index_kwargs)
        
        # for k, v in mm_inputs.items():
            # print("mm_inputs", k, v.shape)
        # print("mm_inputs", mm_inputs)
        batch.update(**mm_inputs)
        labels = [concatenated_feature["labels"] for concatenated_feature in concatenated_features]
        max_label_length = max(len(l) for l in labels)
        logger.debug_rank0("padding_side", padding_side)
        batch["labels"] = [
            label + [self.label_pad_token_id] * (max_label_length - len(label))
            if padding_side == "right"
            else  [self.label_pad_token_id] * (max_label_length - len(label)) + label
            for label in labels
        ]        
        batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)
        batch = super()._prepare_inputs(batch)
        return batch
   
            
    def _prepare_inputs_sft(self, examples):
        texts = [
            self.processing_class.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=True)
            for example in examples
        ]
        image_inputs = []
        for example in examples:
            # imgs, vids = process_vision_info(example["messages"])
            imgs = fetch_image(example)
            image_inputs.append(imgs)
        batch = self.processing_class(text=texts,
                                    images=image_inputs,
                                    return_tensors="pt",
                                    padding=True)
        labels = batch["input_ids"].clone()
        labels[labels == self.processing_class.tokenizer.pad_token_id] = -100
        image_token_id = self.processing_class.tokenizer.convert_tokens_to_ids(self.processing_class.image_token)
        labels[labels == image_token_id] = -100
        batch["labels"] = labels
        # print("batch", batch)
        batch = super()._prepare_inputs(batch)

        return batch    
    def _prepare_inputs_grpo(
        self, inputs: dict[str, Union[torch.Tensor, Any]], unwrapped_model
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        # get raw prompts
        prompts = [x["prompt"] for x in inputs]
        data_sources = [x["data_source"] for x in inputs]

        prompts_text = [
        maybe_apply_chat_template(example, self.processing_class)["prompt"]
        for example in inputs
        ]
        # process raw prompts (text part) into tokenzied text str
        prepared_prompts_text = self.model_attributes.prepare_prompt(self.processing_class, inputs)
        # get image(list) of PIL.PngImagePlugin.PngImageFile
        prepared_images = prepare_images(inputs)
    
        prompt_inputs, additional_output = self.model_attributes.prepare_model_inputs(
            self.processing_class,
            prepared_prompts_text,
            prepared_images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
            return_additional_output=True,
        )
        # prompts_text: a list of str
        # images: a list of PIL.PngImagePlugin.PngImageFile
        # prompt_inputs: transformers.feature_extraction_utils.BatchFeature
        #     dict_keys(['input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw']) 
        
        # prepare device and dtype for prompt_inputs. Do nothing else!
        prompt_inputs = super()._prepare_inputs(prompt_inputs)


        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        # L: depends on prompt length
        # an example with L=644 and per_device_train_batch_size=8:
        # prompt_ids: <class 'torch.Tensor'>, torch.Size([8, 644]), per_device_train_batch_size x L 
        # prompt_mask: <class 'torch.Tensor'>, torch.Size([8, 644]), per_device_train_batch_size x L
            
        # image_grid_thw may be needed for the reward function
        if additional_output is not None:
            assert len(additional_output) == len(inputs)
            for i, (input_i, additional_output_i) in enumerate(zip(inputs, additional_output)):
                input_i.update(additional_output_i)


        # max_prompt_length is only disabled in VLM-R1. Why?
        if self.max_prompt_length is not None:
            assert self.args.use_vllm, "Non-VLLM not supporting this option yet"
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]        
        
            
        # Generate completions using either vLLM or regular generation
        if self.args.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step
            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            all_prompts_text = gather_object(prompts_text)
            '''
            all_image_embeds = gather_object(image_embeds)
            all_image_grid_thws = gather_object(image_grid_thw)
            
            image_urls = pil_images_to_data_urls(images)
            all_image_urls = gather_object(image_urls)
            '''
            #if isinstance(images[0], PIL.PngImagePlugin.PngImageFile):
            #    images = [np.asarray(x).tolist() for x in images]
            #    #images = [pil_image_to_base64(x) for x in images]
            images = [x["image"] for x in inputs]
            
            all_images = gather_object(images)

            ################################################################      
            # https://docs.vllm.ai/en/latest/serving/multimodal_inputs.html
            all_multimodal_inputs = []
            '''
            for prompt, image_embed, image_grid_thw in zip(all_prompts_text, all_image_embeds, all_image_grid_thws):
                all_multimodal_inputs.append({
                    "image_embeds": image_embed,#.detach().cpu().tolist(), 
                    "image_grid_thw": image_grid_thw,#.detach().cpu().tolist(),
                    })
            
            for prompt, image_url in zip(all_prompts_text, all_image_urls):
                all_multimodal_inputs.append({
                    "image": image_url,
                    "mm_processor_kwargs": {k: v.detach().cpu().tolist() for k, v in multimodal_inputs.items()}
                })
            '''
            all_multimodal_inputs = [
                {"prompt": p, 
                 "multi_modal_data": {
                    "image": i
                    }}
                for p, i in zip(all_prompts_text, all_images)
            ]
            #for item in all_multimodal_inputs:
                #assert False, ("Prompt Length: ", len(item["prompt"]), ", Image shape: ", item["multi_modal_data"]["image"].shape)
            '''
            for prompt, image in zip(all_prompts_text, all_images):
                all_multimodal_inputs.append({
                    "image": image,
                    "mm_processor_kwargs": {k: v.detach().cpu().tolist() for k, v in multimodal_inputs.items()}
                })
            '''
            ################################################################      
            
            # 1) Batch‐process both text and images so HF inserts the right number of <image> tokens:
            
            # print("all_prompts_text", all_prompts_text)
            # print("all_images", all_images)




                                      
            #all_images = gather_object(images) # this is added for multimodal
            #.... # help me fill in this part of code to have right all_multimodal_inputs
            #all_multimodal_inputs = gather_object(prompt_inputs)
            #assert False, [self.accelerator.process_index, len(all_multimodal_inputs), type(all_multimodal_inputs[0])] 
            if self.accelerator.is_main_process:
                # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                # prompt individually.
                # len(all_multimodal_inputs): 12
                # self.num_generations: 8
                # len(all_multimodal_inputs[:: self.num_generations]): 2

                #ordered_set_of_prompts = all_prompts_text[:: self.num_generations]
                ordered_set_of_multimodal_inputs = all_multimodal_inputs[:: self.num_generations]
                #assert False, [len(all_multimodal_inputs), len(ordered_set_of_prompts)]
                with profiling_context(self, "vLLM.generate"):
                    all_outputs = self.llm.generate(
                        ordered_set_of_multimodal_inputs,
                        sampling_params=self.sampling_params,
                        use_tqdm=False,
                        )
                    completion_ids = [list(output.token_ids) for outputs in all_outputs for output in outputs.outputs]
                    #assert False, [prompt_ids.shape, len(all_prompts_text),
                    #               len(ordered_set_of_multimodal_inputs), 
                    #               len(ordered_set_of_multimodal_inputs[0]),
                    #               len(completion_ids),
                    #               len(completion_ids[0]), type(completion_ids[0])]
                    '''
                    completion_ids = self.vllm_client.generate(
                        ordered_set_of_multimodal_inputs,
                        n=self.num_generations,
                        repetition_penalty=self.repetition_penalty,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=-1 if self.top_k is None else self.top_k,
                        min_p=0.0 if self.min_p is None else self.min_p,
                        max_tokens=self.max_completion_length,
                        guided_decoding_regex=self.guided_decoding_regex,
                    )
                    '''
                    #assert False, completion_ids
            else:
                completion_ids = [None] * len(all_multimodal_inputs)
            # Broadcast the completions from the main process to all processes, ensuring each process receives its corresponding slice.
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            # 0, 8, 16
            # 0, 8, 12
            # 0, 8, 12
            #print(f"Rank: {self.accelerator.process_index}  Debugging!", self.accelerator.process_index, len(prompts), len(all_multimodal_inputs), len(all_multimodal_inputs_old), len(all_prompts_text), prompt_inputs["input_ids"].shape) 
            completion_ids = completion_ids[process_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            #assert False, [prompt_ids.shape, completion_ids.shape]
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            # Regular generation path
            logger.debug_rank0(f"Rank: {self.accelerator.process_index}  Generating")
                
            # print(f"Rank: {self.accelerator.process_index}  unwrapped")
            if self.max_prompt_length is not None:
                generate_returned_result = unwrapped_model.generate(
                    **{k: v[:, -self.max_prompt_length :] for k, v in prompt_inputs.items() if k not in self.model_attributes.get_non_generate_params()}, 
                    #input_ids=prompt_ids, 
                    #attention_mask=prompt_mask,  
                    generation_config=self.generation_config
                )
            else:
                logger.debug_rank0(f"Rank: {self.accelerator.process_index}  unwrapped_model.generate")
                generate_returned_result = unwrapped_model.generate(
                    **{k: v  for k, v in prompt_inputs.items() if k not in self.model_attributes.get_non_generate_params()}, 
                    #input_ids=prompt_ids, 
                    #attention_mask=prompt_mask, 
                    generation_config=self.generation_config
                )
            logger.debug_rank0(f"Rank: {self.accelerator.process_index}  Generated")
            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            if not self.model_attributes.is_embeds_input():
                prompt_completion_ids = generate_returned_result
                prompt_ids = prompt_completion_ids[:, :prompt_length]
                completion_ids = prompt_completion_ids[:, prompt_length:]
            else:
                # In this case, the input of the LLM backbone is the embedding of the combination of the image and text prompt
                # So the returned result of the `generate` method only contains the completion ids
                completion_ids = generate_returned_result
                prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)


        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
        # Get the multimodal inputs
        multimodal_keywords = self.model_attributes.get_custom_multimodal_keywords()
        multimodal_inputs = {k: prompt_inputs[k] if k in prompt_inputs else None for k in multimodal_keywords}
        

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)
        
        # this is not included in VLM-R1
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        # note: in below, not like VLM-R1,
        #   we removed the shifting handlers like:
        #old_per_token_logps = old_per_token_logps[:, prompt_length - 1:]
        # reason: we now have logits_to_keep
        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep, multimodal_inputs
                )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep, multimodal_inputs
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep, multimodal_inputs
                    )
            
        

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text
        
        '''
        # log to debug image loading and model inference
        if self.accelerator.is_main_process:
            for img_id, img in enumerate(images):
                img.save(f"img_{img_id}.png")
            assert False, completions
        assert False, "Pause"
        '''
        # ───────────────────────────────────────────────────────────────
        # 0.  Set‑up:  encode every data‑source string to an int ID
        # ───────────────────────────────────────────────────────────────
        unique_sources          = sorted({*data_sources})
        source_to_id            = {s: i for i, s in enumerate(unique_sources)}
        data_source_ids         = torch.tensor(
            [source_to_id[s] for s in data_sources], device=device, dtype=torch.long
        )                                           # shape (B,)

        # For each reward func, pre‑compute the *set* of source‑IDs on which it is valid
        allowed_ids_per_func = []
        for f in self.reward_funcs:
            allowed_sources   = [src for src, flist in data_source_reward_func_checker.items()
                                if f in flist]
            # AssertionError: 
            # unique_sources: ['GEOQAV'], 
            # source_to_id.items(): dict_items([('GEOQAV', 0)]), 
            # f: <function accuracy_iou_coco6k_reward at 0x1518d2788b80>, 
            # allowed_sources: ['ViRFT_COCO', 'LISA']
            #assert False, [
            #        unique_sources, 
            #        source_to_id.items(),
            #        f,
            #        allowed_sources]
                    
            allowed_ids_per_func.append(torch.tensor(
                [source_to_id[s] for s in allowed_sources if s in source_to_id], device=device, dtype=torch.long
            ))
        
        # ───────────────────────────────────────────────────────────────
        # 1.  Fast applicability mask  (B  ×  F)
        # ───────────────────────────────────────────────────────────────
        #B, F = len(prompts), len(self.reward_funcs)
        applicable_mask = torch.empty(len(prompts), len(self.reward_funcs), dtype=torch.bool, device=device)

        for i, allowed in enumerate(allowed_ids_per_func):          # ← loop only over F (usually ≪ B)
            # torch.isin does the heavy lifting on the GPU
            applicable_mask[:, i] = torch.isin(data_source_ids, allowed)

        # ───────────────────────────────────────────────────────────────
        # 2.  Run each reward only on the rows where it is applicable
        # ───────────────────────────────────────────────────────────────
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        reward_func_names = []
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                raise NotImplementedError #, "Not supported for now!"
                reward_func_name = f"reward {reward_func.config._name_or_path.split('/')[-1]}"
            else:
                reward_func_name = reward_func.__name__
            reward_func_names.append(unique_sources[0]+"_"+reward_func_name)
            with profiling_context(self, reward_func_name):
                if isinstance(
                    reward_func, nn.Module
                ):  # Module instead of PretrainedModel for compat with compiled models
                    raise NotImplementedError #, "Not supported for now!"
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                else:
                    rows = applicable_mask[:, i].nonzero(as_tuple=True)[0]   # 1‑D LongTensor
                    if rows.numel() == 0:                                    # nothing to do
                        continue
                    
                    # Slice lists/tensors just once  (no Python inner loop)
                    row_idx = rows.tolist()
                    batch_prompts      = [prompts[j]      for j in row_idx]
                    batch_completions  = [completions[j]  for j in row_idx]
                    # batch_reward_kwargs = {k: [inputs[j][k] for j in row_idx]
                    #                     for k in inputs[0] if k not in ("prompt", "completion")}
                    batch_reward_kwargs = {f"{'problem' if k=='prompt' else k}": [inputs[j][k] for j in row_idx]
                                        for k in inputs[0] if k not in [ "completion"]}
                    output_reward_func = reward_func(prompts=batch_prompts,
                         completions=batch_completions,
                         **batch_reward_kwargs)
                    
                    
                    #output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                    # Convert None values to NaN
                    output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                    rewards_per_func[rows, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in batch_reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )
        # If a prompt has no applicable reward functions, issue a warning
        no_applicable_rewards = (~applicable_mask).all(dim=1)
        if no_applicable_rewards.any():
            no_reward_indices = no_applicable_rewards.nonzero(as_tuple=True)[0]
            for idx in no_reward_indices:
                row_data = {key: inputs[idx][key] for key in inputs[idx] if key != "prompt" and key != "completion"}
                row_data["prompt"] = prompts[idx]
                row_data["completion"] = completions[idx]
                row_data["data_source"] = data_sources[idx]
                warnings.warn(
                    f"No applicable reward functions for data source '{data_sources[idx]}' with the following data: {row_data}. "
                    "Please ensure that at least one reward function is applicable to each data source."
                )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func) # (B, F)
        applicable_mask = gather(applicable_mask) # (B, F)

        rewards_per_func_mean = rewards_per_func.mean(0)

        ######
        for idx, reward_func_name in enumerate(reward_func_names):
            with torch.no_grad():
                if not applicable_mask[0][idx]:
                    continue
                ema_prev    = self.loss_ema[reward_func_name]      # tensor
                current_mag = rewards_per_func_mean[idx].item()
                ema_new     = self.ema_alpha * ema_prev + (1 - self.ema_alpha) * current_mag
                self.loss_ema[reward_func_name] = ema_new          # tensor update on GPU
                
                if reward_func_name not in self.dwa_hist:
                    self.dwa_hist[reward_func_name].extend([current_mag])
                else:
                    self.dwa_hist[reward_func_name].append(current_mag)


        with torch.no_grad():
            convergence_rate = {}
            inverse_signal_to_noise = {}
            logger.debug_rank0("self.dwa_hist", self.dwa_hist)
            max_convergence_rate = float('-inf')
            max_inverse_signal_to_noise = float('-inf')
            for t, hist in self.dwa_hist.items():
                hist = list(hist)
                mid = len(hist)//2
                hist = hist[-2*mid:]
                if mid>0:
                    convergence_rate[t] = sum(hist[:mid])/(sum(hist[mid:])+1e-15)
                else:
                    convergence_rate[t] = 1.
                inverse_signal_to_noise[t] = np.std(hist)/(np.mean(hist)+1e-15)
                # if t=="rl":
                #     convergence_rate[t] *= self.preference
                max_convergence_rate = max(max_convergence_rate, convergence_rate[t])
                max_inverse_signal_to_noise = max(max_inverse_signal_to_noise, inverse_signal_to_noise[t])
            # to avoid overflow
            sum_convergence_rate_exp = 0
            sum_inverse_signal_to_noise_exp = 0
            for t, hist in self.dwa_hist.items():
                sum_convergence_rate_exp += math.exp((convergence_rate[t]-max_convergence_rate)/self.softmax_temp)
                sum_inverse_signal_to_noise_exp += math.exp((inverse_signal_to_noise[t]-max_inverse_signal_to_noise)/self.softmax_temp)
    
            mode = "train" if self.model.training else "eval"
            for t in self.loss_ema:
                self._metrics[mode][f"ema_loss/{t}"].append(self.loss_ema[t])      
            for t in convergence_rate:
                self._metrics[mode][f"convergence_rate/{t}"].append(convergence_rate[t])                
            for t in inverse_signal_to_noise:
                self._metrics[mode][f"inverse_signal_to_noise/{t}"].append(inverse_signal_to_noise[t])    
    
        rl_weight = []
        for idx, reward_func_name in enumerate(reward_func_names):
            with torch.no_grad():
                if not applicable_mask[0][idx]:
                    rl_weight.append(0)
                    continue
            if self.args.normalize_loss == "ema":
                ema_new =  [reward_func_name]
                weight = 1. / (ema_new + 1e-15)
                logger.debug_rank0("Normalize by EMA", weight)
                
            elif self.args.normalize_loss == "dwa":
                assert len(convergence_rate)==len(inverse_signal_to_noise)
                relative_convergence_rate = math.exp((convergence_rate[reward_func_name]-max_convergence_rate)/self.softmax_temp)/sum_convergence_rate_exp
                relative_signal_to_noise = math.exp((inverse_signal_to_noise[reward_func_name]-max_inverse_signal_to_noise)/self.softmax_temp)/sum_inverse_signal_to_noise_exp
                if self.adaptive_convergence_instablity_tradeoff:                
                    self.convergence_instablity_tradeoff = 1. -  self.state.global_step/self.max_steps
                    logger.debug_rank0("self.state.global_step", self.state.global_step)
                    logger.debug_rank0("self.max_steps", self.max_steps)
                assert 0<=self.convergence_instablity_tradeoff<=1, f"{self.convergence_instablity_tradeoff},{self.state.global_step},{self.max_steps}"
                
                weight = len(convergence_rate)*(self.convergence_instablity_tradeoff*relative_convergence_rate+(1.-self.convergence_instablity_tradeoff)*relative_signal_to_noise)
                logger.debug_rank0("Normalize by DWA", weight, "relative_convergence_rate=", relative_convergence_rate, "relative_signal_to_noise=", relative_signal_to_noise)
            elif self.args.normalize_loss == "none":
                weight = 1.
            else:
                raise NotImplementedError(f"normalize_loss={self.args.normalize_loss} is not supported")
        
            rl_weight.append(weight)
            
            

       
        


        ######
        logger.debug_rank0("applicable_mask", applicable_mask)
        logger.debug_rank0("rl_weight", rl_weight)
        
        # Calculate re-balanced weights in a vectorized way
        # Get original weights and expand to match batch dimension
        # original_weights = self.reward_weights.to(device) # (F,)
        original_weights = torch.tensor(rl_weight, dtype=torch.float32).to(device) # (F,)
        
        batch_weights = original_weights.unsqueeze(0).expand(len(applicable_mask), -1) # (B, F)
        
        # assert False, [applicable_mask.shape, batch_weights.shape]
        # Set weights of non-applicable functions to 0
        masked_weights = torch.where(applicable_mask, batch_weights, torch.zeros_like(batch_weights))
        # Calculate sum of applicable weights per prompt
        weight_sums = masked_weights.sum(dim=1, keepdim=True) # (B, 1)
        # Avoid division by zero for rows with no applicable rewards
        valid_rows = weight_sums.squeeze(-1) > 0 # (B,)
        normalized_weights = torch.zeros_like(masked_weights)
        normalized_weights[valid_rows] = masked_weights[valid_rows] / weight_sums[valid_rows]
        # Apply normalized weights to rewards and sum
        avg_reward = (rewards_per_func * normalized_weights).nansum(dim=1) # (B,)
        rewards = avg_reward * weight_sums.squeeze(1) # (B,)
        
        # Apply weights to each reward function's output and sum
        #rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        if self.args.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        mode = "eval" if not self.is_in_train else "train"

        if mode == "train":
            #self._total_train_tokens += self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
            self.state.num_input_tokens_seen += self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
        #self._metrics[mode]["num_tokens"] = [self._total_train_tokens]
        self._metrics[mode]["grpo/num_tokens"] = [self.state.num_input_tokens_seen]
        
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["grpo/completion_length"].append(completion_length)

        # Get the names of the reward functions
        reward_func_names = []
        for reward_func in self.reward_funcs:
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                raise NotImplementedError
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            reward_func_names.append(reward_func_name)

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(reward_func_names):
            if not applicable_mask[0][i]:
                continue            
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item() # these are rewards before weighting! mean over non-nan values
            self._metrics[mode][f"grpo/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"grpo/{reward_func_name}/std"].append(std_rewards)
        self._metrics[mode]["grpo"].append(mean_grouped_rewards.mean().item()) # these are rewards after weighting!
        self._metrics[mode]["grpo_std"].append(std_grouped_rewards.mean().item())

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            prompts_to_log = gather_object(prompts_text)
            completions_to_log = gather_object(completions_text)
            rewards_to_log = {
                reward_func_name: rewards_per_func[:, i] for i, reward_func_name in enumerate(reward_func_names)
            }

            if self.accelerator.is_main_process:
                if is_rich_available():
                    print_prompt_completions_sample(
                        prompts_to_log,
                        completions_to_log,
                        rewards_to_log,
                        self.state.global_step,
                        self.num_completions_to_print,
                    )
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                    

                    # For logging
                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": rewards.tolist(),
                    }
                    df = pd.DataFrame(table)
                    if self.args.wandb_log_unique_prompts:
                        df = df.drop_duplicates(subset=["prompt"])
                    wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "multimodal_inputs": multimodal_inputs,
        }
    

 
    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # print(f"Rank: {self.accelerator.process_index}  "
        # f"(local rank: {self.accelerator.local_process_index}, "
        # f"world size: {self.accelerator.num_processes})", "compute loss", inputs)
        # print(f"Rank: {self.accelerator.process_index}  Waiting to compute loss")
        # torch.cuda.synchronize()
        
        # torch.distributed.barrier()
        # self.accelerator.wait_for_everyone() 
        # dummy = torch.zeros(1, device=self.accelerator.device)    
        # torch.distributed.all_reduce(dummy)     
        # torch.cuda.synchronize()
        logger.debug_rank0(f"Rank: {self.accelerator.process_index}  Start computing loss")
        source = inputs.pop("data_source")
        task = inputs.pop("task")
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model
        if self.use_liger_loss:
            # Compute the loss using the liger grpo loss
            return self.compute_liger_loss(model, inputs)

        elif task == "dpo":
            logger.debug_rank0(f"Rank: {self.accelerator.process_index}  DPO LOSS")
            outputs = self._compute_loss_dpo(model, inputs)
            
        elif task=="rl":
            logger.debug_rank0(f"Rank: {self.accelerator.process_index}  GRPO LOSS")
            outputs = self._compute_loss_grpo(model, inputs)
            
        # print(f"Rank: {self.accelerator.process_index}  Waiting for GRPO")
        # self.accelerator.wait_for_everyone()         
        # dummy = torch.zeros(1, device=self.accelerator.device)
        # torch.distributed.all_reduce(dummy)
        # torch.cuda.synchronize() 
        # print(f"Rank: {self.accelerator.process_index}  Waited for GRPO")
             
        elif task == "sft":
            logger.debug_rank0(f"Rank: {self.accelerator.process_index}  SFT LOSS")
            outputs = self._compute_loss_sft(model, inputs)
        else:
            raise NotImplementedError(f"")   
        # torch.distributed.barrier()
        # self.accelerator.wait_for_everyone() 
        # torch.cuda.synchronize()
        logger.debug_rank0(f"Rank: {self.accelerator.process_index}  COMPUTED LOSS")
        mode = "train" if self.model.training else "eval"
        
        # Update EMA
        with torch.no_grad():
            # Suppose raw_loss is a tensor on the GPU
            current_mag = outputs.detach()#.abs()        # keep as tensor
            current_mag = self.accelerator.gather_for_metrics(current_mag).mean().item()
            self._metrics[mode][f"loss/{source}"].append(current_mag)      
            
        if task=="rl":
            return outputs
        
        with torch.no_grad():
            ema_prev    = self.loss_ema[source]      # tensor
    
            ema_new     = self.ema_alpha * ema_prev + (1 - self.ema_alpha) * current_mag
            self.loss_ema[source] = ema_new          # tensor update on GPU
            
            if source not in self.dwa_hist:
                self.dwa_hist[source].extend([current_mag])
            else:
                self.dwa_hist[source].append(current_mag)

            convergence_rate = {}
            inverse_signal_to_noise = {}
            logger.debug_rank0("self.dwa_hist", self.dwa_hist)
            max_convergence_rate = float('-inf')
            max_inverse_signal_to_noise = float('-inf')
            for t, hist in self.dwa_hist.items():
                hist = list(hist)
                mid = len(hist)//2
                hist = hist[-2*mid:]
                if mid>0:
                    convergence_rate[t] = sum(hist[mid:])/(sum(hist[:mid])+1e-15)
                else:
                    convergence_rate[t] = 1.
                inverse_signal_to_noise[t] = np.std(hist)/(np.mean(hist)+1e-15)
                # if t=="rl":
                #     convergence_rate[t] *= self.preference
                max_convergence_rate = max(max_convergence_rate, convergence_rate[t])
                max_inverse_signal_to_noise = max(max_inverse_signal_to_noise, inverse_signal_to_noise[t])
            # to avoid overflow
            sum_convergence_rate_exp = 0
            sum_inverse_signal_to_noise_exp = 0
            for t, hist in self.dwa_hist.items():
                sum_convergence_rate_exp += math.exp((convergence_rate[t]-max_convergence_rate)/self.softmax_temp)
                sum_inverse_signal_to_noise_exp += math.exp((inverse_signal_to_noise[t]-max_inverse_signal_to_noise)/self.softmax_temp)
        if self.args.normalize_loss == "ema":
            ema_new =  [source]
            weight = 1. / (ema_new + 1e-15)
            logger.debug_rank0("Normalize by EMA", weight)
            
        elif self.args.normalize_loss == "dwa":
            assert len(convergence_rate)==len(inverse_signal_to_noise)
            relative_convergence_rate = math.exp((convergence_rate[source]-max_convergence_rate)/self.softmax_temp)/sum_convergence_rate_exp
            relative_signal_to_noise = math.exp((inverse_signal_to_noise[source]-max_inverse_signal_to_noise)/self.softmax_temp)/sum_inverse_signal_to_noise_exp
            if self.adaptive_convergence_instablity_tradeoff:                
                self.convergence_instablity_tradeoff = 1.- self.state.global_step/self.max_steps
                logger.debug_rank0("self.state.global_step", self.state.global_step)
                logger.debug_rank0("self.max_steps", self.max_steps)
                
            assert 0<=self.convergence_instablity_tradeoff<=1, f"{self.convergence_instablity_tradeoff},{self.state.global_step},{self.max_steps}"
            weight = len(convergence_rate)*(self.convergence_instablity_tradeoff*relative_convergence_rate+(1.-self.convergence_instablity_tradeoff)*relative_signal_to_noise)
            logger.debug_rank0("Normalize by DWA", weight, "relative_convergence_rate=", relative_convergence_rate, "relative_signal_to_noise=", relative_signal_to_noise)
        elif self.args.normalize_loss == "none":
            weight = 1.
        else:
            raise NotImplementedError(f"normalize_loss={self.args.normalize_loss} is not supported")
        outputs = outputs * weight
            
            
        mode = "train" if self.model.training else "eval"
        for t in self.loss_ema:
            self._metrics[mode][f"ema_loss/{t}"].append(self.loss_ema[t])      
        for t in convergence_rate:
            self._metrics[mode][f"convergence_rate/{t}"].append(convergence_rate[t])                
        for t in inverse_signal_to_noise:
            self._metrics[mode][f"inverse_signal_to_noise/{t}"].append(inverse_signal_to_noise[t])    
        
        
        return outputs



    def odds_ratio_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""Compute ORPO's odds ratio (OR) loss for batched log probabilities of the policy model."""
        log_odds = (chosen_logps - rejected_logps) - (
            torch.log1p(-torch.exp(chosen_logps)) - torch.log1p(-torch.exp(rejected_logps))
        )
        sft_loss = -chosen_logps
        odds_ratio_loss = -F.logsigmoid(log_odds)
        orpo_loss = sft_loss + self.pref_beta * odds_ratio_loss
        return orpo_loss

    def simpo_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""Compute SimPO loss for batched log probabilities of the policy model."""
        pi_logratios = chosen_logps - rejected_logps
        gamma_logratios = self.simpo_gamma / self.pref_beta
        logits = pi_logratios - gamma_logratios
        simpo_loss = -F.logsigmoid(self.pref_beta * logits)
        return simpo_loss



    def compute_preference_loss(
        self,
        policy_chosen_logps: "torch.Tensor",
        policy_rejected_logps: "torch.Tensor",
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""Compute loss for preference learning."""
        if self.loss_type == "orpo":
            losses = self.odds_ratio_loss(policy_chosen_logps, policy_rejected_logps)
        elif self.loss_type == "simpo":
            losses = self.simpo_loss(policy_chosen_logps, policy_rejected_logps)
        else:
            raise NotImplementedError(f"Unknown loss type: {self.loss_type}.")

        chosen_rewards = self.pref_beta * policy_chosen_logps.to(self.accelerator.device).detach()
        rejected_rewards = self.pref_beta * policy_rejected_logps.to(self.accelerator.device).detach()


        return losses, chosen_rewards, rejected_rewards
    def get_batch_logps(
        self,
        logits: "torch.Tensor",
        labels: "torch.Tensor",
        label_pad_token_id: int = IGNORE_INDEX,
        ld_alpha: Optional[float] = None,
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        r"""Compute the log probabilities of the given labels under the given logits.

        Returns:
            logps: A tensor of shape (batch_size,) containing the sum of log probabilities.
            valid_length: A tensor of shape (batch_size,) containing the number of non-masked tokens.

        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batchsize x seqlen) and labels must have the same shape.")

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id
        labels[labels == label_pad_token_id] = 0  # dummy token
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        valid_length = loss_mask.sum(-1)
        if ld_alpha is not None:
            num_examples = labels.shape[0] // 2
            chosen_lengths = valid_length[:num_examples]
            rejected_lengths = valid_length[num_examples:]
            min_lengths = torch.min(chosen_lengths, rejected_lengths)
            start_positions = torch.argmax(loss_mask.int(), dim=1)
            public_lengths = start_positions + torch.cat([min_lengths, min_lengths], dim=0)

            seq_len = labels.shape[-1]
            position_ids = torch.arange(seq_len, device=per_token_logps.device).expand_as(per_token_logps)

            ld_mask = position_ids < public_lengths.unsqueeze(1)
            front_mask = (ld_mask * loss_mask).float()
            rear_mask = (~ld_mask * loss_mask).float()

            front_logps = (per_token_logps * front_mask).sum(-1)
            rear_logps = (per_token_logps * rear_mask).sum(-1)
            logps = front_logps + ld_alpha * rear_logps
        else:
            logps = (per_token_logps * loss_mask).sum(-1)

        return logps, valid_length

    def concatenated_forward(
        self, model: "PreTrainedModel", batch: dict[str, "torch.Tensor"], is_ref_model: bool = False
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""Compute the sum log probabilities of the labels under given logits if loss_type is not IPO, ORPO or SimPO.

        Otherwise the average log probabilities.
        """

        # concatenated_batch = self.concatenated_inputs(batch, padding_value=self.padding_value)
        all_logits: torch.Tensor = model(**batch, return_dict=True, use_cache=False).logits.to(torch.float32)
        all_logps, valid_length = self.get_batch_logps(
            logits=all_logits, labels=batch["labels"], ld_alpha=(self.ld_alpha if not is_ref_model else None)
        )
        if self.loss_type in ["ipo", "orpo", "simpo"]:
            all_logps = all_logps / valid_length

        batch_size = batch["input_ids"].size(0) // 2
        chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)
        chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)
        chosen_length, _ = valid_length.split(batch_size, dim=0)

        if self.loss_type in ["ipo", "orpo", "simpo"]:
            return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_logps
        else:
            return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_logps / chosen_length


    def get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: dict[str, "torch.Tensor"],
        train_eval: Literal["train", "eval"] = "train",
    ) -> tuple["torch.Tensor", dict[str, "torch.Tensor"]]:
        r"""Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_chosen_logps_avg,
        ) = self.concatenated_forward(model, batch)

        losses, chosen_rewards, rejected_rewards = self.compute_preference_loss(
            policy_chosen_logps,
            policy_rejected_logps,
        )
        sft_loss = -policy_chosen_logps_avg
        if self.ftx_gamma > 1e-6:
            losses += self.ftx_gamma * sft_loss
            if self.bco_gemma > 1e-6:
                # re-weigthing for MPO
                losses /= (self.ftx_gamma + self.bco_gemma + 1.0)

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().item()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().item()
        metrics[f"{prefix}rewards/accuracies"] = (chosen_rewards > rejected_rewards).float().mean().item()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().item()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.mean().item()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.mean().item()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.mean().item()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.mean().item()
        if self.loss_type == "orpo":
            metrics[f"{prefix}sft_loss"] = sft_loss.mean().item()
            metrics[f"{prefix}odds_ratio_loss"] = ((losses - sft_loss) / self.pref_beta).mean().item()

        return losses.mean(), metrics

    def _compute_loss_dpo(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: dict[str, Union[torch.Tensor, Any]],
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor]]]:
        compute_loss_context_manager = (
            autocast(self.accelerator.device.type)
        )

        with compute_loss_context_manager:
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        # force log the metrics
        # self.store_metrics(metrics, train_eval="train")
        loss = loss.to(self.args.device)
        return loss

    def _compute_loss_sft(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        """
        mode = "train" if self.model.training else "eval"
        (loss, outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        if mode == "train":
            # When using padding-free, the attention_mask is not present in the inputs, instead we have cu_seq_lens_q,
            # cu_seq_lens_k, and max_length_k, max_length_q and position_ids.
            if "attention_mask" in inputs:
                num_tokens_in_batch = self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
            elif "position_ids" in inputs:
                local_num_tokens = torch.tensor(inputs["position_ids"].size(1), device=inputs["position_ids"].device)
                num_tokens_in_batch = self.accelerator.gather_for_metrics(local_num_tokens).sum().item()
            else:
                raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
            self._total_train_tokens += num_tokens_in_batch
        self._metrics[mode]["sft/num_tokens"] = [self._total_train_tokens]

        # Compute token accuracy if we have labels and if the model is not using Liger (no logits)
        if "labels" in inputs and not self.args.use_liger_kernel:
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = inputs["labels"][..., 1:].contiguous()

            # Get predictions
            predictions = shift_logits.argmax(dim=-1)
            
            
            # replace -100 so batch_decode won’t overflow
            labels_for_decode = shift_labels.clone()
            labels_for_decode[labels_for_decode == -100] = self.processing_class.tokenizer.pad_token_id

            # get token‐wise predictions
            # decode sequences back to text
            pred_texts = self.processing_class.batch_decode(predictions, skip_special_tokens=True)
            target_texts = self.processing_class.batch_decode(labels_for_decode, skip_special_tokens=True)
            # pred_texts = self.processing_class.batch_decode(predictions, skip_special_tokens=True)
            # target_texts = self.processing_class.batch_decode(shift_labels, skip_special_tokens=True)
            # print("pred_texts", pred_texts)
            # print("target_texts", target_texts)
            # Create mask for non-padding tokens (assuming ignore_index is -100)
            mask = shift_labels != -100

            # Calculate accuracy only on non-padding tokens
            correct_predictions = (predictions == shift_labels) & mask
            total_tokens = mask.sum()
            correct_tokens = correct_predictions.sum()

            # Gather the correct_tokens and total_tokens across all processes
            correct_tokens = self.accelerator.gather_for_metrics(correct_tokens)
            total_tokens = self.accelerator.gather_for_metrics(total_tokens)

            # Compute the mean token accuracy and log it
            total_sum = total_tokens.sum()
            accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
            self._metrics[mode]["sft/mean_token_accuracy"].append(accuracy)


        logger.debug_rank0(f"Rank: {self.accelerator.process_index}  SFT FINISHED")
        return (loss, outputs) if return_outputs else loss

    def _compute_loss_grpo(self, model, inputs):
        # Compute the per-token log probabilities for the model
        # Get the prepared inputs
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        
        # Concatenate for full sequence
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        
        # Get the current policy's log probabilities
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep, inputs["multimodal_inputs"])

        

        # Compute the loss
        # Get the advantages from inputs
        advantages = inputs["advantages"]
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's computation (see
        # _prepare_inputs_grpo) and use per_token_logps.detach() instead.
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
        # Compute the policy ratio and clipped version
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        # Add KL penalty if beta > 0
        if self.beta > 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            per_token_loss = per_token_loss + self.beta * per_token_kl
            # slightly different reduction from VLM-R1
            # mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
        # Compute final loss
        # slightly different reduction from VLM-R1
        #loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        # Log the metrics
        mode = "eval" if not self.is_in_train else "train"
        if self.beta != 0.0:
            self._metrics[mode]["grpo/kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        # Compute the clip ratio
        # open-r1:
        is_clipped = ((coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)) | (
            (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        )
        # different from VLM-R1: is_clipped = (per_token_loss1 < per_token_loss2).float()
                
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["grpo/clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
        logger.debug_rank0(f"Rank: {self.accelerator.process_index}  GRPO FINISHED")
        # self._metrics[mode]["grpo/loss"].append(self.accelerator.gather_for_metrics(loss).mean().item())
        return loss
    
    # have to override original one as we changed compute_loss function
    # warning: we may have sacrified advanced functions in original prediction_step function
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None
    
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "eval" if not self.is_in_train else "train"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics[mode].clear()
    

def pad_without_fast_tokenizer_warning(tokenizer, *pad_args, **pad_kwargs):
    """
    Pads without triggering the warning about how using the pad function is sub-optimal when using a fast tokenizer.
    """

    # To avoid errors when using Feature extractors
    if not hasattr(tokenizer, "deprecation_warnings"):
        return tokenizer.pad(*pad_args, **pad_kwargs)

    # Save the state of the warning, then disable it
    warning_state = tokenizer.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    try:
        padded = tokenizer.pad(*pad_args, **pad_kwargs)
    finally:
        # Restore the state of the warning.
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = warning_state

    return padded

def infer_seqlen(source_len: int, target_len: int, cutoff_len: int) -> tuple[int, int]:
    r"""Compute the real sequence length after truncation by the cutoff_len."""
    if target_len * 2 < cutoff_len:  # truncate source
        max_target_len = cutoff_len
    elif source_len * 2 < cutoff_len:  # truncate target
        max_target_len = cutoff_len - source_len
    else:  # truncate both
        max_target_len = int(cutoff_len * (target_len / (source_len + target_len)))

    new_target_len = min(max_target_len, target_len)
    max_source_len = max(cutoff_len - new_target_len, 0)
    new_source_len = min(max_source_len, source_len)
    return new_source_len, new_target_len