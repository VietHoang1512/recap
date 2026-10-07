import os
import warnings
from unittest.mock import patch
from collections import defaultdict, deque
from contextlib import nullcontext
from typing import Any, Callable, Optional, Sized, Union
from packaging import version
import torch
import torch.nn as nn
from torch.utils.data import Sampler
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
import io
import base64

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

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


logger = logging.getLogger(__name__)


def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    variance = torch.nanmean((tensor - torch.nanmean(tensor, keepdim=True)) ** 2)  # Compute variance ignoring NaNs
    count = torch.sum(~torch.isnan(tensor))  # Count of non-NaN values
    variance *= count / (count - 1)  # Bessel's correction
    return torch.sqrt(variance)


class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility (only affects this sampler).

    Example:
    ```python
    >>> sampler = RepeatRandomSampler(["a", "b", "c", "d", "e", "f", "g"], mini_repeat_count=2, batch_size=3, repeat_count=4)
    >>> list(sampler)
    [4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,

     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6]
    ```

    ```txt
    mini_repeat_count = 3
          -   -   -
         [0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11,      |
                                                                repeat_count = 2
          0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11, ...] |
          ---------   ---------   ---------   ---------
           ---------   ---------   ---------   ---------
            ---------   ---------   ---------   ---------
                         batch_size = 12
    ```
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()  # Create a local random generator
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        # E.g., [2, 4, 3, 1, 0, 6, 5] (num_samples = 7)
        indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()

        #    [2, 4, 3, 1, 0, 6, 5]
        # -> [[2, 4, 3], [1, 0, 6], [5]]  (batch_size = 3)
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]

        #    [[2, 4, 3], [1, 0, 6], [5]]
        # -> [[2, 4, 3], [1, 0, 6]]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count



class GRPOTrainer(Trainer):
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
        #self._total_train_tokens = 0
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
                    self.llm = LLM(
                        model=model_name_or_path, # use the same model to initialize
                        revision=vllm_args.revision,
                        tensor_parallel_size=vllm_args.tensor_parallel_size,
                        device=f"cuda:{self.accelerator.num_processes}",  # take the next GPU idx
                        gpu_memory_utilization=vllm_args.gpu_memory_utilization,
                        dtype=self.model.dtype,
                        # Automatic Prefix Caching caches the KV cache of existing queries, so that a new query can
                        # directly reuse the KV cache if it shares the same prefix with one of the existing queries.
                        # This is particularly useful here because we generate completions from the same prompts.
                        enable_prefix_caching=vllm_args.enable_prefix_caching,
                        enforce_eager=vllm_args.enforce_eager,
                        max_model_len=vllm_args.max_model_len,
                        max_num_seqs=vllm_args.max_num_seqs,
                        max_num_batched_tokens=vllm_args.max_num_batched_tokens,
                        #max_num_batched_tokens=4096, # default was 512; too small for VLM?
                        worker_cls=WeightSyncWorker,
                        #mm_processor_kwargs = ({
                        #    processing_keyword: getattr(self.processing_class, processing_keyword)
                        #    for processing_keyword in self.model_attributes.get_custom_processing_keywords()
                        #}),
                        hf_overrides={processing_keyword: getattr(self.processing_class, processing_keyword) for processing_keyword in self.model_attributes.get_custom_processing_keywords()}
                    )
                # initialize communicator between processes
                self.llm.collective_rpc("init_communicator", args=(vllm_args.host, vllm_args.port, vllm_args.tensor_parallel_size))
                # Set up the communication group for weight broadcasting
                    
                # Guided decoding, if enabled
                # vLLM specific sampling arguments
                self.guided_decoding_regex = args.vllm_guided_decoding_regex
                if self.guided_decoding_regex is not None:
                    guided_decoding = GuidedDecodingParams(backend="outlines", regex=self.guided_decoding_regex)
                else:
                    guided_decoding = None

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
                    guided_decoding=guided_decoding,
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
            
    def _get_train_sampler(self, train_dataset=None) -> Sampler:
        # Returns a sampler that
        # 1. ensures each prompt is repeated across multiple processes. This guarantees that identical prompts are
        #    distributed to different GPUs, allowing rewards to be computed and normalized correctly within each prompt
        #    group. Using the same seed across processes ensures consistent prompt assignment, preventing discrepancies
        #    in group formation.
        # 2. repeats the batch multiple times to allow reusing generations across multiple updates. Refer to
        #    _prepare_inputs to see how the generations are stored and reused.

        # In the following figure, the values are the prompt indices. The first row shows the first sampled batch, the
        # second row shows the second sampled batch, and so on.
        #
        #                                     |     GPU 0     |     GPU 1     |     GPU 2    |
        #
        #               global_step   step     <───────>  num_generations=3
        #                                      <───────────> per_device_train_batch_size=4
        #                ▲   0          0      0   0   0   1   1   1   2   2   2   3   3   3  │
        #  grad_accum=3  │   0          1      4   4   4   5   5   5   6   6   6   7   7   7  │ Generate completions for each prompt
        #                ▼   0          2      8   8   8   9   9   9  10  10  10  11  11  11  │
        #
        #                    1          3      0   0   0   1   1   1   2   2   2   3   3   3  │ The sampled prompts are the same as in the first iteration
        #                    1          4      4   4   4   5   5   5   6   6   6   7   7   7  │ Reuse the completions (here, once, because num_iterations=2)
        #                    1          5      8   8   8   9   9   9  10  10  10  11  11  11  │
        #
        #                    2          6     12  12  12  13  13  13  14  14  14  15  15  15
        #                    2          7     16  16  16  17  17  17  18  18  18  19  19  19
        #                    2          8     20  20  20  21  21  21  22  22  22  23  23  23
        #                                          ...
        effective_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        return RepeatRandomSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=effective_batch_size // self.num_generations,
            repeat_count=self.num_iterations,
            seed=self.args.seed,
        )
    
    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # See _get_train_sampler for an explanation of the sampler.
        return RepeatRandomSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
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
        inputs = [{k:v for k,v in example.items() if v is not None} for example in inputs]
        print("inputs", inputs)
        
        mode = "eval" if not self.is_in_train else "train"
        if mode == "train":            
            buffer_index = self._step % self.args.gradient_accumulation_steps
            buffered_inputs = self._buffered_inputs[buffer_index]
            if self.state.global_step % self.num_iterations == 0 or buffered_inputs is None:
                # buffered_inputs=None can occur when resuming from a checkpoint
                inputs = self._generate_and_score_completions(inputs)
                self._buffered_inputs[buffer_index] = inputs
            else:
                inputs = buffered_inputs
            self._step += 1
        else:
            # In evaluation, we don't reuse completions across multiple updates, so we don't need to buffer inputs.
            inputs = self._generate_and_score_completions(inputs)
        return inputs
    
    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        # get raw prompts
        prompts = [x["prompt"] for x in inputs]
        data_sources = [x["data_source"] for x in inputs]
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
            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            # 0, 8, 16
            # 0, 8, 12
            # 0, 8, 12
            #print("Debugging!", self.accelerator.process_index, len(prompts), len(all_multimodal_inputs), len(all_multimodal_inputs_old), len(all_prompts_text), prompt_inputs["input_ids"].shape) 
            completion_ids = completion_ids[process_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            #assert False, [prompt_ids.shape, completion_ids.shape]
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            # Regular generation path
            with unwrap_model_for_generation(
                self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
            ) as unwrapped_model:
                if self.max_prompt_length is not None:
                    generate_returned_result = unwrapped_model.generate(
                        **{k: v[:, -self.max_prompt_length :] for k, v in prompt_inputs.items() if k not in self.model_attributes.get_non_generate_params()}, 
                        #input_ids=prompt_ids, 
                        #attention_mask=prompt_mask, 
                        generation_config=self.generation_config
                    )
                else:
                    generate_returned_result = unwrapped_model.generate(
                        **{k: v  for k, v in prompt_inputs.items() if k not in self.model_attributes.get_non_generate_params()}, 
                        #input_ids=prompt_ids, 
                        #attention_mask=prompt_mask, 
                        generation_config=self.generation_config
                    )

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
        
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                raise NotImplementedError #, "Not supported for now!"
                reward_func_name = f"reward {reward_func.config._name_or_path.split('/')[-1]}"
            else:
                reward_func_name = reward_func.__name__
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

        # Calculate re-balanced weights in a vectorized way
        # Get original weights and expand to match batch dimension
        original_weights = self.reward_weights.to(device) # (F,)
        batch_weights = original_weights.unsqueeze(0).expand(len(applicable_mask), -1) # (B, F)
        
        #assert False, [applicable_mask.shape, batch_weights.shape]
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
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]
        
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)

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
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item() # these are rewards before weighting! mean over non-nan values
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item()) # these are rewards after weighting!
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

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
    
    def compute_liger_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        # get the last hidden state of the model
        last_hidden_state = self._get_last_hidden_state(model, input_ids, attention_mask, logits_to_keep, inputs["multimodal_inputs"])
        unwrapped_model = self.accelerator.unwrap_model(model)
        # compute loss and metrics using liger grpo loss
        loss, metrics = self.liger_grpo_loss(
            _input=last_hidden_state,
            lin_weight=unwrapped_model.lm_head.weight,
            selected_token_ids=completion_ids,
            attention_mask=completion_mask,
            advantages=inputs["advantages"],
            bias=unwrapped_model.lm_head.bias,
            ref_per_token_logps=inputs["ref_per_token_logps"],
            old_per_token_logps=inputs["old_per_token_logps"],
        )
        # Extract metrics from the liger_grpo_loss output
        # KL divergence is the first metric when beta is non-zero
        mean_kl = metrics[0] if self.beta != 0.0 else None
        clip_ratio = metrics[-1]

        mode = "eval" if not self.is_in_train else "train"
        if self.beta != 0.0:
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
        return loss
 
    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model
        if self.use_liger_loss:
            # Compute the loss using the liger grpo loss
            return self.compute_liger_loss(model, inputs)
        else:
            return self._compute_loss(model, inputs)

    def _compute_loss(self, model, inputs):
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
        # _generate_and_score_completions) and use per_token_logps.detach() instead.
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
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        # Compute the clip ratio
        # open-r1:
        is_clipped = ((coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)) | (
            (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        )
        # different from VLM-R1: is_clipped = (per_token_loss1 < per_token_loss2).float()
                
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
        
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
    
    
    
