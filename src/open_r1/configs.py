from dataclasses import dataclass, field
from typing import Optional, Literal
import trl 


# defining GRPOScriptArguments as inheriting ScriptArguments yet
@dataclass
class GRPOScriptArguments: # do not inherit from (trl.ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        dataset_names: (`list[str]`):
            List of datasets.
        reward_funcs (`list[str]`):
            List of reward functions. See the registry in `open_r1.rewards.get_reward_funcs`.
            Inherited from open-r1: 'accuracy', 'format', 'reasoning_steps', 'cosine',
            'repetition_penalty', 'length', 'tag_count'. Added here: 'accuracy_iou_coco6k',
            'iou_bbox', 'accuracy_sat', 'format_sat', 'accuracy_vilr39k', 'accuracy_thinklite',
            'iou', 'format_think', 'format_think_rec', 'format_answer'.
        cosine_min_value_wrong (`float`):
            Minimum reward for cosine scaling for wrong answers.
        cosine_max_value_wrong (`float`):
            Maximum reward for cosine scaling for wrong answers.
        cosine_min_value_correct (`float`):
            Minimum reward for cosine scaling for correct answers.
        cosine_max_value_correct (`float`):
            Maximum reward for cosine scaling for correct answers.
        cosine_max_len (`int`):
            Maximum length for cosine scaling.
    """
    dataset_names: list[str] = field(
        metadata={"help": "Dataset names."}
    ) # would be something like "../../share_data/???"
    # this is different from list[str]
    # now that each dataset might have different groups of rewards
    # must have the same length as dataset_name list
    
    reward_funcs: list[list[str]] = field(
        #default_factory=lambda: ["accuracy", "format", "tag_count"],
        metadata={
            "help": "List of List of reward functions. "
        },
    )
    cosine_min_value_wrong: float = field(
        default=0.0,
        metadata={"help": "Minimum reward for wrong answers"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Maximum reward for wrong answers"},
    )
    cosine_min_value_correct: float = field(
        default=0.5,
        metadata={"help": "Minimum reward for correct answers"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for correct answers"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "Maximum length for scaling"},
    )
    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )
    repetition_max_penalty: float = field(
        default=-1.0,
        metadata={"help": "Maximum (negative) penalty for for repetition penalty reward"},
    )

    
    # disable now as in conflict with below dataset_dir
    #image_root: Optional[str] = field(
    #    default=None,
    #    metadata={"help": "Root directory of the image"},
    #)
    
    # addtional arguments for data loading
    dataset_dir: Optional[str] = field(
        default=".",
        metadata={"help": "Directory to find data"}
    )
    streaming: bool = field(
        default=False,
        metadata={"help": "Enable dataset streaming."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the pre-processing."},
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the number of examples for each dataset."},
    )
    mix_strategy: Literal["concat", "interleave_under", "interleave_over"] = field(
        default="interleave_under",
        metadata={"help": "Strategy to use in dataset mixing (concat/interleave) (undersampling/oversampling)."},
    )
    interleave_probs: Optional[float] = field(
        default=None,
        metadata={"help": "Probabilities to sample data from datasets. Use commas to separate multiple datasets."},
    )

    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets."},
    )
    split: Optional[str] = field(
        default="train",
    )
    dataset_test_split: Optional[str] = field(
        default="test",
    )
    use_system_prompt: Optional[bool] = field(
        default=False,
    )
    
    # below is from llava, not using for now
    #freeze_backbone:  Optional[bool] = field(
    #    default=False,
    #)
    # Hoang
    eval_dataset_names: Optional[str] = field(default=None, metadata={"help": "Dataset name."})

# TODO: add the shared options with a mixin to reduce code duplication
@dataclass
class GRPOConfig(trl.GRPOConfig, trl.DPOConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    system_prompt: Optional[str] = field(
        default=None, metadata={"help": "The optional system prompt to use for benchmarking."}
    )
    hub_model_revision: Optional[str] = field(
        default="main", metadata={"help": "The Hub model branch to push the model to."}
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )
    
    # this is something that open-r1 newly added
    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    
    # added for model loading
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    
    # added for dataset loader
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pre-trained models downloaded from huggingface.co or modelscope.cn."},
    )
    
    # model 
    freeze_vision_modules:  Optional[bool] = field(
        default=False,
    )
    
    num_completions_to_print: Optional[int] = field(
        default=None,
        metadata={"help": "Number of completions to print with `rich`. If `None`, all completions are logged."},
    )
    
    use_liger_loss: bool = field(
         default=False,
         metadata={"help": "Whether to use the Liger GRPO loss."},
     )
    #  EMA:
    ema_alpha: Optional[float] = field(
        default=.99,
        metadata={"help": "Smoothing factor (e.g., 0.99) to compute the exponential moving average (EMA) of each task's loss magnitude ."},
    )
    normalize_loss: str = field(
        default="none",
        metadata={"help": "Normalize per-domain losses by ema of loss magnitude.", "choices": ["ema", "dwa", "none"],},
    )    
    # DWA:
    iteration_window: Optional[int] = field(
        default=10,
        metadata={"help": "'iteration' loss is averaged over the last 'iteration_window' losses."},
    )
    preference: Optional[float] = field(
        default=1.,
        metadata={"help": "Preference for main task"},
    )
    
    convergence_instablity_tradeoff: Optional[float] = field(
        default=1.,
        metadata={"help": "convergence and instablity tradeoff"},
    )    
    dpo_max_prompt_length: Optional[int] = field(
        default=2048,
        metadata={"help": "Maximum length of the DPO prompt."},
    )
    dpo_max_completion_length: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum length of the DPO completion."},
    )    
    simpo_gamma: float = field(
        default=0.5,
        metadata={"help": "The target reward margin term in SimPO loss."},
    )    
    pref_beta: float = field(
        default=0.1,
        metadata={"help": "The beta parameter in the preference loss."},
    )
    pref_ftx: float = field(
        default=0.0,
        metadata={"help": "The supervised fine-tuning loss coefficient in DPO training."},
    )
    pref_bco_weight: float = field(
        default=0.0,
        metadata={"help": "The Binary Classifier Optimization coefficient in DPO training."},
    )  
    ld_alpha: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Alpha parameter from the LD-DPO paper, which controls the weighting of"
                " the verbose token log-probabilities in responses."
            )
        },
    )    

    softmax_temp: float = field(
        default=1.0,
        metadata={"help": "Temperature hyper-parameter for DWA. Default to 2 like in the original paper."},
    )  
    
    mixed_sampler: Literal["uniform", "default"] = field(
        default="uniform",
        metadata={"help": "Strategy to sample the data from mixed data"},
    )  
# defining GRPOModelConfig as inheriting ModelConfig
# support additional argument: 
#   freeze_vision_modules (bool): whether to freeze vision modules
@dataclass
class GRPOModelConfig(trl.ModelConfig):
    
    hf_hub_token: Optional[str] = field(
        default=None,
        metadata={"help": "Auth token to log in with Hugging Face Hub."},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether to trust the execution of code from datasets/models defined on the Hub or not."},
    )
    
    # added for model loading
    rope_scaling_factor: Optional[float] = field(default=None)
    rope_scaling_type: Optional[str] = field(default=None)
    mm_spatial_pool_stride: Optional[int] = field(default=None)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None)
    mm_resampler_type: Optional[str] = field(default=None)
    
    use_pos_skipping: Optional[bool] = field(default=False)
    pos_skipping_range: Optional[int] = field(default=4096)
    
    model_class_name: Literal["none", "qwen2vl"] = field(
        default="none", 
        metadata={"help": "Used to init model class, format is XXXXForCausalLM. e.g. currently XXXX is chosen from LlavaLlama, LlavaMixtral, LlavaMistral, Llama"})
    
    #vision_tower: Optional[str] = field(default=None)
    # additional arguments for multimodal usage
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image (for QwenVL)"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image (for QwenVL)"},
    )
    #do_rescale: Optional[bool] = field(
    #    default=True
    #) 
    max_anyres_num: Optional[int] = field(
        default=12,
        metadata={"help": "Maximum number of anyres blocks for the image (for InternVL)"},
    )
    use_cache: Optional[bool] = field(
        default=True,
    )

    
    
