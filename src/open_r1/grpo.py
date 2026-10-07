import os
import sys

import datasets
from datasets import load_dataset, DatasetDict, Dataset
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from transformers import TrainingArguments

from trl import TrlParser, get_peft_config


from .configs import GRPOScriptArguments, GRPOConfig, GRPOModelConfig
from .vllm_serve import vLLMScriptArguments

from .utils.wandb_logging import init_wandb_training
from .utils.logging import logging
from .utils.callbacks import get_callbacks
from .rewards import get_reward_funcs
from .loader import _get_merged_dataset
from .models import get_model_processor
from .trainers.grpo_trainer import GRPOTrainer

logger = logging.getLogger(__name__)



# not fixing Qwen2_5 flash attention forward yet

# not defining Func get_vlm_module(model_name_or_path) yet

def main(script_args, training_args, model_args, vllm_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)


    #assert len(script_args.dataset_names) == len(script_args.reward_funcs)
    ################
    # Load tokenizer
    ################
    # do not load for now!
    #tokenizer = get_tokenizer(model_args, training_args)
    
    ################
    # Load reward functions
    ################
    # Get reward functions from the registry
    # this is based on open_r1 reward function registry!
    # this is list[Func]
    # different datasets may have different instances of same rewards
    # later refer to a dictionary to decide which dataset sample uses which rewards
    reward_funcs = get_reward_funcs(script_args)

    ################
    # Load dataset
    ################
    # this is different from VLM-R1's LazySupervisedDataset
    # using load_dataset/load from disk for now, switch to more advanced mergeable dataset
    # from llama_factory later on    
    with training_args.main_process_first(desc="Loading datasets"):
        train_dataset = _get_merged_dataset(model_args, script_args, training_args, using_split=script_args.split, merge=True) 
        eval_dataset = _get_merged_dataset(model_args, script_args, training_args, using_split=script_args.dataset_test_split, merge=True) 
        
    # for now only loading the train split for each dataset 
    # (or in other words, the SPLIT split following script_args)
    print("train_dataset", train_dataset)
    dataset = DatasetDict({script_args.dataset_test_split: eval_dataset, script_args.split: train_dataset})
        
    ################
    # Initialize model kwargs
    ################
    ### This whole part does not exist in VLM-R1 or Visual-RFT!
    
    logger.info("*** Initializing model kwargs ***")
    # get callable torch_dtype instaed of string
    
    # these are handled during model loading
    #model_kwargs = dict(
    #    revision=model_args.model_revision,
    #    trust_remote_code=model_args.trust_remote_code,
    #    attn_implementation=model_args.attn_implementation,
    #    torch_dtype=torch_dtype,
    #    use_cache=False if training_args.gradient_checkpointing else True,
    #)
    #training_args.model_init_kwargs = model_kwargs
    
    model_attributes, model, processor, pad_token_id, customized_kwargs = get_model_processor(model_args, training_args)

    #############################
    # Initialize the GRPO trainer
    #############################
    # different from VLM-R1: 
    # -vlm_module=vlm_module_cls(), 
    # -freeze_vision_modules=model_args.freeze_vision_modules,
    # -attn_implementation=model_args.attn_implementation,
    # -max_pixels=script_args.max_pixels,
    # -min_pixels=script_args.min_pixels,
    # -max_anyres_num=script_args.max_anyres_num,
    # -torch_dtype=model_args.torch_dtype,
    # +callbacks=get_callbacks(training_args, model_args),  
    logger.info(f"dataset {dataset}")     
    trainer = GRPOTrainer(
        model_name_or_path=model_args.model_name_or_path,
        customized_kwargs=customized_kwargs,
        model=model,
        model_attributes=model_attributes,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.split], # default is train; conflict would occur if other values as fixed to be train in DatasetAttr
        eval_dataset=dataset[script_args.dataset_test_split] if (training_args.eval_strategy != "no")  else None,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        processing_class=processor, # open-r1 have this, but not using tokenizer for now
        pad_token_id=pad_token_id,
        vllm_args=vllm_args
    )
    
    # if not training_args.do_eval:
    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    logging.info(f"checkpoint {checkpoint}")
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": ",".join(script_args.dataset_names),
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        #trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)
    
    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)

    ##########
    # Evaluate
    ##########
    # else:
    #     logger.info("*** Evaluate ***")
    #     metrics = trainer.evaluate()
    #     # grab the last history item that has eval_ keys
    #     for record in reversed(trainer.state.log_history):
    #         if any(k.startswith("eval_") for k in record):
    #             metrics.update({k: v for k, v in record.items() if k.startswith("eval_")})
    #             break

        
    #     metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
    #     # enable this would push item to wandb
    #     #trainer.log_metrics("eval", metrics)
        
    #     trainer.save_metrics("eval", metrics)

    
    


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig, vLLMScriptArguments))
    script_args, training_args, model_args, vllm_args = parser.parse_args_and_config()
    if training_args.do_eval:
        setattr(training_args, "reward_funcs", ["accuracy_iou_coco6k","accuracy_sat","format_sat"])
        setattr(training_args, "reward_weights", [1.0, 1.0, 1.0])
        #assert False, [training_args.reward_funcs, training_args.reward_weights]
    main(script_args, training_args, model_args, vllm_args)
