from datasets import Dataset, IterableDataset
from dataclasses import dataclass, Field
from typing import TYPE_CHECKING, Literal, Optional, Union
from transformers import TrainingArguments

from .converter import get_dataset_converter, DatasetAttr


def align_dataset(
    dataset: Union["Dataset", "IterableDataset"], 
    dataset_attr: "DatasetAttr", 
    script_args,#: "Field", 
    training_args: "TrainingArguments") -> Union["Dataset", "IterableDataset"]:
   
    next(iter(dataset))
    try:
        column_names = list(next(iter(dataset)).keys())
    except:
        
        try:
            column_names = list(dataset[script_args.dataset_test_split].features)
        except:
            column_names = list(dataset[script_args.split].features)
    kwargs = {}
    if dataset_attr.load_from == "disk":
        kwargs = dict(
            num_proc=script_args.preprocessing_num_workers,
            load_from_cache_file=(not script_args.overwrite_cache) or (training_args.local_process_index != 0),
            desc="Converting format of dataset",
        )
    elif not script_args.streaming:
        kwargs = dict(
            num_proc=script_args.preprocessing_num_workers,
            load_from_cache_file=(not script_args.overwrite_cache) or (training_args.local_process_index != 0),
            desc="Converting format of dataset",
        )
    
    dataset_converter = get_dataset_converter(dataset_attr.formatting, dataset_attr, script_args)
    return dataset.map(
        dataset_converter,
        batched=False,
        remove_columns=column_names, # remove all original columns
        **kwargs,
    )
    
