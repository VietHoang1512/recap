import json
import os
from dataclasses import dataclass, Field
from typing import TYPE_CHECKING, Literal, Optional, Union
from datasets import load_dataset,  load_from_disk, DatasetDict, concatenate_datasets, interleave_datasets, Value, ClassLabel, Sequence
import numpy as np
if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset, DatasetDict
from torch.utils import data     
from transformers import TrainingArguments
from huggingface_hub import hf_hub_download

from .utils.logging import logging
from .dataset_utils import align_dataset, DatasetAttr


logger = logging.getLogger(__name__)


DATA_CONFIG = "src/open_r1/dataset_info.json"

# Root under which the prepared `save_to_disk` datasets live. `dataset_info.json` refers to it as
# ${DATA_ROOT}; override it to point at wherever `scripts/prepare_data.py` wrote the datasets.
DATA_ROOT = os.environ.setdefault("DATA_ROOT", os.path.abspath("./share_data"))


def resolve_data_path(path: str) -> str:
    """Expand ${DATA_ROOT} (and any other env var or ~) in a dataset_info.json file_name."""
    return os.path.expanduser(os.path.expandvars(path))


FILEEXT2TYPE = {
    "arrow": "arrow",
    "csv": "csv",
    "json": "json",
    "jsonl": "json",
    "parquet": "parquet",
    "txt": "text",
}

def is_env_enabled(env_var: str, default: str = "0") -> bool:
    r"""Check if the environment variable is enabled."""
    return os.getenv(env_var, default).lower() in ["true", "y", "1"]


def use_modelscope() -> bool:
    return is_env_enabled("USE_MODELSCOPE_HUB")

def use_openmind() -> bool:
    return is_env_enabled("USE_OPENMIND_HUB")

def check_version(requirement: str, mandatory: bool = False) -> None:
    r"""Optionally check the package version."""
    if is_env_enabled("DISABLE_VERSION_CHECK") and not mandatory:
        logger.warning_rank0_once("Version checking has been disabled, may lead to unexpected behaviors.")
        return

    if mandatory:
        hint = f"To fix: run `pip install {requirement}`."
    else:
        hint = f"To fix: run `pip install {requirement}` or set `DISABLE_VERSION_CHECK=1` to skip this check."

    require_version(requirement, hint)

def print_dtypes(ds):
    """
    Pretty-print the dtype (or feature type) of every column.
    Works for either a Dataset or the splits inside a DatasetDict.
    """
    def _show(split_name, dataset):
        logger.info_rank0(f"\n--- {split_name} ---")
        for col, feat in dataset.features.items():
            if isinstance(feat, Value):                # primitive -> has .dtype
                dtype = feat.dtype
            elif hasattr(feat, "dtype"):               # e.g. Audio, Image
                dtype = feat.dtype
            else:                                      # complex -> use class name
                dtype = feat.__class__.__name__
            logger.info_rank0(f"{col:<25} {dtype}")

    # Single Dataset
    if not isinstance(ds, DatasetDict):
        _show("dataset", ds)
    # Multiple splits
    else:
        for split_name, split_ds in ds.items():
            _show(split_name, split_ds)

def get_dataset_list(dataset_names: Optional[list[str]], dataset_dir: str, split: str)-> list["DatasetAttr"]:
    r"""Get the attributes of the datasets."""
    if dataset_names is None:
        dataset_names = []

    if dataset_dir == "ONLINE":
        dataset_info = None
    else:
        if dataset_dir.startswith("REMOTE:"):
            config_path = hf_hub_download(repo_id=dataset_dir[7:], filename=DATA_CONFIG, repo_type="dataset")
        else:
            config_path = os.path.join(dataset_dir, DATA_CONFIG) # default: "./dataset_info.json"

        try:
            with open(config_path) as f:
                dataset_info = json.load(f)
        except Exception as err:
            if len(dataset_names) != 0:
                raise ValueError(f"Cannot open {config_path} due to {str(err)}.")

            dataset_info = None

    dataset_list: list[DatasetAttr] = []
    for name in dataset_names:
        if dataset_info is None:  # dataset_dir is ONLINE
            if use_modelscope():
                load_from = "ms_hub"
            elif use_openmind():
                load_from = "om_hub"
            else:
                load_from = "hf_hub"
            dataset_attr = DatasetAttr(load_from, dataset_name=name, split=split)
            dataset_list.append(dataset_attr)
            continue

        if name not in dataset_info:
            raise ValueError(f"Undefined dataset {name} in {DATA_CONFIG}.")

        has_hf_url = "hf_hub_url" in dataset_info[name]
        has_ms_url = "ms_hub_url" in dataset_info[name]
        has_om_url = "om_hub_url" in dataset_info[name]

        if "load_from" in dataset_info[name]:
            dataset_attr = DatasetAttr(dataset_info[name]["load_from"], dataset_name=resolve_data_path(dataset_info[name]["file_name"]), file_ext=dataset_info[name]["file_ext"], split=split)
        elif has_hf_url or has_ms_url or has_om_url:
            if has_ms_url and (use_modelscope() or not has_hf_url):
                dataset_attr = DatasetAttr("ms_hub", dataset_name=dataset_info[name]["ms_hub_url"], file_ext=dataset_info[name]["file_ext"], split=split)
            elif has_om_url and (use_openmind() or not has_hf_url):
                dataset_attr = DatasetAttr("om_hub", dataset_name=dataset_info[name]["om_hub_url"], file_ext=dataset_info[name]["file_ext"], split=split)
            else:
                dataset_attr = DatasetAttr("hf_hub", dataset_name=dataset_info[name]["hf_hub_url"], file_ext=dataset_info[name]["file_ext"], split=split)
        elif "script_url" in dataset_info[name]:
            dataset_attr = DatasetAttr("script", dataset_name=dataset_info[name]["script_url"], file_ext=dataset_info[name]["file_ext"], split=split)
        else:
            # this would be most activated option
            dataset_attr = DatasetAttr("file", dataset_name=resolve_data_path(dataset_info[name]["file_name"]), file_ext=dataset_info[name]["file_ext"], split=split)

        dataset_attr.join(dataset_info[name])
        dataset_list.append(dataset_attr)

    return dataset_list


def _load_single_dataset(
    dataset_attr: "DatasetAttr",
    model_args,#: "Field",
    script_args,#: "Field",
    training_args: "TrainingArguments",
    using_split: str,
) -> Union["Dataset", "IterableDataset"]:
    r"""Load a single dataset and aligns it to the standard format."""
    logger.info_rank0(f"Loading dataset {dataset_attr}...")
    data_path, data_name, data_dir, data_files = None, None, None, None
    
    if dataset_attr.load_from in ["disk", "hf_hub", "ms_hub", "om_hub"]:
        data_path = dataset_attr.dataset_name
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "script":
        data_path = os.path.join(script_args.dataset_dir, dataset_attr.dataset_name)
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "file":
        # expect to have a list of idential files under a local disk folder
        # could be a list of parquet files, for example
        data_files = []
        # dataset_name should be ../../share_data/xxx
        local_path = os.path.join(script_args.dataset_dir, dataset_attr.dataset_name)
        if os.path.isdir(local_path):  # is directory
            for file_name in os.listdir(local_path):
                data_files.append(os.path.join(local_path, file_name))
        elif os.path.isfile(local_path):  # is file
            data_files.append(local_path)
        else:
            raise ValueError(f"File {local_path} not found.")
        
        #data_files = [data_file for data_file in data_files if data_file.endswith(dataset_attr.file_ext)]

        data_path = FILEEXT2TYPE.get(os.path.splitext(data_files[0])[-1][1:], None)
        if data_path is None:
            raise ValueError("Allowed file types: {}.".format(",".join(FILEEXT2TYPE.keys())))

        if any(data_path != FILEEXT2TYPE.get(os.path.splitext(data_file)[-1][1:], None) for data_file in data_files):
            raise ValueError("File types should be identical.")
    else:
        raise NotImplementedError(f"Unknown load type: {dataset_attr.load_from}.")
    if dataset_attr.load_from == "disk":
        dataset = DatasetDict.load_from_disk(
            data_path, 
            #name=data_name, # None
            #data_dir=data_dir, # None
            #data_files=data_files, # None
            #split=dataset_attr.split, # default, train
            #cache_dir=model_args.cache_dir,
           # token=model_args.hf_hub_token,
            #streaming=script_args.streaming,
            #num_proc=script_args.preprocessing_num_workers,
            #trust_remote_code=model_args.trust_remote_code,
        )
    elif dataset_attr.load_from == "ms_hub":
        check_version("modelscope>=1.11.0", mandatory=True)
        from modelscope import MsDataset  # type: ignore
        from modelscope.utils.config_ds import MS_DATASETS_CACHE  # type: ignore

        cache_dir = model_args.cache_dir or MS_DATASETS_CACHE
        dataset = MsDataset.load(
            dataset_name=data_path,
            subset_name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=using_split,
            #cache_dir=cache_dir,
            token=model_args.ms_hub_token,
            use_streaming=script_args.streaming,
        )
        if isinstance(dataset, MsDataset):
            dataset = dataset.to_hf_dataset()

    elif dataset_attr.load_from == "om_hub":
        check_version("openmind>=0.8.0", mandatory=True)
        from openmind import OmDataset  # type: ignore
        from openmind.utils.hub import OM_DATASETS_CACHE  # type: ignore

        cache_dir = model_args.cache_dir or OM_DATASETS_CACHE
        dataset = OmDataset.load_dataset(
            path=data_path,
            name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=using_split,
            #cache_dir=cache_dir,
            token=model_args.om_hub_token,
            streaming=script_args.streaming,
        )
    else:
        # most common case
        dataset = load_dataset(
            path=data_path, # extension type, like "parquet"
            name=data_name, # None
            data_dir=data_dir, # None
            data_files=data_files, # a list of local file paths
            split=using_split, # default, train
            #cache_dir=model_args.cache_dir,
            token=model_args.hf_hub_token,
            streaming=script_args.streaming,
            num_proc=script_args.preprocessing_num_workers,
            trust_remote_code=model_args.trust_remote_code,
        )

    if dataset_attr.num_samples is not None and not script_args.streaming:
        target_num = dataset_attr.num_samples
        indexes = np.random.permutation(len(dataset))[:target_num]  # all samples should be included
        target_num -= len(indexes)
        if target_num > 0:
            expand_indexes = np.random.choice(len(dataset), target_num)
            indexes = np.concatenate((indexes, expand_indexes), axis=0)

        assert len(indexes) == dataset_attr.num_samples, "Sample num mismatched."
        dataset = dataset.select(indexes)
        logger.info_rank0(f"Sampled {dataset_attr.num_samples} examples from dataset {dataset_attr}.")

    # if script_args.max_samples is not None:  # truncate dataset
    #     max_samples = min(script_args.max_samples, len(dataset))
    #     dataset = dataset.select(range(max_samples))

    return align_dataset(dataset, dataset_attr, script_args, training_args)

class MergedIterableDataset(data.IterableDataset):
    def __init__(self, datasets, probs=None, length=None):
        """
        Args:
            datasets (List[Dataset or IterableDataset]): list of Hugging Face datasets.
            probs (List[float], optional): sampling probabilities for interleaving.
            length (int, optional): total number of samples; if None, inferred from sub‑datasets.
        """
        super().__init__()
        self.datasets = datasets
        self.probs = probs or [1/len(datasets)] * len(datasets)
        self._length = length
        # Try to infer length if not provided
        if self._length is None:
            inferred = []
            for ds in self.datasets:
                try:
                    inferred.append(len(ds))
                except (TypeError, AttributeError):
                    inferred = None
                    break
            if inferred is not None:
                self._length = int(sum(inferred))

    def __iter__(self):
        iterators = [iter(ds) for ds in self.datasets]
        while True:
            idx = np.random.choice(len(iterators), p=self.probs)
            try:
                yield next(iterators[idx])
            except StopIteration:
                break

    def __len__(self):
        """
        Returns the total number of samples for Trainer to compute max_steps.
        Raises:
            ValueError: if length is unknown and cannot be inferred.
        """
        if self._length is None:
            raise ValueError(
                "MergedIterableDataset has unknown length; "
                "please specify `length` when initializing or set `max_steps` in TrainingArguments."
            )
        return self._length


def merge_dataset(
    all_datasets: list[Union["Dataset", "IterableDataset"]], 
    script_args,#: "Field", 
    seed: int
) -> Union["Dataset", "IterableDataset"]:
    r"""Merge multiple datasets to a unified dataset."""
    if len(all_datasets) == 1:
        return all_datasets[0]

    elif script_args.mix_strategy == "concat":
        if script_args.streaming:
            logger.warning_rank0_once("The samples between different datasets will not be mixed in streaming mode.")

        return concatenate_datasets(all_datasets)
    elif script_args.mix_strategy == "interleave_custom":
        return MergedIterableDataset(all_datasets, script_args.interleave_probs)
    elif script_args.mix_strategy.startswith("interleave"):
        if not script_args.streaming:
            logger.warning_rank0_once("We recommend using `mix_strategy=concat` in non-streaming mode.")

        return interleave_datasets(
            datasets=all_datasets,
            probabilities=script_args.interleave_probs,
            seed=seed,
            stopping_strategy="first_exhausted" if script_args.mix_strategy.endswith("under") else "all_exhausted",
        )

    else:
        raise ValueError(f"Unknown mixing strategy: {script_args.mix_strategy}.")


def _get_merged_dataset(
    #dataset_names: Optional[list[str]],
    model_args,#: "Field",
    script_args,#: "Field",
    training_args: "TrainingArguments",
    using_split: str,
    merge: bool = True,
)-> Optional[Union["Dataset", "IterableDataset", dict[str, "Dataset"]]]:
    assert using_split in ["train", "test"], f"Invalid {using_split} must be train or test"
    if using_split == "train":
        dataset_names = script_args.dataset_names
    else:
        dataset_names = script_args.eval_dataset_names
    # logger.info_rank0(f"Loading dataset {dataset_names}. split {using_split}")
    if dataset_names is None:
        return None
    
    datasets = {}
    # using_split = script_args.split if not training_args.do_eval else script_args.dataset_test_split
    for dataset_name, dataset_attr in zip(dataset_names, get_dataset_list(dataset_names, script_args.dataset_dir, using_split)):
        logger.info_rank0(f"Loading dataset {dataset_name}. split {using_split}")
        cache_dir = f"./cache/{dataset_name}/"
        if os.path.isdir(cache_dir):
            logger.info_rank0(f"Cache found, loading from {cache_dir}")
            datasets[dataset_name] =  load_from_disk(cache_dir)
            
        else:
            datasets[dataset_name] = _load_single_dataset(dataset_attr, model_args, script_args, training_args, using_split)
            datasets[dataset_name].save_to_disk(cache_dir)
        logger.info_rank0(f"Loaded dataset {dataset_name}. Got {datasets[dataset_name]}. Total sample {len(datasets[dataset_name])}")
        print_dtypes(datasets[dataset_name])
        if isinstance(datasets[dataset_name], DatasetDict):
            datasets[dataset_name] = datasets[dataset_name][using_split]
        # FIXME: hack
        # datasets[dataset_name].select(range(100))
        #assert False, type(datasets[dataset_name]) # datasetdict?

    if merge:
        merged_dataset = merge_dataset(list(datasets.values()), script_args, seed=training_args.seed)
        if script_args.max_samples is not None:  # truncate dataset
            max_samples = min(script_args.max_samples, len(merged_dataset))
            merged_dataset = merged_dataset.shuffle(seed=training_args.seed).select(range(max_samples))
        return merged_dataset
    else:
        return datasets

    
    
    
    
    
