import contextlib
import functools
import time
from typing import Generator

from transformers import Trainer, is_wandb_available


if is_wandb_available():
    import wandb


@contextlib.contextmanager
def profiling_context(trainer: Trainer, name: str) -> Generator[None, None, None]:
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    duration = end_time - start_time

    if "wandb" in trainer.args.report_to and wandb.run is not None and trainer.accelerator.is_main_process:
        wandb.log({f"profiling/Time taken: {trainer.__class__.__name__}.{name}": duration})
    


def profiling_decorator(func: callable) -> callable:
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        with profiling_context(self, func.__name__):
            return func(self, *args, **kwargs)

    return wrapper