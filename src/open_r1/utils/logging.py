import logging
import os
import sys
from functools import lru_cache

# customize logging.Logger functions
def debug_rank0(self: "logging.Logger", *args, **kwargs) -> None:
    """print()-style debug logging, emitted on local rank 0 only.

    Arguments are stringified and space-joined exactly like print(), but only once the
    DEBUG level is actually enabled -- these fire on every training step, so keeping the
    str() lazy matters for anything large (reward tensors, masks, weight dicts).
    """
    if int(os.getenv("LOCAL_RANK", "0")) == 0 and self.isEnabledFor(logging.DEBUG):
        self.debug(" ".join(str(arg) for arg in args), **kwargs)
def info_rank0(self: "logging.Logger", *args, **kwargs) -> None:
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        self.info(*args, **kwargs)
def warning_rank0(self: "logging.Logger", *args, **kwargs) -> None:
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        self.warning(*args, **kwargs)
@lru_cache(None)
def warning_rank0_once(self: "logging.Logger", *args, **kwargs) -> None:
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        self.warning(*args, **kwargs)
logging.Logger.debug_rank0 = debug_rank0
logging.Logger.info_rank0 = info_rank0
logging.Logger.warning_rank0 = warning_rank0
logging.Logger.warning_rank0_once = warning_rank0_once