from transformers.utils.import_utils import _is_package_available
from packaging import version

LIGER_KERNEL_MIN_VERSION = "0.5.6"

# Use same as transformers.utils.import_utils
_rich_available = _is_package_available("rich")
_deepspeed_available = _is_package_available("deepspeed")
_vllm_available = _is_package_available("vllm")
_requests_available = _is_package_available("requests")
_is_liger_kernel_available, _liger_kernel_version = _is_package_available("liger_kernel", return_version=True)

def is_rich_available() -> bool:
    return _rich_available

def is_deepspeed_available() -> bool:
    return _deepspeed_available

def is_vllm_available() -> bool:
    return _vllm_available

def is_requests_available() -> bool:
    return _requests_available

def is_liger_kernel_available(min_version: str = LIGER_KERNEL_MIN_VERSION) -> bool:
    return _is_liger_kernel_available and version.parse(_liger_kernel_version) >= version.parse(min_version)