"""Physical model stores used by the environment-neutral lock resolver."""

from .huggingface import HuggingFaceModelBackend
from .http_file import HttpFileModelBackend
from .local_file import LocalFileModelBackend
from .modelscope import ModelScopeModelBackend
from .python_distribution import PythonDistributionModelBackend
from .torch_hub import TorchHubModelBackend

__all__ = [
    "HuggingFaceModelBackend",
    "HttpFileModelBackend",
    "LocalFileModelBackend",
    "ModelScopeModelBackend",
    "PythonDistributionModelBackend",
    "TorchHubModelBackend",
]
