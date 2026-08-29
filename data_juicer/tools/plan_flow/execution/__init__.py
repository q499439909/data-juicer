"""Stable execution seam and the currently supported backend adapters."""

from .backend import ExecutionBackend
from .local_process import LocalProcessBackend
from .spec import RunHandle, RunResult, RunStatus, RuntimeSpec

__all__ = [
    "ExecutionBackend",
    "LocalProcessBackend",
    "RunHandle",
    "RunResult",
    "RunStatus",
    "RuntimeSpec",
]
