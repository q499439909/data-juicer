from .base import ExecutorBase
from .default_executor import DefaultExecutor
from .factory import ExecutorFactory

__all__ = ["ExecutorBase", "ExecutorFactory", "DefaultExecutor", "RayExecutor", "PartitionedRayExecutor"]


def __getattr__(name):
    """Keep optional Ray executors importable without loading Ray for local runs."""
    if name == "RayExecutor":
        from .ray_executor import RayExecutor

        return RayExecutor
    if name == "PartitionedRayExecutor":
        from .ray_executor_partitioned import PartitionedRayExecutor

        return PartitionedRayExecutor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
