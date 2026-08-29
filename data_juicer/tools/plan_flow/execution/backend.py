"""Execution backend seam used by PlanRunner and backend contract tests."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .spec import RunHandle, RunResult, RunStatus, RuntimeSpec


@runtime_checkable
class ExecutionBackend(Protocol):
    """Lifecycle interface implemented by local, Docker, and future adapters."""

    name: str

    def start(self, spec: RuntimeSpec) -> RunHandle: ...

    def inspect(self, handle: RunHandle) -> RunStatus: ...

    def cancel(self, handle: RunHandle) -> None: ...

    def collect(self, handle: RunHandle) -> RunResult: ...

    def cleanup(self, handle: RunHandle) -> None: ...
