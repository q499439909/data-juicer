"""Download an immutable HTTP artifact without invoking a package manager."""

from __future__ import annotations

import os
import tempfile
import urllib.request
from pathlib import Path

from ..common import FileLock, PlanFlowError, sha256_file


class HttpFileModelBackend:
    name = "http-file"

    def __init__(self, *, cache_root: str | Path | None = None, preloaded_root: str | Path | None = None):
        configured = cache_root or os.environ.get("DSH_MODEL_ARTIFACT_CACHE")
        self.cache_root = Path(configured or Path.home() / ".cache" / "data_juicer" / "model-artifacts").resolve()
        preload = preloaded_root or os.environ.get("DSH_PRELOADED_MODEL_ROOT")
        self.preloaded_root = Path(preload).resolve() if preload else None

    def _target(self, binding: dict) -> Path:
        return self.cache_root / "http-file" / binding["lock_id"] / binding["filename"]

    def prepare(self, binding: dict, *, offline: bool) -> Path:
        candidates = []
        if self.preloaded_root:
            candidates.append(self.preloaded_root / "http-file" / binding["lock_id"] / binding["filename"])
        candidates.append(self._target(binding))
        for candidate in candidates:
            try:
                self.verify(binding, candidate)
                return candidate.resolve()
            except PlanFlowError:
                pass
        if offline:
            raise PlanFlowError("MODEL_NOT_PREPARED", f"Locked artifact is not available: {binding['lock_id']}")
        target = self._target(binding)
        target.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(target.parent / ".prepare.lock"):
            try:
                self.verify(binding, target)
                return target.resolve()
            except PlanFlowError:
                pass
            fd, temporary_name = tempfile.mkstemp(prefix=".download-", dir=target.parent)
            os.close(fd)
            temporary = Path(temporary_name)
            try:
                request = urllib.request.Request(binding["url"], headers={"User-Agent": "DSH-model-lock/1"})
                with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                self.verify(binding, temporary)
                os.replace(temporary, target)
            except PlanFlowError:
                raise
            except Exception as exc:
                raise PlanFlowError("MODEL_DOWNLOAD_BLOCKED", f"Could not download {binding['url']}: {exc}") from exc
            finally:
                temporary.unlink(missing_ok=True)
        return target.resolve()

    def verify(self, binding: dict, path: Path) -> None:
        if not path.is_file() or path.stat().st_size != int(binding["size"]):
            raise PlanFlowError("MODEL_FILE_MISSING", f"Locked artifact is missing or truncated: {path}")
        if sha256_file(path).removeprefix("sha256:") != binding["sha256"]:
            raise PlanFlowError("MODEL_HASH_MISMATCH", f"Locked artifact changed: {binding['lock_id']}")
