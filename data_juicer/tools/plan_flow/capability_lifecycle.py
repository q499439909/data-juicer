"""Approval-gated lifecycle for reusable semantic capabilities and operator artifacts."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .capability_schema import (
    CapabilityCatalog,
    CapabilityDescriptor,
    OperatorArtifact,
    OperatorArtifactCatalog,
)
from .common import FileLock, PlanFlowError, canonical_json, now_iso, read_json, sha256_bytes, write_json_atomic


@dataclass(frozen=True)
class CapabilityProposal:
    proposal_id: str
    capability_id: str
    content_hash: str
    status: str
    path: Path
    reused: bool = False


Validator = Callable[[CapabilityProposal], dict[str, Any]]


class CapabilityLifecycle:
    """Validate a frozen multi-artifact proposal before its single approval gate."""

    def __init__(self, worker_root: str | Path, *, validator: Validator):
        self.worker_root = Path(worker_root).resolve()
        self.root = self.worker_root / "broker-state" / "capability-proposals-v2"
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifacts = OperatorArtifactCatalog(self.worker_root)
        self.capabilities = CapabilityCatalog(self.worker_root)
        self.validator = validator

    def prepare(
        self,
        capability: CapabilityDescriptor,
        artifacts: tuple[OperatorArtifact, ...],
    ) -> CapabilityProposal:
        by_id = {item.artifact_id: item for item in artifacts}
        if len(by_id) != len(artifacts) or tuple(by_id) != capability.operator_artifact_ids:
            raise PlanFlowError(
                "CAPABILITY_ARTIFACT_MISMATCH",
                "Proposal artifacts must exactly match capability.operator_artifact_ids in order",
            )
        try:
            existing = self.capabilities.resolve(capability.capability_id)
        except PlanFlowError as error:
            if error.code != "CAPABILITY_MISSING":
                raise
        else:
            if existing.content_hash != capability.content_hash:
                raise PlanFlowError("CAPABILITY_CONFLICT", "Available capability id has different content")
            for item in artifacts:
                if self.artifacts.resolve(item.artifact_id).content_hash != item.content_hash:
                    raise PlanFlowError("OPERATOR_ARTIFACT_CONFLICT", "Available operator artifact content differs")
            return CapabilityProposal(
                "available-" + capability.content_hash.removeprefix("sha256:")[:24],
                capability.capability_id,
                capability.content_hash,
                "available",
                self.capabilities.root,
                True,
            )

        frozen = {
            "schema_version": 1,
            "capability": capability.to_dict(),
            "operator_artifacts": [item.to_dict() for item in artifacts],
        }
        content_hash = sha256_bytes(canonical_json(frozen))
        proposal_id = "proposal-" + content_hash.removeprefix("sha256:")[:24]
        path = self.root / proposal_id
        with FileLock(self.root / f".{proposal_id}.lock"):
            if path.is_dir():
                stored = read_json(path / "proposal.json")
                if stored.get("content_hash") != content_hash:
                    raise PlanFlowError("CAPABILITY_CONTENT_CHANGED", "Capability proposal id collision")
            else:
                path.mkdir(parents=False)
                write_json_atomic(
                    path / "proposal.json",
                    {
                        **frozen,
                        "proposal_id": proposal_id,
                        "content_hash": content_hash,
                        "status": "staging",
                        "created_at": now_iso(),
                    },
                )
        return self.get(proposal_id)

    def build_and_validate(self, proposal_id: str) -> CapabilityProposal:
        path = self._path(proposal_id)
        with FileLock(self.root / f".{proposal_id}.lock"):
            state = self._load_and_verify(proposal_id)
            if state["status"] == "pending_approval":
                return self._view(path, state)
            if state["status"] != "staging":
                raise PlanFlowError("INVALID_CAPABILITY_STATE", f"Cannot validate from {state['status']}")
            state["status"] = "building"
            write_json_atomic(path / "proposal.json", state)
            proposal = self._view(path, state)
            try:
                report = self.validator(proposal)
                if not isinstance(report, dict) or report.get("passed") is not True:
                    raise PlanFlowError("CAPABILITY_VALIDATION_FAILED", "Capability validation did not pass")
            except Exception:
                state["status"] = "validation_failed"
                write_json_atomic(path / "proposal.json", state)
                raise
            state["status"] = "pending_approval"
            state["validation"] = {**report, "validated_at": now_iso()}
            write_json_atomic(path / "proposal.json", state)
            return self._view(path, state)

    def approve(self, proposal_id: str, expected_content_hash: str, *, note: str) -> CapabilityProposal:
        path = self._path(proposal_id)
        with FileLock(self.root / f".{proposal_id}.lock"):
            state = self._load_and_verify(proposal_id)
            if state["status"] != "pending_approval":
                raise PlanFlowError("CAPABILITY_NOT_VALIDATED", "Capability can be approved only after validation")
            if state["content_hash"] != expected_content_hash:
                raise PlanFlowError("CAPABILITY_CONTENT_CHANGED", "Approval hash does not match proposal content")
            state["status"] = "approved"
            state["approval"] = {
                "content_hash": expected_content_hash,
                "note": str(note),
                "approved_at": now_iso(),
            }
            write_json_atomic(path / "proposal.json", state)
            return self._view(path, state)

    def publish(self, proposal_id: str) -> CapabilityProposal:
        path = self._path(proposal_id)
        with FileLock(self.root / f".{proposal_id}.lock"):
            state = self._load_and_verify(proposal_id)
            if state["status"] == "available":
                return self._view(path, state)
            if state["status"] != "approved":
                raise PlanFlowError("CAPABILITY_APPROVAL_REQUIRED", "Capability proposal is not approved")
            approval = state.get("approval", {})
            if approval.get("content_hash") != state["content_hash"]:
                raise PlanFlowError("CAPABILITY_CONTENT_CHANGED", "Approved content no longer matches proposal")
            state["status"] = "publishing"
            write_json_atomic(path / "proposal.json", state)
            artifacts = tuple(OperatorArtifact.from_dict(item) for item in state["operator_artifacts"])
            capability = CapabilityDescriptor.from_dict(state["capability"])
            for artifact in artifacts:
                self.artifacts.publish(artifact)
            self.capabilities.publish(capability)
            state["status"] = "available"
            state["published_at"] = now_iso()
            write_json_atomic(path / "proposal.json", state)
            return self._view(path, state)

    def get(self, proposal_id: str) -> CapabilityProposal:
        path = self._path(proposal_id)
        return self._view(path, self._load_and_verify(proposal_id))

    def _load_and_verify(self, proposal_id: str) -> dict[str, Any]:
        state = read_json(self._path(proposal_id) / "proposal.json")
        frozen = {
            "schema_version": state.get("schema_version"),
            "capability": state.get("capability"),
            "operator_artifacts": state.get("operator_artifacts"),
        }
        if sha256_bytes(canonical_json(frozen)) != state.get("content_hash"):
            raise PlanFlowError("CAPABILITY_CONTENT_CHANGED", "Frozen capability proposal content changed")
        return state

    def _path(self, proposal_id: str) -> Path:
        if not re.fullmatch(r"proposal-[0-9a-f]{24}", str(proposal_id or "")):
            raise PlanFlowError("INVALID_CAPABILITY", "Invalid capability proposal id")
        return self.root / proposal_id

    @staticmethod
    def _view(path: Path, state: dict[str, Any]) -> CapabilityProposal:
        return CapabilityProposal(
            str(state["proposal_id"]),
            str(state["capability"]["capability_id"]),
            str(state["content_hash"]),
            str(state["status"]),
            path,
        )
