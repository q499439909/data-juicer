"""Account-scoped immutable operator artifacts. Never imports user Python."""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from contextvars import ContextVar
from pathlib import Path

from .common import (
    FileLock,
    PlanFlowError,
    canonical_json,
    read_json,
    write_json_atomic,
    write_text_atomic,
)

current_user: ContextVar[str | None] = ContextVar("operator_user", default=None)
CATEGORIES = frozenset(
    {
        "mapper",
        "filter",
        "selector",
        "deduplicator",
        "aggregator",
        "grouper",
        "pipeline",
    }
)


def safe_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_-]{0,127}", value):
        raise PlanFlowError("INVALID_OPERATOR_ID", "Invalid account, operator, or version identifier")
    return value


def digest(value) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


class UserOperatorStore:
    def __init__(self, root=None, user_id=None):
        self.user_id = safe_id(user_id or current_user.get() or "")
        configured = root or os.environ.get("DSH_USER_DATA_ROOT")
        if not configured:
            raise PlanFlowError(
                "USER_STORE_NOT_CONFIGURED",
                "Set DSH_USER_DATA_ROOT to a persistent account data directory",
            )
        self.root = Path(configured).expanduser().resolve()
        if os.name == "nt" and not str(self.root).startswith("\\\\?\\"):
            raw = str(self.root)
            self.root = Path("\\\\?\\UNC\\" + raw[2:] if raw.startswith("\\\\") else "\\\\?\\" + raw)
        self.home = self.path(self.root / self.user_id)
        self.catalog = self.home / "custom_operators"

    def path(self, path):
        path = Path(path).resolve()
        if not path.is_relative_to(self.root) or path == self.root:
            raise PlanFlowError("USER_PATH_FORBIDDEN", "Operator path escaped the account store")
        # Reject account symlinks/junctions that lead into a different account.
        if path.parts[len(self.root.parts)] != self.user_id:
            raise PlanFlowError("USER_PATH_FORBIDDEN", "Operator path escaped the account")
        return path

    def operator_path(self, category, name):
        if category not in CATEGORIES:
            raise PlanFlowError("INVALID_OPERATOR_TYPE", "Unsupported DJ operator type")
        return self.path(self.catalog / category / safe_id(name))

    def publish(self, source, schema, manifest, contract, report):
        source = source.replace("\r\n", "\n")
        name, category = safe_id(schema["name"]), schema["type"]
        base = self.operator_path(category, name)
        payload = {
            "source": source,
            "schema": schema,
            "manifest": manifest,
            "contract": contract,
        }
        version = digest(payload)
        with FileLock(base / ".lock"):
            pointer = read_json(base / "operator.json") if (base / "operator.json").exists() else {}
            if pointer and pointer.get("contract_hash") != digest(contract):
                raise PlanFlowError(
                    "CONTRACT_CHANGED",
                    "Use a new operator ID for a changed acceptance contract",
                )
            target = self.path(base / "versions" / version)
            if not target.exists():
                # The pointer is the publication boundary; partial unreferenced versions are not discoverable.
                final_target = target
                target = self.path(base / "versions" / f".pending-{uuid.uuid4().hex}")
                write_text_atomic(target / "operator.py", source)
                write_json_atomic(target / "schema.json", schema)
                write_json_atomic(target / "manifest.json", manifest)
                write_json_atomic(
                    target / "dependencies.lock.json",
                    {"requirements": manifest.get("dependencies", [])},
                )
                write_json_atomic(
                    target / "model_refs.json",
                    {"models": manifest.get("model_refs", [])},
                )
                write_json_atomic(target / "validation-contract.json", contract)
                write_json_atomic(target / "validation.json", report)
                for filename, content in manifest.get("assets", {}).items():
                    write_text_atomic(self.path(target / "assets" / filename), content)
                os.replace(target, final_target)
                target = final_target
            else:
                self.resolve(f"user:{name}:{version}", allow_unlisted=True, category=category)
            versions = list(dict.fromkeys([*pointer.get("versions", []), version]))
            previous = pointer.get("current")
            # An experimental revision must not displace an established validated revision.
            previous_valid = (
                previous and read_json(base / "versions" / previous / "validation.json")["status"] == "validated"
            )
            write_json_atomic(
                base / "operator.json",
                {
                    "operator_id": name,
                    "category": category,
                    "versions": versions,
                    "current": previous if previous_valid and report["status"] != "validated" else version,
                    "contract_hash": digest(contract),
                },
            )
        return self.resolve(f"user:{name}:{version}")

    def resolve(self, ref, *, allow_unlisted=False, category=None):
        parts = str(ref).split(":")
        if len(parts) != 3 or parts[0] != "user" or not re.fullmatch(r"[a-f0-9]{64}", parts[2]):
            raise PlanFlowError("INVALID_OPERATOR_REF", "Expected user:<operator_id>:<content_hash>")
        name, version = safe_id(parts[1]), parts[2]
        for kind in [category] if category else sorted(CATEGORIES):
            base = self.operator_path(kind, name)
            pointer = base / "operator.json"
            if not allow_unlisted and (not pointer.exists() or version not in read_json(pointer).get("versions", [])):
                continue
            target = self.path(base / "versions" / version)
            if not target.is_dir():
                continue
            payload = {
                "source": self.path(target / "operator.py").read_text(encoding="utf-8"),
                "schema": read_json(self.path(target / "schema.json")),
                "manifest": read_json(self.path(target / "manifest.json")),
                "contract": read_json(self.path(target / "validation-contract.json")),
            }
            if digest(payload) != version:
                raise PlanFlowError(
                    "OPERATOR_HASH_MISMATCH",
                    "Personal operator artifact changed after validation",
                )
            for filename, content in payload["manifest"].get("assets", {}).items():
                if self.path(target / "assets" / filename).read_text(encoding="utf-8") != content:
                    raise PlanFlowError("OPERATOR_HASH_MISMATCH", "Personal operator asset changed")
            report = read_json(self.path(target / "validation.json"))
            if report.get("status") not in {"validated", "experimental"}:
                raise PlanFlowError("OPERATOR_NOT_AVAILABLE", "Operator has not passed execution tests")
            return {
                **payload["schema"],
                "candidate_id": ref,
                "provider": "user",
                "operator_id": name,
                "version": version,
                "status": report["status"],
                "validation_basis": "user_acceptance_tests",
                "validation_summary": report,
                "runtime_status": "unknown",
                "replaces": payload["manifest"].get("replaces"),
                "_path": str(target / "operator.py"),
                "_manifest": payload["manifest"],
            }
        raise PlanFlowError(
            "OPERATOR_NOT_FOUND",
            "Personal operator version is not available for this account",
        )

    def candidates(self):
        result = []
        for pointer in sorted(self.catalog.glob("*/*/operator.json")):
            data = read_json(self.path(pointer))
            result.append(self.resolve(f"user:{data['operator_id']}:{data['current']}"))
        return result


def public_candidate(candidate):
    return {key: value for key, value in candidate.items() if not key.startswith("_")}


def resolve_bindings(plan, store=None):
    """Resolve exact step references. No caller-provided paths or schema bypasses."""
    from .discovery import operator_schema

    bindings = plan.get("operator_bindings", [])
    if plan.get("recipe", {}).get("custom_operator_paths"):
        raise PlanFlowError("CUSTOM_PATH_FORBIDDEN", "Use operator_bindings, not custom_operator_paths")
    if not bindings:
        return {}
    store = store or UserOperatorStore()
    owner = plan.get("operator_owner")
    if owner and owner != store.user_id:
        raise PlanFlowError("OPERATOR_OWNER_FORBIDDEN", "Plan belongs to a different account")
    steps = plan.get("recipe", {}).get("process", [])
    resolved, indices = {}, set()
    for binding in bindings:
        index = binding.get("step_index")
        if type(index) is not int or index < 0 or index >= len(steps) or index in indices:
            raise PlanFlowError(
                "INVALID_OPERATOR_BINDING",
                "step_index must be unique and inside recipe.process",
            )
        indices.add(index)
        if binding.get("provider") != "user":
            raise PlanFlowError(
                "INVALID_OPERATOR_BINDING",
                "Only user bindings are required; DJ names resolve natively",
            )
        candidate = store.resolve(f"user:{binding.get('operator_id')}:{binding.get('version')}")
        from .user_operator_resources import model_refs

        model_refs(candidate["_manifest"].get("model_refs", []))
        name = candidate["name"]
        if list(steps[index]) != [name] or operator_schema(name):
            raise PlanFlowError(
                "OPERATOR_NAME_CONFLICT",
                "Binding must match its step and must not replace a DJ registration",
            )
        if name in resolved and resolved[name]["version"] != candidate["version"]:
            raise PlanFlowError(
                "OPERATOR_VERSION_CONFLICT",
                "One run cannot load multiple versions of one operator",
            )
        resolved[name] = candidate
    return resolved
