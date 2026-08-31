"""Verified, path-safe access to Data-Juicer run outputs."""

from __future__ import annotations

import hashlib
import mimetypes
import os
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .common import PlanFlowError, is_within, read_json, read_yaml, require_workspace, sha256_file
from .store import PlanStore


@dataclass(frozen=True)
class OpenedAsset:
    path: Path
    media_type: str
    display_name: str
    size: int
    sha256: str


def _asset_id(relative: str) -> str:
    return "asset_" + hashlib.sha256(relative.encode("utf-8")).hexdigest()[:32]


def _media_type(relative: str) -> str:
    if relative.casefold().endswith(".jsonl"):
        return "application/x-ndjson"
    return mimetypes.guess_type(relative)[0] or "application/octet-stream"


def _safe_relative(value: Any) -> str:
    relative = str(value or "").replace("\\", "/")
    candidate = PurePosixPath(relative)
    if not relative or candidate.is_absolute() or ".." in candidate.parts or "." in candidate.parts:
        raise PlanFlowError("INVALID_RESULT_MANIFEST", "Result manifest contains an unsafe output path")
    if candidate.parts[0].endswith(":"):
        raise PlanFlowError("INVALID_RESULT_MANIFEST", "Result manifest contains an absolute output path")
    return candidate.as_posix()


class RunOutputGateway:
    """Hides run layout, manifest validation, and asset resolution behind one interface."""

    def inspect_run(
        self,
        workspace_root: str | Path,
        task_id: str,
        plan_version: str,
        result_ref: str,
    ) -> dict[str, Any]:
        context = self._context(workspace_root, task_id, plan_version, result_ref)
        state = context["state"]
        if state.get("status") == "failed" or state.get("error") or state.get("error_code"):
            return {"eligible": False, "reason": "run_failed"}
        if state.get("status") not in {"succeeded", "cancelled"}:
            return {"eligible": False, "reason": "run_not_complete"}
        manifest_path = context["output_root"] / "result-manifest.json"
        if not manifest_path.is_file() or manifest_path.is_symlink():
            return {"eligible": False, "reason": "no_verified_output"}
        manifest, inventory = self._verified_inventory(context, manifest_path)
        if not inventory:
            return {"eligible": False, "reason": "no_verified_output"}

        view, degraded = self._view(context["output_root"], inventory)
        task = read_yaml(context["task_root"] / "task.yaml")
        title = str(view.get("title") or task.get("title") or "管线结果")[:300]
        summary = view.get("summary") if isinstance(view.get("summary"), dict) else {}
        return {
            "eligible": True,
            "resultRef": str(result_ref),
            "taskId": str(task_id),
            "planVersion": str(plan_version),
            "internalRunId": str(result_ref),
            "title": title,
            "status": "available" if state.get("status") == "succeeded" and not degraded else "partial",
            "createdAt": state.get("created_at"),
            "completedAt": state.get("updated_at") or manifest.get("finished_at"),
            "manifestHash": sha256_file(manifest_path),
            "fileCount": len(inventory),
            "totalBytes": sum(item["size"] for item in inventory.values()),
            "recordCount": summary.get("record_count"),
            "labels": summary.get("labels") if isinstance(summary.get("labels"), dict) else {},
            "metrics": summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {},
            "mediaTypes": self._media_summary(inventory),
            "assets": self._public_assets(inventory, view),
        }

    def open_asset(
        self,
        workspace_root: str | Path,
        task_id: str,
        plan_version: str,
        result_ref: str,
        asset_id: str,
    ) -> OpenedAsset:
        context = self._context(workspace_root, task_id, plan_version, result_ref)
        manifest_path = context["output_root"] / "result-manifest.json"
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise PlanFlowError("RESULT_NOT_FOUND", "Run has no verified result manifest")
        _, inventory = self._verified_inventory(context, manifest_path)
        item = next((entry for entry in inventory.values() if entry["assetId"] == asset_id), None)
        if item is None:
            raise PlanFlowError("ASSET_NOT_FOUND", "Unknown result asset")
        path = context["output_root"] / item["relative"]
        if path.is_symlink() or not path.is_file() or not is_within(path, context["output_root"]):
            raise PlanFlowError("RESULT_INTEGRITY_FAILED", "Result asset is no longer safe to open")
        if path.stat().st_size != item["size"] or sha256_file(path) != item["sha256"]:
            raise PlanFlowError("RESULT_INTEGRITY_FAILED", "Result asset changed after the manifest was created")
        return OpenedAsset(path, item["mediaType"], item["name"], item["size"], item["sha256"])

    def delete_outputs(
        self,
        workspace_root: str | Path,
        task_id: str,
        plan_version: str,
        result_ref: str,
        expected_manifest_hash: str,
    ) -> dict[str, Any]:
        context = self._context(workspace_root, task_id, plan_version, result_ref)
        manifest_path = context["output_root"] / "result-manifest.json"
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise PlanFlowError("RESULT_NOT_FOUND", "Run has no verified result manifest")
        if sha256_file(manifest_path) != expected_manifest_hash:
            raise PlanFlowError("RESULT_CHANGED", "Result manifest changed after deletion was confirmed")
        self._verified_inventory(context, manifest_path)
        shutil.rmtree(context["output_root"])
        return {"deleted": True, "resultRef": str(result_ref)}

    def create_archive(
        self,
        workspace_root: str | Path,
        task_id: str,
        plan_version: str,
        result_ref: str,
        asset_ids: list[str] | None,
    ) -> OpenedAsset:
        context = self._context(workspace_root, task_id, plan_version, result_ref)
        manifest_path = context["output_root"] / "result-manifest.json"
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise PlanFlowError("RESULT_NOT_FOUND", "Run has no verified result manifest")
        _, inventory = self._verified_inventory(context, manifest_path)
        by_id = {item["assetId"]: item for item in inventory.values()}
        if asset_ids is None:
            selected = list(inventory.values())
            display_name = "result-all.zip"
        else:
            unique_ids = list(dict.fromkeys(str(value) for value in asset_ids))
            if not unique_ids or any(value not in by_id for value in unique_ids):
                raise PlanFlowError("INVALID_ARCHIVE_SELECTION", "Archive selection contains an unknown result asset")
            selected = [by_id[value] for value in unique_ids]
            display_name = "result-selection.zip"
        selected.sort(key=lambda item: item["relative"])
        cache_key = hashlib.sha256(
            (sha256_file(manifest_path) + "\0" + "\0".join(item["assetId"] for item in selected)).encode("utf-8")
        ).hexdigest()
        cache_root = context["output_root"] / ".dataset-archives"
        cache_root.mkdir(exist_ok=True)
        target = cache_root / f"{cache_key}.zip"
        if not target.is_file():
            temporary = cache_root / f".{cache_key}.{os.getpid()}.tmp"
            try:
                with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
                    for item in selected:
                        source = context["output_root"] / item["relative"]
                        if source.is_symlink() or not source.is_file() or not is_within(source, context["output_root"]):
                            raise PlanFlowError("RESULT_INTEGRITY_FAILED", "Result asset changed while creating the archive")
                        archive.write(source, item["relative"])
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        return OpenedAsset(target, "application/zip", display_name, target.stat().st_size, sha256_file(target))

    def _context(self, workspace_root: str | Path, task_id: str, plan_version: str, result_ref: str):
        workspace = require_workspace(workspace_root)
        store = PlanStore(workspace)
        task_root = store.task_path(str(task_id))
        run_root = (task_root / "runs" / str(result_ref)).resolve()
        if run_root.parent != (task_root / "runs").resolve() or not (run_root / "run.json").is_file():
            raise PlanFlowError("RUN_NOT_FOUND", "Unknown result run")
        state = read_json(run_root / "run.json")
        if (
            state.get("task_id") != task_id
            or state.get("plan_version") != plan_version
            or state.get("run_id") != result_ref
        ):
            raise PlanFlowError("RUN_NOT_FOUND", "Result run identity does not match")
        output_root = Path(str(state.get("output_dir") or "")).resolve()
        allowed_output_root = (workspace / "outputs").resolve()
        if not output_root.is_dir() or not is_within(output_root, allowed_output_root):
            raise PlanFlowError("PATH_NOT_ALLOWED", "Run output is outside the workspace output root")
        return {
            "workspace": workspace,
            "task_root": task_root,
            "run_root": run_root,
            "output_root": output_root,
            "state": state,
        }

    def _verified_inventory(self, context, manifest_path: Path):
        manifest = read_json(manifest_path)
        outputs = manifest.get("outputs")
        if (
            manifest.get("schema_version") != 1
            or manifest.get("run_id") != context["state"].get("run_id")
            or not isinstance(outputs, list)
            or manifest.get("output_count") != len(outputs)
        ):
            raise PlanFlowError("INVALID_RESULT_MANIFEST", "Result manifest identity or inventory is invalid")
        inventory: dict[str, dict[str, Any]] = {}
        total = 0
        for raw in outputs:
            if not isinstance(raw, dict):
                raise PlanFlowError("INVALID_RESULT_MANIFEST", "Result manifest output entry is invalid")
            relative = _safe_relative(raw.get("path"))
            if relative in inventory:
                raise PlanFlowError("INVALID_RESULT_MANIFEST", "Result manifest contains duplicate paths")
            path = (context["output_root"] / relative).resolve()
            if not is_within(path, context["output_root"]) or path.is_symlink() or not path.is_file():
                raise PlanFlowError("INVALID_RESULT_MANIFEST", "Result manifest references an unsafe or missing file")
            size = path.stat().st_size
            digest = str(raw.get("sha256") or "")
            if raw.get("size_bytes") != size or digest != sha256_file(path):
                raise PlanFlowError("RESULT_INTEGRITY_FAILED", "Result output failed manifest validation")
            total += size
            inventory[relative] = {
                "relative": relative,
                "assetId": _asset_id(relative),
                "name": Path(relative).name,
                "mediaType": _media_type(relative),
                "size": size,
                "sha256": digest,
            }
        if manifest.get("output_size_bytes") != total:
            raise PlanFlowError("INVALID_RESULT_MANIFEST", "Result manifest total size is invalid")
        return manifest, inventory

    @staticmethod
    def _view(output_root: Path, inventory: dict[str, dict[str, Any]]):
        path = output_root / "dataset-view.json"
        if not path.is_file() or path.is_symlink():
            return {"schema_version": 1, "items": [], "documents": [], "summary": {}}, True
        try:
            view = read_json(path)
            if view.get("schema_version") != 1:
                raise ValueError("unsupported schema")
            for collection in ("items", "documents"):
                values = view.get(collection, [])
                if not isinstance(values, list):
                    raise ValueError("invalid collection")
                for item in values:
                    relative = _safe_relative(item.get("asset_path"))
                    if relative not in inventory:
                        raise ValueError("view references an unknown asset")
            return view, False
        except (PlanFlowError, ValueError, TypeError):
            return {"schema_version": 1, "items": [], "documents": [], "summary": {}}, True

    @staticmethod
    def _public_assets(inventory: dict[str, dict[str, Any]], view: dict[str, Any]):
        semantics: dict[str, dict[str, Any]] = {}
        for item in view.get("items", []):
            relative = str(item.get("asset_path", "")).replace("\\", "/")
            semantics[relative] = {
                "itemId": str(item.get("item_id") or _asset_id(relative)),
                "name": str(item.get("display_name") or inventory[relative]["name"]),
                "mediaType": str(item.get("media_type") or inventory[relative]["mediaType"]),
                "labels": [str(label)[:100] for label in item.get("labels", []) if str(label).strip()][:50],
                "metrics": item.get("metrics") if isinstance(item.get("metrics"), dict) else {},
            }
        for document in view.get("documents", []):
            relative = str(document.get("asset_path", "")).replace("\\", "/")
            semantics.setdefault(relative, {
                "itemId": _asset_id(relative),
                "name": inventory[relative]["name"],
                "mediaType": "application/x-ndjson" if document.get("kind") == "jsonl" else inventory[relative]["mediaType"],
                "labels": [],
                "metrics": {},
            })
        result = []
        for relative, item in inventory.items():
            semantic = semantics.get(relative, {
                "itemId": item["assetId"],
                "name": item["name"],
                "mediaType": item["mediaType"],
                "labels": [],
                "metrics": {},
            })
            result.append({
                "assetId": item["assetId"],
                "itemId": semantic["itemId"],
                "name": semantic["name"],
                "mediaType": semantic["mediaType"],
                "size": item["size"],
                "sha256": item["sha256"],
                "labels": semantic["labels"],
                "metrics": semantic["metrics"],
            })
        return result

    @staticmethod
    def _media_summary(inventory: dict[str, dict[str, Any]]):
        result: dict[str, int] = {}
        for item in inventory.values():
            family = item["mediaType"].split("/", 1)[0]
            if item["mediaType"] in {"application/json", "application/x-ndjson"}:
                family = "json"
            result[family] = result.get(family, 0) + 1
        return result
