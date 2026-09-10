"""Offline model bundle export/import for air-gapped native workers."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

from .common import PlanFlowError, read_yaml, write_json_atomic
from .model_lock_resolver import ModelLockResolver


def _installed_path(root: Path, binding: dict) -> Path:
    if binding.get("backend") == "huggingface":
        return root / ("models--" + binding["model_id"].replace("/", "--")) / "snapshots" / binding["revision"]
    if binding.get("backend") == "modelscope":
        return root / "modelscope" / binding["model_id"] / binding["revision"]
    if binding.get("backend") == "torch-hub":
        return root / "torch-hub" / binding["lock_id"] / binding["revision"]
    if binding.get("backend") == "http-file":
        return root / "http-file" / binding["lock_id"]
    return root / binding["lock_id"]


def export_bundle(plan_path: str | Path, destination: str | Path, *, resolver=None) -> dict:
    plan = read_yaml(Path(plan_path).expanduser().resolve())
    bindings = plan.get("model_bindings", [])
    resolver = resolver or ModelLockResolver()
    runtime_bindings = bindings
    if plan.get("operator_bindings"):
        from .user_operator_store import UserOperatorStore, resolve_bindings

        personal = resolve_bindings(plan, UserOperatorStore(user_id=plan["operator_owner"]))
        runtime_bindings = resolver.attach_local_sources(bindings, personal)
    paths = resolver.prepare(runtime_bindings, offline=False)
    target = Path(destination).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    if any(target.iterdir()):
        raise PlanFlowError("MODEL_PATH_FORBIDDEN", f"Export directory must be empty: {target}")
    exported = []
    for binding in bindings:
        source = paths[binding["binding_id"]]
        model_target = target / binding["lock_id"]
        if source.is_dir():
            shutil.copytree(source, model_target)
        else:
            model_target.mkdir()
            shutil.copy2(source, model_target / binding.get("filename", source.name))
        exported.append({key: copy.deepcopy(value) for key, value in binding.items() if key != "consumers"})
    manifest = {"schema_version": 1, "models": exported}
    write_json_atomic(target / "model-bundle.json", manifest)
    return manifest


def import_bundle(bundle_dir: str | Path, preloaded_root: str | Path, *, resolver=None) -> dict:
    source_root = Path(bundle_dir).expanduser().resolve()
    manifest_path = source_root / "model-bundle.json"
    if not manifest_path.is_file():
        raise PlanFlowError("MODEL_FILE_MISSING", f"Model bundle manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("models"), list):
        raise PlanFlowError("MODEL_BINDING_CONFLICT", "Invalid model bundle manifest")
    resolver = resolver or ModelLockResolver()
    destination_root = Path(preloaded_root).expanduser().resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    imported = []
    for binding in manifest["models"]:
        source = (source_root / binding["lock_id"]).resolve()
        if source.parent != source_root or not source.exists():
            raise PlanFlowError("MODEL_PATH_FORBIDDEN", f"Invalid model bundle entry: {binding.get('lock_id')}")
        backend = resolver.backends.get(binding.get("backend"))
        if backend is None:
            raise PlanFlowError("MODEL_BINDING_CONFLICT", f"Unknown model backend: {binding.get('backend')}")
        verify_path = source
        if binding.get("backend") in {"local-file", "http-file"}:
            verify_path = source / binding["filename"]
        backend.verify(binding, verify_path)
        destination = _installed_path(destination_root, binding).resolve()
        if destination == destination_root or not destination.is_relative_to(destination_root):
            raise PlanFlowError("MODEL_PATH_FORBIDDEN", f"Model bundle destination escaped its root: {destination}")
        if destination.exists():
            existing_path = destination
            if binding.get("backend") in {"local-file", "http-file"}:
                existing_path = destination / binding["filename"]
            backend.verify(binding, existing_path)
            imported.append({"lock_id": binding["lock_id"], "status": "reused"})
            continue
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)
        imported.append({"lock_id": binding["lock_id"], "status": "imported"})
    return {"ok": True, "preloaded_root": str(destination_root), "models": imported}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Prepare portable, version-locked DJ model bundles")
    subparsers = parser.add_subparsers(dest="command", required=True)
    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("plan")
    export_parser.add_argument("destination")
    import_parser = subparsers.add_parser("import")
    import_parser.add_argument("bundle")
    import_parser.add_argument("preloaded_root")
    args = parser.parse_args(argv)
    result = (
        export_bundle(args.plan, args.destination)
        if args.command == "export"
        else import_bundle(args.bundle, args.preloaded_root)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
