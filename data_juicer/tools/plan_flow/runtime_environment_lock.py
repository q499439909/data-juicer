"""Freeze and verify the Python runtime independently from model artifacts."""

from __future__ import annotations

import importlib.metadata
import sys
import tomllib
from pathlib import Path

from .common import PlanFlowError, sha256_file


_AUDITED_PACKAGES = {
    "data-juicer",
    "huggingface-hub",
    "modelscope",
    "numpy",
    "opencv-contrib-python",
    "torch",
    "transformers",
    "ultralytics",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def locked_package_versions(lock_path: str | Path | None = None) -> dict[str, str]:
    lock_path = Path(lock_path).resolve() if lock_path else project_root() / "uv.lock"
    try:
        raw = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PlanFlowError("RUNTIME_LOCK_MISSING", f"Cannot read Python lock: {lock_path}") from exc
    return {
        str(item["name"]).casefold(): str(item["version"])
        for item in raw.get("package", [])
        if isinstance(item, dict) and item.get("name") and item.get("version")
    }


def freeze_runtime_lock(lock_path: str | Path | None = None) -> dict:
    """Return a path-free identity for the Python dependency graph."""
    path = Path(lock_path).resolve() if lock_path else project_root() / "uv.lock"
    versions = locked_package_versions(path)
    return {
        "schema_version": 1,
        "manager": "uv",
        "lockfile": "uv.lock",
        "sha256": sha256_file(path).removeprefix("sha256:"),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "audited_packages": {name: versions[name] for name in sorted(_AUDITED_PACKAGES & versions.keys())},
    }


def verify_runtime_lock(
    expected: dict, lock_path: str | Path | None = None, *, required_packages: set[str] | None = None
) -> dict:
    """Fail closed if source lock or an installed audited package has drifted."""
    if not isinstance(expected, dict) or expected.get("schema_version") != 1:
        raise PlanFlowError("RUNTIME_LOCK_MISSING", "Plan has no supported Python runtime lock")
    actual = freeze_runtime_lock(lock_path)
    for field in ("manager", "lockfile", "sha256", "python"):
        if expected.get(field) != actual[field]:
            raise PlanFlowError(
                "RUNTIME_LOCK_MISMATCH",
                f"Python runtime lock differs at {field}: expected {expected.get(field)!r}, got {actual[field]!r}",
            )
    required = {name.casefold() for name in (required_packages or set())}
    for name, version in expected.get("audited_packages", {}).items():
        if name.casefold() not in required:
            continue
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            raise PlanFlowError('RUNTIME_PACKAGE_MISSING', f'Required package {name}=={version} is not installed')
        if installed != version:
            raise PlanFlowError(
                "RUNTIME_PACKAGE_MISMATCH",
                f"Installed {name}=={installed} differs from uv.lock ({version})",
            )
    return actual


def inspect_plan_runtime(plan: dict) -> dict:
    """Resolve the platform-specific uv dependency closure and compare installed metadata.

    No installation, download or environment mutation occurs in this path.
    """
    from packaging.markers import Marker
    from packaging.utils import canonicalize_name
    from .common import canonical_json, sha256_bytes
    lock_path=project_root()/'uv.lock'
    lock=tomllib.loads(lock_path.read_text(encoding='utf-8'))
    packages={}
    for item in lock.get('package',[]):
        if item.get('resolution-markers') and not any(Marker(m).evaluate() for m in item['resolution-markers']):
            continue
        packages.setdefault(canonicalize_name(item['name']),[]).append(item)
    root_package=next((p for p in lock.get('package',[]) if p['name']=='py-data-juicer'),None)
    roots=set()
    for binding in plan.get('model_bindings',[]):
        roots.update(canonicalize_name(p) for p in binding.get('runtime_packages',[]))
        if binding.get('distribution'): roots.add(canonicalize_name(binding['distribution']))
    queue=[*(root_package or {}).get('dependencies',[]),*[{'name':p} for p in sorted(roots)]]
    expected={};issues=[];expanded=set()
    while queue:
        dep=queue.pop()
        if dep.get('marker') and not Marker(dep['marker']).evaluate(): continue
        name=canonicalize_name(dep['name'])
        candidates=[p for p in packages.get(name,[]) if not dep.get('version') or p.get('version')==dep['version']]
        if len(candidates)!=1:
            issues.append({'code':'RUNTIME_LOCK_AMBIGUOUS','package':name,'message':'Required dependency does not resolve uniquely in uv.lock'})
            continue
        item=candidates[0]
        key=(name,item.get('version'),tuple(sorted(dep.get('extra',[]))))
        if key in expanded: continue
        expanded.add(key)
        if name in expected and expected[name]!=item.get('version'):
            issues.append({'code':'RUNTIME_LOCK_AMBIGUOUS','package':name,'message':'Dependency versions conflict in the selected closure'})
            continue
        expected[name]=item['version'];queue.extend(item.get('dependencies',[]))
        for extra in dep.get('extra',[]): queue.extend(item.get('optional-dependencies',{}).get(extra,[]))
    observed={}
    for name,version in sorted(expected.items()):
        try: observed[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: observed[name]=None
        if observed[name]!=version:
            issues.append({'code':'RUNTIME_PACKAGE_MISSING' if observed[name] is None else 'RUNTIME_PACKAGE_MISMATCH',
                           'package':name,'expected':version,'installed':observed[name],
                           'message':f'{name}: expected {version}, installed {observed[name]}'})
    identity={'lock_sha256':sha256_file(lock_path),'python':f'{sys.version_info.major}.{sys.version_info.minor}', 'packages':expected}
    return {'ok':not issues,'runtime_id':sha256_bytes(canonical_json(identity)), 'manifest':identity,
            'installed':observed,'blocking_issues':issues,
            'recovery_actions':[] if not issues else [{'action':'restore_locked_runtime','scope':'maintenance','requires_shared_environment_authority':True,
                'message':'Build or restore an isolated environment from this uv.lock, verify this closure, then retry the same immutable Plan. Do not rewrite uv.lock to match drift.'}]}
