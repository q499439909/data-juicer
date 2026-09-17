"""Path-free build identity and planning health, without loading optional models."""
from __future__ import annotations

import hashlib
import importlib.metadata
import sys
import json
import os
import platform
import uuid
from pathlib import Path

from .common import sha256_file

PROTOCOL = {'name': 'dsh-dj-internal-http', 'version': 1,
            'features': ['trusted-ui-approval', 'owned-workspaces', 'run-idempotency', 'input-profile-v1', 'rank-merge-v1']}


def build_identity(root=None):
    from data_juicer import __version__
    root = Path(root) if root else Path(__file__).resolve().parents[3]
    digest = hashlib.sha256()
    files = sorted(path for path in (root / 'data_juicer').rglob('*')
                   if path.is_file() and path.suffix in {'.py', '.json'} and '__pycache__' not in path.parts)
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b'\0')
        digest.update(bytes.fromhex(sha256_file(path).removeprefix('sha256:')))
    def fingerprint(path): return sha256_file(path) if path.is_file() else None
    packages = sorted({(dist.metadata.get('Name', '').casefold(), dist.version,
                       hashlib.sha256((dist.read_text('RECORD') or '').encode()).hexdigest())
                      for dist in importlib.metadata.distributions()})
    packages_hash = 'sha256:' + hashlib.sha256(json.dumps(packages, separators=(',', ':')).encode()).hexdigest()
    return {'dj_version': __version__, 'source_sha256': 'sha256:' + digest.hexdigest(), 'source_files':len(files),
            'python': platform.python_version(), 'python_executable_sha256':fingerprint(Path(sys.executable)),
            'runtime_packages_sha256':packages_hash, 'runtime_lock_sha256': fingerprint(root / 'uv.lock'),
            'project_sha256': fingerprint(root / 'pyproject.toml'),
            'model_catalog_sha256':fingerprint(root / 'data_juicer/tools/plan_flow/builtin_model_catalog.json')}


def service_contract(execution_mode):
    # Capture once during server creation, not on each request: edits on disk do
    # not change the identity of the code already imported by a running server.
    return {'protocol': PROTOCOL, 'instance_id': str(uuid.uuid4()), 'execution_mode':execution_mode,
            'build':build_identity()}


def planning_health(contract):
    control = bool(os.environ.get('DSH_USER_DATA_ROOT') and os.environ.get('DSH_DJ_INTERNAL_TOKEN'))
    lock = bool(contract['build']['runtime_lock_sha256'])
    return {'ok': control and lock, **contract, 'layers': {
        'process': {'status':'ready'},
        'planning': {'status':'ready' if lock else 'blocked', 'reason':None if lock else 'runtime_lock_missing'},
        'account_control': {'status':'configured' if control else 'blocked'},
        'execution': {'status':'per_plan', 'reason':'GPU, model and operator prerequisites are checked for the selected plan'},
        'observability': {'status':'optional'},
    }}


if __name__ == '__main__':
    print(json.dumps({'protocol':PROTOCOL, 'build':build_identity()}, ensure_ascii=False))
