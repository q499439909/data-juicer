"""Recover interrupted validation jobs without waiting for an Agent poll."""
import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

from .common import read_json, write_json_atomic, now_iso
from .user_operator_store import UserOperatorStore
from .user_operator_validation import UserOperatorValidation
from .execution.supervisor import terminate_tree


def watch(root, user, job_id):
    store = UserOperatorStore(root=root, user_id=user)
    path = store.path(store.home / 'operator_jobs' / f'{job_id}.json')
    while path.exists():
        job = read_json(path)
        if not job.get('cleanup_pending'):
            return
        lost = not UserOperatorValidation._same_process(job.get('supervisor_pid'), job.get('supervisor_create_time'))
        expired = bool(job.get('deadline') and datetime.now(timezone.utc) >= datetime.fromisoformat(job['deadline']))
        if lost or expired:
            if job.get('worker_pid'):
                terminate_tree({'pid':job['worker_pid'], 'pid_create_time':job.get('worker_create_time')})
            if not lost:
                # The live supervisor owns publication and cleanup. Killing its active
                # command unblocks it; never delete files it may still be reading.
                time.sleep(.25)
                continue
            # Re-read to avoid overwriting a completed publication.
            job = read_json(path)
            if not job.get('cleanup_pending'):
                return
            UserOperatorValidation._cleanup(store, store.path(store.home / 'operator_tmp' / job_id), job)
            if job.get('status') == 'testing':
                job.update(status='failed', error='Validation supervisor interrupted' if lost else 'Validation deadline exceeded', error_details={'phase':job.get('phase','preparing')}, finished_at=now_iso())
            write_json_atomic(path, job)
            return
        time.sleep(.25)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root'); parser.add_argument('user'); parser.add_argument('job_id')
    args = parser.parse_args()
    watch(args.root, args.user, args.job_id)
