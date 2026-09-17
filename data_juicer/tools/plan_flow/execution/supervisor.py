"""Detached native deadline watchdog; independent of server and UI polling."""

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock

from ..common import now_iso, read_json, write_json_atomic


def terminate_tree(record):
    import psutil

    try:
        process = psutil.Process(int(record["pid"]))
        expected = record.get("pid_create_time")
        if expected is not None and abs(process.create_time() - expected) >= 0.01:
            return
        processes = process.children(recursive=True) + [process]
    except psutil.NoSuchProcess:
        return
    for item in processes:
        try:
            item.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(processes, timeout=3)
    for item in alive:
        try:
            item.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(alive, timeout=3)
    if alive:
        raise RuntimeError("Worker cleanup did not complete")


def stop_record(record_path, *, timeout=False):
    record_path = Path(record_path)
    record = read_json(record_path)
    state_path = Path(record["run_state_path"])
    lock = state_path.with_suffix(".lock")
    with FileLock(lock):
        state = read_json(state_path)
        if state.get("status") in {"succeeded", "failed", "cancelled"} and not state.get("cleanup_pending"):
            return state
        state.update(status="cancelling", cleanup_pending=True, updated_at=now_iso())
        write_json_atomic(state_path, state)
    terminate_tree(record)
    with FileLock(lock):
        state = read_json(state_path)
        state.update(
            status="failed" if timeout else "cancelled",
            execution_status="failed" if timeout else "cancelled",
            cleanup_pending=False,
            updated_at=now_iso(),
        )
        if timeout:
            state.update(error_code="RUN_TIMEOUT", error="Run deadline exceeded")
        write_json_atomic(state_path, state)
    with FileLock(record_path.with_suffix(".lock")):
        record = read_json(record_path)
        record.update(status="cancelled", finished_at=now_iso())
        write_json_atomic(record_path, record)
    return state


def watch(record_path):
    record_path = Path(record_path)
    from .local_process import LocalProcessBackend

    while record_path.exists():
        record = read_json(record_path)
        state_path = Path(record["run_state_path"])
        if not state_path.exists():
            time.sleep(0.2)
            continue
        state = read_json(state_path)
        if state.get("status") in {"succeeded", "failed", "cancelled"} and not state.get("cleanup_pending"):
            return
        if not LocalProcessBackend._same_process(record):
            with FileLock(state_path.with_suffix(".lock")):
                state = read_json(state_path)
                if state.get("status") not in {"succeeded", "failed", "cancelled"}:
                    state.update(
                        status="failed",
                        execution_status="failed",
                        error_code="RUNNER_LOST",
                        error="Worker exited without a final result",
                        cleanup_pending=False,
                        updated_at=now_iso(),
                    )
                    write_json_atomic(state_path, state)
            return
        deadline = record.get("runtime_spec", {}).get("deadline")
        if state.get("status") == "cancelling" or (
            deadline and datetime.now(timezone.utc) >= datetime.fromisoformat(deadline)
        ):
            stop_record(record_path, timeout=state.get("status") != "cancelling")
            return
        time.sleep(0.25)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("record")
    watch(parser.parse_args().record)
