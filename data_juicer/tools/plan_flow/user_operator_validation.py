"""Bounded asynchronous tests, automatic personal publication, and exact-temp cleanup."""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

from .common import (
    FileLock,
    PlanFlowError,
    now_iso,
    read_json,
    write_json_atomic,
    write_text_atomic,
)
from .user_operator_store import UserOperatorStore, digest, public_candidate, safe_id


def stop_process(process):
    import psutil

    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        parent.terminate()
        _, alive = psutil.wait_procs([*children, parent], timeout=3)
        for child in alive:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(alive, timeout=3)
    except psutil.NoSuchProcess:
        pass
    process.wait(timeout=10)


class UserOperatorValidation:
    def __init__(self):
        self._jobs = {}
        self._lock = threading.Lock()

    def develop(self, proposal, samples, parameters=None, timeout_seconds=180):
        store = UserOperatorStore()
        name, category = safe_id(proposal.get("name", "")), proposal.get("category")
        base = store.operator_path(category, name)
        source = proposal.get("source", "")
        if not isinstance(source, str) or not source.strip() or len(source.encode()) > 256_000:
            raise PlanFlowError("INVALID_OPERATOR_SOURCE", "Provide at most 256 KB of Python source")
        source = source.replace("\r\n", "\n")
        from .operator_authoring import operator_authoring

        operator_authoring.lint(name, category, source)
        if not isinstance(samples, list) or not samples or len(samples) > 100 or len(json.dumps(samples)) > 1_000_000:
            raise PlanFlowError(
                "INVALID_TEST_SAMPLES",
                "Provide 1–100 small JSON records (at most 1 MB)",
            )
        if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 1800:
            raise PlanFlowError("INVALID_TIMEOUT", "Test timeout must be between 1 and 1800 seconds")
        # Copy media fixtures, never test against the original writable business files.
        from .common import sha256_file

        media = []
        media_bytes = 0
        keys = [
            (parameters or {}).get(key, default)
            for key, default in (
                ("image_key", "images"),
                ("audio_key", "audios"),
                ("video_key", "videos"),
            )
        ]
        for row_index, row in enumerate(samples):
            if not isinstance(row, dict):
                raise PlanFlowError("INVALID_TEST_SAMPLES", "Every sample must be an object")
            for key in keys:
                for index, value in enumerate(row.get(key, []) or []):
                    path = Path(value).expanduser()
                    if not path.is_absolute() or not path.is_file():
                        raise PlanFlowError(
                            "INVALID_MEDIA_SAMPLE",
                            "Media samples must reference existing absolute files",
                        )
                    media_bytes += path.stat().st_size
                    if media_bytes > 50_000_000:
                        raise PlanFlowError(
                            "TEST_MEDIA_TOO_LARGE",
                            "Use at most 50 MB of media fixtures per test",
                        )
                    media.append((row_index, key, index, path, sha256_file(path)))
        contract = proposal.get("validation_contract", {})
        if not isinstance(contract, dict) or set(contract) - {
            "purpose",
            "limitations",
            "equals",
            "row_count",
        }:
            raise PlanFlowError(
                "INVALID_VALIDATION_CONTRACT",
                "Contract supports purpose, limitations, equals and row_count",
            )
        if not isinstance(contract.get("equals", []), list):
            raise PlanFlowError(
                "INVALID_VALIDATION_CONTRACT",
                "equals must be a list of {row, field, value} assertions",
            )
        for assertion in contract.get("equals", []):
            if (
                not isinstance(assertion, dict)
                or set(assertion) != {"row", "field", "value"}
                or type(assertion["row"]) is not int
                or assertion["row"] < 0
            ):
                raise PlanFlowError("INVALID_VALIDATION_CONTRACT", "Invalid equality assertion")
        fingerprint = digest({"records": samples, "media": [entry[4] for entry in media]})
        contract = {
            **contract,
            "_sample_fingerprint": fingerprint,
            "_parameters_fingerprint": digest(parameters or {}),
        }
        # Freeze the contract BEFORE any testing, including failed attempts.
        with FileLock(base / ".contract.lock"):
            contract_path = base / "contract.json"
            if contract_path.exists() and read_json(contract_path) != contract:
                raise PlanFlowError(
                    "CONTRACT_CHANGED",
                    "Contract is frozen; changed requirements require a new operator ID",
                )
            write_json_atomic(contract_path, contract)
        from .user_operator_resources import model_refs, validate_assets

        manifest = {
            "dependencies": proposal.get("dependencies", []),
            "model_refs": model_refs(proposal.get("model_refs", []), freeze=True),
            "replaces": proposal.get("replaces"),
            "assets": validate_assets(proposal.get("assets", {})),
        }
        from .user_operator_runtime import dependency_lock

        dependency_lock(manifest["dependencies"])
        job_id = uuid.uuid4().hex[:16]
        temp = store.path(store.home / "operator_tmp" / job_id)
        job_path = store.path(store.home / "operator_jobs" / f"{job_id}.json")
        temp.mkdir(parents=True)
        job = {
            "job_id": job_id,
            "status": "testing",
            "created_at": now_iso(),
            "sample_fingerprint": fingerprint,
            "cleanup_pending": True,
            "supervisor_pid": os.getpid(),
        }
        try:
            copied = copy.deepcopy(samples)
            for row in copied:
                if any(row.get(key) for key in keys):
                    row.setdefault((parameters or {}).get("text_key", "text"), "")
            for number, (row_index, key, index, path, _) in enumerate(media):
                destination = temp / f"fixture-{number}{path.suffix}"
                shutil.copyfile(path, destination)
                copied[row_index][key][index] = str(destination).removeprefix("\\\\?\\")
            write_text_atomic(temp / f"{name}.py", source)
            for filename, content in manifest["assets"].items():
                write_text_atomic(temp / "assets" / filename, content)
            write_text_atomic(
                temp / "input.jsonl",
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in copied),
            )
            write_json_atomic(
                temp / "request.json",
                {"name": name, "category": category, "parameters": parameters or {}},
            )
            # Retain generated source and the frozen contract, not fixtures or raw test logs.
            draft = store.path(base / "drafts" / job_id)
            write_text_atomic(draft / "operator.py", source)
            write_json_atomic(draft / "manifest.json", manifest)
            write_json_atomic(draft / "validation-contract.json", contract)
        except Exception:
            self._cleanup(store, temp, job)
            raise
        import psutil

        job["supervisor_create_time"] = psutil.Process().create_time()
        write_json_atomic(job_path, job)
        cancel = threading.Event()
        with self._lock:
            self._jobs[(store.user_id, job_id)] = cancel
        threading.Thread(
            target=self._run,
            args=(
                store,
                temp,
                job_path,
                job,
                source,
                manifest,
                contract,
                timeout_seconds,
                cancel,
            ),
            daemon=True,
        ).start()
        return {"ok": True, "job": job}

    def get(self, job_id, cancel=False):
        store = UserOperatorStore()
        safe_id(job_id)
        path = store.path(store.home / "operator_jobs" / f"{job_id}.json")
        if cancel:
            event = self._jobs.get((store.user_id, job_id))
            if event:
                event.set()
        job = read_json(path)
        # Recovery only when the original supervisor is no longer alive. Never age-delete active jobs.
        if job.get("cleanup_pending") and (store.user_id, job_id) not in self._jobs:
            import psutil

            if job.get("status") != "testing" or not self._same_process(
                job.get("supervisor_pid"), job.get("supervisor_create_time")
            ):
                pid = job.get("worker_pid")
                if self._same_process(pid, job.get("worker_create_time")):
                    worker = psutil.Process(pid)
                    children = worker.children(recursive=True)
                    for process in [*children, worker]:
                        try:
                            process.kill()
                        except psutil.NoSuchProcess:
                            pass
                    _, alive = psutil.wait_procs([*children, worker], timeout=3)
                    if alive:
                        return {
                            "ok": True,
                            "job": {**job, "recovery": "worker_still_active"},
                        }
                self._cleanup(store, store.path(store.home / "operator_tmp" / job_id), job)
                if job.get("status") == "testing":
                    job.update(status="failed", error="Validation supervisor was interrupted")
                write_json_atomic(path, job)
        return {"ok": True, "job": job}

    @staticmethod
    def _same_process(pid, created):
        import psutil

        try:
            return bool(pid and created and abs(psutil.Process(pid).create_time() - created) < 0.01)
        except psutil.NoSuchProcess:
            return False

    @staticmethod
    def _cleanup(store, temp, job):
        target = store.path(temp)
        if target.parent != store.path(store.home / "operator_tmp"):
            raise PlanFlowError("CLEANUP_PATH_FORBIDDEN", "Only the exact test directory may be removed")
        try:
            if target.exists():
                shutil.rmtree(target)
            job["cleanup_pending"] = False
        except OSError as exc:
            job.update(cleanup_pending=True, cleanup_error=str(exc))

    def _run(self, store, temp, job_path, job, source, manifest, contract, timeout, cancel):
        deadline = time.monotonic() + timeout
        active_process = None
        phase = "runtime_setup"
        try:
            # Third-party dataset cache lock names do not support Windows extended-path prefixes.
            runtime_temp = str(temp)
            if runtime_temp.startswith("\\\\?\\UNC\\"):
                runtime_temp = "\\\\" + runtime_temp[8:]
            elif runtime_temp.startswith("\\\\?\\"):
                runtime_temp = runtime_temp[4:]
            env = os.environ.copy()
            env.update(
                PYTHONUTF8="1",
                PYTHONIOENCODING="utf-8",
                PYTHONDONTWRITEBYTECODE="1",
                TEMP=runtime_temp,
                TMP=runtime_temp,
                TMPDIR=runtime_temp,
                HF_DATASETS_CACHE=".d",
            )
            env["DJ_PRODUCED_DATA_DIR"] = str(Path(runtime_temp) / "produced")
            env["PIP_CACHE_DIR"] = str(Path(runtime_temp) / "pip-cache")
            env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3]) + os.pathsep + env.get("PYTHONPATH", "")

            def run_command(command, mode="runtime"):
                nonlocal active_process, phase
                phase = mode
                with (temp / f"{mode}.log").open("wb") as log:
                    process = subprocess.Popen(
                        command,
                        cwd=temp,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                    active_process = process
                    job["worker_pid"] = process.pid
                    import psutil

                    job["worker_create_time"] = psutil.Process(process.pid).create_time()
                    write_json_atomic(job_path, job)
                    while process.poll() is None:
                        if cancel.wait(0.1) or time.monotonic() > deadline:
                            stop_process(process)
                            raise TimeoutError("cancelled" if cancel.is_set() else "timeout")
                    if process.returncode:
                        error_path = temp / "error.json"
                        message = (
                            read_json(error_path).get("error")
                            if error_path.exists()
                            else f"{mode} worker exited {process.returncode}: "
                            + (temp / f"{mode}.log").read_text(encoding="utf-8", errors="replace")[-1500:]
                        )
                        raise RuntimeError(message)

            from .user_operator_runtime import runtime_python

            python = runtime_python(store, manifest["dependencies"], run_command)
            for mode in ("schema", "execute"):
                run_command(
                    [
                        python,
                        "-X",
                        "utf8",
                        "-m",
                        "data_juicer.tools.plan_flow.user_operator_worker",
                        runtime_temp,
                        mode,
                    ],
                    mode,
                )
            rows = [
                json.loads(line)
                for line in (temp / "validation" / "output.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            phase = "contract"
            if "row_count" in contract and len(rows) != contract["row_count"]:
                raise AssertionError("Output row_count did not meet the frozen contract")
            for assertion in contract.get("equals", []):
                if rows[assertion["row"]].get(assertion["field"]) != assertion["value"]:
                    raise AssertionError(f"Output assertion failed: row {assertion['row']}, field {assertion['field']}")
            accepted = bool(contract.get("purpose") and contract.get("equals"))
            report = {
                "status": "validated" if accepted else "experimental",
                "execution_passed": True,
                "acceptance_passed": accepted,
                "validation_scope": "Declared JSON output assertions only",
                "sample_fingerprint": job["sample_fingerprint"],
                "output_rows": len(rows),
                "limitations": contract.get("limitations", [])
                or ([] if accepted else ["Smoke test only; quality is unverified"]),
            }
            report["cache_snapshot"] = read_json(temp / "cache-snapshot.json")
            if (temp / f"{read_json(temp / 'request.json')['name']}.py").read_text(encoding="utf-8") != source:
                raise RuntimeError("Operator source changed during validation")
            phase = "publish"
            candidate = store.publish(source, read_json(temp / "schema.json"), manifest, contract, report)
            job.update(
                status=report["status"],
                candidate=public_candidate(candidate),
                report=report,
            )
        except Exception as exc:
            job.update(
                status="cancelled" if cancel.is_set() else "failed",
                error=f"{type(exc).__name__}: {exc}"[:2000],
                error_details={"phase": phase},
            )
        finally:
            stopped = True
            if active_process and active_process.poll() is None:
                try:
                    stop_process(active_process)
                except Exception as exc:
                    stopped = False
                    job.update(
                        cleanup_pending=True,
                        cleanup_error=f"Worker termination pending: {exc}",
                    )
            if stopped:
                self._cleanup(store, temp, job)
            job["finished_at"] = now_iso()
            write_json_atomic(job_path, job)
            with self._lock:
                self._jobs.pop((store.user_id, job["job_id"]), None)


validation_jobs = UserOperatorValidation()
