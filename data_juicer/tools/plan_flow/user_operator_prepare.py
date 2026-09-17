"""Bounded subprocess for personal operator model preparation."""

import sys
from pathlib import Path

from .common import read_json, sha256_file, write_json_atomic


def prepare(temp):
    manifest = read_json(temp / "model-request.json")
    from .model_lock_resolver import ModelLockResolver

    request = read_json(temp / "request.json")
    name = request["name"]
    recipe = {"process": [{name: request["parameters"]}]}
    personal = {name: {"_manifest": manifest}}
    resolver = ModelLockResolver()
    bindings = resolver.freeze({"recipe": recipe}, personal=personal)
    runtime_bindings = resolver.attach_local_sources(bindings, personal)
    paths = resolver.prepare(runtime_bindings)
    for ref in manifest["model_refs"]:
        if ref.get("path") or ref.get("files") or ref.get("backend") == "http-file":
            continue
        binding = next(
            item
            for item in bindings
            if item.get("model_id") == ref.get("model_id") and item.get("revision") == ref.get("revision")
        )
        snapshot = paths[binding["binding_id"]]
        ref["files"] = [
            {
                "path": path.relative_to(snapshot).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path).removeprefix("sha256:"),
            }
            for path in sorted(snapshot.rglob("*"))
            if path.is_file()
        ]
    # Re-freeze after discovery so the published user artifact and
    # every future Plan contain exact file identities.
    bindings = resolver.freeze({"recipe": recipe}, personal=personal)
    runtime_bindings = resolver.attach_local_sources(bindings, personal)
    paths = resolver.prepare(runtime_bindings, offline=True)
    materialized, provenance = resolver.materialize(recipe, bindings, paths)
    request["parameters"] = materialized["process"][0][name]
    write_json_atomic(temp / "request.json", request)
    write_json_atomic(temp / "resolved-models.json", provenance)

    write_json_atomic(temp / "model-result.json", manifest)


if __name__ == "__main__":
    prepare(Path(sys.argv[1]))
