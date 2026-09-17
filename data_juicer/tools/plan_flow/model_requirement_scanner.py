"""Audit every built-in ``prepare_model`` call against the formal catalog."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from .model_lock_resolver import load_builtin_model_catalog


def _prepare_model_calls() -> list[dict]:
    ops_root = Path(__file__).resolve().parents[2] / "ops"
    findings = []
    for source in sorted(ops_root.rglob("*.py")):
        try:
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            called = (isinstance(node.func, ast.Name) and node.func.id == "prepare_model") or (
                isinstance(node.func, ast.Attribute) and node.func.attr == "prepare_model"
            )
            if not called:
                continue
            model_type = None
            for keyword in node.keywords:
                if keyword.arg == "model_type" and isinstance(keyword.value, ast.Constant):
                    model_type = keyword.value.value
                    break
            findings.append(
                {
                    "operator": source.stem,
                    "model_type": model_type,
                    "source": source.relative_to(ops_root.parents[1]).as_posix(),
                    "line": node.lineno,
                }
            )
    return findings


def _matches(requirement: dict, finding: dict) -> bool:
    if requirement.get("operator") not in (None, finding["operator"]):
        return False
    if "model_types" in requirement:
        return finding["model_type"] in requirement["model_types"]
    return "model_type" not in requirement or requirement.get("model_type") == finding["model_type"]


def scan(catalog_path: str | Path | None = None) -> dict:
    catalog = load_builtin_model_catalog(catalog_path)
    locked_operators = {consumer["operator"] for model in catalog["models"] for consumer in model.get("consumers", [])}
    blocked = catalog.get("blocked_requirements", [])
    exempt = catalog.get("exempt_requirements", [])
    findings = []
    for finding in _prepare_model_calls():
        blocked_match = next((item for item in blocked if _matches(item, finding)), None)
        exempt_match = next((item for item in exempt if _matches(item, finding)), None)
        if blocked_match:
            finding.update(status="blocked", reason=blocked_match["reason"])
        elif exempt_match:
            finding.update(status="exempt", reason=exempt_match["reason"])
        elif finding["operator"] in locked_operators:
            finding.update(status="locked")
        else:
            finding.update(status="unresolved", reason="no catalog lock, blocker, or reviewed exemption")
        findings.append(finding)
    counts = {
        status: sum(item["status"] == status for item in findings)
        for status in ("locked", "blocked", "exempt", "unresolved")
    }
    return {
        "schema_version": 2,
        "call_site_count": len(findings),
        "status_counts": counts,
        "complete": counts["unresolved"] == 0,
        "findings": findings,
        "note": "Complete means every built-in prepare_model call site is locked, explicitly blocked, or reviewed as non-downloading/external.",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Audit DJ model requirements against the built-in catalog")
    parser.add_argument("--output")
    parser.add_argument("--check", action="store_true", help="fail when any call site remains unresolved")
    args = parser.parse_args(argv)
    result = scan()
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 1 if args.check and not result["complete"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
