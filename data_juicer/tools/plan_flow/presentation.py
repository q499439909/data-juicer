"""Presentation-only model for immutable Data-Juicer plans.

The execution recipe remains the source of truth.  This module only gives its
ordered process entries stable UI identifiers and optional semantic groups.
"""

from __future__ import annotations

from typing import Any


def _operator_name(step: Any, index: int) -> str:
    if not isinstance(step, dict) or len(step) != 1:
        raise ValueError(f"recipe.process[{index}] must contain exactly one operator")
    return str(next(iter(step)))


def _default_groups(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": f"stage-{index + 1:03d}",
            "title": step["operator_name"],
            "summary": "",
            "step_refs": [step["id"]],
        }
        for index, step in enumerate(steps)
    ]


def build_plan_view(
    plan: dict[str, Any], content_hash: str, view_spec: dict[str, Any] | None = None
) -> tuple[dict[str, Any], list[str]]:
    """Build a safe linear PlanView, falling back when presentation input is invalid."""

    process = plan.get("recipe", {}).get("process", [])
    if not isinstance(process, list):
        raise ValueError("recipe.process must be a list")

    steps = []
    for index, item in enumerate(process):
        name = _operator_name(item, index)
        steps.append(
            {
                "id": f"step-{index:03d}",
                "process_index": index,
                "operator_name": name,
                "execution_key": f"op_{index + 1:03d}_{name}",
                "implementation": {"kind": "catalog_operator"},
            }
        )

    warnings: list[str] = []
    groups = _default_groups(steps)
    raw_groups = view_spec.get("groups") if isinstance(view_spec, dict) else None
    if raw_groups is not None:
        try:
            if not isinstance(raw_groups, list):
                raise ValueError("view_spec.groups must be a list")
            owner: dict[int, int] = {}
            normalized: list[tuple[int, dict[str, Any]]] = []
            previous_first = -1
            for group_index, raw in enumerate(raw_groups):
                if not isinstance(raw, dict):
                    raise ValueError(f"groups[{group_index}] must be an object")
                title = str(raw.get("title") or "").strip()
                indexes = raw.get("process_indexes")
                if not title:
                    raise ValueError(f"groups[{group_index}].title is required")
                if not isinstance(indexes, list) or not indexes:
                    raise ValueError(f"groups[{group_index}].process_indexes must be a non-empty list")
                if any(isinstance(item, bool) or not isinstance(item, int) for item in indexes):
                    raise ValueError(f"groups[{group_index}] contains a non-integer process index")
                if indexes != sorted(indexes) or len(set(indexes)) != len(indexes):
                    raise ValueError(f"groups[{group_index}] process indexes must be unique and ordered")
                if indexes[0] < 0 or indexes[-1] >= len(steps):
                    raise ValueError(f"groups[{group_index}] references a missing process index")
                if indexes != list(range(indexes[0], indexes[-1] + 1)):
                    raise ValueError(f"groups[{group_index}] must describe a contiguous linear stage")
                if indexes[0] <= previous_first:
                    raise ValueError("groups must follow recipe execution order")
                previous_first = indexes[0]
                for process_index in indexes:
                    if process_index in owner:
                        raise ValueError(f"process index {process_index} belongs to more than one group")
                    owner[process_index] = group_index
                normalized.append(
                    (
                        indexes[0],
                        {
                            "title": title,
                            "summary": str(raw.get("summary") or "").strip(),
                            "step_refs": [steps[index]["id"] for index in indexes],
                        },
                    )
                )

            for index, step in enumerate(steps):
                if index not in owner:
                    normalized.append(
                        (
                            index,
                            {"title": step["operator_name"], "summary": "", "step_refs": [step["id"]]},
                        )
                    )
            normalized.sort(key=lambda item: item[0])
            groups = [
                {"id": f"stage-{index + 1:03d}", **group}
                for index, (_, group) in enumerate(normalized)
            ]
        except ValueError as exc:
            warnings.append(str(exc))

    return (
        {
            "schema_version": "1.0",
            "plan_id": plan.get("plan_id"),
            "plan_version": plan.get("plan_version"),
            "recipe_content_hash": content_hash,
            "layout": "linear",
            "groups": groups,
            "steps": steps,
        },
        warnings,
    )
