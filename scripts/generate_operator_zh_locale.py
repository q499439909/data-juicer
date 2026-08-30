"""Regenerate the checked-in zh-CN operator-library presentation asset."""

from __future__ import annotations

import json

from data_juicer.tools.plan_flow.discovery import operator_catalog, operator_detail
from data_juicer.tools.plan_flow.localization import (
    LOCALE_PATH,
    translated_operator_name,
    translated_operator_text,
    translated_parameter_name,
    translated_parameter_summary,
)


def main() -> None:
    operators = {}
    for item in operator_catalog()["operators"]:
        detail = operator_detail(item["name"])["operator"]
        display_name = translated_operator_name(item["name"], item["category"])
        copy = translated_operator_text(
            item["name"],
            display_name,
            item["category"],
            item["modalities"],
            item["description"],
        )
        operators[item["name"]] = {
            "display_name": display_name,
            "summary": copy["summary"],
            "description": copy["description"],
            "parameters": {
                parameter["name"]: {
                    "display_name": translated_parameter_name(parameter["name"]),
                    "description": translated_parameter_summary(
                        translated_parameter_name(parameter["name"]), parameter["required"]
                    ),
                }
                for parameter in detail["parameters"]
            },
        }

    LOCALE_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCALE_PATH.write_text(
        json.dumps({"schema_version": 2, "locale": "zh-CN", "operators": operators}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(operators)} operators to {LOCALE_PATH}")


if __name__ == "__main__":
    main()
