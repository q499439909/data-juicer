"""Versioned public planning contract, shared by discovery and validation."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Rule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    field: str = Field(description="Numeric per-image field in __dj__stats__, queried from capability schemas.")
    min: float | None = Field(default=None, allow_inf_nan=False)
    max: float | None = Field(default=None, allow_inf_nan=False)
    min_inclusive: bool = True
    max_inclusive: bool = True


class ImageAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["image_audit"]
    audit_mode: Literal["full", "cascade"] = Field(
        default="full",
        description="Full requires every rule score on every image; cascade marks downstream rules not_evaluated after first rejection.",
    )
    rules: list[Rule] = Field(min_length=1)
    image_key: str = "images"
    copy_kept: bool = False
    output_prefix: str = Field(default="audit", pattern=r"^[a-zA-Z0-9_-]+$")


class PythonStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["python"]
    script: str = Field(
        description="Existing authorized workspace source, frozen and hashed into the Plan artifact bundle."
    )
    arguments: list[str] | dict[str, Any] = Field(
        default_factory=dict,
        description="Tokens ${recipe.output}, ${RUN_OUTPUT}, ${RUN_DIR}; runs in the Run output directory.",
    )


class PackageManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1, description="Existing JSONL result manifest inside RUN_OUTPUT.")
    media_dir: str = Field(min_length=1, description="New or empty media directory inside RUN_OUTPUT.")
    media_keys: list[str] = Field(default_factory=list)


class DatasetPackage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["dataset_package"]
    manifests: list[PackageManifest] = Field(
        min_length=1,
        description="Copy referenced local media into each output package and rewrite JSONL media paths as relative paths.",
    )


class Recipe(BaseModel):
    model_config = ConfigDict(extra="allow")
    dataset_path: str | None = Field(
        default=None,
        description="Local manifest path in the selected workspace. Raw media folders must first use inspect_input.",
    )
    dataset: dict[str, Any] | None = Field(
        default=None,
        description='Alternative to dataset_path: {"configs":[{"type":"local","path":"input.jsonl"}]}. Exactly one input source is required.',
    )
    generated_dataset_config: dict[str, Any] | None = Field(
        default=None,
        description="Advanced DJ input configuration; native controlled input freezing may reject unsupported sources.",
    )
    export_path: str = Field(
        default="processed.jsonl",
        description="Only the basename is used inside the new Run output directory. Never overwrites the input.",
    )
    process: list[
        Annotated[dict[str, dict[str, Any]], Field(json_schema_extra={"minProperties": 1, "maxProperties": 1})]
    ] = Field(
        default_factory=list,
        description='Ordered single-operator objects: [{"clean_links_mapper":{"repl":""}}]. Parameter definitions come from get_capability_schemas.',
    )
    text_keys: str | list[str] = "text"
    image_key: str = "images"
    executor_type: str = "default"
    np: int = Field(default=1, ge=1)
    keep_stats_in_res_ds: bool = Field(
        default=True,
        description="Plan-flow preserves __dj__meta__ and __dj__stats__ in the recipe output by default so downstream steps have one deterministic input. Set false explicitly only when a compact primary dataset and companion *_stats.jsonl are intended.",
    )


class Producer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["recipe", "postprocess"]
    index: int | None = Field(
        default=None, ge=0, description="Required zero-based postprocess index for kind=postprocess."
    )


class AcceptanceCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    kind: Literal["row_count", "field_equals", "audit_consistency", "manual"]
    criterion: str | None = Field(default=None, description="Exact acceptance_criteria text covered by this evidence.")
    output_id: str | None = Field(default=None, description="Required declared output ID for automatic checks.")
    min: int = Field(default=0, ge=0)
    max: int | None = Field(default=None, ge=0)
    field: str | None = None
    value: Any = None


class Output(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    path: str = Field(
        description="Safe relative path within RUN_OUTPUT. Zero records/files is valid unless a positive minimum is explicitly requested."
    )
    format: Literal["file", "json", "jsonl", "directory"] = "file"
    producer: Producer = Field(
        description='{"kind":"recipe"} for export_path, or {"kind":"postprocess","index":0} for a zero-based postprocess index.'
    )
    min_records: int = Field(default=0, ge=0)
    min_files: int = Field(default=0, ge=0)
    required_fields: list[str] = Field(default_factory=list)


class Coverage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str | None = None
    requirement: str = Field(min_length=1)
    status: Literal["covered", "gap", "needs_user_input"]
    evidence: str = Field(
        default="",
        description="Observed evidence or exact capability limitation. Filenames and inventory do not prove media quality.",
    )


class AuditDecision(BaseModel):
    image_id: str
    source: str
    image_index: int
    scores: dict[str, Annotated[float, Field(strict=True, allow_inf_nan=False)] | None]
    checks: dict[str, Literal["passed", "failed", "not_evaluated"]]
    kept: bool
    first_reason: str | None
    all_reasons: list[str]
    all_reasons_complete: bool
    output_image: str | None = None


class AuditStage(BaseModel):
    id: str
    entered: int
    removed: int
    remaining: int
    independent_hits: int


class AuditSummary(BaseModel):
    schema_version: Literal[1]
    audit_mode: Literal["full", "cascade"]
    input: int
    kept: int
    removed: int
    stages: list[AuditStage]
    rules: list[Rule]
    zero_retained_valid: bool
    independent_hits_complete: bool


class PlanDraft(BaseModel):
    # Existing DJ extensions remain usable; semantic validation checks config fields.
    model_config = ConfigDict(extra="allow")
    schema_version: Literal[1] = 1
    user_intent: str = Field(min_length=1)
    modality: str = "unknown"
    recipe: Recipe
    postprocess: list[ImageAudit | PythonStep | DatasetPackage] = Field(default_factory=list)
    expected_outputs: list[Output] = Field(default_factory=list)
    acceptance_checks: list[AcceptanceCheck] = Field(
        default_factory=list,
        description="Manual criteria remain unverified until human review; record counts cannot prove image clarity or other semantic quality.",
    )
    acceptance_criteria: list[str] = Field(default_factory=list)
    risk_notes: list[str] = Field(default_factory=list)
    coverage: list[Coverage] = Field(
        default_factory=list,
        description="One record per requirement. The host records explicit gap acceptance with the fixed Plan approval; needs_user_input must be clarified first.",
    )


def contract():
    return {
        "ok": True,
        "contract_version": "plan-v1.3",
        "schema": PlanDraft.model_json_schema(),
        "image_audit": {
            "input": "Recipe JSONL with every original image and __dj__stats__ arrays in image order. Native image_audit uses Filter.run(reduce=False) and keeps all scores. No earlier dropping or expansion is allowed.",
            "outputs": [
                "audit/decisions.jsonl",
                "audit/kept.jsonl",
                "audit/summary.json",
                "audit/report.md",
                "audit/images/ (copy_kept=true)",
            ],
            "output_schemas": {
                "decisions.jsonl": AuditDecision.model_json_schema(),
                "kept.jsonl": AuditDecision.model_json_schema(),
                "summary.json": AuditSummary.model_json_schema(),
            },
            "example_decision": {
                "image_id": "content-identity",
                "source": "D:/workspace/data/image.jpg",
                "image_index": 0,
                "scores": {"image_watermark_prob": 0.8},
                "checks": {"watermark": "failed"},
                "kept": False,
                "first_reason": "watermark",
                "all_reasons": ["watermark"],
                "all_reasons_complete": True,
            },
            "recipe_example": {
                "postprocess": [
                    {
                        "kind": "image_audit",
                        "copy_kept": True,
                        "rules": [
                            {"id": "watermark", "field": "image_watermark_prob", "max": 0.8, "max_inclusive": False}
                        ],
                    }
                ],
                "acceptance_checks": [{"id": "audit_integrity", "kind": "audit_consistency", "output_id": "summary"}],
            },
            "semantics": "Rules are the single decision source. Bounds default inclusive; use max_inclusive=false for strict < thresholds. Native report.md renders decisions and summary without reevaluating thresholds. Missing scores are never zero or a pass. Full mode fails on missing scores; cascade records not_evaluated after rejection. Empty input and zero retained are valid. stages gives entered/removed/remaining and independent_hits; report and copied images use those same decisions.",
            "limitations": [
                "Does not infer clarity from face detection.",
                "Cascade currently saves decision evaluation, not inference cost: DJ scores the full input. Native execution only.",
            ],
        },
        "dataset_package": {
            "semantics": "Task-specific processing writes result JSONL that references read-only source media. A later generic dataset_package step copies only those selected media into RUN_OUTPUT and rewrites media fields to paths relative to each JSONL manifest.",
            "example": {
                "kind": "dataset_package",
                "manifests": [
                    {"path": "cats.jsonl", "media_dir": "cats"},
                    {"path": "dogs.jsonl", "media_dir": "dogs"},
                ],
            },
        },
        "approval": {
            "tool": "mcp__dj__confirm_plan",
            "identity": ["task_id", "plan_version", "content_hash"],
            "automatic_dispatch": True,
        },
    }
