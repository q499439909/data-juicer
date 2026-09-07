"""Version-matched authoring contracts for account-scoped DJ operators."""

from __future__ import annotations

import ast
from copy import deepcopy

from data_juicer import __version__ as dj_version

from .common import PlanFlowError
from .user_operator_store import CATEGORIES

CONTRACT_VERSION = "dj-operator-authoring/v2"
MODALITIES = frozenset({"generic", "text", "image", "audio", "video", "multimodal"})
_MODALITY_NOTES = {
    "generic": "Use the configured text_key/image_key/audio_key/video_key fields instead of hard-coded keys.",
    "text": "Read and write text through self.text_key.",
    "image": "Read image path lists through self.image_key. Use DJ multimedia loaders when caching decoded images.",
    "audio": "Read audio path lists through self.audio_key.",
    "video": "Read video path lists through self.video_key.",
    "multimodal": "Use self.text_key and the configured image/audio/video keys for each modality.",
}


_MODEL_POLICY = {
    "decision_order": [
        "reuse_a_matching_dj_builtin",
        "none",
        "api_vlm",
        "remote_model",
        "local_model",
    ],
    "strategies": {
        "none": "Prefer a lightweight deterministic implementation when it can meet the validation contract.",
        "api_vlm": (
            "Use the backend-configured API model and credentials. Do not put secrets in source, parameters, "
            "model_refs, or assets; this strategy does not download model weights."
        ),
        "remote_model": (
            "Declare every model as {model_id, revision} in proposal.model_refs, where revision is an immutable "
            "40-64 character hexadecimal commit. Load that same model_id and revision in source."
        ),
        "local_model": (
            "Declare an existing absolute model file path in proposal.model_refs. Validation freezes its SHA256; "
            "do not copy the model into the operator directory."
        ),
    },
    "dependencies": (
        "Declare all additional Python packages as exact name==version pins in proposal.dependencies. "
        "They are installed in the current account's isolated operator runtime."
    ),
    "cache": (
        "Model frameworks download into the configured backend model caches during validation; cached models "
        "are reusable and are not deleted with temporary test files."
    ),
    "assets": "proposal.assets is for small text resources only, never model weights.",
    "confirmation_required": [
        "paid_api",
        "private_or_gated_model",
        "large_model_download",
    ],
    "next_step": (
        "Choose exactly one strategy, fill source_scaffold, declare only the resources it needs, then call "
        "develop_custom_operator. Do not perform extra interface research."
    ),
}


_SPECS = {
    "mapper": ("Mapper", ["process_single"], ["process"]),
    "filter": ("Filter", ["compute_stats_single", "process_single"], ["compute_stats", "process"]),
    "deduplicator": ("Deduplicator", ["compute_hash", "process"], []),
    "selector": ("Selector", ["process"], []),
    "grouper": ("Grouper", ["process"], []),
    "aggregator": ("Aggregator", ["process_single"], []),
    "pipeline": ("Pipeline", ["run"], []),
}


_METHODS = {
    "mapper": """    def process_single(self, sample):
        return sample
""",
    "filter": """    def compute_stats_single(self, sample, context=False):
        sample.setdefault(Fields.stats, {})["replace_with_stat"] = 0.0
        return sample

    def process_single(self, sample):
        return True
""",
    "deduplicator": """    def compute_hash(self, sample):
        return sample

    def process(self, dataset, show_num=0):
        return dataset, []
""",
    "selector": """    def process(self, dataset):
        return dataset
""",
    "grouper": """    def process(self, dataset):
        return list(dataset)
""",
    "aggregator": """    def process_single(self, sample):
        return sample
""",
    "pipeline": """    def run(self, dataset):
        return dataset
""",
}


def _source_scaffold(category: str, base_class: str) -> str:
    fields_import = "from data_juicer.utils.constant import Fields\n" if category == "filter" else ""
    class_name = "ReplaceWithOperatorClass"
    return f'''from data_juicer.ops.base_op import OPERATORS, {base_class}
{fields_import}
OP_NAME = "replace_with_operator_name"


@OPERATORS.register_module(OP_NAME)
class {class_name}({base_class}):
    """Describe the operator's behavior and observable output contract."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_parameters = self.remove_extra_parameters(locals())

{_METHODS[category]}'''


class OperatorAuthoring:
    """Own the DJ-version-specific facts needed to author one valid operator."""

    def get_spec(self, category: str, modality: str = "generic") -> dict:
        if category not in CATEGORIES:
            raise PlanFlowError("INVALID_OPERATOR_TYPE", "Unsupported DJ operator type")
        if modality not in MODALITIES:
            raise PlanFlowError("INVALID_MODALITY", "Unsupported DJ operator modality")
        base_class, required_methods, forbidden_overrides = _SPECS[category]
        return {
            "dj_version": dj_version,
            "contract_version": CONTRACT_VERSION,
            "category": category,
            "modality": modality,
            "base_class": base_class,
            "required_methods": required_methods,
            "forbidden_overrides": forbidden_overrides,
            "registration": "@OPERATORS.register_module(OP_NAME)",
            "optional_integrations": (
                [
                    "from data_juicer.ops.op_fusion import LOADED_IMAGES",
                    "@LOADED_IMAGES.register_module(OP_NAME)",
                ]
                if modality == "image"
                else []
            ),
            "modality_notes": _MODALITY_NOTES[modality],
            "constraints": {
                "register_exactly_one": True,
                "operator_name_must_match": True,
            },
            "model_policy": deepcopy(_MODEL_POLICY),
            "source_scaffold": _source_scaffold(category, base_class),
        }

    def lint(self, name: str, category: str, source: str) -> None:
        """Reject source that cannot satisfy the running DJ operator interface."""
        spec = self.get_spec(category)
        diagnostics = []
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            self._reject(
                [{"line": exc.lineno or 1, "symbol": "python", "message": exc.msg}],
            )
            return

        assignments = {
            target.id: node.value.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        imports = [node for node in tree.body if isinstance(node, ast.ImportFrom)]
        base_imported = any(
            node.module == "data_juicer.ops.base_op"
            and any(
                alias.name == spec["base_class"] and alias.asname in {None, spec["base_class"]} for alias in node.names
            )
            for node in imports
        )
        if not base_imported:
            diagnostics.append(
                {
                    "line": 1,
                    "symbol": f"data_juicer.ops.base_op.{spec['base_class']}",
                    "message": f"Import {spec['base_class']} from the running DJ base_op module",
                    "replacement": f"from data_juicer.ops.base_op import OPERATORS, {spec['base_class']}",
                }
            )
        registry_imported = any(
            node.module in {"data_juicer.ops", "data_juicer.ops.base_op"}
            and any(alias.name == "OPERATORS" and alias.asname in {None, "OPERATORS"} for alias in node.names)
            for node in imports
        )
        if not registry_imported:
            diagnostics.append(
                {
                    "line": 1,
                    "symbol": "data_juicer.ops.base_op.OPERATORS",
                    "message": "Import the running DJ operator registry",
                    "replacement": f"from data_juicer.ops.base_op import OPERATORS, {spec['base_class']}",
                }
            )
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "data_juicer.ops.base_filter":
                diagnostics.append(
                    {
                        "line": node.lineno,
                        "symbol": node.module,
                        "message": f"DJ {dj_version} does not provide this module",
                        "replacement": (f"from data_juicer.ops.base_op import OPERATORS, {spec['base_class']}"),
                    }
                )

        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]

        def is_registration(decorator):
            return (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr == "register_module"
                and isinstance(decorator.func.value, ast.Name)
                and decorator.func.value.id == "OPERATORS"
            )

        all_registrations = [
            decorator for candidate in classes for decorator in candidate.decorator_list if is_registration(decorator)
        ]
        if len(all_registrations) != 1:
            diagnostics.append(
                {
                    "line": all_registrations[0].lineno if all_registrations else 1,
                    "symbol": "OPERATORS.register_module",
                    "message": "Source must contain exactly one DJ operator registration",
                    "replacement": "Keep only @OPERATORS.register_module(OP_NAME) on the operator class",
                }
            )
        matching = [
            node
            for node in classes
            if any(isinstance(base, ast.Name) and base.id == spec["base_class"] for base in node.bases)
        ]
        if len(matching) != 1:
            diagnostics.append(
                {
                    "line": classes[0].lineno if classes else 1,
                    "symbol": spec["base_class"],
                    "message": f"Source must define exactly one {spec['base_class']} subclass",
                    "replacement": spec["base_class"],
                }
            )
        operator_class = matching[0] if len(matching) == 1 else (classes[0] if len(classes) == 1 else None)
        if operator_class is not None:
            methods = {
                node.name: node
                for node in operator_class.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            for method in spec["required_methods"]:
                if method not in methods:
                    diagnostics.append(
                        {
                            "line": operator_class.lineno,
                            "symbol": method,
                            "message": f"{spec['base_class']} subclasses must implement {method}",
                            "replacement": method,
                        }
                    )
            for method in spec["forbidden_overrides"]:
                if method in methods:
                    replacement = f"{method}_single"
                    diagnostics.append(
                        {
                            "line": methods[method].lineno,
                            "symbol": method,
                            "message": f"{spec['base_class']} subclasses must not override {method}",
                            "replacement": replacement,
                        }
                    )

            registrations = []
            for decorator in operator_class.decorator_list:
                if is_registration(decorator):
                    registrations.append(decorator)
            if len(registrations) != 1:
                diagnostics.append(
                    {
                        "line": operator_class.lineno,
                        "symbol": "OPERATORS.register_module",
                        "message": "Operator class must have exactly one DJ registration decorator",
                        "replacement": "@OPERATORS.register_module(OP_NAME)",
                    }
                )
            else:
                call = registrations[0]
                registered_name = None
                if len(call.args) == 1:
                    argument = call.args[0]
                    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                        registered_name = argument.value
                    elif isinstance(argument, ast.Name):
                        registered_name = assignments.get(argument.id)
                if registered_name != name:
                    diagnostics.append(
                        {
                            "line": call.lineno,
                            "symbol": "operator_name",
                            "message": "Registered operator name must match proposal.name",
                            "replacement": name,
                        }
                    )

        from data_juicer.ops import OPERATORS

        if name in OPERATORS.modules:
            diagnostics.append(
                {
                    "line": 1,
                    "symbol": "operator_name",
                    "message": "A personal operator cannot use a DJ built-in registration name",
                    "replacement": "Choose a distinct personal operator name",
                }
            )
        if diagnostics:
            self._reject(diagnostics)

    @staticmethod
    def _reject(diagnostics: list[dict]) -> None:
        raise PlanFlowError(
            "OPERATOR_SOURCE_INCOMPATIBLE",
            "Generated source does not match the running DJ operator interface",
            details={
                "phase": "source_preflight",
                "contract_version": CONTRACT_VERSION,
                "diagnostics": diagnostics,
            },
        )


operator_authoring = OperatorAuthoring()
