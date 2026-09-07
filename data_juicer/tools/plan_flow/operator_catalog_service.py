"""Unified discovery without importing personal operators into the server registry."""

from . import discovery
from .user_operator_store import UserOperatorStore, current_user, public_candidate

_SEARCH_FIELDS = (
    "candidate_id",
    "name",
    "type",
    "tags",
    "description",
    "match_score",
    "matched_requirements",
    "provider",
    "status",
    "version",
)
_SCHEMA_FIELDS = (
    "candidate_id",
    "name",
    "type",
    "parameters",
    "provider",
    "operator_id",
    "status",
    "version",
    "validation_basis",
    "validation_summary",
    "runtime_status",
)
_USER_MIN_QUERY_COVERAGE = 0.25
_QUERY_STOP_WORDS = {
    "a",
    "an",
    "and",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


def select_fields(item, fields):
    return {key: item[key] for key in fields if key in item}


def builtin(item):
    from data_juicer import __version__

    name = item["name"]
    return {
        **item,
        "candidate_id": f"dj:{name}",
        "provider": "dj",
        "operator_id": name,
        "version": __version__,
        "status": "validated",
        "validation_basis": "builtin_release",
        "validation_summary": {"limitations": ["Built-in release, not current-task quality evidence"]},
        "runtime_status": "unknown",
    }


def personal():
    return UserOperatorStore().candidates() if current_user.get() else []


def catalog():
    result = discovery.operator_catalog()
    result["operators"] = [builtin(item) for item in result["operators"]]
    for item in personal():
        result["operators"].append(
            {
                **public_candidate(item),
                "category": item["type"],
                "modalities": [tag for tag in item["tags"] if tag in discovery._CATALOG_MODALITIES] or ["general"],
                "devices": [tag for tag in item["tags"] if tag in {"cpu", "gpu"}] or ["unknown"],
            }
        )
    result["total"] = len(result["operators"])
    result["facets"]["providers"] = ["dj", "user"]
    return result


def schemas(refs):
    operators, missing = [], []
    for ref in dict.fromkeys(refs):
        if ref.startswith("user:"):
            operators.append(public_candidate(UserOperatorStore().resolve(ref)))
        else:
            item = discovery.operator_schema(ref.removeprefix("dj:"))
            if item:
                operators.append(builtin(item))
            else:
                missing.append(ref)
    return {
        "ok": not missing,
        "operators": [select_fields(item, _SCHEMA_FIELDS) for item in operators],
        "missing": missing,
    }


def detail(ref):
    if not ref.startswith("user:"):
        result = discovery.operator_detail(ref.removeprefix("dj:"))
        if result.get("ok"):
            result["operator"] = builtin(result["operator"])
        return result
    item = schemas([ref])["operators"][0]
    item["category"] = item["type"]
    item["parameters"] = [{"name": key, **value} for key, value in item["parameters"].items()]
    return {"ok": True, "operator": item}


def _query_coverage(query_tokens, candidate_tokens):
    """Return comparable lexical coverage without rewarding generic stop words."""
    query_terms = set(query_tokens) - _QUERY_STOP_WORDS
    if not query_terms:
        return 0.0
    candidate_terms = set(candidate_tokens) - _QUERY_STOP_WORDS
    return len(query_terms & candidate_terms) / len(query_terms)


def search(requirements, modality=None, executor_type="default", top_k=3):
    from data_juicer.tools.op_search import OPSearcher

    result = discovery.search_capabilities(requirements, modality, executor_type, top_k)
    users = [
        public_candidate(item)
        for item in personal()
        if not modality or modality in item["tags"] or "multimodal" in item["tags"]
    ]
    corpus = [
        OPSearcher._tokenize(" ".join([item["name"], item["description"], str(item["parameters"])])) or ["operator"]
        for item in users
    ]
    natives = {item["name"]: builtin(item) for item in result["operators"]}
    selected = {}
    for row in result["results"]:
        query = row["requirement"]
        tokens = OPSearcher._tokenize(query)
        user_candidates = []
        for index, item in enumerate(users):
            exact = item["name"] == query or item["candidate_id"] == query
            coverage = 1.0 if exact else _query_coverage(tokens, corpus[index])
            if not exact and coverage < _USER_MIN_QUERY_COVERAGE:
                continue
            user_candidates.append({**item, "match_score": round(coverage, 6)})

        groups = [[natives[name] for name in row["operator_names"]], user_candidates]
        fused = []
        for group in groups:
            for rank, item in enumerate(group):
                fused.append(
                    (
                        item["name"] == query or item["candidate_id"] == query,
                        float(item.get("match_score", 0.0)),
                        item["status"] == "validated",
                        item["provider"] == "dj",
                        -rank,
                        item,
                    )
                )
        fused.sort(key=lambda entry: entry[:5], reverse=True)
        candidates = []
        for entry in fused[: result["top_k"]]:
            item = dict(entry[5])
            item["matched_requirements"] = list(dict.fromkeys([*item.get("matched_requirements", []), query]))
            candidates.append(item)
        row["candidate_ids"] = [item["candidate_id"] for item in candidates]
        row["operator_names"] = [item["name"] for item in candidates]
        row["coverage"] = "candidates" if candidates else "gap"
        row["fallbacks"] = [] if candidates else ["custom_operator"]
        for item in candidates:
            previous = selected.get(item["candidate_id"])
            if previous:
                previous["match_score"] = max(previous.get("match_score", 0.0), item.get("match_score", 0.0))
                previous["matched_requirements"] = list(
                    dict.fromkeys([*previous.get("matched_requirements", []), *item["matched_requirements"]])
                )
            else:
                selected[item["candidate_id"]] = item
    result["operators"] = [select_fields(item, _SEARCH_FIELDS) for item in selected.values()]
    return result
