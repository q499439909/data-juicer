"""Unified discovery without importing personal operators into the server registry."""

from functools import lru_cache

from . import discovery
from .user_operator_store import UserOperatorStore, current_user, public_candidate

_SEARCH_FIELDS = (
    "candidate_id",
    "name",
    "type",
    "tags",
    "description",
    "match_score",
    "ranking",
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
    "model_locks",
    "input_contract",
    "output_contract",
    "score_semantics",
    "limitations",
    "runtime_requirements",
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
        "model_locks": _builtin_model_locks().get(name, []),
    }


@lru_cache(maxsize=1)
def _builtin_model_locks():
    from .model_lock_resolver import load_builtin_model_catalog

    result = {}
    for model in load_builtin_model_catalog()["models"]:
        for consumer in model.get("consumers", []):
            result.setdefault(consumer["operator"], []).append(
                {
                    "provider": "dj",
                    "lock_id": model["lock_id"],
                    "backend": model["backend"],
                    "model_id": model.get("model_id"),
                    "revision": model.get("revision"),
                    "distribution": model.get("distribution"),
                    "version": model.get("version"),
                    "parameter": consumer.get("parameter"),
                    "status": "locked",
                }
            )
    return result


def _personal_model_locks(candidate):
    result = []
    for ref in candidate.get("_manifest", {}).get("model_refs", []):
        result.append(
            {
                "provider": "user",
                "backend": "local-file" if ref.get("path") else "huggingface",
                "model_id": ref.get("model_id"),
                "revision": ref.get("revision"),
                "parameter": ref.get("parameter"),
                "sha256": ref.get("sha256"),
                "status": "locked",
            }
        )
    return result


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
    from .score_contracts import enrich

    operators, missing = [], []
    for ref in dict.fromkeys(refs):
        if ref.startswith("user:"):
            candidate = UserOperatorStore().resolve(ref)
            operators.append({**public_candidate(candidate), "model_locks": _personal_model_locks(candidate)})
        else:
            item = discovery.operator_schema(ref.removeprefix("dj:"))
            if item:
                operators.append(builtin(item))
            else:
                missing.append(ref)
    return {
        "ok": not missing,
        "operators": [select_fields(enrich(item), _SCHEMA_FIELDS) for item in operators],
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
    """Merge per-query source ranks, never compare BM25 with token coverage."""
    from data_juicer.tools.op_search import OPSearcher

    result = discovery.search_capabilities(requirements, modality, executor_type, top_k)
    users = [
        public_candidate(item)
        for item in personal()
        if not modality or modality == "multimodal" or modality in item["tags"] or "multimodal" in item["tags"]
    ]
    corpus = [
        OPSearcher._tokenize(" ".join([item["name"], item["description"], str(item["parameters"])])) for item in users
    ]
    natives = {item["name"]: builtin(item) for item in result["operators"]}
    selected = {}
    for row in result["results"]:
        query = row["requirement"]
        tokens = OPSearcher._tokenize(query)
        personal_hits = []
        for index, item in enumerate(users):
            exact = item["name"] == query or item["candidate_id"] == query
            raw = 1.0 if exact else _query_coverage(tokens, corpus[index])
            if exact or raw >= _USER_MIN_QUERY_COVERAGE:
                personal_hits.append(
                    (item, {"method": "exact_name" if exact else "token_coverage", "raw_score": raw, "exact": exact})
                )
        personal_hits.sort(key=lambda pair: (-int(pair[1]["exact"]), -pair[1]["raw_score"], pair[0]["candidate_id"]))
        for rank, (_, evidence) in enumerate(personal_hits, 1):
            evidence["rank"] = rank
        native_evidence = {item["name"]: item for item in row.get("retrieval", [])}
        hits = [
            (
                natives[name],
                native_evidence.get(name, {"rank": rank, "method": "bm25", "raw_score": None, "exact": name == query}),
            )
            for rank, name in enumerate(row["operator_names"], 1)
        ] + personal_hits
        # Destructive image transforms are not evidence of a detector/quality metric.
        # Keep exact-name requests and explicit editing requirements available.
        editing = any(
            word in query.casefold()
            for word in ("remove", "inpaint", "blur faces", "去除", "去水印", "模糊人脸", "修复")
        )
        if not editing:
            hits = [
                pair
                for pair in hits
                if pair[1]["exact"] or not any(part in pair[0]["name"] for part in ("_remove_mapper", "_blur_mapper"))
            ]
        # Each provider is one retrieval list. Equal source ranks use stable ties;
        # the raw scores are evidence, not a cross-provider quality comparison.
        hits.sort(
            key=lambda pair: (
                -int(pair[1]["exact"]),
                -1 / (60 + pair[1]["rank"]),
                -int(pair[0]["status"] == "validated"),
                pair[0]["provider"],
                pair[0]["candidate_id"],
            )
        )
        candidates = []
        row["ranking"] = []
        for overall_rank, (original, evidence) in enumerate(hits[: result["top_k"]], 1):
            item = dict(original)
            score = 1.0 if evidence["exact"] else 61 / (60 + evidence["rank"])
            rank_info = {
                "requirement": query,
                "method": evidence["method"],
                "raw_score": evidence["raw_score"],
                "source_rank": evidence["rank"],
                "rank": overall_rank,
                "exact": evidence["exact"],
                "fusion_score": round(1 / (60 + evidence["rank"]), 8),
            }
            item.update(match_score=round(score, 6), matched_requirements=[query], ranking=[rank_info])
            row["ranking"].append({"candidate_id": item["candidate_id"], **rank_info})
            candidates.append(item)
        row["candidate_ids"] = [item["candidate_id"] for item in candidates]
        row["operator_names"] = [item["name"] for item in candidates]
        row["coverage"] = "candidates" if candidates else "gap"
        row["fallbacks"] = [] if candidates else ["custom_operator"]
        for item in candidates:
            previous = selected.get(item["candidate_id"])
            if previous:
                previous["match_score"] = max(previous["match_score"], item["match_score"])
                previous["matched_requirements"] = list(dict.fromkeys([*previous["matched_requirements"], query]))
                previous["ranking"].extend(item["ranking"])
            else:
                selected[item["candidate_id"]] = item
    result["operators"] = [select_fields(item, _SEARCH_FIELDS) for item in selected.values()]
    result["ranking_policy"] = {
        "method": "reciprocal_rank_merge",
        "k": 60,
        "exact_first": True,
        "score_semantics": "Rank-derived relevance only, not quality or a calibrated probability",
        "sources": ["bm25", "token_coverage"],
        "ties": ["validated", "provider", "candidate_id"],
    }
    return result
