import hashlib


def text_signature(text: str) -> str:
    normalized = " ".join(str(text).split()).casefold().encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()[:16]
