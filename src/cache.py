"""src/cache.py — Redis answer cache. Fails open: a Redis problem never breaks a query."""

import hashlib
import json
import logging
import re
from typing import Optional

import redis

from src.config import config

logger = logging.getLogger("crag_cache")
_client: Optional[redis.Redis] = None

# Fingerprint of the indexed corpus, set once at API startup. It is part of every cache key,
# so after any re-ingestion + restart, old cached answers become unreachable (they just expire).
_corpus_fingerprint = "unset"

REFUSAL_MARKER = "cannot answer this based on the provided documents"
WARNING_PREFIX = "[warning"


def _get_client() -> Optional[redis.Redis]:
    global _client
    if not config.cache_enabled:
        return None
    if _client is None:
        _client = redis.Redis(
            host=config.redis_host,
            port=config.redis_port,
            socket_connect_timeout=0.3,
            socket_timeout=0.3,
            decode_responses=True,
        )
    return _client


def set_corpus_fingerprint(chunks) -> str:
    """Hash of all indexed chunk ids + texts. Call once at startup with retriever.all_chunks."""
    global _corpus_fingerprint
    h = hashlib.md5()
    for c in sorted(chunks, key=lambda c: c["id"]):
        h.update(c["id"].encode("utf-8"))
        h.update(c["text"].encode("utf-8"))
    _corpus_fingerprint = h.hexdigest()[:12]
    return _corpus_fingerprint


def is_cacheable(answer: str) -> bool:
    """Never cache empty answers, refusals, or answers flagged as unverified: a retry may do better."""
    a = (answer or "").strip().lower()
    if not a:
        return False
    if a.startswith(WARNING_PREFIX):
        return False
    if REFUSAL_MARKER in a:
        return False
    return True


def make_key(query: str, scope: str = "default", web_mode: str = "allowlist") -> str:
    """Key covers normalized query + scope + web mode + cache version + corpus fingerprint."""
    norm = re.sub(r"\s+", " ", query.strip().lower())
    payload = json.dumps(
        {"q": norm, "scope": scope, "web": web_mode, "v": config.cache_version, "c": _corpus_fingerprint},
        sort_keys=True,
    )
    return "crag:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_cached(key: str) -> Optional[dict]:
    client = _get_client()
    if client is None:
        return None
    try:
        raw = client.get(key)
        return json.loads(raw) if raw else None
    except Exception as e:
        logger.warning(f"Cache read failed (ignored): {type(e).__name__}")
        return None


def set_cached(key: str, value: dict, web_used: bool) -> None:
    client = _get_client()
    if client is None:
        return
    ttl = config.cache_ttl_web_seconds if web_used else config.cache_ttl_seconds
    try:
        client.setex(key, ttl, json.dumps(value))
    except Exception as e:
        logger.warning(f"Cache write failed (ignored): {type(e).__name__}")