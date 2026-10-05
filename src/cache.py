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


def make_key(query: str, scope: str = "default", web_mode: str = "allowlist") -> str:
    """Key covers normalized query + scope + web mode + cache version."""
    norm = re.sub(r"\s+", " ", query.strip().lower())
    payload = json.dumps(
        {"q": norm, "scope": scope, "web": web_mode, "v": config.cache_version},
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