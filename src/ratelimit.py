"""src/ratelimit.py — Redis fixed-window rate limits. Fails open: a Redis problem never blocks a query."""

import logging
import time
from typing import Optional, Tuple

from src.cache import _get_client
from src.config import config

logger = logging.getLogger("crag_ratelimit")


def check_rate_limit(ip: str) -> Optional[Tuple[str, int]]:
    """Returns None if the request may proceed, else (scope, retry_after_seconds)."""
    client = _get_client()
    if client is None:
        return None
    try:
        now = int(time.time())

        ip_key = f"crag:rl:ip:{ip}:{now // 60}"
        pipe = client.pipeline()
        pipe.incr(ip_key)
        pipe.expire(ip_key, 90)
        ip_count = pipe.execute()[0]
        if ip_count > config.rate_limit_per_minute:
            return ("ip", 60 - (now % 60))

        day_key = f"crag:rl:global:{time.strftime('%Y%m%d', time.gmtime(now))}"
        pipe = client.pipeline()
        pipe.incr(day_key)
        pipe.expire(day_key, 90000)
        day_count = pipe.execute()[0]
        if day_count > config.rate_limit_daily_global:
            return ("global", 86400 - (now % 86400))
    except Exception as e:
        logger.warning(f"Rate-limit check failed (ignored): {type(e).__name__}")
        return None
    return None
