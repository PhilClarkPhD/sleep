"""
Security utilities for the Mora Sleep Scoring API.

Provides:
- Optional API key authentication
- Rate limiting
- Request logging
"""

import hashlib
import logging
import time
from collections import defaultdict
from typing import Optional

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from app.core.config import settings

logger = logging.getLogger(__name__)

# API key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class RateLimiter:
    """
    Simple in-memory rate limiter.

    For production, consider using Redis-based rate limiting.
    """

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests: dict = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        """Check if a client is allowed to make a request."""
        now = time.time()
        minute_ago = now - 60

        # Clean old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > minute_ago
        ]

        # Check rate limit
        if len(self.requests[client_id]) >= self.requests_per_minute:
            return False

        # Record this request
        self.requests[client_id].append(now)
        return True

    def get_remaining(self, client_id: str) -> int:
        """Get remaining requests for a client."""
        now = time.time()
        minute_ago = now - 60

        recent_requests = [
            req_time for req_time in self.requests[client_id]
            if req_time > minute_ago
        ]
        return max(0, self.requests_per_minute - len(recent_requests))


# Global rate limiter instance
rate_limiter = RateLimiter(requests_per_minute=settings.RATE_LIMIT_PER_MINUTE)


def hash_api_key(key: str) -> str:
    """Hash an API key for comparison."""
    return hashlib.sha256(key.encode()).hexdigest()


async def verify_api_key(
    request: Request,
    api_key: Optional[str] = Security(api_key_header),
) -> Optional[str]:
    """
    Verify the API key if authentication is enabled.

    Returns the client ID (hashed key) if valid, None if auth is disabled.
    Raises HTTPException if auth is enabled but key is invalid.
    """
    # If no API keys configured, allow all requests
    if not settings.API_KEYS:
        return None

    if not api_key:
        logger.warning(f"Missing API key from {get_client_ip(request)}")
        raise HTTPException(
            status_code=401,
            detail="API key required. Include X-API-Key header.",
        )

    # Check against configured keys
    key_hash = hash_api_key(api_key)
    if key_hash not in settings.API_KEY_HASHES:
        logger.warning(f"Invalid API key from {get_client_ip(request)}")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key.",
        )

    return key_hash


async def check_rate_limit(request: Request) -> None:
    """
    Check if the request is within rate limits.

    Raises HTTPException if rate limit exceeded.
    """
    if not settings.RATE_LIMIT_ENABLED:
        return

    client_id = get_client_ip(request)

    if not rate_limiter.is_allowed(client_id):
        remaining = rate_limiter.get_remaining(client_id)
        logger.warning(f"Rate limit exceeded for {client_id}")
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Try again in 60 seconds.",
            headers={"Retry-After": "60"},
        )


def get_client_ip(request: Request) -> str:
    """Get the client IP address from the request."""
    # Check for forwarded header (behind proxy)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()

    # Fall back to direct client
    if request.client:
        return request.client.host

    return "unknown"


def log_request(request: Request, response_status: int, duration_ms: float) -> None:
    """Log request details for monitoring."""
    client_ip = get_client_ip(request)
    method = request.method
    path = request.url.path
    user_agent = request.headers.get("User-Agent", "unknown")[:50]

    logger.info(
        f"{method} {path} - {response_status} - {duration_ms:.1f}ms - {client_ip} - {user_agent}"
    )
