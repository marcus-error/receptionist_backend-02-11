"""
Top-level ASGI CORS middleware.

Wraps the outermost ASGI app (socketio.ASGIApp) so that CORS headers are
injected for *every* HTTP response — including OPTIONS preflight — before
Socket.IO or FastAPI inner middleware has a chance to interfere.

This is the definitive fix for the CORS error observed in production:
  Access to XMLHttpRequest at 'https://api.callzenza.in/...'
  from origin 'https://callzenza.in' has been blocked by CORS policy.
"""
from __future__ import annotations

from typing import Callable


# Origins that are always explicitly allowed (fast path before regex fallback)
_ALLOWED_ORIGINS = {
    "https://callzenza.in",
    "https://www.callzenza.in",
    "https://vr.aminoid.in",
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:5175",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
    "http://127.0.0.1:5175",
    "http://127.0.0.1:3000",
}

_CORS_HEADERS = [
    (b"access-control-allow-methods", b"GET, POST, PUT, DELETE, OPTIONS, PATCH, HEAD"),
    (b"access-control-allow-headers", b"*"),
    (b"access-control-expose-headers", b"*"),
    (b"access-control-max-age", b"86400"),
    (b"access-control-allow-credentials", b"true"),
    (b"vary", b"Origin"),
]


def _is_allowed_origin(origin: str) -> bool:
    """Return True if the request origin should receive CORS headers."""
    if origin in _ALLOWED_ORIGINS:
        return True
    # Permissive fallback: allow any http/https origin
    return origin.startswith("http://") or origin.startswith("https://")


class CorsAsgiMiddleware:
    """
    Minimal ASGI middleware that adds CORS headers to HTTP responses.
    Handles OPTIONS preflight responses directly (200 + CORS headers, no body).
    For all other methods, passes through to the wrapped app and appends headers.
    """

    def __init__(self, app: Callable) -> None:
        self._app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            # WebSocket / lifespan — pass straight through
            await self._app(scope, receive, send)
            return

        # Extract the Origin header
        origin = ""
        for name, value in scope.get("headers", []):
            if name == b"origin":
                origin = value.decode("latin-1", errors="replace")
                break

        if not origin or not _is_allowed_origin(origin):
            # No CORS needed — delegate unchanged
            await self._app(scope, receive, send)
            return

        cors_headers = [(b"access-control-allow-origin", origin.encode())] + _CORS_HEADERS

        if scope.get("method") == "OPTIONS":
            # Respond immediately to preflight — do NOT forward to inner app
            await send({
                "type": "http.response.start",
                "status": 204,
                "headers": cors_headers,
            })
            await send({"type": "http.response.body", "body": b""})
            return

        # For non-OPTIONS requests, wrap the inner `send` to inject CORS headers
        headers_sent = False

        async def send_with_cors(message):
            nonlocal headers_sent
            if message["type"] == "http.response.start" and not headers_sent:
                headers_sent = True
                existing = list(message.get("headers", []))
                existing_names = {h[0].lower() for h in existing}
                extra = [h for h in cors_headers if h[0] not in existing_names]
                message = {**message, "headers": existing + extra}
            await send(message)

        await self._app(scope, receive, send_with_cors)

