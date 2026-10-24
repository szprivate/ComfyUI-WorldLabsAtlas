"""Typed errors. Every one carries a ``kind`` string so callers can branch without isinstance."""

from __future__ import annotations

import json


class AtlasError(Exception):
    """Base class: an API or operation failure, already classified."""

    kind = "api"

    def __init__(self, message: str, kind: str | None = None, **extra):
        super().__init__(message)
        if kind:
            self.kind = kind
        self.extra = {k: v for k, v in extra.items() if v not in (None, "")}

    def to_dict(self) -> dict:
        d = {"error": str(self), "error_kind": self.kind}
        d.update(self.extra)
        return d

    def render(self) -> str:
        return json.dumps(self.to_dict())


class AuthError(AtlasError):
    """No key, or World Labs rejected it (401), or the key lacks a scope (403)."""
    kind = "auth"


class ArgsError(AtlasError):
    """The caller gave something unusable (bad camera, missing file, rejected request shape)."""
    kind = "args"


class BillingError(AtlasError):
    """Out of credits or monthly spend limit reached (402)."""
    kind = "balance"


class RateLimitError(AtlasError):
    """429 / 503 that did not clear after the polite retries. ``retry_after`` is seconds if known."""
    kind = "rate_limit"


class ConcurrencyError(RateLimitError):
    """The project already runs the maximum number of Atlas operations.

    ``extra['in_flight']`` lists what occupies the slots (id, task, age_s) and ``extra['hint']``
    says what to do (wait, or cancel one with ``client.cancel(op_id)``).
    """
    kind = "concurrency"

    @property
    def in_flight(self) -> list:
        return self.extra.get("in_flight") or []


class TransientError(AtlasError):
    """Network trouble, or an answer that may succeed if tried again."""
    kind = "transient"


class OperationFailed(AtlasError):
    """The operation ran and ended in an error (content policy, execution failure ...)."""
    kind = "failed"


class AtlasTimeout(AtlasError):
    """Still running when our timeout ran out. The operation keeps running server-side
    (operations are kept 30 days): ``extra['operation_id']`` collects it later."""
    kind = "timeout"


class DependencyError(AtlasError):
    """An optional Python package is missing (e.g. OpenEXR for depth)."""
    kind = "dependency"
