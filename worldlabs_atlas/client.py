"""Plain HTTP client for the World Labs Atlas developer API (beta). No ComfyUI imports.

The API is a set of view-synthesis *tasks*, each an async *operation*::

    atlasTextToImage   prompt -> image (+ depth)
    images2PosedRGBD   image(s) -> the same image(s) with an estimated camera and depth
    atlasGenerate      posed RGBD context + target cameras -> new views (+ depth)
    splats2Mesh        a Gaussian-splat .spz -> a GLB mesh

Auth is the ``WLT-Api-Key`` header. The key comes from the constructor argument, else the
``WLT_API_KEY`` / ``WORLDLABS_API_KEY`` environment variable. It is never logged or stored.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

import requests

from .errors import (ArgsError, AtlasError, AtlasTimeout, AuthError, BillingError, ConcurrencyError,
                     OperationFailed, RateLimitError, TransientError)

DEFAULT_BASE_URL = "https://api.atlas-beta.worldlabs.ai/api/v2"
MAX_INLINE_BYTES = 10 * 1024 * 1024 - 4096     # inline base64 assets must decode to < 10 MiB
WAIT_SLICE_S = 30                              # the server caps :wait at 30 s

TASK_I2P = "images2PosedRGBD"
TASK_T2I = "atlasTextToImage"
TASK_GEN = "atlasGenerate"
TASK_S2M = "splats2Mesh"
ENV_KEYS = ("WLT_API_KEY", "WORLDLABS_API_KEY")


def resolve_api_key(api_key: Optional[str] = None) -> str:
    """Argument first, then the environment. Returns '' if none."""
    k = (api_key or "").strip()
    if k:
        return k
    for name in ENV_KEYS:
        v = (os.environ.get(name) or "").strip()
        if v:
            return v
    return ""


def canonical_hash(*parts) -> str:
    """Stable sha256 of JSON-able parts: the request hash used for caching."""
    blob = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _body_of(r) -> dict:
    try:
        b = r.json()
        return b if isinstance(b, dict) else {"detail": b}
    except Exception:  # noqa: BLE001
        return {"detail": (getattr(r, "text", "") or "")[:300]}


def http_error(r) -> AtlasError:
    """Classify a non-2xx answer (docs: 'Errors and retries')."""
    st, body = r.status_code, _body_of(r)
    code = str(body.get("code") or "")
    detail = body.get("detail")
    msg = detail if isinstance(detail, str) else (json.dumps(detail)[:300] if detail else "")
    retry_after = (getattr(r, "headers", None) or {}).get("Retry-After")
    if st == 401:
        return AuthError("World Labs rejected the API key (missing, unknown, expired or revoked). "
                         f"Set WLT_API_KEY. {msg}".strip())
    if st == 402:
        if code == "SPEND_LIMIT_EXCEEDED":
            return BillingError("The World Labs monthly spend limit is reached: raise it or wait for the "
                                "reset (platform > Settings > Billing).", "spend_limit", code=code)
        return BillingError(f"The World Labs account is out of credits ({code or '402'}): add credits "
                            f"(platform > Settings > Billing). {msg}".strip(), "balance", code=code)
    if st == 403:
        return AuthError("The API key lacks a scope. It needs tasks.create, operations.read, "
                         f"assets.create and assets.read. {msg}".strip(), "scope")
    if st == 404:
        return AtlasError("Not found, or this Atlas task is not enabled for the account (it is a beta "
                          f"entitlement: ask World Labs for access). {msg}".strip(), "not_found")
    if st == 409:
        return AtlasError(f"Idempotency key / id collision. {msg}".strip(), "api")
    if st == 422:
        return ArgsError(f"World Labs rejected the request shape: {msg}")
    if st == 429 and code == "CONCURRENCY_LIMIT_EXCEEDED":
        return ConcurrencyError("World Labs concurrency limit: the project already has the maximum number of "
                                f"Atlas operations running; retry after {retry_after or 'a minute'} s. {msg}".strip(),
                                code=code, retry_after=retry_after)
    if st == 429:
        return RateLimitError(f"World Labs request rate limit ({code or '429'}); retry after "
                              f"{retry_after or 'a minute'} s. {msg}".strip(), code=code, retry_after=retry_after)
    if st == 503:
        return RateLimitError(f"World Labs capacity is temporarily unavailable. {msg}".strip(),
                              retry_after=retry_after)
    if st >= 500:
        return TransientError(f"World Labs server error {st} {code}: {msg}".strip())
    return AtlasError(f"World Labs error {st} {code}: {msg}".strip(), "api")


def operation_error(op: dict) -> AtlasError:
    err = op.get("error") or {}
    reason = ""
    for d in err.get("details") or []:
        if isinstance(d, dict) and d.get("reason"):
            reason = str(d["reason"])
    msg = err.get("message") or "the task failed"
    oid = op.get("id")
    if reason == "CONTENT_POLICY" or err.get("code") == 3:
        return OperationFailed(f"World Labs declined the content under its policy ({msg}). Change the prompt "
                               "or the picture; a retry is declined again.", "content_refusal", operation_id=oid)
    if reason == "EXECUTION_TIMEOUT" or err.get("code") == 4:
        return OperationFailed(f"The Atlas task exceeded its execution deadline ({msg}). A retry is a new task.",
                               "execution_timeout", operation_id=oid)
    return OperationFailed(f"The Atlas task failed ({reason or err.get('code')}): {msg}", "execution",
                           operation_id=oid)


class WorldLabsClient:
    """One API session. ``session`` and ``sleep`` can be replaced (tests do)."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, session=None,
                 sleep: Callable[[float], None] = time.sleep, concurrency_budget_s: float = 90.0,
                 max_retries: int = 3, wait_slice_s: int = WAIT_SLICE_S):
        self._api_key = (api_key or "").strip()
        self.base_url = (base_url or os.environ.get("WLT_API_BASE_URL") or DEFAULT_BASE_URL).strip().rstrip("/")
        self.session = session or requests.Session()
        self.sleep = sleep
        self.concurrency_budget_s = concurrency_budget_s
        self.max_retries = max_retries
        self.wait_slice_s = max(1, min(int(wait_slice_s), WAIT_SLICE_S))

    def __repr__(self) -> str:  # never show the key
        return f"WorldLabsClient(base_url={self.base_url!r}, key={'set' if self.api_key else 'missing'})"

    @property
    def api_key(self) -> str:
        return resolve_api_key(self._api_key)

    # -- raw requests ---------------------------------------------------------

    def request(self, method: str, path: str, *, json_body=None, params=None, idem: str = "",
                timeout: float = 120.0, probe: bool = False) -> dict:
        """One API call with polite retries: 429/503 are retried (Retry-After honoured, capped at 60 s);
        a full concurrency limit is retried for up to ``concurrency_budget_s`` and then raised as
        :class:`ConcurrencyError` carrying the operations that occupy the slots."""
        key = self.api_key
        if not key:
            raise AuthError("No World Labs API key: pass api_key, or set WLT_API_KEY (or WORLDLABS_API_KEY).")
        headers = {"WLT-Api-Key": key, "Content-Type": "application/json"}
        if idem:
            headers["Idempotency-Key"] = idem
        url = f"{self.base_url}{path}"
        attempt, conc_waited = 0, 0.0
        while True:
            try:
                r = self.session.request(method, url, headers=headers, json=json_body, params=params,
                                         timeout=timeout)
            except requests.Timeout as exc:
                raise TransientError("World Labs did not answer in time.", "timeout") from exc
            except requests.RequestException as exc:
                raise TransientError(f"could not reach World Labs: {exc}") from exc
            if r.status_code in (429, 503) and not probe:
                ra = (getattr(r, "headers", None) or {}).get("Retry-After")
                try:
                    wait = min(float(ra), 60.0)
                except (TypeError, ValueError):
                    wait = min(2.0 * (attempt + 1), 15.0)
                if _body_of(r).get("code") == "CONCURRENCY_LIMIT_EXCEEDED":
                    wait = max(wait, 3.0)
                    if conc_waited + wait <= self.concurrency_budget_s:
                        conc_waited += wait
                        self.sleep(wait)
                        continue
                    err = http_error(r)
                    self._concurrency_hint(err)
                    raise err
                if attempt < self.max_retries:
                    attempt += 1
                    self.sleep(wait)
                    continue
            if r.status_code >= 400:
                raise http_error(r)
            return _body_of(r)

    def _concurrency_hint(self, exc: AtlasError) -> None:
        running = self.in_flight()
        exc.extra["in_flight"] = running
        if running:
            ids = ", ".join(f"{o['id']} ({o['task']}, {o['age_s']} s)" for o in running[:5])
            exc.extra["hint"] = (f"{len(running)} operation(s) are still running: {ids}. Wait for them, or "
                                 "cancel one (client.cancel(operation_id)).")
        else:
            exc.extra["hint"] = ("No operation of this project is listed as running, so the slot is held by a "
                                 "task that has not been released yet (or another key/project of the same "
                                 "account). Wait a minute and retry; do not resubmit in a tight loop.")

    # -- operations -----------------------------------------------------------

    def submit(self, task: str, body: dict) -> dict:
        return self.request("POST", f"/tasks:{task}", json_body=body, idem=f"wlatlas-{uuid.uuid4().hex}")

    def get_operation(self, op_id: str) -> dict:
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", str(op_id)):
            raise ArgsError("operation id looks malformed")
        return self.request("GET", f"/operations/{op_id}")

    def list_operations(self, task: str = "", status: str = "", limit: int = 20) -> list:
        params = {"pageSize": max(1, min(int(limit), 100))}
        if task:
            params["task"] = task
        if status:
            params["status"] = status.upper()
        body = self.request("GET", "/operations", params=params)
        return [{"id": o.get("id"), "task": (o.get("metadata") or {}).get("task"), "done": o.get("done"),
                 "error": (o.get("error") or {}).get("message"), "created": o.get("createdAtSecs")}
                for o in body.get("operations") or []]

    def in_flight(self) -> list:
        """The project's RUNNING operations (id, task, age in s), best effort - never raises."""
        try:
            body = self.request("GET", "/operations", params={"pageSize": 20, "status": "RUNNING"}, probe=True)
        except Exception:  # noqa: BLE001
            return []
        now, out = time.time(), []
        for o in body.get("operations") or []:
            if o.get("done"):
                continue
            out.append({"id": o.get("id"), "task": (o.get("metadata") or {}).get("task"),
                        "age_s": int(now - o["createdAtSecs"]) if o.get("createdAtSecs") else None})
        return out

    def cancel(self, op_id: str) -> dict:
        """POST /operations/{id}:cancel (a finished operation just comes back unchanged)."""
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", str(op_id)):
            raise ArgsError("operation id looks malformed")
        return self.request("POST", f"/operations/{op_id}:cancel")

    def wait(self, op: dict, timeout_s: float = 600.0, on_progress: Optional[Callable[[dict, float], None]] = None,
             cancel_on_abort: bool = True) -> dict:
        """Block until the operation is ``done`` (server-side :wait slices, backoff between polls).

        ``on_progress(op, elapsed_s)`` runs after every poll; if it raises, the operation is cancelled
        (``cancel_on_abort``) and the exception propagates - this is how a UI interrupt stops billing.
        Raises :class:`AtlasTimeout` (the operation keeps running) or :class:`OperationFailed`.
        """
        start = time.monotonic()
        deadline = start + float(timeout_s)
        delay = 2.0
        try:
            while not op.get("done"):
                if time.monotonic() >= deadline:
                    raise AtlasTimeout(f"still running after {int(timeout_s)} s; collect it later with the "
                                       "operation id (operations are kept 30 days).", operation_id=op.get("id"))
                left = max(1, min(self.wait_slice_s, int(deadline - time.monotonic())))
                try:
                    op = self.request("POST", f"/operations/{op['id']}:wait", json_body={"timeoutSecs": left})
                except AtlasError as exc:
                    if exc.kind != "not_found":    # a fresh operation can 404 briefly
                        raise
                    self.sleep(delay)
                    delay = min(delay * 1.5, 10.0)
                    continue
                if on_progress:
                    on_progress(op, time.monotonic() - start)
                if not op.get("done"):
                    self.sleep(delay)
                    delay = min(delay * 1.5, 10.0)
        except AtlasError:
            raise
        except BaseException:
            if cancel_on_abort and op.get("id") and not op.get("done"):
                try:
                    self.cancel(op["id"])
                except Exception:  # noqa: BLE001
                    pass
            raise
        if op.get("error"):
            raise operation_error(op)
        return op

    @staticmethod
    def usage_of(op: dict) -> dict:
        """Whatever the operation says about cost. The API documents no per-task usage field (billing is on
        the platform's Usage page), so this only passes through metadata keys that look like it."""
        md = op.get("metadata") or {}
        return {k: v for k, v in md.items() if re.search(r"usage|credit|cost|price|bill", k, re.I)}

    # -- assets ---------------------------------------------------------------

    def upload_bytes(self, data: bytes, name: str, content_type: str = "application/octet-stream") -> str:
        """Upload bytes as a project asset; returns its assetId."""
        created = self.request("POST", "/assets", json_body={"asset": {"displayName": name[:250]}})
        aid = created.get("id")
        if not aid:
            raise AtlasError("asset creation returned no id", "api")
        grant = self.request("POST", f"/assets/{aid}:createUploadUrl",
                             json_body={"contentType": content_type, "contentLength": len(data)})
        hdrs = dict(grant.get("headers") or {})
        hdrs.setdefault("Content-Type", content_type)
        try:
            r = self.session.request(grant.get("method") or "PUT", grant["uploadUrl"], data=data, headers=hdrs,
                                     timeout=600)
        except requests.RequestException as exc:
            raise TransientError(f"upload to World Labs failed: {exc}") from exc
        if r.status_code >= 400:
            raise AtlasError(f"upload to World Labs failed (HTTP {r.status_code})", "api")
        self.request("POST", f"/assets/{aid}:completeUpload", json_body={"sizeBytes": len(data)})
        return aid

    def upload_file(self, path, content_type: str = "application/octet-stream") -> str:
        p = Path(path)
        return self.upload_bytes(p.read_bytes(), p.name, content_type)

    def image_asset(self, ref) -> dict:
        """A TaskAsset for a picture. ``ref``: an https URL, a data: URI, a local path, or ``bytes``.
        Files under 10 MiB go inline as base64; bigger ones are uploaded as a project asset."""
        if isinstance(ref, (bytes, bytearray)):
            data, name = bytes(ref), "image.png"
        else:
            s = str(ref).strip()
            if s.startswith(("http://", "https://")):
                return {"url": s}
            if s.startswith("data:"):
                return {"base64": s}
            p = Path(os.path.expanduser(s.strip('"')))
            if not p.is_file():
                raise ArgsError(f"image not found: {s}")
            data, name = p.read_bytes(), p.name
        if len(data) <= MAX_INLINE_BYTES:
            return {"base64": base64.b64encode(data).decode("ascii")}
        return {"assetId": self.upload_bytes(data, name, "image/png")}

    def download(self, asset: dict, dest) -> Path:
        """Fetch a task output (``{assetId, url}`` or inline base64) to *dest*. The read URL is signed
        (no API key on it); an expired one is replaced through :createReadUrl."""
        dest = Path(dest)
        if not isinstance(asset, dict):
            raise AtlasError("the response named no file where one was expected", "api")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if asset.get("base64"):
            dest.write_bytes(base64.b64decode(asset["base64"].split(",", 1)[-1]))
            return dest
        url = asset.get("url")
        for attempt in (0, 1):
            if not url or attempt == 1:
                if not asset.get("assetId"):
                    raise AtlasError("the response named no downloadable file", "api")
                url = self.request("POST", f"/assets/{asset['assetId']}:createReadUrl").get("readUrl")
            try:
                r = self.session.get(url, timeout=300)
            except requests.RequestException as exc:
                raise TransientError(f"could not download an Atlas output: {exc}") from exc
            if r.status_code < 400:
                dest.write_bytes(r.content)
                return dest
            if attempt == 1:
                raise AtlasError(f"could not download an Atlas output (HTTP {r.status_code})", "api")
        raise AtlasError("download failed", "api")  # pragma: no cover
