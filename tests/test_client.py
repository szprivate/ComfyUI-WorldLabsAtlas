import pytest
import requests

from conftest import FakeResponse, make_client
from worldlabs_atlas import (ArgsError, AtlasTimeout, AuthError, BillingError, ConcurrencyError, OperationFailed,
                             RateLimitError, WorldLabsClient, canonical_hash, resolve_api_key)
from worldlabs_atlas.client import MAX_INLINE_BYTES


def test_key_from_argument_then_env(monkeypatch):
    monkeypatch.delenv("WLT_API_KEY", raising=False)
    monkeypatch.delenv("WORLDLABS_API_KEY", raising=False)
    assert resolve_api_key() == ""
    monkeypatch.setenv("WORLDLABS_API_KEY", "envkey")
    assert resolve_api_key() == "envkey"
    monkeypatch.setenv("WLT_API_KEY", "wlt")
    assert resolve_api_key() == "wlt"
    assert resolve_api_key("arg") == "arg"
    assert "arg" not in repr(WorldLabsClient(api_key="arg"))


def test_missing_key_raises_auth(monkeypatch):
    monkeypatch.delenv("WLT_API_KEY", raising=False)
    monkeypatch.delenv("WORLDLABS_API_KEY", raising=False)
    c = WorldLabsClient(session=object())
    with pytest.raises(AuthError):
        c.request("GET", "/operations")


def test_auth_header_and_idempotency():
    seen = {}

    def h(m, p, kw):
        seen.update(kw["headers"])
        return FakeResponse(200, {"id": "op1"})
    c = make_client(h)
    c.submit("atlasGenerate", {"a": 1})
    assert seen["WLT-Api-Key"] == "test-key"
    assert seen["Idempotency-Key"].startswith("wlatlas-")


@pytest.mark.parametrize("status,body,exc,kind", [
    (401, {}, AuthError, "auth"),
    (403, {}, AuthError, "scope"),
    (402, {"code": "SPEND_LIMIT_EXCEEDED"}, BillingError, "spend_limit"),
    (402, {"code": "NO_CREDITS"}, BillingError, "balance"),
    (422, {"detail": "bad"}, ArgsError, "args"),
])
def test_http_errors_classified(status, body, exc, kind):
    c = make_client(lambda m, p, kw: FakeResponse(status, body))
    with pytest.raises(exc) as e:
        c.request("GET", "/x")
    assert e.value.kind == kind


def test_rate_limit_retried_with_retry_after():
    n = {"i": 0}

    def h(m, p, kw):
        n["i"] += 1
        return FakeResponse(429, {"code": "RATE"}, {"Retry-After": "7"}) if n["i"] < 3 else FakeResponse(200, {"ok": 1})
    c = make_client(h)
    assert c.request("GET", "/x") == {"ok": 1}
    assert c.sleeps == [7.0, 7.0]


def test_rate_limit_gives_up():
    c = make_client(lambda m, p, kw: FakeResponse(429, {"code": "RATE"}), max_retries=2)
    with pytest.raises(RateLimitError):
        c.request("GET", "/x")
    assert len(c.sleeps) == 2


def test_concurrency_retries_then_reports_in_flight():
    def h(m, p, kw):
        if p == "/operations":
            return FakeResponse(200, {"operations": [{"id": "op_busy", "done": False, "createdAtSecs": 1,
                                                      "metadata": {"task": "atlasGenerate"}}]})
        return FakeResponse(429, {"code": "CONCURRENCY_LIMIT_EXCEEDED"}, {"Retry-After": "5"})
    c = make_client(h, concurrency_budget_s=12)
    with pytest.raises(ConcurrencyError) as e:
        c.submit("atlasGenerate", {})
    assert c.sleeps == [5.0, 5.0]                       # 5+5 <= 12, the third would exceed the budget
    assert e.value.in_flight[0]["id"] == "op_busy"
    assert "op_busy" in e.value.extra["hint"]
    assert e.value.kind == "concurrency" and isinstance(e.value, RateLimitError)


def test_concurrency_without_running_ops_hint():
    def h(m, p, kw):
        if p == "/operations":
            return FakeResponse(200, {"operations": []})
        return FakeResponse(429, {"code": "CONCURRENCY_LIMIT_EXCEEDED"})
    c = make_client(h, concurrency_budget_s=0)
    with pytest.raises(ConcurrencyError) as e:
        c.submit("t", {})
    assert e.value.in_flight == [] and "slot is held" in e.value.extra["hint"]


def test_network_error_is_transient():
    def h(m, p, kw):
        raise requests.ConnectionError("down")
    c = make_client(h)
    with pytest.raises(Exception) as e:
        c.request("GET", "/x")
    assert e.value.kind == "transient"


def test_wait_polls_until_done():
    states = [{"id": "op1", "done": False}, {"id": "op1", "done": False}, {"id": "op1", "done": True, "response": {"r": 1}}]

    def h(m, p, kw):
        assert p == "/operations/op1:wait"
        return FakeResponse(200, states.pop(0))
    c = make_client(h)
    seen = []
    op = c.wait({"id": "op1"}, 600, lambda o, t: seen.append(o["done"]))
    assert op["response"] == {"r": 1} and seen == [False, False, True]
    assert len(c.sleeps) == 2 and c.sleeps[1] > c.sleeps[0]       # backoff


def test_wait_retries_fresh_404():
    seq = [FakeResponse(404, {}), FakeResponse(200, {"id": "op1", "done": True})]
    c = make_client(lambda m, p, kw: seq.pop(0))
    assert c.wait({"id": "op1"})["done"]


def test_wait_timeout_keeps_operation_id():
    c = make_client(lambda m, p, kw: FakeResponse(200, {"id": "op1", "done": False}))
    with pytest.raises(AtlasTimeout) as e:
        c.wait({"id": "op1"}, timeout_s=0)
    assert e.value.extra["operation_id"] == "op1"


def test_wait_failed_operation_content_policy():
    op = {"id": "op1", "done": True, "error": {"code": 3, "message": "no", "details": [{"reason": "CONTENT_POLICY"}]}}
    c = make_client(lambda m, p, kw: FakeResponse(200, op))
    with pytest.raises(OperationFailed) as e:
        c.wait({"id": "op1"})
    assert e.value.kind == "content_refusal"


def test_wait_abort_cancels_operation():
    calls = []

    def h(m, p, kw):
        calls.append(p)
        return FakeResponse(200, {"id": "op1", "done": False})
    c = make_client(h)

    def boom(op, t):
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        c.wait({"id": "op1"}, 600, boom)
    assert "/operations/op1:cancel" in calls


def test_cancel_and_malformed_id():
    c = make_client(lambda m, p, kw: FakeResponse(200, {"id": "op1", "done": True}))
    assert c.cancel("op1")["done"]
    with pytest.raises(ArgsError):
        c.cancel("../x")


def test_download_replaces_expired_url(tmp_path):
    def h(m, p, kw):
        if p == "/assets/a1:createReadUrl":
            return FakeResponse(200, {"readUrl": "https://cdn/new"})
        if p == "https://cdn/old":
            return FakeResponse(403)
        if p == "https://cdn/new":
            return FakeResponse(200, content=b"DATA")
        raise AssertionError(p)
    c = make_client(h)
    out = c.download({"assetId": "a1", "url": "https://cdn/old"}, tmp_path / "x" / "f.bin")
    assert out.read_bytes() == b"DATA"


def test_download_inline_base64(tmp_path):
    import base64
    c = make_client(lambda m, p, kw: None)
    c.download({"base64": base64.b64encode(b"hi").decode()}, tmp_path / "f")
    assert (tmp_path / "f").read_bytes() == b"hi"


def test_image_asset_forms_and_large_upload(tmp_path):
    c = make_client(lambda m, p, kw: None)
    assert c.image_asset("https://x/y.png") == {"url": "https://x/y.png"}
    assert c.image_asset(b"abc")["base64"] == "YWJj"
    with pytest.raises(ArgsError):
        c.image_asset(str(tmp_path / "missing.png"))

    def h(m, p, kw):
        if p == "/assets":
            return FakeResponse(200, {"id": "asset_big"})
        if p.endswith(":createUploadUrl"):
            return FakeResponse(200, {"uploadUrl": "https://up/x", "method": "PUT", "headers": {}})
        if p == "https://up/x":
            return FakeResponse(200)
        if p.endswith(":completeUpload"):
            return FakeResponse(200, {})
        raise AssertionError(p)
    c2 = make_client(h)
    assert c2.image_asset(b"x" * (MAX_INLINE_BYTES + 1)) == {"assetId": "asset_big"}


def test_canonical_hash_is_stable_and_order_independent():
    assert canonical_hash("a", {"x": 1, "y": 2}) == canonical_hash("a", {"y": 2, "x": 1})
    assert canonical_hash("a", {"x": 1}) != canonical_hash("a", {"x": 2})
