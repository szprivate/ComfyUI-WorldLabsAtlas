import io
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worldlabs_atlas import Atlas, WorldLabsClient  # noqa: E402

BASE = "https://api.test/api/v2"


class FakeResponse:
    def __init__(self, status=200, body=None, headers=None, content=b""):
        self.status_code = status
        self._body = body if body is not None else {}
        self.headers = headers or {}
        self.content = content
        self.text = json.dumps(self._body)

    def json(self):
        return self._body


class FakeSession:
    """handler(method, path, kwargs) -> FakeResponse. Records every call; ``path`` is relative to BASE."""

    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    def request(self, method, url, **kw):
        path = url[len(BASE):] if url.startswith(BASE) else url
        self.calls.append((method, path, kw))
        return self.handler(method, path, kw)

    def get(self, url, **kw):
        self.calls.append(("GET", url, kw))
        return self.handler("GET", url, kw)

    def paths(self):
        return [f"{m} {p}" for m, p, _ in self.calls]


def make_client(handler, **kw):
    sleeps = []
    c = WorldLabsClient(api_key="test-key", base_url=BASE, session=FakeSession(handler),
                        sleep=sleeps.append, **kw)
    c.sleeps = sleeps
    return c


def png_bytes(w=8, h=4, color=(200, 100, 50)):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


CAM = {"extrinsics": {"position": [0.0, 1.0, 0.0], "quaternion": [0.0, 0.0, 0.0, 1.0], "coordinateSystem": "rub"},
       "intrinsics": {"fx": 700.0, "fy": 700.0, "cx": 640.0, "cy": 360.0, "width": 1280, "height": 720}}


@pytest.fixture
def world_dir(tmp_path):
    """A saved world with one posed frame that has stored depth/image assets (no mesh files)."""
    wd = tmp_path / "root" / "w1"
    wd.mkdir(parents=True)
    (wd / "view_000.png").write_bytes(png_bytes())
    data = {"version": 1, "name": "w1", "prompt": "a cabin", "prompt_used": "a cabin interior", "frames": [
        {"id": 0, "image": "view_000.png", "source": "posed", "operation": "op0", "camera": CAM,
         "image_asset": "asset_img0", "depth": "depth_000.exr", "depth_asset": "asset_dep0"}], "operations": []}
    (wd / "world.json").write_text(json.dumps(data), encoding="utf-8")
    return wd


@pytest.fixture
def atlas_factory(tmp_path):
    def make(handler):
        client = make_client(handler)
        return Atlas(client, tmp_path / "root"), client
    return make
