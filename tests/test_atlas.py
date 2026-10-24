import json
import struct

import pytest

from conftest import CAM, FakeResponse, png_bytes
from worldlabs_atlas import ArgsError, AtlasError, World


class Server:
    """A tiny fake Atlas backend: records submitted bodies, answers operations with a canned response."""

    def __init__(self, response, op_id="op_1"):
        self.response, self.op_id = response, op_id
        self.submitted = []

    def __call__(self, method, path, kw):
        if path.startswith("/tasks:"):
            self.submitted.append((path.split(":", 1)[1], kw["json"]))
            return FakeResponse(200, {"id": self.op_id, "done": False})
        if path.endswith(":wait"):
            return FakeResponse(200, {"id": self.op_id, "done": True, "response": self.response})
        if path.startswith("https://cdn/"):
            return FakeResponse(200, content=png_bytes())
        raise AssertionError(f"unexpected {method} {path}")


def frame(name, with_depth=False):
    fr = {"imageAsset": {"assetId": f"asset_{name}", "url": f"https://cdn/{name}.png"}, "camera": CAM}
    if with_depth:
        fr["depth"] = {"depthAsset": {"assetId": f"dep_{name}", "url": f"https://cdn/{name}_d.png"}}
    return fr


def test_create_world_maps_request_saves_files_and_caches(atlas_factory):
    srv = Server({"frames": [frame("f0")], "promptUsed": "pu"})
    atlas, client = atlas_factory(srv)
    out = atlas.create_world("My World!", "a cabin", [png_bytes()], timeout_s=30)
    task, body = srv.submitted[0]
    assert task == "images2PosedRGBD"
    assert body["targetResolution"] == [1280, 720] and body["frames"][0]["imageAsset"]["base64"]
    assert out["name"] == "My_World" and out["cached"] is False
    w = World.load(out["world_dir"])
    assert (w.dir / "view_000.png").exists() and w.data["prompt_used"] == "pu" and w.data["prompt"] == "a cabin"
    assert "depth map" in out["mesh_error"]                  # no depth in the fake: recorded, not raised
    assert not list(w.dir.glob("pending/*"))

    n = len(client.session.calls)
    again = atlas.create_world("other name", "a cabin", [png_bytes()])
    assert again["cached"] is True and again["world_dir"] == out["world_dir"]
    assert len(client.session.calls) == n                    # not a single HTTP call
    forced = atlas.create_world("My World!", "a cabin", [png_bytes()], force=True)
    assert forced["world_dir"].endswith("My_World_2") and len(srv.submitted) == 2


def test_create_world_text_prompt_request(atlas_factory):
    srv = Server({"frames": [frame("f0", True)]})
    atlas, _ = atlas_factory(srv)
    atlas.create_world("t", prompt="a beach", aspect_ratio="oops")
    task, body = srv.submitted[0]
    assert task == "atlasTextToImage"
    assert body == {"prompt": "a beach", "aspectRatio": "16:9", "returnDepth": True, "numSamples": 1}


def test_create_world_needs_input(atlas_factory):
    atlas, _ = atlas_factory(Server({}))
    with pytest.raises(ArgsError):
        atlas.create_world("x")
    with pytest.raises(ArgsError):
        atlas.create_world("", "p")


def test_failed_job_leaves_no_pending(atlas_factory):
    def h(m, p, kw):
        if p.startswith("/tasks:"):
            return FakeResponse(200, {"id": "op_f", "done": False})
        return FakeResponse(200, {"id": "op_f", "done": True, "error": {"code": 3, "message": "x"}})
    atlas, _ = atlas_factory(h)
    with pytest.raises(AtlasError) as e:
        atlas.create_world("x", "p")
    assert e.value.kind == "content_refusal" and atlas.pending() == []


def test_timeout_leaves_pending_and_resume_finishes(atlas_factory):
    state = {"done": False}

    def h(m, p, kw):
        if p.startswith("/tasks:"):
            return FakeResponse(200, {"id": "op_t", "done": False})
        if p == "/operations/op_t":
            return FakeResponse(200, {"id": "op_t", "done": False})
        if p.endswith(":wait"):
            if state["done"]:
                return FakeResponse(200, {"id": "op_t", "done": True, "response": {"frames": [frame("f0", True)]}})
            return FakeResponse(200, {"id": "op_t", "done": False})
        return FakeResponse(200, content=png_bytes())
    atlas, _ = atlas_factory(h)
    with pytest.raises(AtlasError) as e:
        atlas.create_world("slow", "p", timeout_s=0)
    assert e.value.kind == "timeout" and e.value.extra["operation_id"] == "op_t"
    assert atlas.pending() == ["op_t"]
    state["done"] = True
    out = atlas.resume("op_t", timeout_s=30)
    assert out["status"] == "ok" and atlas.pending() == [] and out["name"] == "slow"


def test_render_view_request_cache_and_files(atlas_factory, world_dir):
    srv = Server({"frames": [{"imageAsset": {"assetId": "a_new", "url": "https://cdn/new.png"}}]}, "op_r")
    atlas, client = atlas_factory(srv)
    cam_spec = {"position": [0.5, 0, 0], "look_at": [0.5, 0, -2], "fov": 50}
    out = atlas.render_view("w1", cam_spec, timeout_s=30)
    task, body = srv.submitted[0]
    assert task == "atlasGenerate"
    assert body["contextFrames"][0]["imageAsset"] == {"assetId": "asset_img0"}
    assert body["contextFrames"][0]["depth"] == {"depthAsset": {"assetId": "asset_dep0"}}
    assert body["targetCameras"][0]["extrinsics"]["position"] == [0.5, 1.0, 0.0]      # base y=1 added
    assert body["prompt"] == "a cabin interior" and body["enhancePrompt"] is False
    assert body["returnDepth"] is False
    assert out["path"].endswith("render_000.png") and out["cached"] is False
    assert out["fov_deg"] == 50

    n = len(client.session.calls)
    again = atlas.render_view("w1", cam_spec)
    assert again["cached"] is True and again["path"] == out["path"] and len(client.session.calls) == n
    other = atlas.render_view("w1", {**cam_spec, "position": [0.6, 0, 0]})
    assert other["path"].endswith("render_001.png") and len(srv.submitted) == 2
    atlas.render_view("w1", cam_spec, force=True)
    assert len(srv.submitted) == 3
    w = World.load(world_dir)
    assert len(w.data["renders"]) == 3


def test_render_user_prompt_enables_enhance(atlas_factory, world_dir):
    srv = Server({"frames": [{"imageAsset": {"assetId": "a", "url": "https://cdn/a.png"}}]})
    atlas, _ = atlas_factory(srv)
    atlas.render_view("w1", {"forward": 1}, prompt="at night")
    body = srv.submitted[0][1]
    assert body["prompt"] == "at night" and body["enhancePrompt"] is True


def test_render_add_to_world_appends_frame(atlas_factory, world_dir):
    resp = {"frames": [{**frame("n", True)}]}
    srv = Server(resp)
    atlas, _ = atlas_factory(srv)
    out = atlas.render_view("w1", {"forward": 1}, add_to_world=True)
    assert srv.submitted[0][1]["returnDepth"] is True
    assert out["added_to_world"] is True
    w = World.load(world_dir)
    assert [f["id"] for f in w.frames] == [0, 1] and w.frames[1]["depth_asset"] == "dep_n"
    assert (world_dir / "renders" / "render_000.png").exists()


def test_render_without_context_or_bad_camera(atlas_factory, world_dir):
    atlas, _ = atlas_factory(Server({}))
    with pytest.raises(ArgsError):
        atlas.render_view("w1", {})
    with pytest.raises(ArgsError):
        atlas.render_view("nope", {"forward": 1})


def test_expand_world_adds_frames_and_caches(atlas_factory, world_dir):
    srv = Server({"frames": [frame("e0", True), frame("e1", True)]}, "op_e")
    atlas, client = atlas_factory(srv)
    out = atlas.expand_world("w1", [{"forward": 1, "yaw": 10}, {"forward": 2, "yaw": -10}])
    body = srv.submitted[0][1]
    assert len(body["targetCameras"]) == 2 and body["returnDepth"] is True
    assert out["added_frames"] == [1, 2]
    n = len(client.session.calls)
    assert atlas.expand_world("w1", [{"forward": 1, "yaw": 10}, {"forward": 2, "yaw": -10}])["cached"] is True
    assert len(client.session.calls) == n
    with pytest.raises(ArgsError):
        atlas.expand_world("w1", [])
    with pytest.raises(ArgsError):
        atlas.expand_world("w1", [{"forward": 1}] * 9)


def test_splat_to_mesh_uploads_and_caches(atlas_factory, tmp_path):
    spz = tmp_path / "a.spz"
    spz.write_bytes(b"splat")
    seen = []

    def h(m, p, kw):
        seen.append(p)
        if p == "/assets":
            return FakeResponse(200, {"id": "asset_s"})
        if p.endswith(":createUploadUrl"):
            return FakeResponse(200, {"uploadUrl": "https://up/s", "method": "PUT"})
        if p == "https://up/s":
            return FakeResponse(200)
        if p.endswith(":completeUpload"):
            return FakeResponse(200, {})
        if p.startswith("/tasks:"):
            assert kw["json"] == {"splatAsset": {"assetId": "asset_s"}, "textureMode": "none", "targetFaces": 500}
            return FakeResponse(200, {"id": "op_s", "done": False})
        if p.endswith(":wait"):
            return FakeResponse(200, {"id": "op_s", "done": True, "response": {"mesh": {"url": "https://cdn/m.glb"}}})
        return FakeResponse(200, content=b"GLB")
    atlas, _ = atlas_factory(h)
    out = atlas.splat_to_mesh(spz, "mesh1", "none", 500)
    assert open(out["glb"], "rb").read() == b"GLB" and out["cached"] is False
    n = len(seen)
    assert atlas.splat_to_mesh(spz, "mesh1", "none", 500)["cached"] is True and len(seen) == n
    with pytest.raises(ArgsError):
        atlas.splat_to_mesh(tmp_path / "x.ply")


def test_cancel_removes_pending(atlas_factory, world_dir):
    (world_dir / "pending").mkdir()
    (world_dir / "pending" / "op_c.json").write_text("{}")
    atlas, _ = atlas_factory(lambda m, p, kw: FakeResponse(200, {"id": "op_c", "done": False}))
    assert atlas.cancel("op_c")["status"] == "cancelled" and atlas.pending() == []


def test_list_and_resolve_forms(atlas_factory, world_dir):
    atlas, _ = atlas_factory(Server({}))
    assert [w["name"] for w in atlas.list_worlds()] == ["w1"]
    for ref in ("w1", str(world_dir), str(world_dir / "world.json"), {"dir": str(world_dir)}):
        assert atlas.resolve(ref).name == "w1"


# -- mesh ------------------------------------------------------------------------------


def make_depth_world(wd):
    from PIL import Image
    import numpy as np
    Image.new("RGB", (32, 18), (10, 200, 30)).save(wd / "view_000.png")
    Image.fromarray((np.full((18, 32), 128, dtype="uint8"))).save(wd / "depth_000.png")
    data = json.loads((wd / "world.json").read_text())
    f = data["frames"][0]
    f["depth"], f["depth_log"] = "depth_000.png", {"zMin": 1.0, "zMax": 4.0}
    (wd / "world.json").write_text(json.dumps(data))


def test_rebuild_mesh_and_ply_from_saved_depth(atlas_factory, world_dir):
    make_depth_world(world_dir)
    atlas, client = atlas_factory(Server({}))
    out = atlas.rebuild_mesh("w1", mesh_detail=2)
    glb = open(out["glb"], "rb").read()
    magic, version, length = struct.unpack("<III", glb[:12])
    assert magic == 0x46546C67 and version == 2 and length == len(glb)
    assert out["triangles"] > 0 and out["vertices"] > 0
    ply = atlas.export_ply("w1", mesh_detail=2)
    raw = open(ply["ply"], "rb").read()
    assert raw.startswith(b"ply\nformat binary_little_endian") and ply["points"] > 0
    assert len(client.session.calls) == 0                    # free: no API calls
    assert World.load(world_dir).data["mesh"]["triangles"] == out["triangles"]
