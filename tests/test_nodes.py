"""The nodes module must import without ComfyUI/torch, and map widget inputs to the right requests."""

import importlib
import sys
import types

import pytest

from conftest import ROOT
from worldlabs_atlas import AtlasError, comfy_nodes as n


def test_mappings_complete_and_consistent():
    assert set(n.NODE_CLASS_MAPPINGS) == set(n.NODE_DISPLAY_NAME_MAPPINGS)
    assert len(n.NODE_CLASS_MAPPINGS) == 8
    for cls in n.NODE_CLASS_MAPPINGS.values():
        assert cls.CATEGORY == "WorldLabs Atlas"
        it = cls.INPUT_TYPES()
        assert "required" in it
        assert hasattr(cls, cls.FUNCTION)
        assert len(cls.RETURN_TYPES) == len(cls.RETURN_NAMES)


def test_root_init_loads_with_stubbed_comfy(monkeypatch):
    fp = types.ModuleType("folder_paths")
    fp.get_output_directory = lambda: str(ROOT / "_stub_out")
    monkeypatch.setitem(sys.modules, "folder_paths", fp)
    comfy = types.ModuleType("comfy")
    for sub in ("utils", "model_management"):
        m = types.ModuleType(f"comfy.{sub}")
        monkeypatch.setitem(sys.modules, f"comfy.{sub}", m)
        setattr(comfy, sub, m)
    sys.modules["comfy.utils"].ProgressBar = lambda total: types.SimpleNamespace(update_absolute=lambda v: None)
    sys.modules["comfy.model_management"].throw_exception_if_processing_interrupted = lambda: None
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    spec = importlib.util.spec_from_file_location("wl_root_pkg", ROOT / "__init__.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert "WLAtlasRenderView" in mod.NODE_CLASS_MAPPINGS
    nodes = importlib.reload(n)
    assert str(nodes.output_root()).endswith("worldlabs_atlas")
    cb = nodes.progress_callback(60)
    cb({}, 10.0)


def test_render_widgets_become_camera_spec():
    spec = n.WLAtlasRenderView.widget_spec(1, 2, 3, 4, 5, 6, 0)
    assert spec == {"position": [1, 2, 3], "look_at": [4, 5, 6]}
    assert n.WLAtlasRenderView.widget_spec(1, 2, 3, 4, 5, 6, 40)["fov"] == 40
    assert n.WLAtlasExpandWorld.widget_spec(2, 0, 0, 20, 0) == {"forward": 2, "right": 0, "up": 0, "yaw": 20, "pitch": 0}


def test_camera_input_overrides_widgets_and_lists_expand():
    widget = {"forward": 1}
    assert n.camera_specs(None, widget) == [widget]
    c = {"position": [0, 0, 0], "look_at": [0, 0, -1]}
    assert n.camera_specs(c, widget) == [c]
    assert n.camera_specs([c, c], widget) == [c, c]


def test_orbit_node_outputs():
    node = n.WLAtlasCameraOrbit()
    camera, px, py, pz, lx, ly, lz = node.run(0, 0, -3, 2, 0, 0, 0, 1, 40)
    assert camera["look_at"] == [0, 0, -3] and (px, py, pz) == (0, 0, -1) and (lx, ly, lz) == (0, 0, -3)
    camera, *_ = node.run(0, 0, -3, 2, 0, 0, 55, 4, 60)
    assert isinstance(camera, list) and len(camera) == 4 and camera[0]["fov"] == 55


def test_is_changed_hashes_inputs_and_ignore_cache_forces_rerun():
    f = n.WLAtlasRenderView.IS_CHANGED
    a = f({"dir": "x"}, 1, 2, 3, 4, 5, 6, 0, "", False, False, 0, 0, 600, False)
    b = f({"dir": "x"}, 1, 2, 3, 4, 5, 6, 0, "", False, False, 0, 0, 600, False, api_key="changes-nothing")
    c = f({"dir": "x"}, 1.5, 2, 3, 4, 5, 6, 0, "", False, False, 0, 0, 600, False)
    assert a == b and a != c
    forced = f({"dir": "x"}, 1, 2, 3, 4, 5, 6, 0, "", False, False, 0, 0, 600, True)
    assert forced != forced                                   # NaN: ComfyUI always re-runs


def test_guarded_surfaces_hint_and_operation():
    def boom():
        raise AtlasError("busy", "concurrency", hint="wait a minute", operation_id="op_9")
    with pytest.raises(RuntimeError) as e:
        n.guarded(boom)
    msg = str(e.value)
    assert "concurrency" in msg and "wait a minute" in msg and "op_9" in msg


def test_create_node_passes_images_and_flags(monkeypatch, tmp_path, world_dir):
    torch = pytest.importorskip("torch")
    seen = {}

    class FakeAtlas:
        def create_world(self, name, prompt, images, aspect_ratio, mesh_detail, timeout_s, force, on_progress):
            seen.update(name=name, images=images, force=force, aspect=aspect_ratio)
            return {"world_dir": str(world_dir)}
    monkeypatch.setattr(n, "make_atlas", lambda key="", root=None: FakeAtlas())
    img = torch.rand(2, 8, 8, 3)
    world, preview, info = n.WLAtlasCreateWorld().run("nm", "p", "16:9", 4, 60, True, image=img)
    assert len(seen["images"]) == 2 and seen["images"][0][:4] == b"\x89PNG" and seen["force"] is True
    assert world["name"] == "w1" and preview.shape[-1] == 3
