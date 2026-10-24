"""ComfyUI nodes (V1 style) on top of the plain client. Category: "WorldLabs Atlas".

Custom types: ``WORLD`` is a small dict ``{"name", "dir"}`` that points at a world folder on disk;
``CAMERA`` is a camera spec dict (or a list of them, for sweeps) in the world's mesh frame.

torch / folder_paths / comfy are only needed when a node runs, so this module imports without them
(the tests use that).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Optional

from . import camera as cam
from .atlas import MAX_VIEWS, Atlas, World
from .client import WorldLabsClient, canonical_hash
from .errors import AtlasError

try:  # inside ComfyUI
    import folder_paths  # type: ignore
except Exception:  # noqa: BLE001
    folder_paths = None
try:
    import comfy.model_management as _mm  # type: ignore
    import comfy.utils as _cu  # type: ignore
except Exception:  # noqa: BLE001
    _mm = _cu = None

CATEGORY = "WorldLabs Atlas"
POLL_SLICE_S = 5          # poll slice in nodes: an interrupt in the UI is noticed within this


# -- plumbing --------------------------------------------------------------------


def output_root() -> Path:
    env = os.environ.get("WORLDLABS_ATLAS_DIR")
    if env:
        return Path(env).expanduser()
    base = folder_paths.get_output_directory() if folder_paths else str(Path.cwd() / "output")
    return Path(base) / "worldlabs_atlas"


def make_atlas(api_key: str = "", root=None) -> Atlas:
    client = WorldLabsClient(api_key=(api_key or "").strip() or None, wait_slice_s=POLL_SLICE_S)
    return Atlas(client, root or output_root())


def progress_callback(timeout_s: float):
    """A ``on_progress`` for ``client.wait``: moves a ProgressBar and aborts (cancelling the operation)
    when the user interrupts the queue."""
    pbar = _cu.ProgressBar(100) if _cu else None

    def on_progress(op, elapsed):
        if _mm:
            _mm.throw_exception_if_processing_interrupted()
        if pbar:
            pbar.update_absolute(min(95, int(100 * (1 - math.exp(-elapsed / 45.0)))))
    return on_progress


def guarded(fn):
    """Run ``fn``; turn an AtlasError into a RuntimeError ComfyUI shows in the UI."""
    try:
        return fn()
    except AtlasError as exc:
        msg = f"World Labs Atlas [{exc.kind}]: {exc}"
        if exc.extra.get("hint"):
            msg += f"\n{exc.extra['hint']}"
        if exc.extra.get("operation_id"):
            msg += f"\n(operation {exc.extra['operation_id']})"
        raise RuntimeError(msg) from exc


def tensor_to_png_bytes(image, index: int = 0) -> bytes:
    import io
    import numpy as np
    from PIL import Image
    arr = image[index].detach().cpu().numpy() if hasattr(image[index], "detach") else np.asarray(image[index])
    arr = (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr[..., :3]).save(buf, format="PNG")
    return buf.getvalue()


def files_to_tensor(paths, size=(0, 0)):
    """PNG/JPG files -> an IMAGE batch tensor [B, H, W, 3] (all resized to the first / to ``size``)."""
    import numpy as np
    import torch
    from PIL import Image
    arrs, target = [], None
    for p in paths:
        im = Image.open(p).convert("RGB")
        if size and size[0] and size[1]:
            target = (int(size[0]), int(size[1]))
        elif target is None:
            target = im.size
        if im.size != target:
            im = im.resize(target, Image.LANCZOS)
        arrs.append(np.asarray(im).astype("float32") / 255.0)
    return torch.from_numpy(np.stack(arrs))


def depth_file_to_tensor(path):
    """A depth EXR/PNG -> a normalised grey IMAGE [1, H, W, 3] (near = white)."""
    import numpy as np
    import torch
    from . import mesh
    d = mesh.read_exr(Path(path)) if str(path).lower().endswith(".exr") else mesh.read_png_gray(Path(path))
    ok = np.isfinite(d) & (d > 0)
    lo, hi = (float(d[ok].min()), float(d[ok].max())) if ok.any() else (0.0, 1.0)
    n = 1.0 - np.where(ok, (d - lo) / max(hi - lo, 1e-6), 1.0)
    return torch.from_numpy(np.repeat(n[None, ..., None], 3, axis=-1).astype("float32"))


def blank_image(w=64, h=64):
    import torch
    return torch.zeros((1, h, w, 3), dtype=torch.float32)


def camera_specs(camera, widget_spec: Optional[dict]) -> list:
    """The camera specs a node should render: the CAMERA input (dict or list) if connected, else the widgets."""
    if camera:
        specs = list(camera) if isinstance(camera, (list, tuple)) else [camera]
    else:
        specs = [widget_spec]
    specs = [s for s in specs if s]
    if not specs:
        raise RuntimeError("World Labs Atlas: no camera given")
    return specs


def world_handle(w: World) -> dict:
    return w.handle()


def preview_image(world: World):
    p = world.first_image()
    return files_to_tensor([p]) if p else blank_image()


def _hash(*parts) -> str:
    return canonical_hash(*parts)


def _always_new():
    return float("nan")


# -- nodes -------------------------------------------------------------------------


class WLAtlasApiKey:
    """Optional. Passes a key to other nodes; leave it out and they use WLT_API_KEY / WORLDLABS_API_KEY."""

    CATEGORY = CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("api_key",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"api_key": ("STRING", {"default": "", "multiline": False,
                                                    "tooltip": "A key typed here is saved INSIDE the workflow "
                                                               "file. Prefer the WLT_API_KEY environment variable."})}}

    def run(self, api_key):
        return (api_key.strip(),)

    @classmethod
    def IS_CHANGED(cls, api_key):
        return _hash("key", api_key)


_KEY_OPT = ("STRING", {"forceInput": True, "tooltip": "From 'Atlas API Key'. Empty: environment variable."})


class WLAtlasCreateWorld:
    CATEGORY = CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("WORLD", "IMAGE", "STRING")
    RETURN_NAMES = ("world", "preview", "info")
    DESCRIPTION = ("Make an Atlas world from a picture and/or a text prompt. Billed to your World Labs credits; "
                   "an identical request is served from disk.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "name": ("STRING", {"default": "my_world"}),
                    "prompt": ("STRING", {"default": "", "multiline": True,
                                          "tooltip": "With no image: the place to imagine. With an image: a scene description for later renders."}),
                    "aspect_ratio": (["16:9", "9:16", "4:3", "3:4", "1:1"], {"default": "16:9"}),
                    "mesh_detail": ("INT", {"default": 4, "min": 1, "max": 16, "tooltip": "Depth-grid stride in pixels: 2 fine, 4 default, 8 light."}),
                    "timeout_s": ("INT", {"default": 600, "min": 30, "max": 7200}),
                    "ignore_cache": ("BOOLEAN", {"default": False, "tooltip": "Run (and bill) again even if the same request was already made."}),
                },
                "optional": {"image": ("IMAGE",), "api_key": _KEY_OPT}}

    @classmethod
    def IS_CHANGED(cls, name, prompt, aspect_ratio, mesh_detail, timeout_s, ignore_cache, image=None, api_key=""):
        if ignore_cache:
            return _always_new()
        img = _hash(tensor_to_png_bytes(image).hex()) if image is not None else ""
        return _hash("create", name, prompt, aspect_ratio, mesh_detail, img)

    def run(self, name, prompt, aspect_ratio, mesh_detail, timeout_s, ignore_cache, image=None, api_key=""):
        atlas = make_atlas(api_key)
        images = [tensor_to_png_bytes(image, i) for i in range(len(image))] if image is not None else []
        out = guarded(lambda: atlas.create_world(name, prompt, images, aspect_ratio, mesh_detail, timeout_s,
                                                 bool(ignore_cache), progress_callback(timeout_s)))
        w = World.load(out["world_dir"])
        return (world_handle(w), preview_image(w), json.dumps(out, indent=1))


class WLAtlasLoadWorld:
    CATEGORY = CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("WORLD", "IMAGE", "STRING")
    RETURN_NAMES = ("world", "preview", "info")
    DESCRIPTION = "Load a world made earlier: a folder name under output/worldlabs_atlas, or a full folder path."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"world": ("STRING", {"default": "", "tooltip": "Folder name or path of a world (contains world.json)."})}}

    @classmethod
    def IS_CHANGED(cls, world):
        try:
            f = make_atlas().resolve(world).dir / "world.json"
            return _hash("load", str(f), f.stat().st_mtime)
        except Exception:  # noqa: BLE001
            return _always_new()

    def run(self, world):
        atlas = make_atlas()
        w = guarded(lambda: atlas.resolve(world))
        info = {"name": w.name, "dir": str(w.dir), "frames": len(w.frames), "glb": str(w.glb) if w.glb else "",
                "renders": len(w.data.get("renders") or [])}
        return (world_handle(w), preview_image(w), json.dumps(info, indent=1))


class WLAtlasCameraOrbit:
    CATEGORY = CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("CAMERA", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("camera", "pos_x", "pos_y", "pos_z", "look_x", "look_y", "look_z")
    DESCRIPTION = ("Cameras on a circle around a centre, looking at it. count > 1 makes a sweep from yaw to "
                   "yaw_end; Render View renders every camera (one billed task each).")

    @classmethod
    def INPUT_TYPES(cls):
        f = lambda d, lo=-1000.0, hi=1000.0, step=0.05: ("FLOAT", {"default": d, "min": lo, "max": hi, "step": step})
        return {"required": {
            "center_x": f(0.0), "center_y": f(0.0), "center_z": f(-3.0),
            "distance": f(2.0, 0.01, 1000.0),
            "yaw": f(0.0, -360.0, 360.0, 1.0), "pitch": f(0.0, -85.0, 85.0, 1.0),
            "fov": f(0.0, 0.0, 120.0, 1.0),
            "count": ("INT", {"default": 1, "min": 1, "max": MAX_VIEWS}),
            "yaw_end": f(40.0, -360.0, 360.0, 1.0)}}

    def run(self, center_x, center_y, center_z, distance, yaw, pitch, fov, count, yaw_end):
        specs = cam.orbit_sweep([center_x, center_y, center_z], distance, yaw, yaw_end, count, pitch, fov)
        first = specs[0]
        return (specs if len(specs) > 1 else specs[0], *first["position"], *first["look_at"])


class WLAtlasRenderView:
    CATEGORY = CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "IMAGE", "WORLD", "STRING")
    RETURN_NAMES = ("image", "depth", "world", "paths")
    DESCRIPTION = ("Ask Atlas for a new 1280x720 picture of the world from a camera. One billed task per camera; "
                   "identical requests are served from disk. Atlas gives no alpha/coverage, so there is no MASK.")

    @classmethod
    def INPUT_TYPES(cls):
        f = lambda d, lo=-1000.0, hi=1000.0, step=0.05: ("FLOAT", {"default": d, "min": lo, "max": hi, "step": step})
        return {"required": {
                    "world": ("WORLD",),
                    "pos_x": f(0.5), "pos_y": f(0.0), "pos_z": f(0.0),
                    "look_x": f(0.5), "look_y": f(0.0), "look_z": f(-1.0),
                    "fov": f(0.0, 0.0, 120.0, 1.0),
                    "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Empty: reuse the world's own description (cheaper, consistent)."}),
                    "add_to_world": ("BOOLEAN", {"default": False, "tooltip": "Also add the view (with depth) to the world and rebuild world.glb."}),
                    "return_depth": ("BOOLEAN", {"default": False}),
                    "output_width": ("INT", {"default": 0, "min": 0, "max": 8192, "tooltip": "0 = native 1280"}),
                    "output_height": ("INT", {"default": 0, "min": 0, "max": 8192, "tooltip": "0 = native 720"}),
                    "timeout_s": ("INT", {"default": 600, "min": 30, "max": 7200}),
                    "ignore_cache": ("BOOLEAN", {"default": False}),
                },
                "optional": {"camera": ("CAMERA", {"tooltip": "Overrides the position / look_at widgets. A list renders each."}),
                             "api_key": _KEY_OPT}}

    @classmethod
    def IS_CHANGED(cls, world, pos_x, pos_y, pos_z, look_x, look_y, look_z, fov, prompt, add_to_world, return_depth,
                   output_width, output_height, timeout_s, ignore_cache, camera=None, api_key=""):
        if ignore_cache:
            return _always_new()
        return _hash("render", world, camera, [pos_x, pos_y, pos_z, look_x, look_y, look_z], fov, prompt,
                     add_to_world, return_depth, output_width, output_height)

    @staticmethod
    def widget_spec(pos_x, pos_y, pos_z, look_x, look_y, look_z, fov):
        spec = {"position": [pos_x, pos_y, pos_z], "look_at": [look_x, look_y, look_z]}
        if fov:
            spec["fov"] = fov
        return spec

    def run(self, world, pos_x, pos_y, pos_z, look_x, look_y, look_z, fov, prompt, add_to_world, return_depth,
            output_width, output_height, timeout_s, ignore_cache, camera=None, api_key=""):
        atlas = make_atlas(api_key)
        specs = camera_specs(camera, self.widget_spec(pos_x, pos_y, pos_z, look_x, look_y, look_z, fov))
        if len(specs) > MAX_VIEWS:
            raise RuntimeError(f"World Labs Atlas: at most {MAX_VIEWS} cameras per Render View node")
        results = []
        for spec in specs:
            results.append(guarded(lambda s=spec: atlas.render_view(
                world, s, fov=0.0, prompt=prompt, add_to_world=bool(add_to_world), depth=bool(return_depth),
                timeout_s=timeout_s, force=bool(ignore_cache), on_progress=progress_callback(timeout_s))))
        images = files_to_tensor([r["path"] for r in results], (output_width, output_height))
        depth_paths = [r["depth_path"] for r in results if r.get("depth_path")]
        w = atlas.resolve(world)
        depth = depth_file_to_tensor(depth_paths[0]) if depth_paths else blank_image()
        return (images, depth, world_handle(w), json.dumps([r["path"] for r in results]))


class WLAtlasExpandWorld:
    CATEGORY = CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("WORLD", "IMAGE", "STRING")
    RETURN_NAMES = ("world", "new_views", "info")
    DESCRIPTION = ("Grow the world: Atlas invents new views at new cameras and they are added to the mesh. "
                   "Needs camera movement (a pure turn gives no depth). 1-8 views, billed per view batch.")

    @classmethod
    def INPUT_TYPES(cls):
        f = lambda d, lo=-100.0, hi=100.0, step=0.1: ("FLOAT", {"default": d, "min": lo, "max": hi, "step": step})
        return {"required": {
                    "world": ("WORLD",),
                    "forward": f(2.0), "right": f(0.0), "up": f(0.0),
                    "yaw": f(20.0, -180.0, 180.0, 1.0), "pitch": f(0.0, -85.0, 85.0, 1.0),
                    "fov": f(0.0, 0.0, 120.0, 1.0),
                    "prompt": ("STRING", {"default": "", "multiline": True}),
                    "mesh_detail": ("INT", {"default": 4, "min": 1, "max": 16}),
                    "timeout_s": ("INT", {"default": 600, "min": 30, "max": 7200}),
                    "ignore_cache": ("BOOLEAN", {"default": False}),
                },
                "optional": {"camera": ("CAMERA", {"tooltip": "Overrides the move widgets. A list adds each view (max 8)."}),
                             "api_key": _KEY_OPT}}

    @classmethod
    def IS_CHANGED(cls, world, forward, right, up, yaw, pitch, fov, prompt, mesh_detail, timeout_s, ignore_cache,
                   camera=None, api_key=""):
        if ignore_cache:
            return _always_new()
        return _hash("expand", world, camera, [forward, right, up, yaw, pitch], fov, prompt, mesh_detail)

    @staticmethod
    def widget_spec(forward, right, up, yaw, pitch):
        return {"forward": forward, "right": right, "up": up, "yaw": yaw, "pitch": pitch}

    def run(self, world, forward, right, up, yaw, pitch, fov, prompt, mesh_detail, timeout_s, ignore_cache,
            camera=None, api_key=""):
        atlas = make_atlas(api_key)
        specs = camera_specs(camera, self.widget_spec(forward, right, up, yaw, pitch))
        out = guarded(lambda: atlas.expand_world(world, specs, fov=fov, prompt=prompt, mesh_detail=mesh_detail,
                                                 timeout_s=timeout_s, force=bool(ignore_cache),
                                                 on_progress=progress_callback(timeout_s)))
        w = atlas.resolve(world)
        ids = set(out.get("added_frames") or [])
        paths = [w.dir / f["image"] for f in w.frames if f["id"] in ids]
        new = files_to_tensor(paths) if paths else blank_image()
        return (world_handle(w), new, json.dumps(out, indent=1))


class WLAtlasExportWorld:
    CATEGORY = CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("WORLD", "STRING", "STRING")
    RETURN_NAMES = ("world", "glb_path", "ply_path")
    OUTPUT_NODE = True
    DESCRIPTION = ("Write the world's mesh (GLB, textured) and optionally a coloured point cloud (PLY) from its "
                   "saved depth maps, under output/worldlabs_atlas/<name>/. No API call, no cost.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "world": ("WORLD",),
            "rebuild_mesh": ("BOOLEAN", {"default": False, "tooltip": "Rebuild world.glb with the settings below."}),
            "export_ply": ("BOOLEAN", {"default": True}),
            "mesh_detail": ("INT", {"default": 4, "min": 1, "max": 16}),
            "max_jump": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 10.0, "step": 0.01}),
            "conf_min": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
            "max_depth": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.5,
                                    "tooltip": "Drop everything farther than this (0 = keep all); cuts a sky."})}}

    @classmethod
    def IS_CHANGED(cls, world, rebuild_mesh, export_ply, mesh_detail, max_jump, conf_min, max_depth):
        return _hash("export", world, rebuild_mesh, export_ply, mesh_detail, max_jump, conf_min, max_depth)

    def run(self, world, rebuild_mesh, export_ply, mesh_detail, max_jump, conf_min, max_depth):
        atlas = make_atlas()
        w = guarded(lambda: atlas.resolve(world))
        if rebuild_mesh or not (w.glb and w.glb.exists()):
            guarded(lambda: atlas.rebuild_mesh(w, mesh_detail, max_jump, conf_min, max_depth))
            w = atlas.resolve(world)
        ply = ""
        if export_ply:
            ply = guarded(lambda: atlas.export_ply(w, mesh_detail, conf_min, max_depth))["ply"]
        return {"ui": {"text": [str(w.glb)]}, "result": (world_handle(w), str(w.glb) if w.glb else "", ply)}


class WLAtlasSplatToMesh:
    CATEGORY = CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("glb_path",)
    OUTPUT_NODE = True
    DESCRIPTION = "Gaussian-splat .spz (e.g. a Marble export) -> GLB mesh with Atlas splats2Mesh. Billed."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "spz_path": ("STRING", {"default": ""}),
                    "name": ("STRING", {"default": ""}),
                    "texture_mode": (["image_texture", "vertex_color", "none"], {"default": "image_texture"}),
                    "target_faces": ("INT", {"default": 0, "min": 0, "max": 100000000}),
                    "timeout_s": ("INT", {"default": 900, "min": 30, "max": 7200}),
                    "ignore_cache": ("BOOLEAN", {"default": False})},
                "optional": {"api_key": _KEY_OPT}}

    @classmethod
    def IS_CHANGED(cls, spz_path, name, texture_mode, target_faces, timeout_s, ignore_cache, api_key=""):
        if ignore_cache:
            return _always_new()
        try:
            mt = os.path.getmtime(spz_path)
        except OSError:
            mt = 0
        return _hash("splat", spz_path, mt, name, texture_mode, target_faces)

    def run(self, spz_path, name, texture_mode, target_faces, timeout_s, ignore_cache, api_key=""):
        atlas = make_atlas(api_key)
        out = guarded(lambda: atlas.splat_to_mesh(spz_path, name, texture_mode, target_faces, timeout_s,
                                                  bool(ignore_cache), progress_callback(timeout_s)))
        return {"ui": {"text": [out["glb"]]}, "result": (out["glb"],)}


NODE_CLASS_MAPPINGS = {
    "WLAtlasApiKey": WLAtlasApiKey,
    "WLAtlasCreateWorld": WLAtlasCreateWorld,
    "WLAtlasLoadWorld": WLAtlasLoadWorld,
    "WLAtlasCameraOrbit": WLAtlasCameraOrbit,
    "WLAtlasRenderView": WLAtlasRenderView,
    "WLAtlasExpandWorld": WLAtlasExpandWorld,
    "WLAtlasExportWorld": WLAtlasExportWorld,
    "WLAtlasSplatToMesh": WLAtlasSplatToMesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WLAtlasApiKey": "Atlas API Key",
    "WLAtlasCreateWorld": "Atlas Create World",
    "WLAtlasLoadWorld": "Atlas Load World",
    "WLAtlasCameraOrbit": "Atlas Camera from Orbit",
    "WLAtlasRenderView": "Atlas Render View",
    "WLAtlasExpandWorld": "Atlas Expand World",
    "WLAtlasExportWorld": "Atlas Export World (GLB / PLY)",
    "WLAtlasSplatToMesh": "Atlas Splat to Mesh (.spz)",
}
