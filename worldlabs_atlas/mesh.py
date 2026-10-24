"""Posed RGBD views -> a textured depth mesh (GLB) or a coloured point cloud (PLY).

Needs ``numpy`` and ``Pillow`` (both ship with ComfyUI) and, to read Atlas depth maps, either the
``OpenEXR`` package or an OpenCV build with EXR support. Imports are lazy so the rest of the package
works without them.
"""

from __future__ import annotations

import json
import math
import os
import struct
from pathlib import Path
from typing import Optional

from .camera import normalize_camera
from .errors import AtlasError, DependencyError


def read_exr(path: Path):
    """float32 [H, W] (first channel) of an EXR. Row 0 is the top row."""
    import numpy as np
    try:
        import OpenEXR  # type: ignore
        f = OpenEXR.File(str(path))
        chans = f.channels()
        key = "Y" if "Y" in chans else "R" if "R" in chans else sorted(chans)[0]
        return np.asarray(chans[key].pixels, dtype=np.float32)
    except ImportError:
        pass
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    try:
        import cv2  # type: ignore
        a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if a is not None:
            return (a[..., 0] if a.ndim == 3 else a).astype("float32")
    except ImportError:
        pass
    raise DependencyError("cannot read the depth EXR: install the OpenEXR package "
                          "(pip install OpenEXR) into ComfyUI's Python environment.")


def read_png_gray(path: Path):
    import numpy as np
    from PIL import Image
    im = Image.open(path)
    a = np.asarray(im)
    if a.ndim == 3:
        a = a[..., 0]
    top = 65535.0 if a.dtype == np.uint16 or im.mode.startswith("I;16") else 255.0
    return a.astype("float32") / top


def decode_depth(depth_file: Optional[Path], depth_meta: Optional[dict] = None):
    """Linear depth [H, W] float32 from a saved depth file (EXR linear, or inverted-log PNG)."""
    import numpy as np
    if depth_file is None:
        return None
    if str(depth_file).lower().endswith(".exr"):
        return read_exr(depth_file)
    v = read_png_gray(depth_file)              # white = zMin, black = zMax, log-spaced
    zmin, zmax = float((depth_meta or {})["zMin"]), float((depth_meta or {})["zMax"])
    return np.exp(math.log(zmax) + v * (math.log(zmin) - math.log(zmax))).astype("float32")


def image_size(path: Path):
    from PIL import Image
    with Image.open(path) as im:
        return im.size


def frame_mesh(image_size_wh, K: dict, depth, conf, cam: dict, base: dict, stride: int = 4,
               max_jump: float = 0.15, conf_min: float = 0.1, max_depth: float = 0.0):
    """Unproject one posed RGBD view into (positions[N,3], uvs[N,2], triangles[M,3]) in the *base* frame
    (the first picture's camera: origin, looking down -Z).

    ``depth`` is z-depth along the optical axis. Quads whose neighbouring depths jump by more than
    ``max_jump`` (relative) are cut, so the mesh does not stretch across silhouettes.
    """
    import numpy as np
    H, W = depth.shape
    s = max(1, int(stride))
    ys, xs = np.arange(s // 2, H, s), np.arange(s // 2, W, s)
    d = depth[np.ix_(ys, xs)].astype("float64")
    ok = np.isfinite(d) & (d > 1e-6)
    if max_depth and max_depth > 0:
        ok &= d <= max_depth
    if conf is not None:
        c = conf[np.ix_(np.minimum(ys, conf.shape[0] - 1), np.minimum(xs, conf.shape[1] - 1))]
        ok &= c >= conf_min
    d = np.where(ok, d, 1.0)
    sx, sy = K["width"] / W, K["height"] / H            # depth grid vs. camera grid
    u, v = np.meshgrid((xs + 0.5) * sx, (ys + 0.5) * sy)
    x = (u - K["cx"]) / K["fx"] * d
    y = -(v - K["cy"]) / K["fy"] * d
    p_cam = np.stack([x, y, -d], axis=-1)                # RUB: camera looks down -Z
    R, t = np.array(cam["R"]), np.array(cam["t"])
    R0, t0 = np.array(base["R"]), np.array(base["t"])
    local = (R0.T @ ((p_cam.reshape(-1, 3) @ R.T + t) - t0).T).T     # world -> the base camera's frame
    uv = np.stack([(xs[None, :] + 0.5) / W * np.ones((len(ys), 1)),
                   (ys[:, None] + 0.5) / H * np.ones((1, len(xs)))], -1)
    gh, gw = d.shape
    idx = np.arange(gh * gw).reshape(gh, gw)
    a, b, c_, e = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    okq = ok[:-1, :-1] & ok[:-1, 1:] & ok[1:, :-1] & ok[1:, 1:]
    zs = np.stack([d[:-1, :-1], d[:-1, 1:], d[1:, :-1], d[1:, 1:]])
    okq &= (zs.max(0) - zs.min(0)) <= max_jump * zs.min(0)
    qa, qb, qc, qe = a[okq], b[okq], c_[okq], e[okq]
    tris = np.concatenate([np.stack([qa, qc, qb], -1), np.stack([qb, qc, qe], -1)])
    used = np.unique(tris)
    remap = -np.ones(gh * gw, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return (local[used].astype("float32"), uv.reshape(-1, 2)[used].astype("float32"),
            remap[tris].astype("uint32"))


def _pad4(b: bytes, fill: bytes = b"\x00") -> bytes:
    return b + fill * ((4 - len(b) % 4) % 4)


def write_glb(path: Path, parts: list) -> dict:
    """A GLB of textured meshes. Each part: ``{"positions","uvs","indices","image": bytes, "mime", "name"}``.
    Materials are unlit (the picture's own colours)."""
    import numpy as np
    blob = bytearray()
    views, accs, meshes, mats, texs, imgs, nodes = [], [], [], [], [], [], []

    def add_view(raw: bytes, target: Optional[int] = None) -> int:
        off = len(blob)
        blob.extend(_pad4(raw))
        v = {"buffer": 0, "byteOffset": off, "byteLength": len(raw)}
        if target:
            v["target"] = target
        views.append(v)
        return len(views) - 1

    for i, p in enumerate(parts):
        pos = np.ascontiguousarray(p["positions"], dtype="<f4")
        uv = np.ascontiguousarray(p["uvs"], dtype="<f4")
        ind = np.ascontiguousarray(p["indices"], dtype="<u4").reshape(-1)
        a_pos = len(accs)
        accs.append({"bufferView": add_view(pos.tobytes(), 34962), "componentType": 5126, "count": len(pos),
                     "type": "VEC3", "min": [float(v) for v in pos.min(0)], "max": [float(v) for v in pos.max(0)]})
        a_uv = len(accs)
        accs.append({"bufferView": add_view(uv.tobytes(), 34962), "componentType": 5126, "count": len(uv),
                     "type": "VEC2"})
        a_ix = len(accs)
        accs.append({"bufferView": add_view(ind.tobytes(), 34963), "componentType": 5125, "count": len(ind),
                     "type": "SCALAR"})
        imgs.append({"bufferView": add_view(p["image"]), "mimeType": p.get("mime", "image/png")})
        texs.append({"source": len(imgs) - 1, "sampler": 0})
        mats.append({"name": p.get("name", f"view{i}"), "doubleSided": True,
                     "pbrMetallicRoughness": {"baseColorTexture": {"index": len(texs) - 1}, "metallicFactor": 0.0,
                                              "roughnessFactor": 1.0},
                     "extensions": {"KHR_materials_unlit": {}}})
        meshes.append({"name": p.get("name", f"view{i}"),
                       "primitives": [{"attributes": {"POSITION": a_pos, "TEXCOORD_0": a_uv}, "indices": a_ix,
                                       "material": len(mats) - 1, "mode": 4}]})
        nodes.append({"mesh": len(meshes) - 1, "name": p.get("name", f"view{i}")})
    gltf = {"asset": {"version": "2.0", "generator": "ComfyUI-WorldLabsAtlas"},
            "extensionsUsed": ["KHR_materials_unlit"], "scene": 0,
            "scenes": [{"nodes": list(range(len(nodes)))}], "nodes": nodes, "meshes": meshes, "materials": mats,
            "textures": texs, "images": imgs,
            "samplers": [{"magFilter": 9729, "minFilter": 9729, "wrapS": 33071, "wrapT": 33071}],
            "accessors": accs, "bufferViews": views, "buffers": [{"byteLength": len(blob)}]}
    js = _pad4(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b" ")
    binary = bytes(blob)
    total = 12 + 8 + len(js) + 8 + len(binary)
    out = bytearray(struct.pack("<III", 0x46546C67, 2, total))
    out += struct.pack("<II", len(js), 0x4E4F534A) + js
    out += struct.pack("<II", len(binary), 0x004E4942) + binary
    Path(path).write_bytes(bytes(out))
    return {"vertices": int(sum(len(p["positions"]) for p in parts)),
            "triangles": int(sum(len(p["indices"]) for p in parts)), "bytes": total}


def write_ply(path: Path, positions, colors) -> dict:
    """Binary little-endian PLY point cloud: float xyz + uchar rgb."""
    import numpy as np
    pos = np.ascontiguousarray(positions, dtype="<f4")
    col = np.ascontiguousarray(colors, dtype="u1")
    n = len(pos)
    head = ("ply\nformat binary_little_endian 1.0\n"
            f"element vertex {n}\nproperty float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode("ascii")
    rec = np.empty(n, dtype=[("p", "<f4", 3), ("c", "u1", 3)])
    rec["p"], rec["c"] = pos, col
    Path(path).write_bytes(head + rec.tobytes())
    return {"points": int(n)}


def _usable_frames(world: dict) -> list:
    return [f for f in world["frames"] if f.get("depth") and f.get("camera")]


def build_glb(wd: Path, world: dict, out: Path, stride: int = 4, max_jump: float = 0.15,
              conf_min: float = 0.1, max_depth: float = 0.0) -> dict:
    """Write ``out`` (a GLB) from every frame of a world dict that has a depth map."""
    frames = _usable_frames(world)
    if not frames:
        raise AtlasError("none of the world's frames has a depth map, so there is nothing to mesh", "args")
    base = normalize_camera(frames[0]["camera"])
    parts = []
    for fr in frames:
        cam = normalize_camera(fr["camera"])
        depth = decode_depth(wd / fr["depth"], fr.get("depth_log") or {})
        conf = read_png_gray(wd / fr["confidence"]) if fr.get("confidence") else None
        img_path = wd / fr["image"]
        w, h = image_size(img_path)
        pos, uv, tri = frame_mesh((w, h), cam["K"], depth, conf, cam, base, stride, max_jump, conf_min, max_depth)
        if len(tri) == 0:
            continue
        parts.append({"positions": pos, "uvs": uv, "indices": tri, "image": img_path.read_bytes(),
                      "mime": "image/jpeg" if img_path.suffix.lower() in (".jpg", ".jpeg") else "image/png",
                      "name": f"view_{fr['id']:03d}"})
    if not parts:
        raise AtlasError("the depth maps gave no usable surface (all depth invalid or cut by the confidence / "
                         "jump limits); try a lower conf_min or a larger max_jump", "args")
    stats = write_glb(out, parts)
    stats.update(stride=stride, max_jump=max_jump, conf_min=conf_min)
    return stats


def build_ply(wd: Path, world: dict, out: Path, stride: int = 4, conf_min: float = 0.1,
              max_depth: float = 0.0) -> dict:
    """Write ``out`` (a PLY point cloud, colours from the pictures) from the world's depth frames."""
    import numpy as np
    from PIL import Image
    frames = _usable_frames(world)
    if not frames:
        raise AtlasError("none of the world's frames has a depth map, so there is nothing to export", "args")
    base = normalize_camera(frames[0]["camera"])
    pts, cols = [], []
    for fr in frames:
        cam = normalize_camera(fr["camera"])
        depth = decode_depth(wd / fr["depth"], fr.get("depth_log") or {})
        conf = read_png_gray(wd / fr["confidence"]) if fr.get("confidence") else None
        img = np.asarray(Image.open(wd / fr["image"]).convert("RGB"))
        H, W = depth.shape
        pos, uv, tri = frame_mesh((img.shape[1], img.shape[0]), cam["K"], depth, conf, cam, base, stride,
                                  max_jump=1e9, conf_min=conf_min, max_depth=max_depth)
        if not len(pos):
            continue
        px = np.clip((uv[:, 0] * img.shape[1]).astype(int), 0, img.shape[1] - 1)
        py = np.clip((uv[:, 1] * img.shape[0]).astype(int), 0, img.shape[0] - 1)
        pts.append(pos)
        cols.append(img[py, px])
    if not pts:
        raise AtlasError("the depth maps gave no usable points", "args")
    stats = write_ply(out, np.concatenate(pts), np.concatenate(cols))
    stats.update(stride=stride, conf_min=conf_min)
    return stats
