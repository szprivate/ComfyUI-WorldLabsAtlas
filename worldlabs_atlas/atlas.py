"""High-level Atlas workflows on top of :class:`WorldLabsClient`.

A *world* is a folder on disk (``world.json`` + pictures, depth maps, ``world.glb``). The API itself has
no server-side "world" object - it is a set of view-synthesis tasks - so the world is built here:
posed RGBD views become a textured depth mesh in the first picture's camera frame.

Every call that costs credits is *cached by request hash*: the hash covers the task and its full request
body (pictures, context frames, cameras, prompt), so repeating an identical call returns the files from
disk and bills nothing. Pass ``force=True`` to run it again.

Folder layout (``world.json`` is compatible with worlds written by agentY)::

    <root>/<name>/world.json  view_000.png  depth_000.exr  conf_000.png  world.glb
                  renders/render_000.png ...   pending/<operation_id>.json
    <root>/_cache/<hash>.json          create-world / splat-mesh request cache
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

from . import camera as cam
from . import mesh
from .client import (TASK_GEN, TASK_I2P, TASK_S2M, TASK_T2I, WorldLabsClient, canonical_hash)
from .errors import ArgsError, AtlasError, AtlasTimeout, DependencyError

ProgressFn = Optional[Callable[[dict, float], None]]
MAX_VIEWS = 8


def default_root() -> Path:
    env = os.environ.get("WORLDLABS_ATLAS_DIR")
    return Path(env).expanduser() if env else Path.home() / "worldlabs_atlas"


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(name or "").strip()).strip("_")[:60]


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# -- a world on disk ----------------------------------------------------------------


class World:
    """The world.json of one folder, with helpers. ``data`` is the JSON dict itself."""

    def __init__(self, directory: Path, data: dict):
        self.dir = Path(directory)
        self.data = data

    # identity
    @property
    def name(self) -> str:
        return self.data.get("name") or self.dir.name

    @property
    def frames(self) -> list:
        return self.data.setdefault("frames", [])

    @property
    def glb(self) -> Optional[Path]:
        g = self.data.get("glb")
        return self.dir / g if g else None

    def handle(self) -> dict:
        """The small JSON-able handle that nodes pass around as the WORLD type."""
        return {"name": self.name, "dir": str(self.dir)}

    # persistence
    @classmethod
    def load(cls, directory) -> "World":
        d = Path(directory)
        f = d / "world.json"
        if not f.exists():
            raise ArgsError(f"no Atlas world in {d} (world.json missing)")
        return cls(d, json.loads(f.read_text(encoding="utf-8")))

    def save(self) -> None:
        (self.dir / "world.json").write_text(json.dumps(self.data, indent=1), encoding="utf-8")

    # frames
    def posed_frames(self) -> list:
        return [f for f in self.frames if f.get("depth") and f.get("camera")]

    def first_posed(self) -> dict:
        first = next((f for f in self.frames if f.get("camera")), None)
        if not first:
            raise ArgsError("the world has no posed frame to move from")
        return cam.normalize_camera(first["camera"])

    def first_image(self) -> Optional[Path]:
        for f in self.frames:
            p = self.dir / f.get("image", "")
            if f.get("image") and p.exists():
                return p
        return None

    def requests(self) -> dict:
        return self.data.setdefault("requests", {})


# -- the workflows ------------------------------------------------------------------


class Atlas:
    """``Atlas(client, root)``: create / render / expand / export, with request-hash caching."""

    def __init__(self, client: Optional[WorldLabsClient] = None, root=None):
        self.client = client or WorldLabsClient()
        self.root = Path(root) if root else default_root()

    # -- locating worlds -------------------------------------------------------

    def world_dir(self, name: str) -> Path:
        return self.root / slug(name)

    def resolve(self, ref) -> World:
        """A world by name (under the root), or by the path of its folder / world.json / world.glb;
        also accepts a handle dict (``{"dir": ...}``)."""
        if isinstance(ref, World):
            return ref
        if isinstance(ref, dict):
            ref = ref.get("dir") or ref.get("name")
        s = str(ref or "").strip().strip('"')
        if not s:
            raise ArgsError("name the world (a folder name under the output root, or a path)")
        p = Path(s).expanduser()
        if p.is_file():
            p = p.parent
        if p.is_dir() and (p / "world.json").exists():
            return World.load(p)
        d = self.world_dir(s)
        if not (d / "world.json").exists():
            raise ArgsError(f"no Atlas world '{s}' (looked in {d}); list_worlds() shows what exists")
        return World.load(d)

    def list_worlds(self) -> list:
        rows = []
        for f in sorted(self.root.glob("*/world.json")):
            try:
                w = json.loads(f.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            rows.append({"name": w.get("name"), "dir": str(f.parent),
                         "glb": str(f.parent / w["glb"]) if w.get("glb") else "",
                         "frames": len(w.get("frames") or []), "prompt": (w.get("prompt") or "")[:120],
                         "created": w.get("created")})
        return rows

    def pending(self) -> list:
        return [p.stem for p in self.root.glob("*/pending/*.json")]

    # -- request cache ---------------------------------------------------------

    def _cache_file(self, key: str) -> Path:
        return self.root / "_cache" / f"{key}.json"

    def _cache_get(self, key: str) -> Optional[dict]:
        f = self._cache_file(key)
        if f.exists():
            try:
                return json.loads(f.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                return None
        return None

    def _cache_put(self, key: str, rec: dict) -> None:
        f = self._cache_file(key)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(rec), encoding="utf-8")

    def _unique_dir(self, name: str) -> Path:
        base = slug(name)
        if not base:
            raise ArgsError("give the world a name")
        d, i = self.root / base, 2
        while (d / "world.json").exists():
            d = self.root / f"{base}_{i}"
            i += 1
        return d

    # -- pending jobs (resumable) ---------------------------------------------

    @staticmethod
    def _pending_path(wd: Path, op_id: str) -> Path:
        return wd / "pending" / f"{op_id}.json"

    def _save_pending(self, wd: Path, op: dict, job: dict) -> None:
        p = self._pending_path(wd, op["id"])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(job, indent=1), encoding="utf-8")

    def _run(self, job: dict, op: dict, timeout_s: float, on_progress: ProgressFn) -> dict:
        """Persist the job (so a timeout is resumable), wait, finish."""
        wd = Path(job["world_dir"])
        wd.mkdir(parents=True, exist_ok=True)
        self._save_pending(wd, op, job)
        try:
            op = self.client.wait(op, timeout_s, on_progress)
        except AtlasTimeout as exc:
            exc.extra.setdefault("operation_id", op.get("id"))
            raise
        except AtlasError as exc:
            if exc.kind in ("content_refusal", "execution", "execution_timeout"):
                self._pending_path(wd, op["id"]).unlink(missing_ok=True)   # nothing to collect
            raise
        return self._finish(job, op)

    def resume(self, operation_id: str, timeout_s: float = 120.0, on_progress: ProgressFn = None) -> dict:
        """Collect a job that timed out (or was started elsewhere on this machine)."""
        op_id = str(operation_id).strip()
        op = self.client.get_operation(op_id)
        for f in self.root.glob(f"*/pending/{op_id}.json"):
            job = json.loads(f.read_text(encoding="utf-8"))
            op = self.client.wait(op, timeout_s, on_progress)
            return self._finish(job, op)
        state = "failed" if op.get("error") else "done" if op.get("done") else "running"
        return {"status": state, "operation_id": op_id, "task": (op.get("metadata") or {}).get("task"),
                "note": "no saved job record for this operation on this machine, so nothing was downloaded"}

    def cancel(self, operation_id: str) -> dict:
        op = self.client.cancel(operation_id)
        for f in self.root.glob(f"*/pending/{operation_id}.json"):
            f.unlink(missing_ok=True)
        return {"status": "cancelled" if op.get("error") or not op.get("done") else "already_finished",
                "operation_id": operation_id}

    # -- frames and meshes -----------------------------------------------------

    def _frame_entry(self, world: World, i: int, fr: dict, src: str, op_id: str) -> dict:
        wd, c = world.dir, self.client
        stem = f"{i:03d}"
        img = c.download(fr["imageAsset"], wd / f"view_{stem}.png")
        camera = None
        if fr.get("camera"):
            n = cam.normalize_camera(fr["camera"])
            camera = cam.camera_to_api(n["R"], n["t"], n["K"])
        entry = {"id": i, "image": img.name, "source": src, "operation": op_id, "camera": camera,
                 "image_asset": (fr["imageAsset"] or {}).get("assetId")}
        d = fr.get("depth")
        if d:
            if d.get("depthAsset"):
                entry["depth"] = c.download(d["depthAsset"], wd / f"depth_{stem}.exr").name
                entry["depth_asset"] = d["depthAsset"].get("assetId")
            elif d.get("logDepthAsset"):
                entry["depth"] = c.download(d["logDepthAsset"], wd / f"depth_{stem}.png").name
                entry["depth_log"] = {"zMin": d.get("zMin"), "zMax": d.get("zMax")}
                entry["depth_asset"] = d["logDepthAsset"].get("assetId")
            if d.get("confidenceAsset"):
                entry["confidence"] = c.download(d["confidenceAsset"], wd / f"conf_{stem}.png").name
        return entry

    def build_mesh(self, world: World, stride: int = 4, max_jump: float = 0.15, conf_min: float = 0.1,
                   max_depth: float = 0.0) -> dict:
        """(Re)build ``world.glb`` from the saved depth maps. No API call, no cost."""
        glb = world.dir / "world.glb"
        stats = mesh.build_glb(world.dir, world.data, glb, max(1, int(stride)), float(max_jump),
                               float(conf_min), float(max_depth))
        world.data["glb"] = glb.name
        world.data["mesh"] = {**stats, "frame": "first view's camera: origin, looking down -Z, Y up (metres as "
                                                 "Atlas scales it)"}
        world.data.pop("mesh_error", None)
        return stats

    def _try_mesh(self, world: World, opts: dict) -> Optional[str]:
        """Build the mesh; a failure (e.g. no OpenEXR) is recorded, not raised - the credits are spent
        and the pictures are saved either way."""
        try:
            self.build_mesh(world, **opts)
            return None
        except (DependencyError, AtlasError, ImportError, OSError, ValueError) as exc:
            world.data["mesh_error"] = str(exc)
            return str(exc)

    def rebuild_mesh(self, world, mesh_detail: int = 4, max_jump: float = 0.15, conf_min: float = 0.1,
                     max_depth: float = 0.0) -> dict:
        w = self.resolve(world)
        stats = self.build_mesh(w, mesh_detail, max_jump, conf_min, max_depth)
        w.save()
        return {"glb": str(w.glb), **stats}

    def export_ply(self, world, mesh_detail: int = 4, conf_min: float = 0.1, max_depth: float = 0.0) -> dict:
        w = self.resolve(world)
        out = w.dir / "world.ply"
        stats = mesh.build_ply(w.dir, w.data, out, max(1, int(mesh_detail)), float(conf_min), float(max_depth))
        w.data["ply"] = out.name
        w.save()
        return {"ply": str(out), **stats}

    # -- finishing a job -------------------------------------------------------

    def _finish(self, job: dict, op: dict) -> dict:
        kind, wd = job["kind"], Path(job["world_dir"])
        resp, op_id = op.get("response") or {}, op.get("id", "")
        usage = self.client.usage_of(op)
        if kind == "splat_mesh":
            glb = wd / f"{job['stem']}.glb"
            self.client.download(resp.get("mesh") or {}, glb)
            self._pending_path(wd, op_id).unlink(missing_ok=True)
            out = {"status": "ok", "glb": str(glb), "operation_id": op_id, "usage": usage or None,
                   "anchor_positions": (resp.get("anchorPositions") or [])[:16]}
            if job.get("cache_key"):
                self._cache_put(job["cache_key"], {"glb": str(glb)})
            return out
        if kind == "render":
            return self._finish_render(job, op)
        frames = resp.get("frames") or []
        if not frames:
            raise AtlasError("the operation succeeded but returned no frames", "api", operation_id=op_id)
        wd.mkdir(parents=True, exist_ok=True)
        if kind == "create":
            world = World(wd, {"version": 1, "name": job["name"], "provider": "worldlabs-atlas", "created": _now(),
                               "prompt": job.get("prompt", ""), "prompt_used": resp.get("promptUsed"),
                               "frames": [], "operations": []})
            src = job["source"]
        else:
            world = World.load(wd)
            src = "generated"
            if resp.get("promptUsed") and not world.data.get("prompt_used"):
                world.data["prompt_used"] = resp["promptUsed"]
        first = len(world.frames)
        added = []
        for j, fr in enumerate(frames):
            if not fr.get("camera") and job.get("cameras"):
                fr = {**fr, "camera": job["cameras"][j]}
            entry = self._frame_entry(world, first + j, fr, src, op_id)
            world.frames.append(entry)
            added.append(entry["id"])
        world.data.setdefault("operations", []).append({"id": op_id, "task": job["task"], "when": _now()})
        if kind == "create" and src == "text" and not all(f.get("depth") for f in world.frames):
            raise AtlasError("Atlas returned the picture without depth; the world cannot be meshed.", "api",
                             operation_id=op_id)
        mesh_error = self._try_mesh(world, job.get("mesh_opts") or {}) if job.get("build_mesh", True) else None
        if job.get("request_key"):
            world.requests()[job["request_key"]] = {"added_frames": added, "operation": op_id, "when": _now()}
        world.save()
        self._pending_path(wd, op_id).unlink(missing_ok=True)
        if kind == "create" and job.get("cache_key"):
            self._cache_put(job["cache_key"], {"world_dir": str(wd)})
        return self._summary(world, op_id, usage, {"mesh_error": mesh_error, "added_frames": added})

    def _summary(self, world: World, op_id: str, usage: dict, extra: Optional[dict] = None) -> dict:
        out = {"status": "ok", "name": world.name, "world_dir": str(world.dir),
               "glb": str(world.glb) if world.glb and world.glb.exists() else "",
               "frames": [{"id": f["id"], "image": str(world.dir / f["image"]), "has_depth": bool(f.get("depth"))}
                          for f in world.frames],
               "mesh": world.data.get("mesh"), "prompt_used": world.data.get("prompt_used"),
               "operation_id": op_id, "usage": usage or None, "cached": False,
               "usage_note": "Atlas reports no per-task cost; credits show on the World Labs platform's Usage page."}
        if extra:
            out.update({k: v for k, v in extra.items() if v is not None})
        return out

    # -- create ----------------------------------------------------------------

    def create_world(self, name: str, prompt: str = "", images: Sequence = (), aspect_ratio: str = "16:9",
                     mesh_detail: int = 4, timeout_s: float = 600.0, force: bool = False,
                     on_progress: ProgressFn = None) -> dict:
        """Make a world from picture(s) (paths, URLs or bytes) and/or a text prompt.

        With pictures: ``images2PosedRGBD`` (camera + depth estimated per picture; several pictures must
        show the SAME place). With only a prompt: ``atlasTextToImage``. Returns a summary dict
        (``world_dir``, ``glb``, ``frames`` ...); ``cached`` is true when an identical request was already
        run and nothing was billed.
        """
        if not slug(name):
            raise ArgsError("give the world a name")
        imgs = [p for p in (images or []) if (isinstance(p, (bytes, bytearray)) or str(p).strip())]
        if not imgs and not str(prompt or "").strip():
            raise ArgsError("give pictures or a prompt")
        mesh_opts = {"stride": max(1, int(mesh_detail))}
        if imgs:
            task, source = TASK_I2P, "posed"
            body = {"frames": [{"imageAsset": self.client.image_asset(p)} for p in imgs],
                    "targetResolution": list(cam.GRID)}
        else:
            task, source = TASK_T2I, "text"
            ar = aspect_ratio if aspect_ratio in ("16:9", "9:16", "4:3", "3:4", "1:1") else "16:9"
            body = {"prompt": prompt, "aspectRatio": ar, "returnDepth": True, "numSamples": 1}
        key = canonical_hash(task, body)
        if not force:
            hit = self._cache_get(key)
            if hit and (Path(hit["world_dir"]) / "world.json").exists():
                w = World.load(hit["world_dir"])
                return self._summary(w, "", {}, {"cached": True, "cache_key": key})
        wd = self._unique_dir(name)
        op = self.client.submit(task, body)
        job = {"kind": "create", "task": task, "source": source, "name": wd.name, "world_dir": str(wd),
               "prompt": prompt, "mesh_opts": mesh_opts, "cache_key": key}
        out = self._run(job, op, timeout_s, on_progress)
        out["cache_key"] = key
        return out

    # -- context + cameras -----------------------------------------------------

    def context_frames(self, world: World, ctx: list) -> list:
        out = []
        for f in ctx:
            if not f.get("depth_asset"):
                raise ArgsError(f"frame {f['id']} has no stored depth asset (made outside Atlas?)")
            d_asset = {"assetId": f["depth_asset"]}
            depth = ({"logDepthAsset": d_asset, **f["depth_log"]} if f.get("depth_log") else {"depthAsset": d_asset})
            image = ({"assetId": f["image_asset"]} if f.get("image_asset")
                     else self.client.image_asset(str(world.dir / f["image"])))
            out.append({"imageAsset": image, "depth": depth, "camera": f["camera"]})
        return out

    @staticmethod
    def pick_context(world: World, api_cam: dict, n: int = 4) -> list:
        """The posed views with depth nearest the target camera (position, then where they look)."""
        tgt = cam.normalize_camera(api_cam)
        tpos, tfwd = tgt["t"], [-v for v in cam.col(tgt["R"], 2)]
        scored = []
        for f in world.posed_frames():
            c = cam.normalize_camera(f["camera"])
            fwd = [-v for v in cam.col(c["R"], 2)]
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(c["t"], tpos)))
            ang = math.acos(max(-1.0, min(1.0, cam.dot(fwd, tfwd))))
            scored.append((dist + 2.0 * ang, f["id"], f))
        scored.sort(key=lambda s: (s[0], s[1]))
        return [f for _s, _i, f in scored[:n]]

    # -- render one view ---------------------------------------------------------

    def render_view(self, world, camera_spec: dict, fov: float = 0.0, prompt: str = "", add_to_world: bool = False,
                    depth: bool = False, mesh_detail: int = 4, timeout_s: float = 600.0, force: bool = False,
                    on_progress: ProgressFn = None) -> dict:
        """Ask Atlas for ONE new 1280x720 picture of the world from a camera (``atlasGenerate``).

        ``camera_spec`` is in the world's mesh frame (see :mod:`worldlabs_atlas.camera`). With
        ``add_to_world`` the view (with depth) joins the world and ``world.glb`` is rebuilt, so later
        renders stay consistent with it. One call is one billed task; an identical repeat is served from
        disk (``cached`` true).
        """
        w = self.resolve(world)
        base = w.first_posed()
        api_cam, (_R, _t), fov_used = cam.resolve_camera(base, camera_spec, fov)
        ctx = self.pick_context(w, api_cam)
        if not ctx:
            raise ArgsError("the world has no view with depth to use as context")
        use_prompt = prompt or w.data.get("prompt_used") or w.data.get("prompt") or None
        body = {"contextFrames": self.context_frames(w, ctx), "targetCameras": [api_cam],
                "returnDepth": bool(depth or add_to_world), "enhancePrompt": not bool(use_prompt and not prompt)}
        if use_prompt:
            body["prompt"] = use_prompt
        # the key covers the inputs (cameras, prompt, flags), not the auto-picked context: a world that grew
        # since (add_to_world / expand) must not make an identical request bill again
        key = canonical_hash(TASK_GEN, "render", api_cam, use_prompt, bool(prompt), bool(depth), bool(add_to_world))
        rec = w.requests().get(key)
        if rec and not force and (w.dir / rec.get("file", "")).is_file():
            return self._render_result(w, rec, cached=True)
        op = self.client.submit(TASK_GEN, body)
        job = {"kind": "render", "task": TASK_GEN, "name": w.name, "world_dir": str(w.dir), "camera": api_cam,
               "fov": round(fov_used, 3), "spec": camera_spec, "prompt": use_prompt, "add_to_world": bool(add_to_world),
               "depth": bool(depth), "mesh_opts": {"stride": max(1, int(mesh_detail))}, "request_key": key}
        return self._run(job, op, timeout_s, on_progress)

    def _render_result(self, w: World, rec: dict, cached: bool) -> dict:
        return {"status": "ok", "world": w.name, "path": str(w.dir / rec["file"]),
                "depth_path": str(w.dir / rec["depth_file"]) if rec.get("depth_file") else "",
                "camera_world": rec.get("camera_world"), "camera_mesh_frame": rec.get("camera_mesh_frame"),
                "fov_deg": rec.get("fov_deg"), "operation_id": rec.get("operation", ""), "cached": cached,
                "added_to_world": rec.get("added_frame") is not None,
                "glb": str(w.glb) if w.glb and w.glb.exists() else ""}

    def _finish_render(self, job: dict, op: dict) -> dict:
        wd, op_id = Path(job["world_dir"]), op.get("id", "")
        frames = (op.get("response") or {}).get("frames") or []
        if not frames:
            raise AtlasError("the operation succeeded but returned no frame", "api", operation_id=op_id)
        fr = frames[0]
        if not fr.get("camera"):
            fr = {**fr, "camera": job["camera"]}
        world = World.load(wd)
        rdir = wd / "renders"
        rdir.mkdir(parents=True, exist_ok=True)
        idx = max([int(m.group(1)) for f in rdir.glob("render_*.png")
                   if (m := re.match(r"render_(\d+)", f.stem))] + [-1]) + 1
        raw = rdir / f"render_{idx:03d}.png"
        added, depth_file, mesh_error = None, "", None
        if job.get("add_to_world"):
            entry = self._frame_entry(world, max([f["id"] for f in world.frames] + [-1]) + 1, fr, "render", op_id)
            shutil.copyfile(wd / entry["image"], raw)
            world.frames.append(entry)
            added = entry["id"]
            mesh_error = self._try_mesh(world, job.get("mesh_opts") or {})
            if entry.get("depth"):
                depth_file = entry["depth"]
        else:
            self.client.download(fr["imageAsset"], raw)
            d = fr.get("depth") or {}
            if job.get("depth") and (d.get("depthAsset") or d.get("logDepthAsset")):
                asset = d.get("depthAsset") or d["logDepthAsset"]
                dp = rdir / (raw.stem + ("_depth.exr" if d.get("depthAsset") else "_depth.png"))
                self.client.download(asset, dp)
                depth_file = f"renders/{dp.name}"
        base = world.first_posed()
        mp = cam.camera_to_mesh_pose(base, fr["camera"])
        mesh_cam = {"position": [round(v, 5) for v in mp[1]],
                    "rotation_matrix": [[round(v, 6) for v in r] for r in mp[0]]}
        rec = {"file": f"renders/{raw.name}", "depth_file": depth_file, "operation": op_id, "camera_world": fr["camera"],
               "camera_mesh_frame": mesh_cam, "fov_deg": job.get("fov"), "added_frame": added, "when": _now(),
               "prompt": job.get("prompt")}
        world.data.setdefault("renders", []).append({k: v for k, v in rec.items() if k != "prompt"})
        if job.get("request_key"):
            world.requests()[job["request_key"]] = rec
        world.save()
        self._pending_path(wd, op_id).unlink(missing_ok=True)
        out = self._render_result(world, rec, cached=False)
        out["usage"] = self.client.usage_of(op) or None
        out["usage_note"] = "Atlas reports no per-task cost; credits show on the World Labs platform's Usage page."
        if mesh_error:
            out["mesh_error"] = mesh_error
        return out

    # -- expand ----------------------------------------------------------------

    def expand_world(self, world, cameras: Sequence[dict], fov: float = 0.0, prompt: str = "",
                     context: Optional[Sequence[int]] = None, mesh_detail: int = 4, timeout_s: float = 600.0,
                     force: bool = False, on_progress: ProgressFn = None) -> dict:
        """Grow the world: Atlas invents 1-8 new views at new cameras (``atlasGenerate``) and they are
        added (with depth) to the world and its mesh. ``cameras`` are camera specs in the mesh frame.
        Pure turns on the spot cannot give depth - include some movement."""
        w = self.resolve(world)
        if not cameras or len(cameras) > MAX_VIEWS:
            raise ArgsError(f"give 1 to {MAX_VIEWS} cameras")
        posed = w.posed_frames()
        if context:
            ctx = [f for f in posed if f["id"] in {int(c) for c in context}]
        else:
            ctx = posed[:1] + [f for f in posed[-3:] if f is not posed[0]]
        if not ctx:
            raise ArgsError("no context frames with depth to give Atlas")
        base = w.first_posed()
        cams = [cam.resolve_camera(base, c, fov)[0] for c in cameras]
        use_prompt = prompt or w.data.get("prompt_used") or w.data.get("prompt") or None
        key = canonical_hash(TASK_GEN, "expand", cams, use_prompt, bool(prompt), sorted(int(c) for c in context or []))
        if not force and key in w.requests():
            return self._summary(w, w.requests()[key].get("operation", ""), {},
                                 {"cached": True, "added_frames": w.requests()[key].get("added_frames")})
        body = {"contextFrames": self.context_frames(w, ctx), "targetCameras": cams, "returnDepth": True,
                "enhancePrompt": not bool(use_prompt and not prompt)}
        if use_prompt:
            body["prompt"] = use_prompt
        op = self.client.submit(TASK_GEN, body)
        job = {"kind": "expand", "task": TASK_GEN, "name": w.name, "world_dir": str(w.dir), "cameras": cams,
               "mesh_opts": {"stride": max(1, int(mesh_detail))}, "request_key": key}
        return self._run(job, op, timeout_s, on_progress)

    # -- splat -> mesh -----------------------------------------------------------

    def splat_to_mesh(self, splat_path, name: str = "", texture_mode: str = "image_texture", target_faces: int = 0,
                      timeout_s: float = 900.0, force: bool = False, on_progress: ProgressFn = None) -> dict:
        """A Gaussian-splat ``.spz`` -> a GLB mesh (``splats2Mesh``). Saved under ``<root>/<name>/``."""
        p = Path(os.path.expanduser(str(splat_path).strip().strip('"')))
        if not p.is_file():
            raise ArgsError(f"splat file not found: {splat_path}")
        if p.suffix.lower() != ".spz":
            raise ArgsError("splats2Mesh takes a .spz splat file")
        if texture_mode not in ("image_texture", "vertex_color", "none"):
            raise ArgsError("texture_mode must be image_texture, vertex_color or none")
        key = canonical_hash(TASK_S2M, hashlib.sha256(p.read_bytes()).hexdigest(), texture_mode, int(target_faces or 0))
        if not force:
            hit = self._cache_get(key)
            if hit and Path(hit["glb"]).exists():
                return {"status": "ok", "glb": hit["glb"], "cached": True, "operation_id": ""}
        wd = self.root / (slug(name or p.stem) or "mesh")
        aid = self.client.upload_file(p, "application/octet-stream")
        body = {"splatAsset": {"assetId": aid}, "textureMode": texture_mode}
        if target_faces and int(target_faces) > 0:
            body["targetFaces"] = int(target_faces)
        op = self.client.submit(TASK_S2M, body)
        job = {"kind": "splat_mesh", "task": TASK_S2M, "world_dir": str(wd), "stem": slug(name or p.stem) or "mesh",
               "source": str(p), "cache_key": key}
        out = self._run(job, op, timeout_s, on_progress)
        out["cached"] = False
        return out
