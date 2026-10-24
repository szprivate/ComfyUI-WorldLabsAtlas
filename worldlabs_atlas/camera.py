"""Camera math, pure Python (no numpy).

Conventions
-----------
* The *mesh frame* is ``world.glb``'s own: the first picture's camera is the origin, looking down -Z,
  +Y up, +X right, metres as Atlas scales them ("RUB": right-up-back).
* The API wants cameras in the world's own frame: ``{"extrinsics": {"position", "quaternion" (xyzw),
  "coordinateSystem"}, "intrinsics": {fx, fy, cx, cy, width, height}}`` (camera-to-world).
* A *camera spec* is what a user writes, in the mesh frame. One of::

      {"position": [x, y, z], "look_at": [x, y, z], "roll": deg}
      {"matrix": 4x4 or 16 numbers (row-major camera-to-world, camera looks down -Z, +Y up)}
      {"forward": m, "right": m, "up": m, "yaw": deg, "pitch": deg, "roll": deg}   (relative to the first view)

  plus an optional ``"fov"`` (vertical degrees).
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

from .errors import ArgsError

GRID = (1280, 720)              # the model's camera grid: Atlas renders exactly this size
MIN_FOV, MAX_FOV = 10.0, 120.0


# -- small vector / matrix helpers (3x3 row-major lists) -------------------------


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def unit(a):
    n = math.sqrt(dot(a, a)) or 1.0
    return [x / n for x in a]


def mm(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def mv(a, v):
    return [sum(a[i][k] * v[k] for k in range(3)) for i in range(3)]


def transpose(a):
    return [[a[j][i] for j in range(3)] for i in range(3)]


def col(m, j):
    return [m[0][j], m[1][j], m[2][j]]


def rot_x(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return [[1, 0, 0], [0, c, -s], [0, s, c]]


def rot_y(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return [[c, 0, s], [0, 1, 0], [-s, 0, c]]


def rot_z(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return [[c, -s, 0], [s, c, 0], [0, 0, 1]]


def quat_to_mat(q):
    x, y, z, w = [float(v) for v in q]
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def mat_to_quat(m):
    t = m[0][0] + m[1][1] + m[2][2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return [(m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s, 0.25 * s]
    if m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        return [0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s, (m[2][1] - m[1][2]) / s]
    if m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        return [(m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s, (m[0][2] - m[2][0]) / s]
    s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
    return [(m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s, (m[1][0] - m[0][1]) / s]


def orthonormal(m):
    """The rotation nearest a matrix that may carry scale: Gram-Schmidt on its columns."""
    x = unit(col(m, 0))
    y0 = col(m, 1)
    y = unit([y0[i] - dot(x, y0) * x[i] for i in range(3)])
    z = cross(x, y)
    return [[x[i], y[i], z[i]] for i in range(3)]


def look_at_matrix(position, target, up=(0.0, 1.0, 0.0)):
    """Rotation (camera axes as columns) of a camera at ``position`` looking at ``target``.
    The camera looks down its own -Z. Looking straight along ``up`` falls back to the -Z axis as up."""
    back = unit([p - t for p, t in zip(position, target)])
    if sum(c * c for c in cross(list(up), back)) < 1e-9:
        up = (0.0, 0.0, -1.0)
    right = unit(cross(list(up), back))
    true_up = cross(back, right)
    return [[right[0], true_up[0], back[0]], [right[1], true_up[1], back[1]], [right[2], true_up[2], back[2]]]


# -- API camera <-> internal pose ----------------------------------------------


def normalize_camera(cam: dict) -> dict:
    """An API PinholeCamera -> ``{"R": 3x3, "t": [3], "K": {...}}`` in RUB."""
    ex, K = cam["extrinsics"], cam["intrinsics"]
    R = quat_to_mat(ex["quaternion"])
    if str(ex.get("coordinateSystem", "rub")).lower() == "rdf":      # right-down-forward -> right-up-back
        R = mm(R, [[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    return {"R": R, "t": [float(v) for v in ex["position"]],
            "K": {k: K[k] for k in ("fx", "fy", "cx", "cy", "width", "height")}}


def camera_to_api(R, t, K: dict) -> dict:
    return {"extrinsics": {"position": [round(float(v), 6) for v in t],
                           "quaternion": [round(float(v), 8) for v in mat_to_quat(R)],
                           "coordinateSystem": "rub"},
            "intrinsics": {"fx": K["fx"], "fy": K["fy"], "cx": K["cx"], "cy": K["cy"],
                           "width": int(K["width"]), "height": int(K["height"])}}


def grid_intrinsics(base: dict, fov: float = 0.0) -> dict:
    """Intrinsics on the model's 1280 x 720 grid, from the first picture's lens, or ``fov`` (vertical
    degrees) if given."""
    k0 = base["K"]
    gw, gh = GRID
    sx, sy = gw / k0["width"], gh / k0["height"]
    K = {"fx": k0["fx"] * sx, "fy": k0["fy"] * sy, "cx": gw / 2.0, "cy": gh / 2.0, "width": gw, "height": gh}
    if fov and fov > 0:
        fy = (gh / 2.0) / math.tan(math.radians(fov) / 2.0)
        K.update(fy=fy, fx=fy)
    return K


def fov_of(K: dict) -> float:
    return math.degrees(2.0 * math.atan((K["height"] / 2.0) / K["fy"]))


def mesh_pose_to_camera(base: dict, K: dict, R, t) -> dict:
    """A camera pose in the MESH frame -> an Atlas API camera in the world's own frame."""
    return camera_to_api(mm(base["R"], R), [a + b for a, b in zip(mv(base["R"], t), base["t"])], K)


def camera_to_mesh_pose(base: dict, cam: dict):
    """The inverse: an API camera of the world -> (R, t) in the mesh frame."""
    n = normalize_camera(cam)
    r0t = transpose(base["R"])
    return mm(r0t, n["R"]), mv(r0t, [a - b for a, b in zip(n["t"], base["t"])])


# -- camera specs ----------------------------------------------------------------


def relative_pose(spec: dict):
    """(R, t) of a relative move from the first view (mesh frame: identity at the origin).
    ``forward`` / ``right`` move along the horizontal look direction / its right, ``up`` along +Y;
    ``yaw`` > 0 turns left, ``pitch`` > 0 looks up."""
    fw, rt, up = float(spec.get("forward", 0) or 0), float(spec.get("right", 0) or 0), float(spec.get("up", 0) or 0)
    t = [rt, up, -fw]
    R = mm(mm(rot_y(float(spec.get("yaw", 0) or 0)), rot_x(float(spec.get("pitch", 0) or 0))),
           rot_z(float(spec.get("roll", 0) or 0)))
    return R, t


def _matrix4(m):
    rows = [list(map(float, r)) for r in m] if m and isinstance(m[0], (list, tuple)) else None
    if rows is None:
        flat = [float(x) for x in m]
        if len(flat) != 16:
            raise ArgsError("matrix needs a 4x4 (16 numbers, row-major)")
        rows = [flat[i * 4:(i + 1) * 4] for i in range(4)]
    if len(rows) != 4 or any(len(r) != 4 for r in rows):
        raise ArgsError("matrix needs a 4x4 (16 numbers, row-major)")
    return [r[:3] for r in rows[:3]], [rows[0][3], rows[1][3], rows[2][3]]


def mesh_pose_of_spec(spec: dict):
    """(R, t) in the mesh frame for a camera spec (see the module doc)."""
    if not isinstance(spec, dict) or not spec:
        raise ArgsError("give a camera: position + look_at, a matrix, or a relative move "
                        "(forward / right / up / yaw / pitch / roll)")
    if spec.get("matrix") is not None:
        A, t = _matrix4(spec["matrix"])
        return orthonormal(A), t
    if spec.get("position") is not None:
        pos = [float(x) for x in spec["position"]]
        if spec.get("look_at") is None:
            raise ArgsError("an absolute camera needs look_at next to position")
        tgt = [float(x) for x in spec["look_at"]]
        if dot([a - b for a, b in zip(pos, tgt)], [a - b for a, b in zip(pos, tgt)]) < 1e-12:
            raise ArgsError("look_at is the camera's own position")
        R = look_at_matrix(pos, tgt)
        if spec.get("roll"):
            R = mm(R, rot_z(float(spec["roll"])))
        return R, pos
    return relative_pose(spec)


def resolve_camera(base: dict, spec: dict, fov: float = 0.0):
    """``(api_camera, (R, t) in the mesh frame, fov_used)`` for a camera spec.
    ``base`` is ``normalize_camera`` of the world's first posed frame."""
    R, t = mesh_pose_of_spec(spec)
    use_fov = float(fov or spec.get("fov") or 0.0)
    use_fov = max(MIN_FOV, min(MAX_FOV, use_fov)) if use_fov else 0.0
    K = grid_intrinsics(base, use_fov)
    if not use_fov:
        use_fov = fov_of(K)
    return mesh_pose_to_camera(base, K, R, t), (R, [float(x) for x in t]), use_fov


# -- helpers for sweeps -----------------------------------------------------------


def orbit_camera(center: Sequence[float], distance: float, yaw: float, pitch: float = 0.0,
                 fov: float = 0.0) -> dict:
    """A camera spec on a sphere around ``center``, looking at it.

    ``yaw`` 0 puts the camera on the +Z side of the centre (in front of it, looking toward -Z like the
    first picture); positive yaw swings it round to the +X side. ``pitch`` > 0 raises it above the centre.
    """
    d = max(1e-6, float(distance))
    y, p = math.radians(yaw), math.radians(pitch)
    c = [float(v) for v in center]
    pos = [c[0] + d * math.sin(y) * math.cos(p), c[1] + d * math.sin(p), c[2] + d * math.cos(y) * math.cos(p)]
    out = {"position": [round(v, 6) for v in pos], "look_at": [round(v, 6) for v in c]}
    if fov:
        out["fov"] = float(fov)
    return out


def orbit_sweep(center, distance, yaw_start, yaw_end, count, pitch=0.0, fov=0.0) -> list:
    """``count`` orbit cameras with yaw from ``yaw_start`` to ``yaw_end`` (inclusive)."""
    n = max(1, int(count))
    if n == 1:
        return [orbit_camera(center, distance, yaw_start, pitch, fov)]
    return [orbit_camera(center, distance, yaw_start + (yaw_end - yaw_start) * i / (n - 1), pitch, fov)
            for i in range(n)]


def spec_hash_view(spec: dict) -> dict:
    """A spec reduced to rounded numbers, so tiny float noise does not change a cache key."""
    def r(v):
        if isinstance(v, (list, tuple)):
            return [r(x) for x in v]
        if isinstance(v, float):
            return round(v, 5)
        return v
    return {k: r(v) for k, v in sorted(spec.items())}
