"""Optional MCP server on the same client (``pip install "comfyui-worldlabs-atlas[mcp]"``).

Run:  ``python -m worldlabs_atlas.mcp_server``  (stdio). Key: WLT_API_KEY / WORLDLABS_API_KEY.
Worlds are saved under ``WORLDLABS_ATLAS_DIR`` (default ``~/worldlabs_atlas``).
"""

from __future__ import annotations

import json
from typing import Optional

from .atlas import Atlas
from .client import WorldLabsClient
from .errors import AtlasError


def _atlas() -> Atlas:
    return Atlas(WorldLabsClient())


def _run(fn) -> str:
    try:
        return json.dumps(fn())
    except AtlasError as exc:
        return exc.render()


def build_server():
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover
        raise SystemExit('The MCP server needs the "mcp" package: pip install "comfyui-worldlabs-atlas[mcp]"') from exc

    mcp = FastMCP("worldlabs-atlas")

    @mcp.tool()
    def create_world(name: str, prompt: str = "", image_paths: Optional[list] = None, aspect_ratio: str = "16:9",
                     mesh_detail: int = 4, timeout_s: int = 600, force: bool = False) -> str:
        """Make an Atlas world from picture(s) (local paths / https URLs) and/or a text prompt. Billed;
        an identical earlier request is returned from disk (cached=true)."""
        return _run(lambda: _atlas().create_world(name, prompt, image_paths or [], aspect_ratio, mesh_detail,
                                                  timeout_s, force))

    @mcp.tool()
    def render_view(world: str, camera: dict, fov: float = 0.0, prompt: str = "", add_to_world: bool = False,
                    depth: bool = False, timeout_s: int = 600, force: bool = False) -> str:
        """Render one new 1280x720 view of a world. camera (mesh frame, first picture = origin looking down -Z):
        {"position":[x,y,z],"look_at":[x,y,z]} | {"matrix":4x4} | {"forward","right","up","yaw","pitch","roll"}."""
        return _run(lambda: _atlas().render_view(world, camera, fov, prompt, add_to_world, depth, 4, timeout_s, force))

    @mcp.tool()
    def expand_world(world: str, cameras: list, fov: float = 0.0, prompt: str = "", timeout_s: int = 600,
                     force: bool = False) -> str:
        """Add 1-8 invented views (camera specs as in render_view) to a world and its mesh. Billed."""
        return _run(lambda: _atlas().expand_world(world, cameras, fov, prompt, None, 4, timeout_s, force))

    @mcp.tool()
    def list_worlds() -> str:
        """Worlds saved on this machine."""
        return _run(lambda: {"worlds": _atlas().list_worlds(), "pending_operations": _atlas().pending()})

    @mcp.tool()
    def list_operations(task: str = "", status: str = "", limit: int = 20) -> str:
        """Recent operations of the World Labs project (also ones started elsewhere)."""
        return _run(lambda: {"operations": _atlas().client.list_operations(task, status, limit)})

    @mcp.tool()
    def fetch(operation_id: str, timeout_s: int = 120, cancel: bool = False) -> str:
        """Collect an operation that timed out (downloads + saves it), or cancel it (cancel=true)."""
        a = _atlas()
        return _run(lambda: a.cancel(operation_id) if cancel else a.resume(operation_id, timeout_s))

    @mcp.tool()
    def splat_to_mesh(splat_path: str, name: str = "", texture_mode: str = "image_texture", target_faces: int = 0,
                      timeout_s: int = 900) -> str:
        """Gaussian-splat .spz -> GLB mesh (splats2Mesh). Billed."""
        return _run(lambda: _atlas().splat_to_mesh(splat_path, name, texture_mode, target_faces, timeout_s))

    @mcp.tool()
    def rebuild_mesh(world: str, mesh_detail: int = 4, max_jump: float = 0.15, conf_min: float = 0.1,
                     max_depth: float = 0.0) -> str:
        """Rebuild a world's GLB from its saved depth maps. Free."""
        return _run(lambda: _atlas().rebuild_mesh(world, mesh_detail, max_jump, conf_min, max_depth))

    return mcp


def main() -> None:  # pragma: no cover
    build_server().run()


if __name__ == "__main__":  # pragma: no cover
    main()
