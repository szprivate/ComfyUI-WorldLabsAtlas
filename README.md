# ComfyUI-WorldLabsAtlas

World Labs **Atlas** (the Marble 2 beta developer API) for ComfyUI: turn a picture or a prompt into a 3D-aware
"world", then ask for **new pictures of the same place from any camera** - consistent backgrounds and
environments, angle sweeps, first frames for image-to-video.

The package has three layers that share one client:

1. `worldlabs_atlas/` - a plain-Python client (no ComfyUI imports): auth, uploads, operations, polling with
   backoff, cancel, camera math, mesh export, request caching.
2. ComfyUI nodes (category **WorldLabs Atlas**).
3. An optional MCP server on the same client.

> Atlas is a **beta** API and needs a World Labs account with credits. Nothing here is affiliated with World Labs.

## What Atlas actually is

The API is not a "make me a world" service. It is a few async view-synthesis *tasks*:

| task | does |
|---|---|
| `atlasTextToImage` | prompt -> image + depth |
| `images2PosedRGBD` | picture(s) -> the same picture(s) with an estimated camera and depth |
| `atlasGenerate` | posed RGBD context + new target cameras -> new views (+ depth) |
| `splats2Mesh` | Gaussian splat `.spz` -> GLB mesh |

So a **world** here is a folder on your disk (`world.json`, the pictures, depth maps, a textured depth mesh
`world.glb`). There is no server-side world id; the WORLD socket carries `{name, dir}`. Atlas returns images and
depth, not splats, so there is no PLY/SPZ *download*: the PLY export is a coloured point cloud built from the
depth maps. The mesh is in the first picture's camera frame: the camera at the origin looking down -Z, +Y up,
metres as Atlas scales them. Camera positions in the nodes are in that frame.

## Install

**ComfyUI-Manager**: search "WorldLabs Atlas" (once published to the Comfy Registry), or

```
cd ComfyUI/custom_nodes
git clone https://github.com/szprivate/ComfyUI-WorldLabsAtlas
pip install -r ComfyUI-WorldLabsAtlas/requirements.txt     # with ComfyUI's Python
```

Restart ComfyUI. `requests`, `numpy` and `Pillow` are normally already there. `OpenEXR` is needed to read Atlas
depth maps for the mesh/PLY. Without it worlds are still created and pictures saved; the mesh step reports a
clear error (and `Export World` builds it later, for free, once you install it).

## API key

Create a project API key on the World Labs platform with the scopes `tasks.create`, `operations.read`,
`assets.create`, `assets.read`. Then, **preferably**, set an environment variable before starting ComfyUI:

```
setx WLT_API_KEY "wlt_..."          # Windows (new shell afterwards); WORLDLABS_API_KEY works too
export WLT_API_KEY=wlt_...          # Linux / macOS
```

Alternatively add an **Atlas API Key** node and wire it into the nodes' optional `api_key` input. Careful: a key
typed into a node is saved inside the workflow file. The key is never logged or written to disk by this package.

Optional: `WLT_API_BASE_URL` (default `https://api.atlas-beta.worldlabs.ai/api/v2`), `WORLDLABS_ATLAS_DIR`
(where worlds are saved; default `ComfyUI/output/worldlabs_atlas`, or `~/worldlabs_atlas` outside ComfyUI).

## Nodes

Custom socket types: `WORLD` (a `{name, dir}` handle) and `CAMERA` (a camera dict, or a list of them for a sweep).
All nodes that call the API take an optional `api_key` (STRING) and a `timeout_s`.

| node | inputs | outputs |
|---|---|---|
| **Atlas API Key** | `api_key` | `api_key` STRING |
| **Atlas Create World** | `name`, `prompt`, `aspect_ratio` (prompt-only), `mesh_detail`, `timeout_s`, `ignore_cache`; optional `image` (a batch = several views of the *same* place), `api_key` | `world` WORLD, `preview` IMAGE (first frame), `info` STRING (JSON) |
| **Atlas Load World** | `world` (folder name under the output root, or a full path) | `world`, `preview`, `info` |
| **Atlas Camera from Orbit** | `center_x/y/z`, `distance`, `yaw`, `pitch`, `fov`, `count` (1-8), `yaw_end` | `camera` CAMERA (a list when `count` > 1), `pos_x/y/z`, `look_x/y/z` of the first camera |
| **Atlas Render View** | `world`, `pos_x/y/z`, `look_x/y/z`, `fov` (0 = the first picture's lens), `prompt`, `add_to_world`, `return_depth`, `output_width/height` (0 = native 1280x720), `timeout_s`, `ignore_cache`; optional `camera` (overrides the position/look_at widgets; a list renders one view each, max 8), `api_key` | `image` IMAGE (batch), `depth` IMAGE (normalised, near = white; black unless `return_depth`), `world`, `paths` STRING (JSON list) |
| **Atlas Expand World** | `world`, `forward/right/up`, `yaw`, `pitch`, `fov`, `prompt`, `mesh_detail`, `timeout_s`, `ignore_cache`; optional `camera` (list = several views, max 8), `api_key` | `world`, `new_views` IMAGE, `info` STRING |
| **Atlas Export World (GLB / PLY)** | `world`, `rebuild_mesh`, `export_ply`, `mesh_detail`, `max_jump`, `conf_min`, `max_depth` | `world`, `glb_path`, `ply_path` (free: no API call) |
| **Atlas Splat to Mesh (.spz)** | `spz_path`, `name`, `texture_mode`, `target_faces`, `timeout_s`, `ignore_cache`; optional `api_key` | `glb_path` STRING |

Notes:

* **Render View** always yields 1280x720 (the model's camera grid); `output_width/height` resize afterwards. Atlas gives
  no alpha or coverage, so there is no MASK output. There is no seed in the API: to get another take of an
  identical request, switch on `ignore_cache` (it bills again).
* **Expand World** takes cameras, not pictures: new pictures of a different place cannot be aligned to an existing
  world; make a second world instead. A pure turn on the spot gives no depth, so include some movement.
* Keep a camera a believable step (a few metres, under ~60 degrees of turn) from an existing view: Atlas invents what
  the pictures never showed. `add_to_world` lets later renders stay consistent with earlier new views.
* Camera dict forms: `{"position": [x,y,z], "look_at": [x,y,z]}`, `{"matrix": 4x4 camera-to-world}`, or a move
  relative to the first picture `{"forward","right","up","yaw","pitch","roll"}`; any of them with `"fov"`.
* Long operations block the worker thread (fine in ComfyUI) but poll every ~5 s, move a progress bar, and
  **interrupting the queue cancels the operation** on World Labs' side.

An example workflow is in `example_workflows/world_orbit_sweep.json` (picture -> world -> 4-view orbit sweep).

## Costs, caching, limits

* Atlas reports no per-task cost; credits (1,000 per US$) show on the World Labs platform's Usage page. Every
  Create World, Render View, Expand World and Splat to Mesh call is **one billed task** (a Render View with a 4-camera
  list is four).
* **Nothing is billed twice for the same request.** Each call is cached by request hash (pictures, cameras, prompt,
  flags) in the world's folder: re-running an unchanged graph, restarting ComfyUI, or calling the same render again
  returns the files from disk. `ignore_cache` re-runs on purpose. Create World's cache is keyed on the request, not
  the name, so renaming does not make a second world.
* Rate limits (429/503) are retried politely (Retry-After honoured). A full **concurrency** limit is retried for ~90 s,
  then raised as an error that names the operations occupying the slots and how to cancel one. A job that times
  out keeps running; its `operation_id` is in the error and `Atlas.resume(id)` (or the MCP `fetch`) downloads it
  later (operations are kept 30 days).
* Errors carry a kind: `auth`, `scope`, `balance`, `spend_limit`, `not_found` (task not enabled: beta entitlement),
  `rate_limit`, `concurrency`, `content_refusal`, `execution`, `execution_timeout`, `timeout`, `args`,
  `dependency`, `transient`. ComfyUI shows them as the node error message.

## Python client

```python
from worldlabs_atlas import Atlas, WorldLabsClient

atlas = Atlas(WorldLabsClient(), root="D:/worlds")        # key: WLT_API_KEY / WORLDLABS_API_KEY or api_key=...
w = atlas.create_world("cabin", prompt="a small beach cabin", images=["cabin.png"])
r = atlas.render_view("cabin", {"position": [0.8, 0, 0.3], "look_at": [0.8, 0, -3]})
print(r["path"], r["cached"])
atlas.expand_world("cabin", [{"forward": 2, "yaw": 20}])
atlas.rebuild_mesh("cabin"); atlas.export_ply("cabin")
atlas.client.list_operations(limit=10); atlas.client.cancel("op_id")
```

## MCP server (optional)

```
pip install "comfyui-worldlabs-atlas[mcp]"      # or: pip install mcp   (in a checkout)
python -m worldlabs_atlas.mcp_server            # stdio; tools: create_world, render_view, expand_world,
                                                # list_worlds, list_operations, fetch, splat_to_mesh, rebuild_mesh
```

Example `mcp.json` entry: `{"command": "python", "args": ["-m", "worldlabs_atlas.mcp_server"], "env": {"WLT_API_KEY": "..."}}`.

## Tests

```
pip install pytest requests numpy Pillow
python -m pytest -q
```

All offline (a fake HTTP session): client polling/retries/errors/cancel, camera math, request mapping, caching,
mesh export, and loading the nodes module with stubbed `folder_paths` / `comfy`.

## License

MIT, (c) szprivate. See `LICENSE`.
