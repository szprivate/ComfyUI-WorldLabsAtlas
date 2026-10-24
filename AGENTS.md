# Notes for coding agents

Standalone repo (not part of the agentY release stack): work on `main`, small commits, tests must pass.

- `worldlabs_atlas/` is the plain-Python client and must never import ComfyUI. Only `comfy_nodes.py` touches
  `folder_paths` / `comfy` / `torch`, and it imports them lazily or in try/except so tests run without them.
- Never print, log, commit or write an API key. Live tests read `WORLDLABS_API_KEY` from the environment only.
- Anything that costs credits is cached by request hash (`Atlas._cache_*`, `world.json["requests"]`). Keep that
  true for new calls, and keep live tests to a minimum (they bill).
- Tests are offline (fake HTTP session): `python -m pytest -q`.
- `world.json` is compatible with worlds written by agentY (agenty_core); do not rename its keys.
