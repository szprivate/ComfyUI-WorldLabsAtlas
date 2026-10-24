"""ComfyUI-WorldLabsAtlas: World Labs Atlas nodes for ComfyUI.

ComfyUI imports this folder as a package; the real code is in ``worldlabs_atlas/``, which must be
importable as a top-level package, so this folder is put on ``sys.path`` first.
"""

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from worldlabs_atlas.comfy_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
