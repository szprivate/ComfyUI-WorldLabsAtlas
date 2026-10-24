"""World Labs Atlas: a plain-Python client, ComfyUI nodes and an optional MCP server.

``import worldlabs_atlas`` never imports ComfyUI; the nodes live in ``worldlabs_atlas.comfy_nodes``.
"""

from .atlas import Atlas, World, default_root
from .client import (DEFAULT_BASE_URL, TASK_GEN, TASK_I2P, TASK_S2M, TASK_T2I, WorldLabsClient, canonical_hash,
                     resolve_api_key)
from .errors import (ArgsError, AtlasError, AtlasTimeout, AuthError, BillingError, ConcurrencyError,
                     DependencyError, OperationFailed, RateLimitError, TransientError)

__version__ = "0.1.0"

__all__ = [
    "Atlas", "World", "default_root", "WorldLabsClient", "resolve_api_key", "canonical_hash", "DEFAULT_BASE_URL",
    "TASK_GEN", "TASK_I2P", "TASK_S2M", "TASK_T2I",
    "AtlasError", "AuthError", "ArgsError", "BillingError", "RateLimitError", "ConcurrencyError", "TransientError",
    "OperationFailed", "AtlasTimeout", "DependencyError",
]
