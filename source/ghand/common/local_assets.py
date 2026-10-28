"""Resolve bundled cube and ground resources without simulator imports or downloads."""

from functools import lru_cache
import hashlib
import json
from pathlib import Path

ASSET_ROOT = Path(__file__).resolve().parents[3] / "assets/official"


@lru_cache(maxsize=1)
def asset_paths():
    """Verify every dependent USD/texture once, then map original URLs to local files."""
    manifest = json.loads((ASSET_ROOT / "manifest.json").read_text())
    paths = {}
    for url, entry in manifest["files"].items():
        path = ASSET_ROOT / entry["local"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Bundled asset checksum mismatch: {path}")
        paths[url] = str(path)
    return paths


def localize_assets(value):
    """Replace known resource URLs in saved settings; preserve unrelated values/types."""
    if isinstance(value, str):
        return asset_paths().get(value, value)
    if isinstance(value, dict):
        return {key: localize_assets(item) for key, item in value.items()}
    if isinstance(value, list):
        return [localize_assets(item) for item in value]
    if isinstance(value, tuple):
        return tuple(localize_assets(item) for item in value)
    return value


def bundled_asset(relative_path):
    """Get a verified bundled resource by its path below assets/official."""
    path = str(ASSET_ROOT / relative_path)
    if path not in asset_paths().values():
        raise ValueError(f"Asset is not listed in the bundle manifest: {relative_path}")
    return path
