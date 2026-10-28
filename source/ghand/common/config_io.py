# SPDX-License-Identifier: BSD-3-Clause
"""Load saved YAML settings and restore task configuration without importing the simulator."""

from __future__ import annotations

from pathlib import Path

from .local_assets import localize_assets


def load_config_yaml(path):
    """Load plain YAML plus Python tuple tags, without arbitrary Python constructors."""
    import yaml

    class TupleSafeLoader(yaml.SafeLoader):
        pass

    TupleSafeLoader.add_constructor(
        "tag:yaml.org,2002:python/tuple", lambda loader, node: tuple(loader.construct_sequence(node, deep=True))
    )
    with Path(path).open() as stream:
        result = yaml.load(stream, Loader=TupleSafeLoader)
    if not isinstance(result, dict):
        raise ValueError(f"Expected a configuration mapping in {path}")
    return localize_assets(result)


def _primitive(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, (tuple, list)):
        return all(_primitive(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, (str, int, float)) and _primitive(v) for k, v in value.items())
    return False


def restore_config(config, saved):
    """Restore data into an existing task config; never import serialized callables/classes.

    Existing structured objects and dictionaries containing them are traversed. Plain
    data dictionaries (e.g. joint positions) are replaced, allowing changed regex keys.
    Unsupported/unknown fields are reported in the output provenance.
    """
    report = {"restored": [], "skipped": []}

    def restore(obj, data, prefix=""):
        for key, value in data.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            exists = key in obj if isinstance(obj, dict) else hasattr(obj, key)
            if str(key).startswith("_") or not exists:
                report["skipped"].append(path)
                continue
            current = obj[key] if isinstance(obj, dict) else getattr(obj, key)
            if callable(current):
                report["skipped"].append(path)
            elif isinstance(value, dict) and (
                hasattr(current, "__dict__") or isinstance(current, dict) and not _primitive(current)
            ):
                restore(current, value, path)
            elif _primitive(current) and _primitive(value):
                if isinstance(current, tuple) and isinstance(value, (tuple, list)):
                    value = tuple(value)
                if isinstance(obj, dict):
                    obj[key] = value
                else:
                    setattr(obj, key, value)
                report["restored"].append(path)
            else:
                report["skipped"].append(path)

    restore(config, saved)
    return report
