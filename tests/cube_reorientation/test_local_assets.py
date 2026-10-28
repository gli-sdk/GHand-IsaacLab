"""Saved environment files resolve bundled assets without simulator imports."""

from pathlib import Path
import tempfile
import unittest

from ghand.common.config_io import load_config_yaml

PROJECT = Path(__file__).resolve().parents[2]
URL = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Props/Blocks/DexCube/dex_cube_instanceable.usd"


class LocalAssetTests(unittest.TestCase):
    def test_saved_remote_cube_resolves_to_project_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "env.yaml"
            path.write_text(f"object_cfg:\n  spawn:\n    usd_path: {URL}\nscale: !!python/tuple [0.6, 0.6, 0.6]\n")
            cfg = load_config_yaml(path)
            self.assertEqual(
                cfg["object_cfg"]["spawn"]["usd_path"],
                str(PROJECT / "assets/official/Isaac/Props/Blocks/DexCube/dex_cube_instanceable.usd"),
            )
            self.assertEqual(cfg["scale"], (0.6, 0.6, 0.6))

    def test_unrelated_values_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "env.yaml"
            path.write_text("weight: 0.026453952\ncustom_asset: custom.usd\n")
            self.assertEqual(load_config_yaml(path), {"weight": 0.026453952, "custom_asset": "custom.usd"})
