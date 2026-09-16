# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests: never import a real runtime or open a device."""

import builtins
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import run_dataflow_passthrough_hardware as runner


class RuntimeEnvironmentTest(unittest.TestCase):
    def test_preserves_roots_and_selects_source_before_import(self):
        api = Mock()
        original_import = builtins.__import__

        def import_runtime(name, *args, **kwargs):
            if name == "ttrt.runtime":
                self.assertEqual(os.environ["TT_METAL_HOME"], "/source")
                self.assertEqual(os.environ["TT_METAL_RUNTIME_ROOT"], "/install")
                os.environ["TT_METAL_RUNTIME_ROOT"] = "/wheel"
                os.environ["TT_METAL_RUNTIME_ROOT_EXTERNAL"] = "/install"
                return SimpleNamespace(_ttmlir_runtime=SimpleNamespace(runtime=api))
            return original_import(name, *args, **kwargs)

        with (
            patch.dict(os.environ, {"TT_METAL_RUNTIME_ROOT": "/install"}, clear=True),
            patch("builtins.__import__", side_effect=import_runtime),
            patch.object(runner.ctypes, "CDLL"),
            patch.object(
                runner,
                "loaded_libraries",
                return_value={"/install/lib/libtt_metal.so": "hash"},
            ),
        ):
            runner.load_runtime(Path("/source"))
            self.assertEqual(os.environ["TT_METAL_RUNTIME_ROOT"], "/install")
            self.assertNotIn("TT_METAL_RUNTIME_ROOT_EXTERNAL", os.environ)
            api.set_metal_home.assert_called_once_with("/source")

    def test_missing_root_does_not_load_runtime(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(
            runner.ctypes, "CDLL"
        ) as load:
            with self.assertRaisesRegex(RuntimeError, "matched installed runtime"):
                runner.load_runtime(Path("/source"))
            load.assert_not_called()

    def test_mismatched_loaded_library_is_rejected(self):
        fake = SimpleNamespace(_ttmlir_runtime=SimpleNamespace(runtime=Mock()))
        with (
            patch.dict(os.environ, {"TT_METAL_RUNTIME_ROOT": "/install"}, clear=True),
            patch.dict(
                "sys.modules", {"ttrt": SimpleNamespace(), "ttrt.runtime": fake}
            ),
            patch.object(runner.ctypes, "CDLL"),
            patch.object(
                runner,
                "loaded_libraries",
                return_value={"/wheel/libtt_metal.so": "hash"},
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "outside the requested"):
                runner.load_runtime(Path("/source"))


if __name__ == "__main__":
    unittest.main()
