# RUN: %python %s %runtime_python_dir

# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ctypes
from pathlib import Path
import sys
import sysconfig


def test_runtime_import():
    """Exercise Python bindings without querying or opening a device."""
    library = Path(sysconfig.get_config_var("LIBDIR")) / sysconfig.get_config_var(
        "LDLIBRARY"
    )
    ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
    import _ttmlir_runtime as runtime

    assert callable(runtime.binary.load_binary_from_path)
    assert callable(runtime.runtime.open_mesh_device)
    assert runtime.runtime.WorkaroundEnv.get(False, False, False) is not None


if __name__ == "__main__":
    sys.path.insert(0, sys.argv[1])
    test_runtime_import()
