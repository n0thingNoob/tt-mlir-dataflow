# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ctypes
from pathlib import Path
import sysconfig


def test_runtime_import():
    """Exercise Python bindings without querying or opening a device."""
    library = Path(sysconfig.get_config_var("LIBDIR")) / sysconfig.get_config_var(
        "LDLIBRARY"
    )
    ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
    from ttrt.runtime import _ttmlir_runtime as runtime

    assert callable(runtime.binary.load_binary_from_path)
    assert callable(runtime.runtime.open_mesh_device)
    assert runtime.runtime.WorkaroundEnv.get(False, False, False) is not None
