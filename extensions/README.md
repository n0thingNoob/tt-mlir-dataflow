# Out-of-tree extensions

Optional compiler extensions live behind CMake feature flags so the default
tt-mlir build remains compatible with upstream.

## Astraia

Astraia adds research passes for automatic spatial dataflow mapping on top of
the D2M dialect. During local development, keep the two repositories as sibling
checkouts and configure this repository with an explicit source path:

```bash
cmake -G Ninja -B build -S . \
  -DTTMLIR_ENABLE_ASTRAIA=ON \
  -DTTMLIR_ASTRAIA_SOURCE_DIR=../Astraia
```

After compatible revisions have been published, the Astraia repository can be
pinned as a submodule at `extensions/astraia`; that is also the default source
path used by CMake.

The extension is linked into `ttmlir-opt` only when enabled. No Astraia source
or pass registration is part of a default build.
