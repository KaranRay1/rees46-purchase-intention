#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# macOS only: let LightGBM use the OpenMP runtime that ships with PyTorch.
#
# Why: LightGBM's macOS wheel needs libomp.dylib and only looks for it in
# Homebrew / MacPorts folders. Without Homebrew it fails to import
# ("Library not loaded: @rpath/libomp.dylib"). PyTorch already bundles a
# libomp.dylib, and using that one also avoids loading two OpenMP runtimes in
# the same process (a known cause of crashes when torch and LightGBM mix).
#
# What it does: adds PyTorch's lib folder to LightGBM's library search path
# (an rpath) and re-signs the library ad hoc, as macOS requires after editing.
# Safe to run more than once. Re-run after reinstalling lightgbm or torch.
# Undo: pip install --force-reinstall lightgbm
#
# Usage:  bash scripts/fix_macos_lightgbm.sh            (uses .venv/bin/python)
#         PYTHON=/path/to/python bash scripts/fix_macos_lightgbm.sh
# -----------------------------------------------------------------------------
set -euo pipefail

if [[ "$(uname)" != "Darwin" ]]; then
  echo "Not macOS; nothing to do."
  exit 0
fi

PYTHON="${PYTHON:-.venv/bin/python}"
SITE="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
LIB="$SITE/lightgbm/lib/lib_lightgbm.dylib"
TORCH_LIB="$SITE/torch/lib"
RPATH="@loader_path/../../torch/lib"

[[ -f "$LIB" ]] || { echo "LightGBM not found at $LIB"; exit 1; }
[[ -f "$TORCH_LIB/libomp.dylib" ]] || { echo "PyTorch libomp.dylib not found in $TORCH_LIB"; exit 1; }

if otool -l "$LIB" | grep -q "$RPATH"; then
  echo "rpath already present."
else
  install_name_tool -add_rpath "$RPATH" "$LIB"
  codesign --force --sign - "$LIB"
  echo "Added rpath $RPATH and re-signed $LIB"
fi

"$PYTHON" -c 'import lightgbm; print("lightgbm", lightgbm.__version__, "imports OK")'
