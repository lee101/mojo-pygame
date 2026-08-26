"""ctypes bridge to the single Mojo shared library."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
LIB = os.environ.get("MOJOPYGAME_LIB") or os.path.join(
    ROOT, "dist", "libmojo-pygame.so"
)

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mpg_blit_rgba": ([I] * 18, None),
    "mpg_rect_collisions": ([I, I, I, I], I),
    "mpg_group_collisions": ([I, I, I, I, I], I),
    "mpg_mask_overlap": ([I] * 9, I),
    "mpg_mask_overlap_area": ([I] * 10, I),
    "mpg_mask_overlap_into": ([I] * 10, I),
    "mpg_mix_i16": ([I] * 8, None),
    "mpg_mix_i16_range": ([I] * 9, None),
    "mpg_resample_linear_i16": ([I, I, I, I, I, F], None),
}


class BuildError(RuntimeError):
    pass


def _mojo_command() -> list[str]:
    override = os.environ.get("MOJOPYGAME_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi):
        return [
            pixi,
            "run",
            "--manifest-path",
            os.path.join(ROOT, "pixi.toml"),
            "mojo",
        ]
    raise BuildError("mojo not found; set MOJOPYGAME_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    if os.environ.get("MOJOPYGAME_LIB") and os.path.exists(LIB) and not force:
        return LIB
    sources = [
        os.path.join(path, name)
        for path, _, names in os.walk(SRC)
        for name in names
        if name.endswith(".mojo")
    ]
    if not sources:
        if os.path.exists(LIB):
            return LIB
        raise BuildError(f"no Mojo sources found under {SRC}")
    if not force and os.path.exists(LIB):
        if os.path.getmtime(LIB) >= max(os.path.getmtime(path) for path in sources):
            return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    command = _mojo_command() + [
        "build",
        "--emit",
        "shared-lib",
        "-I",
        SRC,
        os.path.join(SRC, "capi.mojo"),
        "-o",
        LIB,
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    if result.returncode or not os.path.exists(LIB):
        raise BuildError((result.stderr or result.stdout).strip()[:4000])
    return LIB


_loaded: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _loaded
    if _loaded is None:
        _loaded = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_loaded, name)
            function.argtypes = argtypes
            function.restype = restype
    return _loaded


def addr(array: np.ndarray) -> int:
    return int(array.ctypes.data)


def main() -> int:
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
