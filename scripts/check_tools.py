#!/usr/bin/env python3
"""Check raster's required and optional development tools."""

from __future__ import annotations

import shutil
import subprocess
import sys


TOOLS: tuple[tuple[str, bool], ...] = (
    ("cmake", True),
    ("ninja", True),
    ("c++", True),
    ("just", True),
    ("clang-format", False),
    ("clang-tidy", False),
    ("cppcheck", False),
    ("ccache", False),
    ("sccache", False),
    ("mold", False),
    ("ld.lld", False),
)


def version(executable: str) -> str:
    try:
        result = subprocess.run(
            [executable, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "version unavailable"
    line = (result.stdout or result.stderr).splitlines()
    return line[0].strip() if line else "version unavailable"


def main() -> int:
    missing_required: list[str] = []
    for executable, required in TOOLS:
        path = shutil.which(executable)
        label = "required" if required else "optional"
        if path is None:
            print(f"MISSING ({label}): {executable}")
            if required:
                missing_required.append(executable)
            continue
        print(f"FOUND  ({label}): {path} — {version(path)}")

    pkg_config = shutil.which("pkg-config")
    if pkg_config is None:
        print("MISSING (required for SDL3 discovery): pkg-config")
        missing_required.append("pkg-config")
    else:
        result = subprocess.run(
            [pkg_config, "--modversion", "sdl3"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            print(f"FOUND  (required): SDL3 {result.stdout.strip()}")
        else:
            print("MISSING (required): SDL3 development package (pkg-config sdl3)")
            missing_required.append("SDL3 development package")

    if missing_required:
        print(f"Missing required tools: {', '.join(missing_required)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
