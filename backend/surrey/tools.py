"""Finding the external binaries this project shells out to.

Neither is assumed to be on PATH: on Windows a package manager can install ffmpeg
without creating the PATH shim, and a script that only asks PATH then reports
installed software as missing.

So: look on PATH, then in the places a package manager actually puts things, then
under the project's own tools directory, and if none of that finds it, say where
it looked rather than only that it failed.

An environment variable overrides everything, because the next machine will put
these somewhere none of this anticipates.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "data" / "tools"

FFMPEG_ENV = "SURREY_FFMPEG"
CONVERTER_ENV = "SURREY_POTREECONVERTER"

CONVERTER_URL = (
    "https://github.com/potree/PotreeConverter/releases/download/"
    "2.1.3/PotreeConverter_2.1.3_x64_windows.zip"
)


def _winget_packages() -> Path:
    return Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"


def _first_existing(candidates) -> Path | None:
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    return None


def _executables(root: Path, stem: str) -> list[Path]:
    """Files named `stem`, preferring a .exe. Directories are not binaries.

    A glob for "PotreeConverter" matches the directory it lives in, and returning
    that produces a path that looks fine and fails the moment it is run.
    """
    if not root.exists():
        return []
    hits = [p for p in root.glob(f"**/{stem}.exe") if p.is_file()]
    hits += [p for p in root.glob(f"**/{stem}") if p.is_file()]
    return hits


def find_ffmpeg(required: bool = True) -> Path | None:
    """Locate ffmpeg without assuming it is on PATH."""
    tried: list[str] = []

    env = os.environ.get(FFMPEG_ENV)
    tried.append(f"${FFMPEG_ENV}={env or 'unset'}")
    found = _first_existing([env])
    if found:
        return found

    on_path = shutil.which("ffmpeg")
    tried.append(f"PATH -> {on_path or 'not found'}")
    if on_path:
        return Path(on_path)

    # winget installs into a versioned directory and does not always create the
    # shim it says it creates.
    pkgs = _winget_packages()
    tried.append(f"winget packages under {pkgs}")
    if pkgs.exists():
        matches = sorted(p for p in pkgs.glob("Gyan.FFmpeg*/**/bin/ffmpeg.exe") if p.is_file())
        if matches:
            return matches[-1]

    local = _executables(TOOLS, "ffmpeg")
    tried.append(f"project tools under {TOOLS}")
    if local:
        return local[0]

    if not required:
        return None
    raise SystemExit(
        "ffmpeg not found. Looked in:\n  "
        + "\n  ".join(tried)
        + "\n\nInstall it with:\n  winget install --exact --id Gyan.FFmpeg\n"
        f"or point {FFMPEG_ENV} at the binary directly."
    )


def find_potree_converter(required: bool = True) -> Path | None:
    """Locate PotreeConverter, same rules."""
    tried: list[str] = []

    env = os.environ.get(CONVERTER_ENV)
    tried.append(f"${CONVERTER_ENV}={env or 'unset'}")
    found = _first_existing([env])
    if found:
        return found

    on_path = shutil.which("PotreeConverter")
    tried.append(f"PATH -> {on_path or 'not found'}")
    if on_path:
        return Path(on_path)

    local = _executables(TOOLS, "PotreeConverter")
    tried.append(f"project tools under {TOOLS}")
    if local:
        return local[0]

    if not required:
        return None
    raise SystemExit(
        "PotreeConverter not found. Looked in:\n  "
        + "\n  ".join(tried)
        + f"\n\nIt is BSD-2-Clause with a prebuilt Windows binary:\n  {CONVERTER_URL}\n"
        f"Unzip it under {TOOLS} or point {CONVERTER_ENV} at the binary."
    )


def tool_report() -> dict:
    """What is present, for an artefact to record rather than a reader to guess."""
    ff = find_ffmpeg(required=False)
    pc = find_potree_converter(required=False)
    return {
        "ffmpeg": str(ff) if ff else None,
        "ffmpeg_on_path": shutil.which("ffmpeg") is not None,
        "potree_converter": str(pc) if pc else None,
        "potree_converter_on_path": shutil.which("PotreeConverter") is not None,
    }


def gpu_busy() -> tuple[bool, str]:
    """Whether the card is in use, so a timing is not taken beside a training run."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout
        name, util, mem = [x.strip() for x in out.strip().split(",")]
        return int(util) > 15 or int(mem) > 600, f"{name}, {util}% busy, {mem} MiB used"
    except (OSError, ValueError, subprocess.SubprocessError):
        return False, "nvidia-smi unavailable"
