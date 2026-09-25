"""Locating external binaries without assuming PATH.

winget installed ffmpeg on this machine, said it had changed PATH, and never
created the shim. So `ffmpeg` resolved to nothing while `ffmpeg.exe` sat on disk
and worked, and a script that assumes the name would report software as missing
when it is installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey import tools  # noqa: E402


def test_a_directory_with_the_right_name_is_not_mistaken_for_a_binary(tmp_path: Path):
    # A glob for "PotreeConverter" matches the directory it lives in. Returning
    # that gives a path that looks fine and fails the moment it is executed.
    (tmp_path / "PotreeConverter").mkdir()
    assert tools._executables(tmp_path, "PotreeConverter") == []

    real = tmp_path / "PotreeConverter" / "PotreeConverter.exe"
    real.write_bytes(b"")
    assert tools._executables(tmp_path, "PotreeConverter") == [real]


def test_an_exe_is_preferred_over_an_extensionless_file(tmp_path: Path):
    (tmp_path / "bin").mkdir()
    plain = tmp_path / "bin" / "ffmpeg"
    exe = tmp_path / "bin" / "ffmpeg.exe"
    plain.write_bytes(b"")
    exe.write_bytes(b"")
    assert tools._executables(tmp_path, "ffmpeg")[0] == exe


def test_first_existing_ignores_directories(tmp_path: Path):
    d = tmp_path / "thing"
    d.mkdir()
    assert tools._first_existing([str(d)]) is None
    f = tmp_path / "thing.exe"
    f.write_bytes(b"")
    assert tools._first_existing([str(d), str(f)]) == f


def test_a_missing_tool_says_where_it_looked(monkeypatch, tmp_path: Path):
    # "not found" on its own sends someone to reinstall software that is already
    # installed. The message has to name the places that were checked.
    monkeypatch.delenv(tools.FFMPEG_ENV, raising=False)
    monkeypatch.setattr(tools.shutil, "which", lambda _: None)
    monkeypatch.setattr(tools, "TOOLS", tmp_path)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    with pytest.raises(SystemExit) as e:
        tools.find_ffmpeg()
    message = str(e.value)
    assert "Looked in" in message
    assert "PATH" in message
    assert tools.FFMPEG_ENV in message
    assert "winget install" in message


def test_a_missing_tool_can_be_asked_about_without_raising(monkeypatch, tmp_path: Path):
    monkeypatch.delenv(tools.FFMPEG_ENV, raising=False)
    monkeypatch.setattr(tools.shutil, "which", lambda _: None)
    monkeypatch.setattr(tools, "TOOLS", tmp_path)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert tools.find_ffmpeg(required=False) is None


def test_the_environment_variable_wins(monkeypatch, tmp_path: Path):
    # The next machine will put these somewhere none of this anticipates.
    f = tmp_path / "my-ffmpeg.exe"
    f.write_bytes(b"")
    monkeypatch.setenv(tools.FFMPEG_ENV, str(f))
    assert tools.find_ffmpeg() == f


def test_tool_report_distinguishes_found_from_on_path():
    rep = tools.tool_report()
    assert set(rep) == {"ffmpeg", "ffmpeg_on_path", "potree_converter", "potree_converter_on_path"}
    for key in ("ffmpeg", "potree_converter"):
        if rep[key] is not None:
            assert Path(rep[key]).is_file(), f"{key} must point at a file"
