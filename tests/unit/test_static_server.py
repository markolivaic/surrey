"""The local server answers byte ranges the way the viewer's host must."""

from __future__ import annotations

import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.static_server import serve  # noqa: E402


@pytest.fixture()
def server(tmp_path: Path):
    (tmp_path / "octree.bin").write_bytes(bytes(range(256)) * 4)
    srv = serve(str(tmp_path), 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def get(url: str, rng: str | None = None):
    req = urllib.request.Request(url, headers={"Range": rng} if rng else {})
    return urllib.request.urlopen(req, timeout=10)


def test_a_range_comes_back_as_206_with_only_those_bytes(server):
    r = get(f"{server}/octree.bin", "bytes=10-19")
    assert r.status == 206
    assert r.headers["Content-Range"] == "bytes 10-19/1024"
    assert r.read() == bytes(range(10, 20))
    assert r.headers["Access-Control-Allow-Origin"] == "*"


def test_open_and_suffix_ranges(server):
    assert get(f"{server}/octree.bin", "bytes=1020-").read() == bytes([252, 253, 254, 255])
    assert get(f"{server}/octree.bin", "bytes=-2").read() == bytes([254, 255])


def test_no_range_is_the_whole_file_and_a_bad_range_is_416(server):
    r = get(f"{server}/octree.bin")
    assert r.status == 200 and len(r.read()) == 1024
    with pytest.raises(urllib.error.HTTPError) as e:
        get(f"{server}/octree.bin", "bytes=5000-5001")
    assert e.value.code == 416


def test_a_mounted_folder_answers_ranges_and_cannot_be_escaped(tmp_path: Path):
    page, data = tmp_path / "page", tmp_path / "data"
    page.mkdir()
    data.mkdir()
    (page / "index.html").write_text("page")
    (data / "octree.bin").write_bytes(bytes(range(100)))
    (tmp_path / "secret.txt").write_text("no")
    srv = serve(str(page), 0, mounts={"/data": str(data)})
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        assert get(f"{base}/index.html").read() == b"page"
        r = get(f"{base}/data/octree.bin", "bytes=5-9")
        assert r.status == 206 and r.read() == bytes(range(5, 10))
        with pytest.raises(urllib.error.HTTPError):
            get(f"{base}/data/../secret.txt")
    finally:
        srv.shutdown()
