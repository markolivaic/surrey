"""A local static server that answers HTTP Range requests, for the viewer and its data.

Python's http.server ignores the Range header and sends the whole file with 200. An
octree loader asks for a node's bytes by range, so against that server every node
request would download the entire octree file. Clone-and-run needs a server that
behaves like the host will, so this one answers a single byte range with 206 and
Content-Range, and sends CORS headers so a page on another local port can read it.

Mounts put a second folder under a URL prefix, so the page and the octrees built by
scripts/pipeline/build_octree.py are served together the way the host serves them:

Run:
    python -m surrey.static_server --dir web/dist --mount /data=data/viewer --port 8770
"""

from __future__ import annotations

import argparse
import functools
import http.server
import os
import re

RANGE = re.compile(r"bytes=(\d*)-(\d*)$")


class RangeHandler(http.server.SimpleHTTPRequestHandler):
    mounts: dict[str, str] = {}

    def translate_path(self, path):
        clean = path.split("?", 1)[0].split("#", 1)[0]
        for prefix, folder in self.mounts.items():
            if clean == prefix or clean.startswith(prefix + "/"):
                rest = clean[len(prefix) :].lstrip("/")
                target = os.path.realpath(os.path.join(folder, *rest.split("/")))
                root = os.path.realpath(folder)
                # A request may not climb out of the mounted folder.
                return (
                    target
                    if target == root or target.startswith(root + os.sep)
                    else os.path.join(root, "__denied__")
                )
        return super().translate_path(path)

    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".bin": "application/octet-stream",
        ".json": "application/json",
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".wasm": "application/wasm",
    }

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header(
            "Access-Control-Expose-Headers", "Content-Range, Content-Length, Accept-Ranges"
        )
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def do_OPTIONS(self):  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Headers", "Range")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.end_headers()

    def send_head(self):
        header = self.headers.get("Range")
        if not header:
            return super().send_head()
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()
        m = RANGE.match(header.strip())
        try:
            f = open(path, "rb")  # noqa: SIM115 - handed to copyfile and closed by the base class
        except OSError:
            self.send_error(404, "File not found")
            return None
        size = os.fstat(f.fileno()).st_size
        if not m or (m.group(1) == "" and m.group(2) == ""):
            f.close()
            self.send_error(416, "Unsupported range")
            return None
        if m.group(1) == "":
            start, end = max(0, size - int(m.group(2))), size - 1
        else:
            start = int(m.group(1))
            end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
        if start >= size or start > end:
            f.close()
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        return _Slice(f, end - start + 1)


class _Slice:
    """A file object that stops after n bytes, so copyfile sends only the range."""

    def __init__(self, f, n: int):
        self.f, self.left = f, n

    def read(self, size: int = -1) -> bytes:
        if self.left <= 0:
            return b""
        size = self.left if size < 0 else min(size, self.left)
        data = self.f.read(size)
        self.left -= len(data)
        return data

    def close(self):
        self.f.close()


def serve(
    directory: str, port: int, host: str = "127.0.0.1", mounts: dict[str, str] | None = None
) -> http.server.ThreadingHTTPServer:
    handler_class = type("MountedRangeHandler", (RangeHandler,), {"mounts": dict(mounts or {})})
    handler = functools.partial(handler_class, directory=directory)
    return http.server.ThreadingHTTPServer((host, port), handler)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=".")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument(
        "--mount", action="append", default=[], help="URL prefix and folder, e.g. /data=data/viewer"
    )
    args = ap.parse_args()
    mounts = dict(m.split("=", 1) for m in args.mount)
    server = serve(args.dir, args.port, mounts=mounts)
    print(f"serving {args.dir} on http://127.0.0.1:{args.port} with Range support")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
