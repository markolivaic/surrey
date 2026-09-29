"""Turn one screen recording into an MP4, a GIF and a poster.

The MP4 is H.264, yuv420p and faststart, under 5 MB; the GIF is under 8 MB and links to it; the poster
is the JPEG of the first frame. The media is one uninterrupted recording of the running page. ffmpeg is
located by backend/surrey/tools.py rather than assumed to be on PATH.

Run:
    python scripts/pipeline/make_walkthrough.py --source recording.webm
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.tools import find_ffmpeg  # noqa: E402

OUT_DIR = ROOT / "docs" / "walkthrough"
STEM = "app-walkthrough"
DEFAULT_ARTIFACT = ROOT / "data" / "artifacts" / "walkthrough.json"

MP4_MAX_BYTES = 5_000_000
GIF_MAX_BYTES = 8_000_000
# A point cloud is close to noise to an encoder, so neither file fits its limit at the usual settings:
# at CRF 26 the first recording of the page came to 13 MB, and a 960 wide GIF at 12 frames a second to
# 34 MB. The MP4 is therefore encoded in two passes at the bitrate that fills nine tenths of its limit
# over the clip's own length, and the GIF is drawn smaller, slower and with fewer colours, undithered.
MP4_FILL = 0.9
GIF_FPS = 8
GIF_WIDTH = 640
GIF_COLOURS = 96

MAGIC = {
    ".mp4": lambda b: b[4:8] == b"ftyp",
    ".gif": lambda b: b[:6] in (b"GIF89a", b"GIF87a"),
    ".jpg": lambda b: b[:3] == b"\xff\xd8\xff",
}


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"ffmpeg failed:\n{' '.join(cmd)}\n{proc.stderr[-2000:]}")


def probe(ffmpeg: Path, path: Path) -> dict:
    """Stream details, read back from the file rather than assumed from the flags."""
    ffprobe = ffmpeg.with_name(ffmpeg.name.replace("ffmpeg", "ffprobe"))
    if not ffprobe.exists():
        return {}
    proc = subprocess.run(
        [str(ffprobe), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return {}
    doc = json.loads(proc.stdout)
    video = next((s for s in doc.get("streams", []) if s.get("codec_type") == "video"), {})
    return {
        "codec": video.get("codec_name"),
        "pix_fmt": video.get("pix_fmt"),
        "width": video.get("width"),
        "height": video.get("height"),
        "duration_s": round(float(doc.get("format", {}).get("duration", 0.0)), 2),
    }


def repo_relative(path: Path) -> str:
    """A repo-relative path where possible, the full one otherwise.

    Outputs usually land in docs/walkthrough, but a trial run writes to a scratch
    directory outside the repository and relative_to raises there.
    """
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def check(path: Path, limit: int | None = None) -> dict:
    blob = path.read_bytes()[:16]
    ok = MAGIC[path.suffix](blob)
    size = path.stat().st_size
    return {
        "path": repo_relative(path),
        "bytes": size,
        "magic_ok": bool(ok),
        "within_limit": None if limit is None else bool(size <= limit),
        "limit_bytes": limit,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source",
        type=Path,
        required=True,
        help="the raw recording, for example a Playwright webm",
    )
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--stem", default=STEM)
    ap.add_argument("--gif-fps", type=int, default=GIF_FPS)
    ap.add_argument("--gif-width", type=int, default=GIF_WIDTH)
    ap.add_argument("--gif-colours", type=int, default=GIF_COLOURS)
    ap.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="seconds cut from the front: the page loading, before its first screen has drawn",
    )
    ap.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    args = ap.parse_args()

    if not args.source.exists():
        raise SystemExit(f"{args.source} does not exist")
    ffmpeg = find_ffmpeg()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    mp4 = args.out_dir / f"{args.stem}.mp4"
    gif = args.out_dir / f"{args.stem}.gif"
    poster = args.out_dir / f"{args.stem}-poster.jpg"

    t0 = time.time()
    print(f"ffmpeg  {ffmpeg}")

    # H.264 in yuv420p with the index at the front, so it plays on the first frame
    # in a browser instead of after the whole file arrives. Two passes at the bitrate the limit allows.
    seconds = (probe(ffmpeg, args.source).get("duration_s") or 0.0) - args.start
    if seconds <= 0:
        raise SystemExit(f"cannot read the length of {args.source}")
    kbps = int(MP4_MAX_BYTES * MP4_FILL * 8 / seconds / 1000)
    passlog = args.out_dir / "_x264pass"
    common = ["-c:v", "libx264", "-preset", "slow", "-b:v", f"{kbps}k", "-pix_fmt", "yuv420p"]
    common += ["-passlogfile", str(passlog), "-an"]
    source = ["-ss", f"{args.start:.2f}", "-i", str(args.source)]
    run([str(ffmpeg), "-y", *source, *common, "-pass", "1", "-f", "mp4", os.devnull])
    run([str(ffmpeg), "-y", *source, *common, "-pass", "2"] + ["-movflags", "+faststart", str(mp4)])
    for leftover in args.out_dir.glob("_x264pass*"):
        leftover.unlink()

    # Two passes for the GIF: build a palette from the whole clip, then map to it.
    # One pass picks 256 colours from the first frame and the rest of the clip
    # banding is then permanent.
    palette = args.out_dir / "_palette.png"
    vf = f"fps={args.gif_fps},scale={args.gif_width}:-1:flags=lanczos"
    run(
        [
            str(ffmpeg),
            "-y",
            "-i",
            str(mp4),
            "-vf",
            f"{vf},palettegen=max_colors={args.gif_colours}:stats_mode=diff",
            str(palette),
        ]
    )
    run(
        [
            str(ffmpeg),
            "-y",
            "-i",
            str(mp4),
            "-i",
            str(palette),
            "-lavfi",
            f"{vf} [x]; [x][1:v] paletteuse=dither=none:diff_mode=rectangle",
            str(gif),
        ]
    )
    palette.unlink(missing_ok=True)

    run([str(ffmpeg), "-y", "-i", str(mp4), "-frames:v", "1", "-q:v", "3", str(poster)])

    files = [check(mp4, MP4_MAX_BYTES), check(gif, GIF_MAX_BYTES), check(poster)]
    result = {
        # Where the recording and the encoder were, without the machine's own absolute paths.
        "source": repo_relative(args.source),
        "ffmpeg": ffmpeg.name,
        "ffmpeg_on_path": False,
        "mp4": {**files[0], **probe(ffmpeg, mp4)},
        "gif": files[1],
        "poster": files[2],
        "gif_fps": args.gif_fps,
        "gif_width": args.gif_width,
        "gif_colours": args.gif_colours,
        "seconds_cut_from_the_front": args.start,
        "mp4_kbps": kbps,
        "seconds": round(time.time() - t0, 1),
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")

    for f in files:
        state = "ok" if f["magic_ok"] else "BAD MAGIC"
        if f["within_limit"] is False:
            state = f"OVER LIMIT ({f['limit_bytes'] / 1e6:.0f} MB)"
        print(f"  {Path(f['path']).name:34s} {f['bytes'] / 1e6:>7.2f} MB  {state}")
    print(
        f"  codec {result['mp4'].get('codec')}  pix_fmt {result['mp4'].get('pix_fmt')}  "
        f"{result['mp4'].get('width')}x{result['mp4'].get('height')}  "
        f"{result['mp4'].get('duration_s')} s"
    )
    print(f"wrote {args.artifact}")

    bad = [f for f in files if not f["magic_ok"] or f["within_limit"] is False]
    if bad:
        print("\nsome outputs do not meet the standard")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
