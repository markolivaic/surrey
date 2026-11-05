"""Turn one screen recording into an MP4, an animation for the README, and a poster.

The MP4 is H.264, yuv420p and faststart, at the recording's own size. The animation is an animated WebP
880 px wide, the width of the README column. The poster is the JPEG of the first frame. Each stays under
its limit, so the clone stays small. ffmpeg is located by backend/surrey/tools.py rather than assumed to
be on PATH.

Run:
    python scripts/pipeline/make_walkthrough.py --source recording.webm --keep 21.5
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

from surrey.tools import find_ffmpeg  # noqa: E402

OUT_DIR = ROOT / "docs" / "walkthrough"
STEM = "app-walkthrough"
DEFAULT_ARTIFACT = ROOT / "data" / "artifacts" / "walkthrough.json"

MP4_MAX_BYTES = 5_000_000
WEBP_MAX_BYTES = 5_000_000
# A point cloud is close to noise to an encoder, so no fixed quality fits a limit across clips. The MP4 is
# encoded in two passes at the bitrate that fills nine tenths of its limit over the clip's own length, and
# the animation at the highest of these qualities that comes in under its limit.
MP4_FILL = 0.9
WEBP_FPS = 8
WEBP_WIDTH = 880
WEBP_QUALITIES = (40, 35, 30, 25, 20)


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"ffmpeg failed:\n{' '.join(cmd)}\n{proc.stderr[-2000:]}")


def probe(ffmpeg: Path, path: Path) -> dict:
    """Stream details, read back from the file rather than assumed from the flags."""
    ffprobe = ffmpeg.with_name(ffmpeg.name.replace("ffmpeg", "ffprobe"))
    proc = subprocess.run(
        [str(ffprobe), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(f"ffprobe failed on {path}:\n{proc.stderr[-500:]}")
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
    """A repo-relative path where possible, the full one otherwise, for a trial run outside the repo."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


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
    ap.add_argument(
        "--keep",
        type=float,
        required=True,
        help="seconds kept from the end; what comes before is the page loading",
    )
    ap.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    args = ap.parse_args()

    if not args.source.exists():
        raise SystemExit(f"{args.source} does not exist")
    ffmpeg = find_ffmpeg()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    mp4 = args.out_dir / f"{args.stem}.mp4"
    webp = args.out_dir / f"{args.stem}.webp"
    poster = args.out_dir / f"{args.stem}-poster.jpg"
    print(f"ffmpeg  {ffmpeg}")

    total = probe(ffmpeg, args.source)["duration_s"]
    start = total - args.keep
    if start < 0:
        raise SystemExit(f"{args.source} holds {total} s, less than the {args.keep} s to keep")

    # The index at the front, so it plays on the first frame in a browser instead of after the whole
    # file arrives.
    kbps = int(MP4_MAX_BYTES * MP4_FILL * 8 / args.keep / 1000)
    passlog = args.out_dir / "_x264pass"
    common = ["-c:v", "libx264", "-preset", "slow", "-b:v", f"{kbps}k", "-pix_fmt", "yuv420p"]
    common += ["-passlogfile", str(passlog), "-an"]
    source = ["-ss", f"{start:.2f}", "-i", str(args.source)]
    run([str(ffmpeg), "-y", *source, *common, "-pass", "1", "-f", "mp4", os.devnull])
    run([str(ffmpeg), "-y", *source, *common, "-pass", "2", "-movflags", "+faststart", str(mp4)])
    for leftover in args.out_dir.glob("_x264pass*"):
        leftover.unlink()

    vf = f"fps={WEBP_FPS},scale={WEBP_WIDTH}:-2:flags=lanczos"
    quality = None
    for quality in WEBP_QUALITIES:
        run(
            [str(ffmpeg), "-y", *source, "-vf", vf, "-c:v", "libwebp_anim", "-q:v", str(quality)]
            + ["-compression_level", "6", "-loop", "0", "-an", str(webp)]
        )
        print(f"  animation at quality {quality}: {webp.stat().st_size / 1e6:.2f} MB")
        if webp.stat().st_size <= WEBP_MAX_BYTES:
            break

    run([str(ffmpeg), "-y", "-i", str(mp4), "-frames:v", "1", "-q:v", "3", str(poster)])

    info = probe(ffmpeg, mp4)
    result = {
        "source": repo_relative(args.source),
        "seconds_cut_from_the_front": round(start, 2),
        "mp4": {"path": repo_relative(mp4), "bytes": mp4.stat().st_size, "kbps": kbps, **info},
        "animation": {
            "path": repo_relative(webp),
            "bytes": webp.stat().st_size,
            "fps": WEBP_FPS,
            "width": WEBP_WIDTH,
            "quality": quality,
        },
        "poster": {"path": repo_relative(poster), "bytes": poster.stat().st_size},
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")

    for f in (mp4, webp, poster):
        print(f"  {f.name:34s} {f.stat().st_size / 1e6:>7.2f} MB")
    print(
        f"  {info['codec']} {info['pix_fmt']} {info['width']}x{info['height']} {info['duration_s']} s"
    )
    print(f"wrote {args.artifact}")
    if mp4.stat().st_size > MP4_MAX_BYTES or webp.stat().st_size > WEBP_MAX_BYTES:
        print("an output is over its limit")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
