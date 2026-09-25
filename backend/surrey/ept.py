"""Read an Entwine Point Tile octree over HTTP and hand back true-metre points.

Used against s3://usgs-lidar-public, which is US Government Public Domain and
answers anonymously. Never point this at s3://usgs-lidar without the `-public`
suffix: that bucket is Requester Pays and costs real money.

Two traps live in here, both of them quiet.

1. Hierarchy entries with count == -1 are not empty nodes. They are pointers to
   a further hierarchy page keyed by that node. Treating them as empty caps the
   walk at the first page and silently returns a fraction of the cloud.

2. USGS 3DEP EPT deliveries are published in EPSG:3857, Web Mercator, whose
   horizontal unit is not a metre. At San Francisco's latitude one Web Mercator
   unit is about 0.79 m on the ground, so raw XY is inflated by about 1.26x
   while Z stays in real metres. A model trained on a true-metre cloud sees
   every neighbourhood stretched sideways and flattened. Reprojecting is not
   cosmetic here, it decides whether the transfer result means anything, so
   `read_aoi` always returns a true-metre CRS and records which one.
"""

from __future__ import annotations

import io
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

USER_AGENT = "surrey/0.1 (portfolio research; github.com/markolivaic)"
PUBLIC_ROOT = "https://s3-us-west-2.amazonaws.com/usgs-lidar-public"


def assert_public_bucket(base: str) -> None:
    """Refuse the Requester Pays bucket, which differs by one suffix and bills.

    s3://usgs-lidar and s3://usgs-lidar-public hold the same kind of data and
    only one of them is free. The check is on "usgs-lidar/" with the slash, so
    the public bucket passes and the paid one does not.
    """
    if "usgs-lidar/" in base:
        raise ValueError(
            "that is the Requester Pays bucket and it costs money, use usgs-lidar-public"
        )


def _get(url: str, timeout: float = 60.0, attempts: int = 4) -> bytes:
    """One public object, tried again after a dropped connection. A walk reads thousands of these, and
    one reset from the far end should not end it."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            if attempt == attempts - 1:
                raise
            time.sleep(2.0 * (attempt + 1))
    raise AssertionError("unreachable")


@dataclass(frozen=True)
class NodeKey:
    d: int
    x: int
    y: int
    z: int

    @classmethod
    def parse(cls, s: str) -> NodeKey:
        d, x, y, z = (int(v) for v in s.split("-"))
        return cls(d, x, y, z)

    def __str__(self) -> str:
        return f"{self.d}-{self.x}-{self.y}-{self.z}"

    def bounds(self, cube: np.ndarray) -> np.ndarray:
        """Bounding box of this node inside the octree's root cube."""
        step = (cube[3:] - cube[:3]) / (2**self.d)
        lo = cube[:3] + step * np.array([self.x, self.y, self.z], dtype=float)
        return np.concatenate([lo, lo + step])


@dataclass
class EptSource:
    base: str
    info: dict = field(default_factory=dict)

    @classmethod
    def open(cls, name_or_url: str) -> EptSource:
        base = name_or_url.rstrip("/")
        if not base.startswith("http"):
            base = f"{PUBLIC_ROOT}/{base}"
        assert_public_bucket(base)
        info = json.loads(_get(f"{base}/ept.json"))
        return cls(base=base, info=info)

    @property
    def cube(self) -> np.ndarray:
        return np.asarray(self.info["bounds"], dtype=float)

    @property
    def conforming(self) -> np.ndarray:
        return np.asarray(self.info["boundsConforming"], dtype=float)

    @property
    def epsg(self) -> int:
        return int(self.info["srs"]["horizontal"])

    @property
    def point_count(self) -> int:
        return int(self.info["points"])

    def walk(
        self,
        root: str = "0-0-0-0",
        max_depth: int | None = None,
        keep: Callable[[NodeKey], bool] | None = None,
    ) -> dict[NodeKey, int]:
        """Walk the hierarchy, following count == -1 pointers to further pages.

        `keep(NodeKey) -> bool` prunes whole subtrees. Pruning at the node level
        is what makes an area of interest cheap: a node that cannot overlap the
        area of interest has no descendants that can.
        """
        found: dict[NodeKey, int] = {}
        pages = [root]
        seen_pages: set[str] = set()
        while pages:
            page = pages.pop()
            if page in seen_pages:
                continue
            seen_pages.add(page)
            entries = json.loads(_get(f"{self.base}/ept-hierarchy/{page}.json"))
            for key_s, count in entries.items():
                key = NodeKey.parse(key_s)
                if max_depth is not None and key.d > max_depth:
                    continue
                if keep is not None and not keep(key):
                    continue
                if count == -1:
                    # A pointer to another hierarchy page, not an empty node.
                    pages.append(key_s)
                else:
                    found[key] = int(count)
        return found

    def node_points(self, key: NodeKey) -> np.ndarray:
        """Structured array of one node's points, in the EPT's own CRS."""
        import laspy

        raw = _get(f"{self.base}/ept-data/{key}.laz")
        las = laspy.read(io.BytesIO(raw))
        from .flags import flag_bits

        out = np.empty(
            len(las.points),
            dtype=[
                ("x", "f8"),
                ("y", "f8"),
                ("z", "f8"),
                ("intensity", "u2"),
                ("classification", "u1"),
                ("return_number", "u1"),
                ("number_of_returns", "u1"),
                ("flags", "u1"),
            ],
        )
        out["x"] = np.asarray(las.x)
        out["y"] = np.asarray(las.y)
        out["z"] = np.asarray(las.z)
        out["intensity"] = np.asarray(las.intensity)
        out["classification"] = np.asarray(las.classification)
        out["return_number"] = np.asarray(las.return_number)
        out["number_of_returns"] = np.asarray(las.number_of_returns)
        out["flags"] = flag_bits(las)
        return out


def overlaps_xy(node_bounds: np.ndarray, aoi: tuple[float, float, float, float]) -> bool:
    return not (
        node_bounds[3] <= aoi[0]
        or node_bounds[0] >= aoi[2]
        or node_bounds[4] <= aoi[1]
        or node_bounds[1] >= aoi[3]
    )


def true_metre_epsg(lon: float, lat: float) -> int:
    """UTM zone containing this longitude, northern or southern hemisphere."""
    zone = int((lon + 180.0) // 6.0) + 1
    return (32600 if lat >= 0 else 32700) + zone
