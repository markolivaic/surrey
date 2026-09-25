"""Octree geometry and the guards around it. No network in here."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.ept import (  # noqa: E402
    PUBLIC_ROOT,
    EptSource,
    NodeKey,
    assert_public_bucket,
    overlaps_xy,
    true_metre_epsg,
)

CUBE = np.array([0.0, 0.0, 0.0, 1024.0, 1024.0, 1024.0])


def test_node_key_round_trips_through_its_string_form():
    assert str(NodeKey.parse("3-5-6-7")) == "3-5-6-7"
    assert NodeKey.parse("3-5-6-7") == NodeKey(3, 5, 6, 7)


def test_the_root_node_covers_the_whole_cube():
    assert NodeKey(0, 0, 0, 0).bounds(CUBE).tolist() == CUBE.tolist()


def test_depth_one_splits_the_cube_into_octants():
    assert NodeKey(1, 0, 0, 0).bounds(CUBE).tolist() == [0, 0, 0, 512, 512, 512]
    assert NodeKey(1, 1, 1, 1).bounds(CUBE).tolist() == [512, 512, 512, 1024, 1024, 1024]


def test_children_of_a_node_tile_it_without_gaps_or_overlap():
    parent = NodeKey(2, 1, 1, 1)
    pb = parent.bounds(CUBE)
    volume = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                c = NodeKey(3, parent.x * 2 + dx, parent.y * 2 + dy, parent.z * 2 + dz)
                cb = c.bounds(CUBE)
                assert cb[0] >= pb[0] and cb[3] <= pb[3]
                volume += float(np.prod(cb[3:] - cb[:3]))
    assert volume == pytest.approx(float(np.prod(pb[3:] - pb[:3])))


def test_overlaps_xy_ignores_z_and_excludes_touching_edges():
    node = np.array([0.0, 0.0, -500.0, 10.0, 10.0, 500.0])
    assert overlaps_xy(node, (5.0, 5.0, 15.0, 15.0))
    assert not overlaps_xy(node, (20.0, 20.0, 30.0, 30.0))
    # A node whose edge only touches the area contributes no interior points.
    assert not overlaps_xy(node, (10.0, 0.0, 20.0, 10.0))


def test_a_deep_node_far_from_the_area_is_pruned():
    aoi = (0.0, 0.0, 8.0, 8.0)
    assert overlaps_xy(NodeKey(7, 0, 0, 0).bounds(CUBE), aoi)
    assert not overlaps_xy(NodeKey(7, 60, 60, 0).bounds(CUBE), aoi)


def test_true_metre_epsg_picks_the_utm_zone():
    assert true_metre_epsg(-122.42, 37.77) == 32610  # San Francisco
    assert true_metre_epsg(-122.85, 49.19) == 32610  # Surrey, same zone
    assert true_metre_epsg(15.98, 45.81) == 32633  # Zagreb
    assert true_metre_epsg(151.2, -33.87) == 32756  # Sydney, southern


def test_the_requester_pays_bucket_is_refused():
    # The paid bucket differs from the free one by a suffix and charges real money.
    with pytest.raises(ValueError, match="Requester Pays|costs money"):
        assert_public_bucket("https://s3-us-west-2.amazonaws.com/usgs-lidar/CA_SanFrancisco_1_B23")
    with pytest.raises(ValueError):
        assert_public_bucket("s3://usgs-lidar/CA_SanFrancisco_1_B23")


def test_the_public_bucket_is_not_refused():
    # A guard that rejected every URL would pass the test above on its own. This
    # is the other half of the rule: the free bucket has to get through.
    assert_public_bucket(f"{PUBLIC_ROOT}/CA_SanFrancisco_1_B23")
    assert_public_bucket("https://s3-us-west-2.amazonaws.com/usgs-lidar-public/AK_Fairbanks")
    assert "usgs-lidar-public" in PUBLIC_ROOT


def test_opening_a_paid_bucket_fails_before_any_request():
    # EptSource.open must refuse without touching the network. If it reached the
    # fetch, this would raise a URL error rather than a ValueError.
    with pytest.raises(ValueError):
        EptSource.open("https://s3-us-west-2.amazonaws.com/usgs-lidar/CA_SanFrancisco_1_B23")


def test_hierarchy_pointer_entries_are_not_treated_as_empty_nodes():
    # count == -1 means "there is another hierarchy page keyed by this node".
    # Reading it as an empty node caps the walk at the first page. This drives the
    # walk with a fake fetcher so the behaviour is pinned without the network.
    pages = {
        "0-0-0-0": {"0-0-0-0": 100, "1-0-0-0": -1, "1-1-0-0": 7},
        "1-0-0-0": {"1-0-0-0": 50, "2-0-0-0": 3},
    }
    src = EptSource(
        base="test://",
        info={
            "bounds": CUBE.tolist(),
            "boundsConforming": CUBE.tolist(),
            "srs": {"horizontal": 3857},
            "points": 160,
        },
    )
    import surrey.ept as ept

    original = ept._get
    ept._get = lambda url, timeout=60.0: (
        __import__("json").dumps(pages[url.split("/")[-1].removesuffix(".json")]).encode()
    )
    try:
        found = src.walk()
    finally:
        ept._get = original

    keys = {str(k) for k in found}
    assert "1-0-0-0" in keys, "the pointer node itself carries points on its own page"
    assert "2-0-0-0" in keys, "the second page was never read"
    assert found[NodeKey.parse("2-0-0-0")] == 3
    assert -1 not in found.values()
