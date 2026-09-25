# PotreeConverter 2.1.3: LAS extra bytes come back wrong

A reproduction kept for the record. It has not been reported upstream.

## What happens

`scripts/investigations/repro_potreeconverter_extra_bytes.py` writes a LAS 1.4 point format 7 file of 200,000
synthetic points (no project data) with N uint8 extra byte attributes, attribute k holding
10 times (k + 1) on every point, converts it with `--encoding UNCOMPRESSED --attributes rgb x0 ...`,
and reads the extra attributes back from the first node of the octree. Results on Windows x64,
in `docs/potreeconverter_extra_bytes.json`:

| extra attributes | written (laspy reads them back) | metadata min and max | read from the octree |
|---|---|---|---|
| 2 | 10, 20 | 10, 20 | 20, 0 |
| 4 | 10, 20, 30, 40 | 10, 20, 30, 40 | 40, 0, 0, 0 |

The first extra attribute comes back holding another attribute's value and the rest come back
as zeros. The metadata still reports the right minimum and maximum for each attribute, so
nothing looks wrong until a node is decoded. An earlier test on real points, requesting three of
four extra attributes, returned the third attribute's values in the first slot, so the offset
depends on how many extra attributes the file has and how many are requested. The same happens
with `--encoding BROTLI`.

## A crash beside it

The same kind of file at 20,000 points with two extra attributes made the converter exit with
0xC0000005 (access violation); without extra attributes it converted. On project data a 9.0
million point file crashed while chunking whenever extra attributes were requested, three runs
out of three, and converted when split into two files.

## What surrey does instead

Predictions travel in standard LAS fields, which come back exact on a file of known values:
classification and user data for the first variant, and intensity, point source id and scan
angle, class in the low byte and confidence in the high byte, for the next three
(`scripts/pipeline/build_octree.py`). That caps the viewer at four variants
(`docs/design/directions.md`). Every built octree is checked point by point against the cache
and the prediction files by `scripts/pipeline/check_octree_decode.py`.
