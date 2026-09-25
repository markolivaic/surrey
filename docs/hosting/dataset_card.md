---
license: other
license_name: per-file
license_link: LICENSE_MODEL.md
pretty_name: Point cloud octrees for a Surrey-trained lidar segmentation model viewed over San Francisco
tags:
  - lidar
  - point-cloud
  - potree
size_categories:
  - n<1K
---

# Viewer data: San Francisco and one Surrey tile as octrees

These files are what a static web page streams by HTTP byte range to draw lidar points in 3D.
Each area is one folder holding a PotreeConverter 2.1.3 octree with Brotli encoding
(`metadata.json`, `hierarchy.bin`, `octree.bin`) and a `manifest.json` with the SHA-256 of every
file, the attributes, the variants and where the points came from.

They are built locally by `scripts/pipeline/build_octree.py` in the project repository from public
sources and saved predictions, so a clone never depends on this copy.

## What is in a point

- position, metres, UTM zone 10N for San Francisco
- colour: NAIP 2022 at the point, San Francisco only
- for each of up to four model variants, the predicted class and the confidence of that
  prediction, packed into standard LAS fields (the manifest's `slots` names them)

The colours a page draws from the class and confidence are predictions of a model trained on
lidar from Surrey, British Columbia. They are not measurements and not ground truth.

## Licence, per file

| Files | Content | Terms |
|---|---|---|
| San Francisco folders: positions | USGS 3DEP delivery `CA_SanFrancisco_1_B23` | United States Government public domain |
| San Francisco folders: colour | NAIP 2022 imagery, USDA Farm Production and Conservation | United States Government public domain |
| Every folder: class and confidence | predictions made by this project's models | CC BY-NC 4.0, non-commercial, in `LICENSE_MODEL.md` beside this card |
| Surrey folder: positions | held-out tile of the Hugging Face mirror mbendjilali/DALES-2, lidar flown over Surrey | Open Government Licence, City of Surrey: "Contains information licensed under the Open Government License – City of Surrey." |

The Surrey folder holds points and predictions only. It contains no DALES annotation and no mbendjilali/DALES-2 label:
the build never reads the label array, and `scripts/pipeline/check_octree_decode.py` confirms the class
field equals the model's prediction on every decoded point. The upstream DALES annotations carry
"Creative Commons 3.0" with a separate non-commercial restriction, which is why they are not here.

The predictions, and the model weights behind them, have their own licence, separate from the
MIT licence of the project's code: CC BY-NC 4.0, set out in `LICENSE_MODEL.md` in this repository.
The reason is upstream. The models were trained on mbendjilali/DALES-2 labels, whose DALES layer
carries that non-commercial restriction; M0prime and M1 also on swissSURFACE3D (source: Federal
Office of Topography swisstopo, ©swisstopo), with the BC Freshwater Atlas used to leave DALES water
out of their training:

> Contains information licensed under the Open Government Licence – British Columbia.

Commercial use of the predictions needs the upstream restriction settled first, as
`LICENSE_MODEL.md` says. The project's THIRD_PARTY_NOTICES.md sets out every layer.

No NOAA data is stored here. The water null is not a do-nothing baseline: it knows each point's absolute height in NAVD88 and the sea level at the moment of flight. The model knows neither; it sees height above the floor of a 40 m block. The sea level used by the project's water null comes from NOAA
CO-OPS station 9414290 and is credited on the page that uses it.
