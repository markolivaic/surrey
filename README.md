# surrey

A point cloud segmentation model trained on lidar flown over Surrey, British Columbia, run on San Francisco and scored there against references that owe nothing to it.

[![ci](https://github.com/markolivaic/surrey/actions/workflows/ci.yml/badge.svg)](https://github.com/markolivaic/surrey/actions/workflows/ci.yml)

![The first screen of the page: the Sunset in 3D, coloured by the published model's classes, beside the lidar standard's class table](docs/design/readme/first_screen.webp)

![Cards 1 and 3 side by side: Golden Gate Park under the published model, where the trees are called vegetation, and under the old model with its input as delivered, where the forest is called building](docs/design/readme/cards_1_and_3.webp)

The colours are the model's predictions, not measurements. Left, the published model; right, the old model with the input fault described below.

## What this is

- A PointNet++ style network trained on DALES-2 (Bendjilali and others, 2026), a re-annotated aerial lidar benchmark in fifteen classes, flown over Surrey, British Columbia, Canada, not over Dayton.
- The published model, M0 seed A, run on eight areas of the USGS 3DEP San Francisco delivery and scored per area against the delivery's ground class, building footprints and NAIP vegetation, each beside its null.
- Three training seeds, every figure given as the median with the lowest and highest seed.
- A static web page that streams the real points, with a divider between the aerial photograph and the model's classes.

## What this is NOT

- **Not ground truth.** The colours on San Francisco are predictions of a model trained in another country, not measurements and not part of any USGS deliverable.
- **Not a labelled benchmark in San Francisco.** None exists for this delivery. Ground is scored against a vendor's automated class, buildings against 2D footprints of uncertain age, vegetation against an NDVI mask.
- **Not an untouched test.** San Francisco was also the development set: the input fault below was found from its figures.
- **Not a model that knows water or bridges.** The training labels have no word for either.
- **Not the full page from a fresh clone.** The real point clouds are built by the pipeline. A clone shows the page on a synthetic FIXTURE octree, labelled FIXTURE on screen.
- **Not new.** Airborne lidar segmentation losing accuracy from one city to another is known (Xie and others, 2021), DALES has been a source domain before (Wang and others, 2024), and softmax confidence is a known weak signal for unfamiliar inputs (Hendrycks and Gimpel, 2017). This is a careful measurement of one case, and a page to see it.

## Walkthrough

[![The walkthrough: the first screen, the six cards and the free view with the divider](docs/walkthrough/app-walkthrough.gif)](docs/walkthrough/app-walkthrough.mp4)

One uninterrupted recording of the running page: the first screen, the six cards, and the free view with the divider dragged across. Every state transition is rendered by the running application. No screen is mocked. Its first frame is the [poster](docs/walkthrough/app-walkthrough-poster.jpg).

## At home, on held-out Surrey

On the 11 held-out Surrey tiles, M0 reaches a point accuracy of 0.9285 (0.9273 to 0.9288), against 0.4797 for calling everything ground. Mean IoU is 0.5233 (0.52 to 0.527) over the fifteen classes and 0.8162 (0.8125 to 0.8164) over the six coarse groups the page draws.

On the original DALES, KPConv reached a mean IoU of 0.811 (Varney and others, 2020), over eight classes of a different annotation; here the classes are fifteen and the points voxel-thinned, so the two numbers rank nothing.

## San Francisco

Median over three seeds, the lowest and highest in brackets, each class beside the figure that catches the opposite mistake.

| Area | NAIP vegetation cores called vegetation | NAIP not-vegetation cores called vegetation | Delivery ground called ground | Ground precision, floor | Ground precision, ceiling |
|---|---|---|---|---|---|
| Golden Gate Park | 0.9722 (0.9704 to 0.9724) | 0.8468 (0.8411 to 0.853) | 0.9425 (0.9366 to 0.9779) | 0.7445 (0.743 to 0.7507) | 0.9665 (0.9661 to 0.9687) |
| Sunset | 0.9465 (0.9418 to 0.9471) | 0.036 (0.0331 to 0.0361) | 0.9647 (0.9588 to 0.9744) | 0.7466 (0.7442 to 0.7474) | 0.9619 (0.9595 to 0.9641) |
| Mission | 0.9828 (0.9803 to 0.9837) | 0.0328 (0.0275 to 0.033) | 0.9692 (0.9582 to 0.9872) | 0.7185 (0.6916 to 0.7228) | 0.8828 (0.8494 to 0.8881) |
| Russian Hill | 0.9313 (0.9201 to 0.9318) | 0.0442 (0.0423 to 0.0447) | 0.9139 (0.9028 to 0.9354) | 0.7724 (0.7689 to 0.7745) | 0.952 (0.9457 to 0.9536) |
| SoMa | 0.9854 (0.9834 to 0.9862) | 0.0313 (0.0281 to 0.0316) | 0.9922 (0.9828 to 0.9958) | 0.6924 (0.6746 to 0.6966) | 0.8414 (0.8197 to 0.8461) |
| Financial District | 0.9969 (0.9962 to 0.997) | 0.116 (0.1139 to 0.1245) | 0.9784 (0.9726 to 0.9861) | 0.6583 (0.6505 to 0.6626) | 0.9102 (0.8989 to 0.9157) |
| Embarcadero | 0.9985 (0.9985 to 0.9987) | 0.05 (0.0461 to 0.0518) | 0.8318 (0.8095 to 0.8722) | 0.3489 (0.3447 to 0.3627) | 0.5321 (0.5263 to 0.5533) |
| Bay Bridge | no core | 0.0312 (0.0276 to 0.0379) | no core | no core | no core |

Buildings, where the footprint null calls every point more than 2 m up a building. Lift is the share of the room above the null's precision that the model takes.

| Area | Footprints scored | Building lift | Precision against footprints | The null's precision | Footprint covered | The null's coverage |
|---|---|---|---|---|---|---|
| Sunset | 33 | 0.6699 (0.6634 to 0.6765) | 0.8482 (0.8452 to 0.8512) | 0.5401 (0.5401 to 0.5401) | 0.9877 (0.9877 to 0.9896) | 0.9936 (0.9936 to 0.9936) |
| Mission | 56 | 0.6136 (0.6023 to 0.6151) | 0.8317 (0.8268 to 0.8323) | 0.5644 (0.5644 to 0.5644) | 0.9776 (0.9776 to 0.9777) | 0.9937 (0.9937 to 0.9937) |
| Russian Hill | 77 | 0.5579 (0.5531 to 0.5623) | 0.8333 (0.8316 to 0.835) | 0.623 (0.623 to 0.623) | 0.984 (0.9834 to 0.9844) | 0.9935 (0.9935 to 0.9935) |
| SoMa | 26 | 0.5621 (0.5594 to 0.5631) | 0.8455 (0.8446 to 0.8459) | 0.6472 (0.6472 to 0.6472) | 0.722 (0.7203 to 0.7222) | 0.7511 (0.7511 to 0.7511) |
| Financial District | 23 | 0.1456 (0.1452 to 0.1972) | 0.7234 (0.7232 to 0.7401) | 0.6762 (0.6762 to 0.6762) | 0.8248 (0.8222 to 0.8354) | 0.8541 (0.8541 to 0.8541) |
| Embarcadero | 6 | 0.0443 (0.0382 to 0.0499) | 0.2034 (0.1984 to 0.2081) | 0.1666 (0.1666 to 0.1666) | 0.8506 (0.8506 to 0.8541) | 0.8749 (0.8749 to 0.8749) |

Water is called ground, 0.9618 (0.9089 to 0.9724) at Embarcadero, and the Bay Bridge deck building, 0.9983 (0.9982 to 0.9984). Confidence separates these mistakes from right ground less well than mistakes at home, AUROC 0.7404 against 0.9375, on 3 of 3 seeds.

## Why the building lift moved

Against the first round, the building lift fell in the Mission, on Russian Hill, in SoMa and in the Financial District, where the retrained model covers more of each footprint, less precisely: a trade, by a reading written beforehand. The new block iterator moved it, not the voting ([docs/method.md](docs/method.md)).

## The Financial District, the one area outside the range

An area lies outside what training showed when its median 40 m block spans more height than the training blocks' 99th percentile, 46.19 m. The Financial District's spans 99.82 m, no other area's more than 34.53 m. Buildings were expected to score lower here, and do on 3 of 3 seeds, lift 0.1456 (0.1452 to 0.1972) against 0.5884 (0.5808 to 0.5886).

## The input that broke the first model

The first model read a fourth input, the return ratio. The return fields of DALES-2 as published on Hugging Face hold 0, so it was 0 on all 395,996,290 prepared DALES-2 points, and real on all but 3 of 43,938,643 San Francisco points. Given that input as delivered, the old model called the forest building:

| Area | NAIP vegetation cores called vegetation, input as delivered | the same, return ratio set to 0 |
|---|---|---|
| Golden Gate Park | 0.0002 | 0.9707 |
| Sunset | 0.0 | 0.9475 |
| Mission | 0.0 | 0.9822 |
| Russian Hill | 0.0 | 0.9303 |
| SoMa | 0.0 | 0.9859 |
| Financial District | 0.0 | 0.9971 |
| Embarcadero | 0.0 | 0.9984 |

The published model has three inputs; `tests/integrity/test_input_channels.py` fails any model with an input constant in its training data.

## Water, and the old M1

M1, the first round's attempt at water, learned it from Swiss tiles where 0.8586 to 0.9999 of the labelled water is synthetic fill. Real water there holds at most a median of 3.16 points per square metre, against 28.1 at Embarcadero. Its lift over the water null is negative on every seed, and it is not trained again ([docs/method.md](docs/method.md)).

## A bigger model

A control with 11,505,423 parameters beats the highest small seed in 0 of 6 areas with footprints, where the rule written beforehand asked for 4 ([docs/method.md](docs/method.md)).

## Reading the figures

A figure moved only when every seed of one model lies beyond every seed of the other ([docs/method.md](docs/method.md) gives how often chance does that). Before the first commit, the order of rules and results is self-reported; from the first commit on, the git history shows it.

## Why this exists

Most lidar segmentation results are reported on tiles from the same survey the model learned from. That tells you how well it learned one place. It does not tell you what happens when the same model is pointed at a city it has never seen, which is what anyone who wants to use it will do.

I wanted to see that gap on real data, not on a benchmark split. Surrey is where the labels are; San Francisco has a public national lidar delivery and no labels at all. So the question became how to measure a model where nothing is labelled, without fooling myself. Each class is scored against something the model never touched: the delivery's own ground class, building footprints, vegetation from aerial imagery. Each figure stands beside a null, and the rules for reading it were written before the numbers.

This would be wrong if the references are wrong, and they are imperfect: the footprints sit off the lidar, the vegetation mask is a colour index, and San Francisco is also where the input fault was found. Those limits are written down below, next to the numbers they affect.

## The name

Surrey, British Columbia, is where the lidar was flown; every picture is San Francisco. DALES, which DALES-2 re-annotates, is the Dayton Annotated LiDAR Earth Scan, named after the University of Dayton, where it was annotated, and not flown over Dayton either.

## How it works

```mermaid
flowchart LR
  D[DALES-2, Surrey tiles] --> PT[prep_tiles.py]
  PT --> TR[train.py: M0, three seeds, and the large control]
  F[USGS 3DEP San Francisco] --> PF[fetch_sf_aoi.py, prep_tiles.py]
  TR --> PR[predict_cache.py, the mean of the voted passes]
  PF --> PR
  PR --> SC[score_variant.py, against references and nulls]
  SC --> RG[build_quantities.py: data/artifacts/quantities.json]
  RG --> RM[render_readme.py: this README and docs/method.md]
  RG --> PG[page_figures.py: the page's figures]
  PR --> OC[build_octree.py: octrees]
  OC --> WEB[web/: the page]
  PG --> WEB
```

| Path | What it holds |
|---|---|
| `backend/surrey/` | The model, the block cache, inference and voting, the metrics, the palette and its rules, the registry of every quantity (`quantities.py`) |
| `scripts/pipeline.py` | Every step from the public sources to this README and the page, in order, with what each reads and writes |
| `scripts/pipeline/`, `scripts/investigations/` | The steps, and the measurements that each answered one question |
| `data/artifacts/` | The committed artefacts every figure is read from; `quantities.json` is the registry over all of them |
| `docs/readme.template.md`, `docs/method.template.md` | This README and the method notes with each figure named by registry key |
| `web/` | The page: its own three.js loader for Potree 2 octrees, the eye-dome pass, the cards, the free view |
| `tests/unit/`, `tests/integrity/`, `tests/smoke/` | Behavioural tests, guards over documents and artefacts, and the built page in a real browser |
| `docs/phase4_criteria.md` | The retraining's rules, written before it ran |

## Quick start

A fresh clone runs the tests and shows the page on the FIXTURE octree, with no key or account. It needs
Python 3.12 and Node 24.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux and macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest tests/unit tests/integrity -q
cd web
npm ci
npm run build
cd ..
python -m playwright install chromium
python -m pytest tests/smoke -q
python scripts/pipeline/serve_page.py
```

`serve_page.py` prints the fixture page, `index.html?fixture=1`, and the real page, which needs the
octrees the pipeline builds. The tests use the CPU build of torch 2.9.1; the models were trained with
torch 2.11 on NVIDIA cards, each recorded in its training artefact.

```bash
python scripts/pipeline.py              # the steps, with what each reads and writes
python scripts/pipeline.py --check      # which outputs are already on disk
python scripts/pipeline.py --run all    # all of it, in order; hours, and an NVIDIA card
```

PotreeConverter 2.1.3 and ffmpeg are looked for on PATH, or named in `.env.example`.

## Verification

Each row is a command a reader runs from the repository root. The results are what
`python scripts/pipeline/verify.py` printed in the working copy, on Windows 11, AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD, NVIDIA GeForce RTX 3050 Laptop GPU, Python 3.12.1, on
2026-09-26. A test is counted once however many cases a parametrisation makes of it.

| Check | Command | Result |
|---|---|---|
| behavioural tests | `python -m pytest tests/unit -q` | 166 tests; cases 175 passed |
| integrity tests | `python -m pytest tests/integrity -q` | 51 tests; cases 55 passed |
| README is generated | `python scripts/pipeline/render_readme.py --check` | passes |
| page figures | `python scripts/pipeline/page_figures.py` | passes |
| viewer tests | `npm test (in web/)` | 5 passed, 0 failed |
| viewer build | `npm run build (in web/)` | passes |
| page smoke test | `python -m pytest tests/smoke -q` | 1 test; cases 1 passed |

## Limitations and failure modes

- San Francisco was the development set as well as the test.
- The footprints sit about two metres off the lidar; part of the offset is not explained.
- In Golden Gate Park the park's false vegetation figure rests on only 13,847 points.
- The old four channel model, M0prime and M1 are from the first round and appear only on the bug and water cards.
- Frame rates were measured on one laptop.

## References

These references motivate the design; they do not validate the numbers here.

- Varney, N., Asari, V. K., Graehling, Q. DALES: A Large-Scale Aerial LiDAR Data Set for Semantic Segmentation. CVPR Workshops, 2020. arXiv 2004.11985.
- Singer, N. M., Asari, V. K. DALES Objects: A Large Scale Benchmark Dataset for Instance Segmentation in Aerial Lidar. IEEE Access 9, 97495 to 97504, 2021.
- Bendjilali, M., Peyran, C., Velumani, K., Mauri, A., Luminari, N., Alliez, P. DALES 2: A Renovated Aerial LiDAR Benchmark for 3D Scene Understanding. CVPR Workshops (USM3D), 2026.
- Xie, Y., Schindler, K., Tian, J., Zhu, X. X. Exploring Cross-City Semantic Segmentation of ALS Point Clouds. ISPRS Archives XLIII-B2-2021, 247 to 254, 2021.
- Wang, P., Yao, W., Shao, J., He, Z. Test-time adaptation for geospatial point cloud semantic segmentation with distinct domain shifts. arXiv 2407.06043, 2024.
- Hendrycks, D., Gimpel, K. A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks. ICLR, 2017.
- Qi, C. R., Yi, L., Su, H., Guibas, L. J. PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space. NeurIPS, 2017.
- ASPRS. LAS Specification 1.4 R15, 09 July 2019, Table 17, ASPRS Standard Point Classes.
- Boucheny, C. Interactive scientific visualization of large datasets: towards a perceptive-based approach. PhD thesis, Université Joseph Fourier, 2009. The source of eye-dome lighting.

## License

Code: MIT, in LICENSE. Model weights and predictions: CC BY-NC 4.0, in LICENSE_MODEL.md, because the labels' upstream DALES layer carries a non-commercial restriction. Data, the quoted class table and model inputs carry their own terms, in THIRD_PARTY_NOTICES.md.
