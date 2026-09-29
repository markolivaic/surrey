---
license: cc-by-nc-4.0
library_name: pytorch
pipeline_tag: other
tags:
  - lidar
  - point-cloud
  - semantic-segmentation
---

# surrey: a PointNet++ style lidar segmentation model trained on DALES-2

Weights of the models in the project repository, github.com/markolivaic/surrey. The README there
gives every figure, each with its null and its seed range; this card gives none, so it cannot drift
from them.

## Files

| File | What it is |
|---|---|
| `m0_seedA.pt` | The published model, M0 seed A: the one the page and the README's cards show |
| `m0_seedB.pt`, `m0_seedC.pt` | The same recipe with two other seeds, the range behind every figure |
| `m0_big_seedA.pt` | The capacity control, a wider network with the same recipe; not adopted |

Each file holds the network's tensors and a few plain values (channels, width, recipe, seed, step and
the first 16 hex digits of the source training checkpoint's SHA-256). It loads with
`torch.load(..., weights_only=True)`, which is what `surrey.inference.load_checkpoint` does by default.

## Input and output

- Input: points in 40 m blocks, three values each: x and y centred on the block, z above the block's
  floor, in metres. No intensity, no return numbers, no colour.
- Output: fifteen DALES-2 classes per point. The class names and their order are in
  `backend/surrey/data.py` of the repository.
- Inference averages the softmax over several seeded passes, each drawing its own sample centroids
  and its own dealing of points into blocks (`backend/surrey/inference.py`).

## Training data and what it does not know

Trained on DALES-2 (Bendjilali and others, 2026), aerial lidar flown over Surrey, British Columbia,
Canada. The labels have no class for water or bridges, so the model has no way to call either.

Run on USGS 3DEP lidar over San Francisco, the model's classes are predictions, not measurements, and
not part of any USGS deliverable. How far they hold there, and where they do not, is the subject of the
repository.

## Licence

CC BY-NC 4.0. Every model here learned from the DALES-2 labels, which revise the DALES annotations,
whose terms exclude commercial use; the restriction is passed on. `LICENSE_MODEL.md` in the repository
sets out why and what to credit.
