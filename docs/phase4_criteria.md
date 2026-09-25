# Phase 4: what runs, how it is read, written before any of it runs

Every model in `data/artifacts` today was predicted by an iterator that cut each block into contiguous
runs at the source's full density, with one pass per point. M1 learned water from Swiss points that
were almost all synthetic. Phase 4 trains M0 again, predicts every model again with both inference
faults fixed, and scores the lot. This file is the plan and the reading rules; the commit that adds it
is its date, and `data/artifacts/logs/gpu_chain_phase4.log` records each run against it.

## What changes for every prediction

- The iterator is `CachedTile.iter_full` as it is now: random partitions of the whole block, each point
  predicted once per pass, each tensor drawn as a training crop is (`tests/unit/test_cache.py`).
- Voting. The network draws its set abstraction centroids at random, so one pass over the same points
  gives a different class to a share of them from one seed to the next. A prediction is the softmax
  averaged over N passes, pass v seeded with v for both the centroid draws and the dealing of blocks
  (`surrey.inference.vote_probabilities`), the test-time voting point cloud segmentation models use.
- N is chosen by a rule, on the eleven held-out Surrey tiles and before any San Francisco figure
  exists (`scripts/pipeline/choose_votes.py`, into `data/artifacts/votes.json`): N is one of 1, 2, 4,
  8 and 16, the smallest at which the coarse classes of the N-pass vote and the 2N-pass vote agree on
  at least 0.99 of the held-out points, pooled, and 16 if none of 1, 2, 4 or 8 does. The new M0 seed A
  decides it. Every model is then predicted with that same N, the old ones included.
- No prepared tile holds a point with a dropped LAS flag (`backend/surrey/flags.py`): synthetic and
  withheld points are gone from every source.

## What is trained

| Run | Seeds | Data | Notes |
|---|---|---|---|
| M0 | A, B, C | the 29 DALES-2 training tiles | class weights from those 29 tiles only |
| M0 big | A | as M0 | the capacity control, width 512, batch 8, learning rate 0.004, 6000 steps, as the earlier large run (`docs/rented_gpu.md`) |

Before each long run, 200 steps of its configuration are timed on the card and the time goes into the
chain log before the run starts.

## What is predicted again without training

- The old four channel checkpoint, as delivered (`returns_ood`) and with the return ratio set to 0
  (`ret0`).
- The old M0prime and M1, three seeds each. They were trained on the laptop with class weights from all
  40 DALES-2 tiles and on Swiss tiles that still held their synthetic water. They stay as the record of
  what happened, and they are read only against each other: M1 against M0prime, never against the new
  M0. They are not scored on the Swiss tiles again, since those tiles no longer hold the water they were
  trained on; the earlier Swiss artefacts stay as they are.
- The other runs of the earlier rented round (`m0_big_2x_seedA`, `m0_big_lr2_seedA`,
  `m0_rented_small_seedA`, `m1_big_seedA`, `m1_densityaug_seedA`) are not run again. Their artefacts
  stay as the record of that round and are not set against a Phase 4 figure.

## Where each step runs

- On the card: `scripts/pipeline/rented_push.sh HOST PORT --phase4` sends the code, the committed
  artefacts, the San Francisco tiles as the laptop prepared them and the old checkpoints, each hashed
  (SHA-256) first. `scripts/pipeline/rented_setup.sh --m0-only` builds DALES-2 there.
  `scripts/pipeline/phase4_chain.sh` checks the hashes, trains, chooses N, predicts every model and
  hashes every checkpoint and prediction it leaves.
- Back on the laptop: `scripts/pipeline/rented_pull.sh HOST PORT` brings them back and checks every
  hash. A mismatch stops everything after it. The laptop scores.

## How each figure is read

Both sides, for every model: the eleven held-out Surrey tiles and the eight San Francisco areas.

- Vegetation: recall, the share of NAIP vegetation cores called vegetation, beside false vegetation,
  the share of NAIP not-vegetation cores called vegetation. Neither is quoted without the other.
- Ground: recall, beside the ground precision band (the floor and ceiling `score_variant.py` writes).
- Buildings: lift over the footprint null, with the null, the coverage and the number of footprints
  counted beside it.
- Mean IoU over the 15 classes on held-out Surrey, and over the coarse groups on both sides. Any
  comparison with published DALES results names that DALES-2 is a re-annotated benchmark, not the
  original DALES.
- Water and bridge deck, for the old M1 and M0prime: by `backend/surrey/m1_criteria.py` as written, on
  Embarcadero and Bay Bridge only, at the lower edge, the height edge and the polygon edge.

## Seeds, base rates and the old pair

- Every figure is the median over seeds with the minimum and maximum beside it, as
  `backend/surrey/seeds.py` says, and a claim that does not hold on every seed is written as holding on
  k of n.
- Every figure stands beside its base rate: the do-nothing or null figure for the same population, and
  the agreement between the N-pass and 2N-pass votes from `votes.json`.
- M1 against M0prime is read on every metric in `SF_SIDE_EFFECT_METRICS` and every Surrey class, by the
  drop rule in `backend/surrey/seeds.py`, and all of it is reported, not only water.
- The large model is read by the rule in `docs/rented_gpu.md`, with the three new small seeds' range
  beside it.

## One measurement without a model

The density of real water, measured returns only, on the Swiss tiles against the water San Francisco's
delivery labels, per area, on the source files and on the prepared tiles
(`scripts/investigations/real_water_density.py`, into `data/artifacts/real_water_density.json`). It
explains in the README why M1 is not trained again.

## Confounds, with the percentile fixed now

The confounds `scripts/pipeline/input_range_check.py` measures (block height range, ground relief,
building height) are read again on the new predictions. An area counts as outside what training
showed when its median block lies above the 99th percentile of the 29 training tiles' blocks for that
measure. The percentile is fixed here, before any Phase 4 figure exists.

## What stops the chain

A failed hash, a training run that records an interruption it cannot resume exactly, or a prepared
tile that fails `tests/integrity/test_prepared_flags.py`. Each is written to the chain log with the
time, and nothing after it runs until it is resolved.
