# Phase 6: the neighbour search in fp32, M0 trained again, written before any of it runs

This is the last technical round. After it the project is frozen: anything found later goes into the
README's limitations, not into another rented card. The commit that adds this file is its date.

## What was wrong

Training and prediction run under fp16 autocast. The neighbour search in `backend/surrey/model.py`
expanded a squared distance as |q|^2 + |r|^2 - 2 q.r, and under autocast the q.r product ran in half
precision. A point's z is its height above the floor of its 40 m block, up to about 200 m in the
Financial District, and at a square near 40,000 m^2 half precision steps in 32 m^2, while neighbours
15 cm apart differ by a fraction of a square metre. So the higher a point, the closer its neighbours
came to a random pick, in all six searches of a forward pass: three set abstraction levels (k = 32)
and three feature propagation levels (k = 3).

`scripts/investigations/precision_audit.py` measured it on the published Phase 4 checkpoint, on the
four tallest Financial District blocks and the two tallest Sunset blocks, against an exact pass with
autocast off and neighbours found in float64 (`data/artifacts/precision_audit.json`). On the Financial
District blocks the Phase 4 model's predicted class matched the exact pass on 0.5886 of the points and
on 0.1343 of the points more than 100 m up; the first set abstraction level kept 0.5579 of the exact
neighbours, 0.1303 more than 100 m up. On the Sunset blocks, whose points stand below 25 m, the class
matched on 0.9798.

## The fix

`knn_idx` computes the distances in fp32 with autocast off, on coordinates centred on the reference
cloud's mean. A shift does not change a distance, and the centring makes the squares being subtracted
smaller. Measured the same way, with the fix the class matches the exact pass on 0.9985 of the
Financial District points and 0.9972 of those more than 100 m up, and every one of the six searches
keeps at least 0.9994 of the exact neighbours more than 100 m up. The fp32 search without the centring
keeps 0.9973 at the last level more than 100 m up, which is why the centring is there.
`tests/unit/test_knn_precision.py` holds it: under fp16 autocast, on a synthetic tower up to 250 m, the
search finds the exact neighbours on the CPU and on a CUDA card, and the old expansion fails the same
test.

The fault was in training as well as in prediction, so the fix is not applied to the old checkpoints:
a model trained with one neighbour search and read with another is a third thing. M0 is trained again.

## What runs on the card

| Run | Seeds | As in Phase 4 except |
|---|---|---|
| M0 | A, B, C | the neighbour search |
| M0 large, the capacity control | A | the neighbour search |

The data, the class weights, the steps, the batch and the learning rates are Phase 4's, and train.py is
unchanged. Before each run, 200 steps are timed on the card and the estimate goes into the chain log.
The number of voting passes is chosen again by the same rule (`docs/phase4_criteria.md`), on the new
M0 seed A. The four new models are predicted on the 11 held-out Surrey tiles and the 8 San Francisco
areas with that number of passes.

`scripts/pipeline/phase6_chain.sh` runs it and stops on the first failure: a failed input hash, the
neighbour search test failing on the card, a training run that cannot resume exactly, or a prepared
tile holding a dropped flag.

## What is not run again

The four channel checkpoint (the input bug, `returns_ood` and `ret0`), M0prime and M1 keep their Phase 4
predictions. They were trained and predicted with the old neighbour search, and they are labelled so
wherever they appear. Each is read only against its own pair: the bug card sets the same checkpoint and
the same search against itself with one input changed, and M1 is read only against M0prime. The fault
does not decide either comparison, since both sides of each carry it.

## How the results come back

The card uploads its outputs, with their SHA-256 list, straight to a private Hugging Face dataset the
owner creates for this round, using a login only the owner puts on the card. The laptop downloads them
with `scripts/pipeline/phase6_pull.sh`, which checks every hash. The instance is released as soon as the
hashes match. Nothing comes back through the vast.ai SSH proxy.

## How it is read

- Phase 6 M0 replaces Phase 4 M0 in the README and on the page. Phase 4 M0's scores stay in the git
  history as the record of the old search.
- The effect of the fix is Phase 6 M0 against Phase 4 M0, on every San Francisco metric and every held-out
  Surrey class, by the drop rule of `backend/surrey/seeds.py`: a figure moved only when every seed of one
  lies beyond every seed of the other. With three seeds against three, chance alone does that 1 time in
  10 in one direction or the other, so the number of comparisons and the count expected by chance are
  reported beside the moves, and no single move is read as an effect on its own.
- The Financial District claim, written before the first training (buildings score lower there than in
  the built areas inside the training range), is read again on the Phase 6 seeds by the same rule. If it
  no longer holds, the README says the old search accounted for it. If it holds, it stands with the
  exact search. Beside it, as figures that need no footprint: the share of points more than 100 m and
  more than 150 m up called each class.
- The large model is read by the rule in `docs/rented_gpu.md`, against the three new small seeds.

## Scoring changes, fixed now and applied to every model scored after this

Every committed score is computed again with these, the old models' included, so every pair is still
read on one scoring. The predictions of the old models do not change.

1. Ground precision. A predicted ground point is judged wherever the delivery gives a ground surface
   within reach, as before, and also wherever the delivery labels the point itself as something other
   than unassigned: water and bridge deck called ground count as wrong. The share of predicted ground
   still unjudged is reported beside the band.
2. False vegetation. Where the mask rule (`scripts/pipeline/naip_mask_choice.py`) does not decide, false
   vegetation takes the higher of the two masks, as recall takes the lower, so both halves of the pair
   read the harsher mask.
3. Buildings. Points the delivery labels water (9) or bridge deck (17) leave the footprint scoring
   population; the bridge deck keeps its own figure.
4. Ground includes the delivery's ignored ground (20) wherever ground is scored.
5. Lift is (p - p0) / (1 - p0), the share of the room above the null's precision the model takes, and is
   not reported where the null calls nothing a building. The water measure against its physical null is
   named a gain over the null, not a lift.
6. The height gate. The primary gate stays the label-free floor (the 5th percentile of z in 10 m cells),
   the same on both sides of the ocean. For the building figures, a second gate from the delivery's own
   class 2 surface is reported beside it in every built area, because the label-free floor sits on the
   roofs of the tallest blocks.
7. Entropy is normalised by the logarithm of the model's own class count.
8. Every seed comparison reports its two-sided chance and, for a set of comparisons, the count expected
   by chance.

## What stops the round

A failed hash in either direction, the neighbour search test failing on the card, an interruption a
training run cannot resume exactly, or a prepared tile that fails `tests/integrity/test_prepared_flags.py`.
Each is written to the chain log with the time, and nothing after it runs until it is resolved.
