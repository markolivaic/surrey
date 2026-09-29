# Method notes

The detail behind the README's short sections. Like the README, this file is generated from
`docs/method.template.md` by `scripts/pipeline/render_readme.py`, and every figure in it is a registry key.

## Seeds and base rates

Every model figure is the median of three training seeds, with the lowest and highest seed beside it
(`backend/surrey/seeds.py`), and a claim written before training is reported as holding on k of n seeds,
never pooled. Every figure stands beside its null or base rate: the do-nothing or null figure for the
same population.

Three seeds are few. A fourth run of the same recipe lands outside the range of three half the time,
since each of four exchangeable runs is equally likely to be the lowest or the highest. So a figure
outside another model's range is not by itself a difference. A figure is read as moved only when every
seed of one model lies beyond every seed of the other, the drop rule of `backend/surrey/seeds.py`. With
three seeds on each side, chance alone does that with probability
0.1, in one direction or the other. Two recipes
are compared on 69 figures, so about
6.9 moves are expected between them by chance
alone, and `scripts/pipeline/compare_recipes.py` reports that count beside the moves it finds.

Each prediction is the softmax averaged over 4 passes. The number was chosen by a rule written
before any San Francisco figure existed (`docs/phase4_criteria.md`): the smallest of 1, 2, 4, 8 and 16
whose coarse classes agree with twice as many passes on at least 0.99 of the held-out Surrey points. It
was chosen again for Phase 6 by the same rule, on the new seed A, and came out the same. 4
passes and twice as many agree on 0.9905 of the points. The larger set of passes contains
the smaller, so this agreement is higher than two separate sets would reach: it says the vote has
settled, not that it is right.

Before this repository's first commit, the order in which rules were written and results appeared is
self-reported, in the dates the project wrote into its own artefacts. From the first commit on, the git
history shows it: each rented round's rules are in a commit before its results.

## The San Francisco references

Ground is scored against the delivery's ground, classes 2 and 20. Recall is the share of it called
ground. Precision is a band: its floor counts only the delivery's ground as right, its ceiling also counts
points the delivery left unassigned that lie on its ground surface (`backend/surrey/reference.py`). A
predicted ground point is judged where the delivery gives a ground surface within reach or labels the
point itself, so water and bridge deck called ground count as wrong. The share of predicted ground left
unjudged is 0.2467 at Embarcadero and
0.003 in the Financial District.

The vegetation figures read NAIP 2022 cores between 2 and 25 m above the ground floor. A rule fixed
beforehand chooses between two masks, one registered to the lidar and one not
(`scripts/pipeline/naip_mask_choice.py`); where it cannot, recall takes the lower of the two and false
vegetation the higher, so both halves of the pair read the harsher mask. In Golden Gate Park the
not-vegetation cores hold 13,847 points, a share of
0.0015 of the area, 17.79 m up at the median,
with the share of multiple returns canopy has, so the park's false vegetation figure is not read.

The building figures take points more than 2 m above a label-free floor, the 5th percentile of z in 10 m
cells, the same surface on both sides of the ocean. That floor sits on the roofs of the tallest blocks, so
a second gate, from the delivery's own ground surface, stands beside it: in the Financial District the
lift is 0.2156 with the first gate and
0.205 with the second. Points the delivery labels
water or bridge deck are not in the building population; the deck has its own figure. The footprints are
moved by 1.584 m east and -0.792 m
north, the median over the 3 areas whose own correlation
found an offset (`scripts/pipeline/crs_check.py`); the other areas are scored with that offset untested.

## The neighbour search, before and after

The search expanded a squared distance as |q|^2 + |r|^2 - 2 q.r, and under fp16 autocast the product ran
in half precision, where a square near 40,000 m^2 steps in 32 m^2. It now runs in fp32 with autocast off,
on coordinates centred on the reference cloud. `scripts/investigations/precision_audit.py` measured Phase
4 M0 seed A on the four tallest Financial District blocks against an exact pass in float64; each cell is
the share of the exact neighbours a level kept, for points more than 100 m up.

| Level | fp16, as trained | fp32, not centred | fp32, centred |
|---|---|---|---|
| sa1 | 0.1303 | 0.9979 | 0.9994 |
| sa2 | 0.2838 | 0.9997 | 0.9999 |
| sa3 | 0.3519 | 0.9999 | 1.0 |
| fp1 | 0.1132 | 0.9973 | 0.9994 |
| fp2 | 0.2076 | 0.9995 | 0.9998 |
| fp3 | 0.3094 | 0.9998 | 1.0 |

The predicted class matched the exact pass on 0.5886 of those
blocks' points before and 0.9985 after; on the two tallest Sunset
blocks, below 25 m, on 0.9798 and
0.9998. The fault was in training too, so M0 was trained again rather
than read with the new search (`docs/phase6_criteria.md`). Against Phase 4 M0, by the drop rule, these
moved, 8 of 69
where chance alone moves 6.9:

| Where | Figure | Moved |
|---|---|---|
| Bay Bridge | bridge deck called building | better |
| Embarcadero | bridge deck called building | better |
| Embarcadero | building lift | better |
| Embarcadero | ground precision, floor | worse |
| Embarcadero | not-vegetation cores called vegetation | worse |
| Financial District | building lift | better |
| Financial District | ground precision, floor | worse |
| Financial District | raised unassigned points called ground | worse |

## The building lift, first round against Phase 4

Lift is (p - p0) / (1 - p0), a precision measure, so it cannot tell a model that finds fewer buildings
from one that finds more of them less precisely. When the lift fell from the first round to Phase 4, the
two were set side by side, median and range over three seeds each
(`scripts/investigations/building_breakdown.py`). The reading was written into `docs/phase4_criteria.md`
after the drop had been seen and before this breakdown ran: coverage up with precision down is a trade,
both down is worse, anything else is written as measured. Both sides carry the old neighbour search and
the scoring from before Phase 6, as the commits d62078f and 5790e15 hold them, since the first round's
predictions were not kept.

| Area | Lift, first round to Phase 4 | Precision, first round | Precision, Phase 4 | Coverage, first round | Coverage, Phase 4 | Reading |
|---|---|---|---|---|---|---|
| Sunset | 0.7011 to 0.6699, did not move | 0.8627 (0.8444 to 0.8692) | 0.8482 (0.8452 to 0.8512) | 0.9028 (0.8659 to 0.9125) | 0.9877 (0.9877 to 0.9896) | precision did not move, coverage rose |
| Mission | 0.66 to 0.6136, fell | 0.8519 (0.8377 to 0.8558) | 0.8317 (0.8268 to 0.8323) | 0.8825 (0.8745 to 0.9113) | 0.9776 (0.9776 to 0.9777) | a trade: more of each footprint covered, less precisely |
| Russian Hill | 0.651 to 0.5579, fell | 0.8683 (0.856 to 0.8777) | 0.8333 (0.8316 to 0.835) | 0.8542 (0.846 to 0.8681) | 0.984 (0.9834 to 0.9844) | a trade: more of each footprint covered, less precisely |
| SoMa | 0.6206 to 0.5621, fell | 0.8665 (0.8605 to 0.8681) | 0.8455 (0.8446 to 0.8459) | 0.5946 (0.5885 to 0.6119) | 0.722 (0.7203 to 0.7222) | a trade: more of each footprint covered, less precisely |
| Financial District | 0.3306 to 0.1456, fell | 0.7813 (0.7512 to 0.7814) | 0.7234 (0.7232 to 0.7401) | 0.7508 (0.7467 to 0.7542) | 0.8248 (0.8222 to 0.8354) | a trade: more of each footprint covered, less precisely |
| Embarcadero | 0.0159 to 0.0443, rose | 0.1773 (0.1737 to 0.18) | 0.2034 (0.1984 to 0.2081) | 0.8054 (0.7754 to 0.8095) | 0.8506 (0.8506 to 0.8541) | precision rose, coverage rose |

Inference alone separates part of it, on Phase 4 seed A only. In the Financial District its lift is
0.2814 with the first round's block iterator and one
pass, 0.1447 with the new iterator and one pass, and
0.1452 with 4 passes: on that seed the iterator moved
it and the voting did not. The same change of iterator raises accuracy at home from
0.8932 to
0.9181, and mean IoU over fifteen
classes from 0.4261 to
0.4846. The rest, from the first round's
0.3306 (0.2384 to
0.3307) to 0.2814, lies
with the seeds and the retraining, whose class weights came from 29 tiles
instead of 40 and whose inputs hold no withheld points. Separating those needs
training, and was not done.

## The capacity control

A model with 11,505,423 parameters, against the published
725,967, was trained on the same rented card as the three small seeds in Phase 6,
on the same data and the same number of points seen, with the batch and the learning rate scaled as
`docs/rented_gpu.md` sets out. The rule for adopting it was written before it first ran: its building
lift beats the highest small seed in at least 4 of the 6 areas with
footprints. It does in 0. At home it reads
0.9295 against 0.9283
(0.9281 to
0.9285) for the small seeds. The rule is not met,
and the published model is unchanged.

## Water, and the old pair

M1 was the first round's attempt at water: M0prime's recipe plus swissSURFACE3D tiles from Zurich and
Geneva, where water and bridge decks are labelled. Most of that water was never measured. Where a lake
returns no pulse, the survey fills it with points on a regular 25 cm grid and flags them synthetic:
0.8586 to 0.9999 of the water
on the four Swiss tiles M1 learned from (`scripts/investigations/las_flags_audit.py`), a flag read only
after training. Measured returns alone, water holds a median of
3.16 points per square metre on the Swiss tile with the
most of it and 0.02 at Zug, against
28.1 at Embarcadero and
30.58 at Bay Bridge
(`scripts/investigations/real_water_density.py`). With that little real water to learn from, M1 is not
trained again.

What M1 took from the fill is not settled. 0.952 of the
Swiss water points sit on the fill's grid. Moving the Zug water off it takes seed B's water IoU from
0.9304 to 0.2454,
and leaves seeds A and C where they were, 0.8448 to
0.8435 and 0.9386
to 0.922. And water lies at the floor of its 40 m block on
both sides, a median of 0.0 m above it in Switzerland and
0.1 m in San Francisco, where flat ground lies too; height above
the floor is the only height the model reads (`scripts/investigations/water_input_test.py`).

At Zug, on a tile it was never trained on, M1 reaches a water IoU of
0.9304 (0.8448 to
0.9386). That tile's water is 0.993 fill.
At Embarcadero M1 finds 0.0019
(0.0003 to 0.007) of the water.
The rule written before training asked for a positive gain over the water null on every seed at both
places, where the gain is (IoU - IoU0) / (1 - IoU0) against the null's IoU; the water null calls every
point below the sea level at NOAA gauge 9414290 at the moment of flight, plus half a metre, water.

| area | seed A | seed B | seed C | result |
|---|---|---|---|---|
| Embarcadero | -1.4314 | -1.4439 | -1.4478 | positive on 0 of 3 seeds |
| Bay Bridge | -1.4652 | -1.4655 | -1.4386 | positive on 0 of 3 seeds |

The water null is not a do-nothing baseline: it knows each point's absolute height in NAVD88 and the sea level at the moment of flight. The model knows neither; it sees height above the floor of a 40 m block.

M1 and M0prime are the old pair, trained on the laptop with class weights from all 40 tiles and on the
Swiss tiles as they were, predicted again in Phase 4 with the same voting as every other model, and
scored with the Phase 6 rules. Both carry the old neighbour search, and they are read only against each
other. By the drop rule, 11 of
69 figures moved, where chance alone moves
6.9 (`scripts/pipeline/compare_recipes.py`):

| Where | Figure | Moved |
|---|---|---|
| Bay Bridge | water called ground | better |
| Embarcadero | delivery ground called ground | worse |
| Financial District | building lift | worse |
| Financial District | vegetation cores called vegetation | worse |
| Sunset | vegetation cores called vegetation | worse |
| held-out Surrey | IoU, annex | worse |
| held-out Surrey | IoU, ground | worse |
| held-out Surrey | IoU, habitat | better |
| held-out Surrey | IoU, pickup | worse |
| held-out Surrey | IoU, traffic-pole | better |
| held-out Surrey | IoU, vegetation | worse |
