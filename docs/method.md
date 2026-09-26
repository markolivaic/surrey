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
three seeds on each side, chance alone produces that 1 time in 20 in a given direction, so among the
many figures here some such moves are expected by chance.

Each prediction is the softmax averaged over 4 passes. The number was chosen by a rule written
before any San Francisco figure existed (`docs/phase4_criteria.md`): the smallest of 1, 2, 4, 8 and 16
whose coarse classes agree with twice as many passes on at least 0.99 of the held-out Surrey points.
4 passes and twice as many agree on 0.9908 of them, which is also the share of a
prediction that another draw of the passes could change.

Before this repository's first commit, the order in which rules were written and results appeared is
self-reported, in the dates the project wrote into its own artefacts. From the first commit on, the git
history shows it: the retraining's rules are in the first commit, its results in a later one.

## The vegetation reference

The vegetation figures read NAIP 2022 cores between 2 and 25 m above the ground floor. Recall takes the
lower of two masks, one registered to the lidar and one not, because a rule fixed beforehand could not
choose between them (`scripts/pipeline/naip_mask_choice.py`); false vegetation reads the unregistered
mask. In Golden Gate Park the not-vegetation cores hold 13,847 points, a share of
0.0015 of the area, 17.79 m up at the median,
so the park's false vegetation figure rests on little (`scripts/investigations/not_vegetation_heights.py`).

## The building lift, first round against the retraining

Lift is precision against the footprints over the null's precision, so it cannot tell a model that finds
less from one that finds more, less precisely. The first round's M0 (its scores as the first commit
holds them) and the retrained M0 are set side by side, median and range over three seeds each, with the
reading written into `docs/phase4_criteria.md` before the comparison ran: coverage up with precision
down is a trade, both down is worse, anything else is written as measured
(`scripts/investigations/building_breakdown.py`).

| Area | Lift, first round to retrained | Precision, first round | Precision, retrained | Coverage, first round | Coverage, retrained | Reading |
|---|---|---|---|---|---|---|
| Sunset | 0.7011 to 0.6699, did not move | 0.8627 (0.8444 to 0.8692) | 0.8482 (0.8452 to 0.8512) | 0.9028 (0.8659 to 0.9125) | 0.9877 (0.9877 to 0.9896) | precision did not move, coverage rose |
| Mission | 0.66 to 0.6136, fell | 0.8519 (0.8377 to 0.8558) | 0.8317 (0.8268 to 0.8323) | 0.8825 (0.8745 to 0.9113) | 0.9776 (0.9776 to 0.9777) | a trade: more of each footprint covered, less precisely |
| Russian Hill | 0.651 to 0.5579, fell | 0.8683 (0.856 to 0.8777) | 0.8333 (0.8316 to 0.835) | 0.8542 (0.846 to 0.8681) | 0.984 (0.9834 to 0.9844) | a trade: more of each footprint covered, less precisely |
| SoMa | 0.6206 to 0.5621, fell | 0.8665 (0.8605 to 0.8681) | 0.8455 (0.8446 to 0.8459) | 0.5946 (0.5885 to 0.6119) | 0.722 (0.7203 to 0.7222) | a trade: more of each footprint covered, less precisely |
| Financial District | 0.3306 to 0.1456, fell | 0.7813 (0.7512 to 0.7814) | 0.7234 (0.7232 to 0.7401) | 0.7508 (0.7467 to 0.7542) | 0.8248 (0.8222 to 0.8354) | a trade: more of each footprint covered, less precisely |
| Embarcadero | 0.0159 to 0.0443, rose | 0.1773 (0.1737 to 0.18) | 0.2034 (0.1984 to 0.2081) | 0.8054 (0.7754 to 0.8095) | 0.8506 (0.8506 to 0.8541) | precision rose, coverage rose |

Inference alone separates part of it, on the retrained seed A. In the Financial District its lift is
0.2814 with the first round's block iterator and one
pass, 0.1447 with the new iterator and one pass, and
0.1452 with 4 passes: the iterator moved it, the voting
did not. The same change of iterator raises accuracy at home from
0.8932 to
0.9181, and mean IoU over fifteen
classes from 0.4261 to
0.4846. The rest, from the first round's
0.3306 (0.2384 to
0.3307) to 0.2814, lies
with the seeds and the retraining, whose class weights now come from 29
tiles instead of 40 and whose inputs hold no withheld points. Separating those
needs training, and was not done.

## The capacity control

A model with 11,505,423 parameters, against the published
725,967, was trained on the same rented card as the three small seeds, on the same
data and the same number of points seen, with the batch and the learning rate scaled as
`docs/rented_gpu.md` sets out. The rule for adopting it was written before it ran: its building lift
beats the highest small seed in at least 4 of the 6 areas with
footprints. It does in 0. At home it reads
0.9301 against 0.9285
(0.9273 to
0.9288) for the small seeds. The rule is not met,
and the published model is unchanged.

## Water, and the old pair

M1 was the first round's attempt at water: M0prime's recipe plus swissSURFACE3D tiles from Zurich and
Geneva, where water and bridge decks are labelled. Most of that water was never measured. Where a lake
returns no pulse, the survey fills it with points on a regular 25 cm grid and flags them synthetic:
0.8586 to 0.9999 of the water
on the four Swiss tiles M1 learned from (`scripts/investigations/las_flags_audit.py`), a flag read only
after training. M1 learned the fill. Measured returns alone, water holds a median of
3.16 points per square metre on the Swiss tile with the
most of it and 0.02 at Zug, against
28.1 at Embarcadero and
30.58 at Bay Bridge
(`scripts/investigations/real_water_density.py`). With that little real water to learn from, M1 is not
trained again.

At Zug, on a tile it was never trained on, M1 reaches a water IoU of
0.9304 (0.8448 to
0.9386). That tile's water is 0.993 fill.
At Embarcadero M1 finds 0.0019
(0.0003 to 0.007) of the water.
The rule written before training asked for a positive lift over the water null on every seed at both
places; the water null calls every point below the sea level at NOAA gauge 9414290 at the moment of
flight, plus half a metre, water.

| area | seed A | seed B | seed C | result |
|---|---|---|---|---|
| Embarcadero | -1.4314 | -1.4439 | -1.4478 | positive on 0 of 3 seeds |
| Bay Bridge | -1.4652 | -1.4655 | -1.4386 | positive on 0 of 3 seeds |

The water null is not a do-nothing baseline: it knows each point's absolute height in NAVD88 and the sea level at the moment of flight. The model knows neither; it sees height above the floor of a 40 m block.

M1 and M0prime are the old pair, trained on the laptop with class weights from all 40 tiles and on the
Swiss tiles as they were, then predicted again with the same voting as every other model. They are read
only against each other. By the drop rule above, M1 falls against M0prime on ground at Embarcadero, on
buildings and vegetation in the Financial District, on vegetation in the Sunset and on four classes at
home, and gains at Bay Bridge, where less water is called ground, and on two classes at home
(`scripts/pipeline/compare_recipes.py`).
