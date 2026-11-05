# Method notes

The detail behind the README. Like the README, this file is generated from `docs/method.template.md`
by `scripts/pipeline/render_readme.py`, and every figure in it is a registry key.

## Seeds, voting and the drop rule

Every model figure is the median of three training seeds, with the lowest and highest seed beside it
(`backend/surrey/seeds.py`). A claim written before training is reported as holding on k of n seeds,
never pooled.

Three seeds are few. A fourth run of the same recipe lands outside the range of three half the time,
since each of four exchangeable runs is equally likely to be the lowest or the highest, so a figure
outside another model's range is not by itself a difference. A figure is read as moved only when every
seed of one model lies beyond every seed of the other. With three seeds a side, chance alone does that
with probability 0.1, in one direction or the
other, and no comparison can go below that. Two recipes are compared on
69 figures, so about
6.9 moves are expected between them by chance;
the figures are correlated, so that count is context, not a test.

Each prediction is the softmax averaged over 4 passes. The number was chosen by a rule written
before any San Francisco figure existed (`docs/criteria/retraining.md`): the smallest of 1, 2, 4, 8 and 16
whose coarse classes agree with twice as many passes on at least 99% of the held-out Surrey points. It
was chosen again by the same rule for the published model and came out the same; 4 passes
and twice as many agree on 99% of the points. The larger set of passes contains the
smaller, so this agreement says the vote has settled, not that it is right. Because the vote count and
the training checks both used the held-out Surrey tiles, those tiles served as validation as well as
test.

The order in which rules were written and results appeared is the author's own record, in the dates
written into the artefacts and the criteria files.

## At home, per class

Held-out Surrey IoU per class, voted, for the published model and the fp16-search model:

| Class | Drawn as | Share of held-out points | IoU, published model | IoU, fp16-search model | Moved |
|---|---|---|---|---|---|
| ground | ground | 0.4797 | 0.9386 (0.9385 to 0.9399) | 0.9394 (0.9384 to 0.9403) | no detectable move |
| vegetation | vegetation | 0.0602 | 0.5076 (0.5041 to 0.5077) | 0.5051 (0.5049 to 0.5065) | no detectable move |
| car | vehicle | 0.0063 | 0.694 (0.6852 to 0.7041) | 0.6917 (0.6889 to 0.71) | no detectable move |
| powerline | pole | 0.0019 | 0.9008 (0.8985 to 0.9095) | 0.9034 (0.9017 to 0.9107) | no detectable move |
| fence | fence | 0.0049 | 0.5133 (0.5051 to 0.5153) | 0.5136 (0.511 to 0.5136) | no detectable move |
| tree | vegetation | 0.2767 | 0.9093 (0.9062 to 0.9097) | 0.9069 (0.9052 to 0.9095) | no detectable move |
| pickup | vehicle | 0.001 | 0.2531 (0.2427 to 0.2583) | 0.245 (0.2361 to 0.2454) | no detectable move |
| van-truck | vehicle | 0.0011 | 0.322 (0.3073 to 0.3316) | 0.3289 (0.2978 to 0.3414) | no detectable move |
| heavy-duty | vehicle | 0.0002 | 0.0421 (0.0109 to 0.0862) | 0.0155 (0.0108 to 0.0436) | no detectable move |
| utility-pole | pole | 0.0003 | 0.2528 (0.1988 to 0.2582) | 0.2598 (0.1806 to 0.2712) | no detectable move |
| light-pole | pole | 0.0002 | 0.1981 (0.193 to 0.2101) | 0.2197 (0.2052 to 0.2371) | no detectable move |
| traffic-pole | pole | 0.0001 | 0.2841 (0.2781 to 0.2974) | 0.3109 (0.2993 to 0.322) | fell |
| habitat | building | 0.1201 | 0.8933 (0.8925 to 0.895) | 0.8932 (0.8893 to 0.8947) | no detectable move |
| complex | building | 0.0423 | 0.6787 (0.6584 to 0.6816) | 0.6781 (0.6726 to 0.6794) | no detectable move |
| annex | building | 0.0051 | 0.4594 (0.4396 to 0.4624) | 0.4587 (0.452 to 0.4659) | no detectable move |

The DALES-2 paper's Table 5 (Bendjilali and others, 2026) gives 0.2922,
0.4228 and 0.6894 mean IoU for RandLA-Net, KPConv
and PTv3 on the same 15 classes and tiles, recorded with its per-class values in
`data/artifacts/published_baselines.json`. The paper does not say which inputs or subsampling its
models used.

## The San Francisco references

Eight areas of the USGS 3DEP San Francisco delivery are scored, each about 300 m across. A ninth,
`channel`, was fetched and dropped, because the delivery labels too little of it water or bridge deck
to measure either (`scripts/pipeline/fetch_sf_aoi.py` gives the shares).

Ground is scored against the delivery's ground, classes 2 and 20. Recall is the share of it called
ground. Precision is a band: its floor counts only the delivery's ground as right, its ceiling also counts
points the delivery left unassigned that lie on its ground surface (`backend/surrey/reference.py`). A
predicted ground point is judged where the delivery gives a ground surface within reach or labels the
point itself, so water and bridge deck called ground count as wrong. The share of predicted ground left
unjudged is 25% at Embarcadero and
0.3% in the Financial District.

The vegetation figures read NAIP 2022 cores between 2 and 25 m above the ground floor. The vegetation
raster is shrunk by how far a crown of the tallest height scored leans in an aerial photograph, so a
core is a point well inside a patch the photograph shows one way or the other. A rule fixed beforehand
chooses between two masks, one registered to the lidar and one not
(`scripts/pipeline/naip_mask_choice.py`); where it cannot, recall takes the lower of the two and false
vegetation the higher, so both halves of the pair read the harsher mask.

In Golden Gate Park the not-vegetation cores of the unregistered mask hold 13,847
points, 0.2% of the area, 17.79 m up at the
median, with the share of multiple returns canopy has, so the photograph probably misses trees there
and the park's false vegetation figure is not read. That figure comes from the registered mask, the
higher of the two there, whose cores hold 29,191
points.

The confidence figures compare like with like as far as the data allow. At home a hard mistake is any
labelled non-ground point within half a metre of the ground called ground, against human labels; at
Embarcadero it is the delivery's water called ground, against the vendor's class. AUROC is the chance
that a random right ground call is more confident than a random mistake.

## The NDVI threshold

The vegetation reference calls NAIP green above an NDVI threshold of 0.075, fixed by
Otsu's method from the image alone before any prediction was read, on imagery
337 days older than the lidar.
`scripts/pipeline/ndvi_sensitivity.py` rebuilds both masks at other thresholds and reads the three seeds
against them, vegetation found / false vegetation, median over seeds:

| Area | NDVI 0.05 | NDVI 0.075 (Otsu) | NDVI 0.1 | NDVI 0.15 | NDVI 0.2 |
|---|---|---|---|---|---|
| Golden Gate Park | 0.9723 / 0.91 | 0.9726 / 0.9429 | 0.9736 / 0.9549 | 0.9743 / 0.9621 | 0.9767 / 0.9495 |
| Sunset | 0.9464 / 0.0487 | 0.947 / 0.0561 | 0.954 / 0.0654 | 0.9614 / 0.0877 | 0.9734 / 0.1318 |
| Mission | 0.9787 / 0.0423 | 0.9835 / 0.0467 | 0.9849 / 0.0543 | 0.9871 / 0.0808 | 0.9899 / 0.136 |
| Russian Hill | 0.928 / 0.0589 | 0.9293 / 0.0648 | 0.9359 / 0.0732 | 0.9652 / 0.1058 | 0.9823 / 0.1724 |
| SoMa | 0.9884 / 0.0259 | 0.9882 / 0.0281 | 0.9889 / 0.032 | 0.9958 / 0.0644 | 0.9968 / 0.1325 |
| Financial District | 0.995 / 0.1148 | 0.9964 / 0.1177 | 0.9971 / 0.1223 | 0.997 / 0.1378 | 0.9948 / 0.1546 |
| Embarcadero | 0.9981 / 0.0596 | 0.9984 / 0.0588 | 0.9992 / 0.0589 | 0.9991 / 0.0617 | 0.9994 / 0.0621 |

Vegetation found barely moves. False vegetation rises with the threshold: a higher threshold moves
more points into the not-vegetation cores, and the model calls some of them vegetation.

## Rules beside the model

`scripts/pipeline/rule_baselines.py` scores three one-line rules with the model's own scoring, so
every San Francisco figure has something a reader could write down beside it: ground below half a
metre above the label-free floor, vegetation 2-25 m up, and building wherever a point stands more than
2 m above the delivery's ground and NAIP is not green there. The vegetation rule scores 1.0 on both
sides of the cores by construction, which is what the model's pair has to beat.

| Area | Delivery ground called ground, model / rule | Ground precision, floor, model / rule | NAIP vegetation cores called vegetation, model / rule | NAIP not-vegetation cores called vegetation, model / rule |
|---|---|---|---|---|
| Golden Gate Park | 0.9681 / 0.8292 | 0.7487 / 0.7689 | 0.9726 / 1.0 | 0.9429 / 1.0 |
| Sunset | 0.9658 / 0.7202 | 0.7449 / 0.72 | 0.947 / 1.0 | 0.0561 / 1.0 |
| Mission | 0.9779 / 0.9704 | 0.7095 / 0.6775 | 0.9835 / 1.0 | 0.0467 / 1.0 |
| Russian Hill | 0.9162 / 0.6495 | 0.7687 / 0.6815 | 0.9293 / 1.0 | 0.0648 / 1.0 |
| SoMa | 0.9903 / 0.982 | 0.6928 / 0.6625 | 0.9882 / 1.0 | 0.0281 / 1.0 |
| Financial District | 0.9868 / 0.9764 | 0.6245 / 0.5792 | 0.9964 / 1.0 | 0.1177 / 1.0 |
| Embarcadero | 0.8407 / 0.8395 | 0.168 / 0.1742 | 0.9984 / 1.0 | 0.0588 / 1.0 |
| Bay Bridge | no core | no core | no core | 0.042 / 1.0 |

The model beats the ground rule on recall where the ground is not flat, and the building rule's lift
everywhere but Embarcadero, where the rule's 0.1868 is above the model's
0.1307.

## Easy errors

Delivery points left unassigned more than 3 m up but called ground were to be read per area: a rate
above every held-out Surrey tile's, a transfer failure; inside that range, a weakness the model has at
home too.

| Area | Rate | Above the home range | Inside it |
|---|---|---|---|
| Golden Gate Park | 0.0003 (0.0003 to 0.0004) | 0 of 3 seeds | 3 of 3 seeds |
| Sunset | 0.0075 (0.0061 to 0.0101) | 0 of 3 seeds | 3 of 3 seeds |
| Mission | 0.0964 (0.0759 to 0.1265) | 1 of 3 seeds | 2 of 3 seeds |
| Russian Hill | 0.0084 (0.0072 to 0.0179) | 0 of 3 seeds | 3 of 3 seeds |
| SoMa | 0.0984 (0.0858 to 0.1227) | 2 of 3 seeds | 1 of 3 seeds |
| Financial District | 0.0506 (0.0469 to 0.0777) | 0 of 3 seeds | 3 of 3 seeds |
| Embarcadero | 0.1934 (0.1716 to 0.2274) | 3 of 3 seeds | 0 of 3 seeds |

## How uncertain the building figures are, and where lift falls

`scripts/pipeline/building_intervals.py` resamples footprints 1,000 times, each eligible point going
with its nearest footprint, for a 95% interval on precision and lift per area. It also pools 40 m
blocks by how much height they span. For seed A:

| Block spans | Financial District: lift (points) | Other built areas: lift (points) |
|---|---|---|
| under 15 m | 0.3488 (150,242) | 0.5611 (3,436,675) |
| 15 to 30 m | 0.3328 (438,125) | 0.5778 (5,917,335) |
| 30 m to the training blocks' 99th percentile | -0.0229 (164,970) | 0.6042 (1,277,492) |
| from there to the tallest training block | 0.1466 (1,629,258) | 0.2497 (415,095) |
| taller than any training block | 0.1807 (2,390,815) | no footprint (158,201) |

Lift falls with height span in the other built areas too, so the Financial District's low figure is
consistent with heights the network was not trained on. It is low in the district's short blocks as
well, which height alone does not explain; the district's footprints could not be registered to the
lidar, and the tallest blocks elsewhere, at Embarcadero, hold no footprint at all.

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
on coordinates centred on the reference cloud. `archive/investigations/precision_audit.py` measured the
fp16-search model, seed A, on the four tallest Financial District blocks against an exact pass in
float64; each cell is the share of the exact neighbours a level kept, for points more than 100 m up.

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
0.9998. The fault was in training too, so the model was trained
again rather than read with the new search (`docs/criteria/neighbour_search.md`). Against the
fp16-search model, by the drop rule, these moved, 9 of
69 where chance alone moves
6.9:

| Where | Figure | fp16 search | fp32 search | Moved |
|---|---|---|---|---|
| Bay Bridge | bridge deck called building | 0.9983 (0.9982 to 0.9984) | 0.3631 (0.2432 to 0.5152) | better |
| Embarcadero | bridge deck called building | 0.9993 (0.998 to 0.9998) | 0.2421 (0.1948 to 0.2708) | better |
| Embarcadero | building lift | 0.0844 (0.0721 to 0.0973) | 0.1307 (0.1209 to 0.1366) | better |
| Embarcadero | ground precision, floor | 0.1852 (0.183 to 0.1891) | 0.168 (0.1665 to 0.1749) | worse |
| Embarcadero | not-vegetation cores called vegetation | 0.05 (0.0461 to 0.0518) | 0.0588 (0.0532 to 0.066) | worse |
| Financial District | building lift | 0.1456 (0.1452 to 0.1972) | 0.2156 (0.2038 to 0.2186) | better |
| Financial District | ground precision, floor | 0.6583 (0.6505 to 0.6626) | 0.6245 (0.5863 to 0.6289) | worse |
| Financial District | raised unassigned points called ground | 0.0292 (0.027 to 0.0344) | 0.0506 (0.0469 to 0.0777) | worse |
| held-out Surrey | IoU, traffic-pole | 0.3109 (0.2993 to 0.322) | 0.2841 (0.2781 to 0.2974) | worse |

Of the Financial District's 233,408 points more than 150 m
above the ground, the fp16-search model called 0.9469
(0.82 to
0.9553) pole; the published recipe calls
0.8424
(0.7665 to
0.8519) building over its three seeds
(`archive/investigations/high_points_class.py`).

## The return ratio, area by area

One checkpoint with one input changed, so single figures and no seed range:

| Area | Vegetation cores called vegetation, as delivered | the same, ratio set to 0 | Not-vegetation cores called vegetation, as delivered | the same, ratio set to 0 |
|---|---|---|---|---|
| Golden Gate Park | 0.0002 | 0.9707 | 0.0 † | 0.866 † |
| Sunset | 0.0 | 0.9475 | 0.0 | 0.0373 |
| Mission | 0.0 | 0.9822 | 0.0 | 0.0343 |
| Russian Hill | 0.0 | 0.9303 | 0.0 | 0.049 |
| SoMa | 0.0 | 0.9859 | 0.0 | 0.0357 |
| Financial District | 0.0 | 0.9971 | 0.0012 | 0.1208 |
| Embarcadero | 0.0 | 0.9984 | 0.0 | 0.0646 |

† As in the README's San Francisco table, not read.

## The capacity control

A model with 11,505,423 parameters, against the published
725,967, was trained on the same rented card as the three small seeds, on the same
data and the same number of points seen, so for fewer steps, with the batch and the learning rate scaled
as `docs/rented_gpu.md` sets out. The rule for adopting it was written before it first ran: its building
lift beats the highest small seed in at least 4 of the 6 areas with
footprints. It does in 0. At home it reads
0.9295 against 0.9283
(0.9281 to
0.9285) for the small seeds. The rule is not met at
this training budget, and the published model is unchanged.

## Water, and the old pair

M1 was the old attempt at water: M0prime's recipe plus swissSURFACE3D tiles from Zurich and Geneva,
where water and bridge decks are labelled. Most of that water was never measured. Where a lake returns
no pulse, the survey fills it with points on a regular 25 cm grid and flags them synthetic:
86% to 100% of the
water on the four Swiss tiles M1 learned from (`archive/investigations/las_flags_audit.py`), a flag read
only after training. Measured returns alone, water holds a median of
3.16 points per square metre on the Swiss tile with the
most of it and 0.02 at Zug, against
28.1 at Embarcadero and
30.58 at Bay Bridge
(`archive/investigations/real_water_density.py`). With that little real water to learn from, M1 is not
trained again.

What M1 took from the fill is not settled. 95% of the
Swiss water points sit on the fill's grid. Moving the Zug water off it takes seed B's water IoU from
0.9304 to 0.2454,
and leaves seeds A and C where they were, 0.8448 to
0.8435 and 0.9386
to 0.922. And water is flat on both sides, a median of
0.0 m above the 1st percentile of height in its 40 m block in
Switzerland and 0.1 m in San Francisco. Flat ground has the same
local shape, and local shape is all the model sees: it reads coordinates only as offsets between
neighbouring points (`archive/investigations/water_input_test.py`, `tests/unit/test_model_invariance.py`).

At Zug, on a tile it was never trained on, M1 reaches a water IoU of
0.9304 (0.8448 to
0.9386). That tile's water is 99%
fill. At Embarcadero M1 finds 0.2%
(under 0.1% to 0.7%) of the water.
The rule written before training asked for a positive gain over the water null on every seed at both
places, where the gain is (IoU - IoU0) / (1 - IoU0) against the null's IoU; the water null calls every
point below the sea level at NOAA gauge 9414290 at the moment of flight, plus half a metre, water.

| area | seed A | seed B | seed C | result |
|---|---|---|---|---|
| Embarcadero | -1.4314 | -1.4439 | -1.4478 | positive on 0 of 3 seeds |
| Bay Bridge | -1.4652 | -1.4655 | -1.4386 | positive on 0 of 3 seeds |

The water null is not a do-nothing baseline: it knows each point's absolute height in NAVD88 and the sea level at the moment of flight. The model knows neither; it sees only where points lie relative to each other.

M1 and M0prime are the old pair, trained on the laptop with class weights from all 40 tiles and on the
Swiss tiles as they were, predicted again with the same voting as every other model, and scored with the
same rules as the published model. Both carry the old neighbour search, and they are read only against
each other. By the drop rule, 12 of
69 figures moved, where chance alone moves
6.9 (`scripts/pipeline/compare_recipes.py`):

| Where | Figure | M0prime | M1 | Moved |
|---|---|---|---|---|
| Bay Bridge | water called ground | 0.9395 (0.9384 to 0.9549) | 0.8904 (0.7794 to 0.9327) | better |
| Embarcadero | delivery ground called ground | 0.7961 (0.7924 to 0.8039) | 0.7506 (0.7322 to 0.792) | worse |
| Financial District | building lift | 0.13 (0.1272 to 0.1381) | 0.0789 (0.0786 to 0.0944) | worse |
| Financial District | vegetation cores called vegetation | 0.9922 (0.9836 to 0.9956) | 0.9692 (0.9419 to 0.972) | worse |
| Sunset | vegetation cores called vegetation | 0.9383 (0.9377 to 0.9423) | 0.9317 (0.9239 to 0.9342) | worse |
| held-out Surrey | IoU, accuracy | 0.9224 (0.9207 to 0.9225) | 0.9132 (0.9107 to 0.9204) | worse |
| held-out Surrey | IoU, annex | 0.393 (0.3768 to 0.4057) | 0.3506 (0.3116 to 0.3558) | worse |
| held-out Surrey | IoU, complex | 0.649 (0.6397 to 0.6496) | 0.607 (0.5996 to 0.6296) | worse |
| held-out Surrey | IoU, ground | 0.9353 (0.9346 to 0.939) | 0.9286 (0.9266 to 0.9342) | worse |
| held-out Surrey | IoU, habitat | 0.8806 (0.8781 to 0.883) | 0.8702 (0.8687 to 0.8777) | worse |
| held-out Surrey | IoU, pickup | 0.0828 (0.081 to 0.1168) | 0.0391 (0.0305 to 0.0449) | worse |
| held-out Surrey | IoU, vegetation | 0.4736 (0.4709 to 0.4752) | 0.461 (0.4495 to 0.4648) | worse |
