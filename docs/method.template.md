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
with probability {{compare.m0_vs_m0fp16.chance.two_sided_chance_of_a_move}}, in one direction or the
other, and no comparison can go below that. Two recipes are compared on
{{compare.m0_vs_m0fp16.chance.comparisons}} figures, so about
{{compare.m0_vs_m0fp16.chance.moves_expected_by_chance@1}} moves are expected between them by chance;
the figures are correlated, so that count is context, not a test.

Each prediction is the softmax averaged over {{votes.n}} passes. The number was chosen by a rule written
before any San Francisco figure existed (`docs/criteria/retraining.md`): the smallest of 1, 2, 4, 8 and 16
whose coarse classes agree with twice as many passes on at least 99% of the held-out Surrey points. It
was chosen again by the same rule for the published model and came out the same; {{votes.n}} passes
and twice as many agree on {{votes.agreement_4%}} of the points. The larger set of passes contains the
smaller, so this agreement says the vote has settled, not that it is right. Because the vote count and
the training checks both used the held-out Surrey tiles, those tiles served as validation as well as
test.

The order in which rules were written and results appeared is the author's own record, in the dates
written into the artefacts and the criteria files.

## At home, per class

Held-out Surrey IoU per class, voted, for the published model and the fp16-search model:

{{table:surrey_classes}}

The DALES-2 paper's Table 5 (Bendjilali and others, 2026) gives {{published.dales2.randlanet.miou_15}},
{{published.dales2.kpconv.miou_15}} and {{published.dales2.ptv3.miou_15}} mean IoU for RandLA-Net, KPConv
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
unjudged is {{seeds.m0.sf.embarcadero.unjudged_ground_share.median%}} at Embarcadero and
{{seeds.m0.sf.financial.unjudged_ground_share.median%}} in the Financial District.

The vegetation figures read NAIP 2022 cores between 2 and 25 m above the ground floor. The vegetation
raster is shrunk by how far a crown of the tallest height scored leans in an aerial photograph, so a
core is a point well inside a patch the photograph shows one way or the other. A rule fixed beforehand
chooses between two masks, one registered to the lidar and one not
(`scripts/pipeline/naip_mask_choice.py`); where it cannot, recall takes the lower of the two and false
vegetation the higher, so both halves of the pair read the harsher mask.

In Golden Gate Park the not-vegetation cores of the unregistered mask hold {{notveg.sf.ggpark.points}}
points, {{notveg.sf.ggpark.share_of_area%}} of the area, {{notveg.sf.ggpark.height_median_m}} m up at the
median, with the share of multiple returns canopy has, so the photograph probably misses trees there
and the park's false vegetation figure is not read. That figure comes from the registered mask, the
higher of the two there, whose cores hold {{naip.sf.ggpark.registered.not_vegetation_core_points}}
points.

The confidence figures compare like with like as far as the data allow. At home a hard mistake is any
labelled non-ground point within half a metre of the ground called ground, against human labels; at
Embarcadero it is the delivery's water called ground, against the vendor's class. AUROC is the chance
that a random right ground call is more confident than a random mistake.

## The NDVI threshold

The vegetation reference calls NAIP green above an NDVI threshold of {{naip.sf.ndvi_threshold}}, fixed by
Otsu's method from the image alone before any prediction was read, on imagery
{{provenance.sf.days_from_imagery_to_lidar}} days older than the lidar.
`scripts/pipeline/ndvi_sensitivity.py` rebuilds both masks at other thresholds and reads the three seeds
against them, vegetation found / false vegetation, median over seeds:

{{table:ndvi}}

Vegetation found barely moves. False vegetation rises with the threshold: a higher threshold moves
more points into the not-vegetation cores, and the model calls some of them vegetation.

## Rules beside the model

`scripts/pipeline/rule_baselines.py` scores three one-line rules with the model's own scoring, so
every San Francisco figure has something a reader could write down beside it: ground below half a
metre above the label-free floor, vegetation 2-25 m up, and building wherever a point stands more than
2 m above the delivery's ground and NAIP is not green there. The vegetation rule scores {{rules.sf.sunset.vegetation.core_called_vegetation}} on both
sides of the cores by construction, which is what the model's pair has to beat.

{{table:rules}}

The model beats the ground rule on recall where the ground is not flat, and the building rule's lift
everywhere but Embarcadero, where the rule's {{rules.sf.embarcadero.building.lift}} is above the model's
{{seeds.m0.sf.embarcadero.building_lift_global_shift.median}}.

## Easy errors

Delivery points left unassigned more than 3 m up but called ground were to be read per area: a rate
above every held-out Surrey tile's, a transfer failure; inside that range, a weakness the model has at
home too.

{{table:easy_errors}}

## How uncertain the building figures are, and where lift falls

`scripts/pipeline/building_intervals.py` resamples footprints 1,000 times, each eligible point going
with its nearest footprint, for a 95% interval on precision and lift per area. It also pools 40 m
blocks by how much height they span. For seed A:

{{table:by_height}}

Lift falls with height span in the other built areas too, so the Financial District's low figure is
consistent with heights the network was not trained on. It is low in the district's short blocks as
well, which height alone does not explain; the district's footprints could not be registered to the
lidar, and the tallest blocks elsewhere, at Embarcadero, hold no footprint at all.

The building figures take points more than 2 m above a label-free floor, the 5th percentile of z in 10 m
cells, the same surface on both sides of the ocean. That floor sits on the roofs of the tallest blocks, so
a second gate, from the delivery's own ground surface, stands beside it: in the Financial District the
lift is {{seeds.m0.sf.financial.building_lift_global_shift.median}} with the first gate and
{{seeds.m0.sf.financial.building_lift_delivery_gate.median}} with the second. Points the delivery labels
water or bridge deck are not in the building population; the deck has its own figure. The footprints are
moved by {{crs.sf.footprint_offset_global.east_m}} m east and {{crs.sf.footprint_offset_global.north_m}} m
north, the median over the {{crs.sf.footprint_offset_global.areas_used}} areas whose own correlation
found an offset (`scripts/pipeline/crs_check.py`); the other areas are scored with that offset untested.

## The neighbour search, before and after

The search expanded a squared distance as |q|^2 + |r|^2 - 2 q.r, and under fp16 autocast the product ran
in half precision, where a square near 40,000 m^2 steps in 32 m^2. It now runs in fp32 with autocast off,
on coordinates centred on the reference cloud. `archive/investigations/precision_audit.py` measured the
fp16-search model, seed A, on the four tallest Financial District blocks against an exact pass in
float64; each cell is the share of the exact neighbours a level kept, for points more than 100 m up.

| Level | fp16, as trained | fp32, not centred | fp32, centred |
|---|---|---|---|
| sa1 | {{precision.financial.before.sa1_neighbours_kept_above_100m}} | {{precision.financial.fp32_uncentred.sa1_neighbours_kept_above_100m}} | {{precision.financial.after.sa1_neighbours_kept_above_100m}} |
| sa2 | {{precision.financial.before.sa2_neighbours_kept_above_100m}} | {{precision.financial.fp32_uncentred.sa2_neighbours_kept_above_100m}} | {{precision.financial.after.sa2_neighbours_kept_above_100m}} |
| sa3 | {{precision.financial.before.sa3_neighbours_kept_above_100m}} | {{precision.financial.fp32_uncentred.sa3_neighbours_kept_above_100m}} | {{precision.financial.after.sa3_neighbours_kept_above_100m}} |
| fp1 | {{precision.financial.before.fp1_neighbours_kept_above_100m}} | {{precision.financial.fp32_uncentred.fp1_neighbours_kept_above_100m}} | {{precision.financial.after.fp1_neighbours_kept_above_100m}} |
| fp2 | {{precision.financial.before.fp2_neighbours_kept_above_100m}} | {{precision.financial.fp32_uncentred.fp2_neighbours_kept_above_100m}} | {{precision.financial.after.fp2_neighbours_kept_above_100m}} |
| fp3 | {{precision.financial.before.fp3_neighbours_kept_above_100m}} | {{precision.financial.fp32_uncentred.fp3_neighbours_kept_above_100m}} | {{precision.financial.after.fp3_neighbours_kept_above_100m}} |

The predicted class matched the exact pass on {{precision.financial.before.class_agreement}} of those
blocks' points before and {{precision.financial.after.class_agreement}} after; on the two tallest Sunset
blocks, below 25 m, on {{precision.sunset.before.class_agreement}} and
{{precision.sunset.after.class_agreement}}. The fault was in training too, so the model was trained
again rather than read with the new search (`docs/criteria/neighbour_search.md`). Against the
fp16-search model, by the drop rule, these moved, {{compare.m0_vs_m0fp16.chance.moves_seen}} of
{{compare.m0_vs_m0fp16.chance.comparisons}} where chance alone moves
{{compare.m0_vs_m0fp16.chance.moves_expected_by_chance@1}}:

{{table:moves_m0_vs_m0fp16}}

Of the Financial District's {{tall.sf.financial.m0_seedA.above_150m.points}} points more than 150 m
above the ground, the fp16-search model called {{tall.sf.financial.m0fp16_seeds.above_150m.called_pole.median}}
({{tall.sf.financial.m0fp16_seeds.above_150m.called_pole.min}} to
{{tall.sf.financial.m0fp16_seeds.above_150m.called_pole.max}}) pole; the published recipe calls
{{tall.sf.financial.m0_seeds.above_150m.called_building.median}}
({{tall.sf.financial.m0_seeds.above_150m.called_building.min}} to
{{tall.sf.financial.m0_seeds.above_150m.called_building.max}}) building over its three seeds
(`archive/investigations/high_points_class.py`).

## The return ratio, area by area

One checkpoint with one input changed, so single figures and no seed range:

{{table:bug}}

† As in the README's San Francisco table, not read.

## The capacity control

A model with {{train.m0_big_seedA.parameters}} parameters, against the published
{{train.m0_seedA.parameters}}, was trained on the same rented card as the three small seeds, on the same
data and the same number of points seen, so for fewer steps, with the batch and the learning rate scaled
as `docs/rented_gpu.md` sets out. The rule for adopting it was written before it first ran: its building
lift beats the highest small seed in at least 4 of the {{capacity.areas_with_footprints}} areas with
footprints. It does in {{capacity.areas_big_above_small_max}}. At home it reads
{{scores.m0_big_seedA.surrey.test.accuracy}} against {{seeds.m0.surrey.test.accuracy_from_prediction_cache.median}}
({{seeds.m0.surrey.test.accuracy_from_prediction_cache.min}} to
{{seeds.m0.surrey.test.accuracy_from_prediction_cache.max}}) for the small seeds. The rule is not met at
this training budget, and the published model is unchanged.

## Water, and the old pair

M1 was the old attempt at water: M0prime's recipe plus swissSURFACE3D tiles from Zurich and Geneva,
where water and bridge decks are labelled. Most of that water was never measured. Where a lake returns
no pulse, the survey fills it with points on a regular 25 cm grid and flags them synthetic:
{{flags.swiss_training.water.synthetic.min%}} to {{flags.swiss_training.water.synthetic.max%}} of the
water on the four Swiss tiles M1 learned from (`archive/investigations/las_flags_audit.py`), a flag read
only after training. Measured returns alone, water holds a median of
{{realwater.zurich2683_1246.source.density_median}} points per square metre on the Swiss tile with the
most of it and {{realwater.zug2680_1225.source.density_median}} at Zug, against
{{realwater.embarcadero.source.density_median}} at Embarcadero and
{{realwater.baybridge.source.density_median}} at Bay Bridge
(`archive/investigations/real_water_density.py`). With that little real water to learn from, M1 is not
trained again.

What M1 took from the fill is not settled. {{lattice.swiss_training.water.share_on_the_lattice%}} of the
Swiss water points sit on the fill's grid. Moving the Zug water off it takes seed B's water IoU from
{{lattice_test.m1_seedB.as_prepared.water.iou}} to {{lattice_test.m1_seedB.water_jittered.water.iou}},
and leaves seeds A and C where they were, {{lattice_test.m1_seedA.as_prepared.water.iou}} to
{{lattice_test.m1_seedA.water_jittered.water.iou}} and {{lattice_test.m1_seedC.as_prepared.water.iou}}
to {{lattice_test.m1_seedC.water_jittered.water.iou}}. And water is flat on both sides, a median of
{{water_input.both.switzerland.z_median}} m above the 1st percentile of height in its 40 m block in
Switzerland and {{water_input.both.san_francisco.z_median}} m in San Francisco. Flat ground has the same
local shape, and local shape is all the model sees: it reads coordinates only as offsets between
neighbouring points (`archive/investigations/water_input_test.py`, `tests/unit/test_model_invariance.py`).

At Zug, on a tile it was never trained on, M1 reaches a water IoU of
{{swiss_holdout.m1_seeds.water.iou.median}} ({{swiss_holdout.m1_seeds.water.iou.min}} to
{{swiss_holdout.m1_seeds.water.iou.max}}). That tile's water is {{flags.2680_1225.water.synthetic%}}
fill. At Embarcadero M1 finds {{water.m1_seeds.embarcadero.recall.median%}}
({{water.m1_seeds.embarcadero.recall.min%}} to {{water.m1_seeds.embarcadero.recall.max%}}) of the water.
The rule written before training asked for a positive gain over the water null on every seed at both
places, where the gain is (IoU - IoU0) / (1 - IoU0) against the null's IoU; the water null calls every
point below the sea level at NOAA gauge 9414290 at the moment of flight, plus half a metre, water.

{{table:m1_water}}

The water null is not a do-nothing baseline: it knows each point's absolute height in NAVD88 and the sea level at the moment of flight. The model knows neither; it sees only where points lie relative to each other.

M1 and M0prime are the old pair, trained on the laptop with class weights from all 40 tiles and on the
Swiss tiles as they were, predicted again with the same voting as every other model, and scored with the
same rules as the published model. Both carry the old neighbour search, and they are read only against
each other. By the drop rule, {{compare.m1_vs_m0prime.chance.moves_seen}} of
{{compare.m1_vs_m0prime.chance.comparisons}} figures moved, where chance alone moves
{{compare.m1_vs_m0prime.chance.moves_expected_by_chance@1}} (`scripts/pipeline/compare_recipes.py`):

{{table:moves_m1_vs_m0prime}}
