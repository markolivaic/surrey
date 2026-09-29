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
{{compare.m0_vs_m0phase4.chance.two_sided_chance_of_a_move}}, in one direction or the other. Two recipes
are compared on {{compare.m0_vs_m0phase4.chance.comparisons}} figures, so about
{{compare.m0_vs_m0phase4.chance.moves_expected_by_chance}} moves are expected between them by chance
alone, and `scripts/pipeline/compare_recipes.py` reports that count beside the moves it finds.

Each prediction is the softmax averaged over {{votes.n}} passes. The number was chosen by a rule written
before any San Francisco figure existed (`docs/phase4_criteria.md`): the smallest of 1, 2, 4, 8 and 16
whose coarse classes agree with twice as many passes on at least 0.99 of the held-out Surrey points. It
was chosen again for Phase 6 by the same rule, on the new seed A, and came out the same. {{votes.n}}
passes and twice as many agree on {{votes.agreement_4}} of the points. The larger set of passes contains
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
unjudged is {{seeds.m0.sf.embarcadero.unjudged_ground_share.median}} at Embarcadero and
{{seeds.m0.sf.financial.unjudged_ground_share.median}} in the Financial District.

The vegetation figures read NAIP 2022 cores between 2 and 25 m above the ground floor. A rule fixed
beforehand chooses between two masks, one registered to the lidar and one not
(`scripts/pipeline/naip_mask_choice.py`); where it cannot, recall takes the lower of the two and false
vegetation the higher, so both halves of the pair read the harsher mask. In Golden Gate Park the
not-vegetation cores hold {{notveg.sf.ggpark.points}} points, a share of
{{notveg.sf.ggpark.share_of_area}} of the area, {{notveg.sf.ggpark.height_median_m}} m up at the median,
with the share of multiple returns canopy has, so the park's false vegetation figure is not read.

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
on coordinates centred on the reference cloud. `scripts/investigations/precision_audit.py` measured Phase
4 M0 seed A on the four tallest Financial District blocks against an exact pass in float64; each cell is
the share of the exact neighbours a level kept, for points more than 100 m up.

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
{{precision.sunset.after.class_agreement}}. The fault was in training too, so M0 was trained again rather
than read with the new search (`docs/phase6_criteria.md`). Against Phase 4 M0, by the drop rule, these
moved, {{compare.m0_vs_m0phase4.chance.moves_seen}} of {{compare.m0_vs_m0phase4.chance.comparisons}}
where chance alone moves {{compare.m0_vs_m0phase4.chance.moves_expected_by_chance}}:

{{table:moves_m0_vs_m0phase4}}

## The building lift, first round against Phase 4

Lift is (p - p0) / (1 - p0), a precision measure, so it cannot tell a model that finds fewer buildings
from one that finds more of them less precisely. When the lift fell from the first round to Phase 4, the
two were set side by side, median and range over three seeds each
(`scripts/investigations/building_breakdown.py`). The reading was written into `docs/phase4_criteria.md`
after the drop had been seen and before this breakdown ran: coverage up with precision down is a trade,
both down is worse, anything else is written as measured. Both sides carry the old neighbour search and
the scoring from before Phase 6, as the commits d62078f and 5790e15 hold them, since the first round's
predictions were not kept.

{{table:building_change}}

Inference alone separates part of it, on Phase 4 seed A only. In the Financial District its lift is
{{buildings.financial.lift.seed_a.m0_seedA_legacy_iter}} with the first round's block iterator and one
pass, {{buildings.financial.lift.seed_a.m0_seedA_1vote}} with the new iterator and one pass, and
{{buildings.financial.lift.seed_a.four_votes}} with {{votes.n}} passes: on that seed the iterator moved
it and the voting did not. The same change of iterator raises accuracy at home from
{{diagnostic.m0_seedA_legacy_iter.surrey.test.accuracy_labelled_voxel_points}} to
{{diagnostic.m0_seedA_1vote.surrey.test.accuracy_labelled_voxel_points}}, and mean IoU over fifteen
classes from {{diagnostic.m0_seedA_legacy_iter.surrey.test.miou_15}} to
{{diagnostic.m0_seedA_1vote.surrey.test.miou_15}}. The rest, from the first round's
{{buildings.financial.lift.old.median}} ({{buildings.financial.lift.old.min}} to
{{buildings.financial.lift.old.max}}) to {{buildings.financial.lift.seed_a.m0_seedA_legacy_iter}}, lies
with the seeds and the retraining, whose class weights came from {{provenance.dales.train.tiles}} tiles
instead of {{provenance.dales.tiles}} and whose inputs hold no withheld points. Separating those needs
training, and was not done.

## The capacity control

A model with {{train.m0_big_seedA.parameters}} parameters, against the published
{{train.m0_seedA.parameters}}, was trained on the same rented card as the three small seeds in Phase 6,
on the same data and the same number of points seen, with the batch and the learning rate scaled as
`docs/rented_gpu.md` sets out. The rule for adopting it was written before it first ran: its building
lift beats the highest small seed in at least 4 of the {{capacity.areas_with_footprints}} areas with
footprints. It does in {{capacity.areas_big_above_small_max}}. At home it reads
{{scores.m0_big_seedA.surrey.test.accuracy}} against {{seeds.m0.surrey.test.accuracy_from_prediction_cache.median}}
({{seeds.m0.surrey.test.accuracy_from_prediction_cache.min}} to
{{seeds.m0.surrey.test.accuracy_from_prediction_cache.max}}) for the small seeds. The rule is not met,
and the published model is unchanged.

## Water, and the old pair

M1 was the first round's attempt at water: M0prime's recipe plus swissSURFACE3D tiles from Zurich and
Geneva, where water and bridge decks are labelled. Most of that water was never measured. Where a lake
returns no pulse, the survey fills it with points on a regular 25 cm grid and flags them synthetic:
{{flags.swiss_training.water.synthetic.min}} to {{flags.swiss_training.water.synthetic.max}} of the water
on the four Swiss tiles M1 learned from (`scripts/investigations/las_flags_audit.py`), a flag read only
after training. Measured returns alone, water holds a median of
{{realwater.zurich2683_1246.source.density_median}} points per square metre on the Swiss tile with the
most of it and {{realwater.zug2680_1225.source.density_median}} at Zug, against
{{realwater.embarcadero.source.density_median}} at Embarcadero and
{{realwater.baybridge.source.density_median}} at Bay Bridge
(`scripts/investigations/real_water_density.py`). With that little real water to learn from, M1 is not
trained again.

What M1 took from the fill is not settled. {{lattice.swiss_training.water.share_on_the_lattice}} of the
Swiss water points sit on the fill's grid. Moving the Zug water off it takes seed B's water IoU from
{{lattice_test.m1_seedB.as_prepared.water.iou}} to {{lattice_test.m1_seedB.water_jittered.water.iou}},
and leaves seeds A and C where they were, {{lattice_test.m1_seedA.as_prepared.water.iou}} to
{{lattice_test.m1_seedA.water_jittered.water.iou}} and {{lattice_test.m1_seedC.as_prepared.water.iou}}
to {{lattice_test.m1_seedC.water_jittered.water.iou}}. And water lies at the floor of its 40 m block on
both sides, a median of {{water_input.both.switzerland.z_median}} m above it in Switzerland and
{{water_input.both.san_francisco.z_median}} m in San Francisco, where flat ground lies too; height above
the floor is the only height the model reads (`scripts/investigations/water_input_test.py`).

At Zug, on a tile it was never trained on, M1 reaches a water IoU of
{{swiss_holdout.m1_seeds.water.iou.median}} ({{swiss_holdout.m1_seeds.water.iou.min}} to
{{swiss_holdout.m1_seeds.water.iou.max}}). That tile's water is {{flags.2680_1225.water.synthetic}} fill.
At Embarcadero M1 finds {{water.m1_seeds.embarcadero.recall.median}}
({{water.m1_seeds.embarcadero.recall.min}} to {{water.m1_seeds.embarcadero.recall.max}}) of the water.
The rule written before training asked for a positive gain over the water null on every seed at both
places, where the gain is (IoU - IoU0) / (1 - IoU0) against the null's IoU; the water null calls every
point below the sea level at NOAA gauge 9414290 at the moment of flight, plus half a metre, water.

{{table:m1_water}}

The water null is not a do-nothing baseline: it knows each point's absolute height in NAVD88 and the sea level at the moment of flight. The model knows neither; it sees height above the floor of a 40 m block.

M1 and M0prime are the old pair, trained on the laptop with class weights from all 40 tiles and on the
Swiss tiles as they were, predicted again in Phase 4 with the same voting as every other model, and
scored with the Phase 6 rules. Both carry the old neighbour search, and they are read only against each
other. By the drop rule, {{compare.m1_vs_m0prime.chance.moves_seen}} of
{{compare.m1_vs_m0prime.chance.comparisons}} figures moved, where chance alone moves
{{compare.m1_vs_m0prime.chance.moves_expected_by_chance}} (`scripts/pipeline/compare_recipes.py`):

{{table:moves_m1_vs_m0prime}}
