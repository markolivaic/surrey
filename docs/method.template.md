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

Each prediction is the softmax averaged over {{votes.n}} passes. The number was chosen by a rule written
before any San Francisco figure existed (`docs/phase4_criteria.md`): the smallest of 1, 2, 4, 8 and 16
whose coarse classes agree with twice as many passes on at least 0.99 of the held-out Surrey points.
{{votes.n}} passes and twice as many agree on {{votes.agreement_4}} of them.

Before this repository's first commit, the order in which rules were written and results appeared is
self-reported, in the dates the project wrote into its own artefacts. From the first commit on, the git
history shows it: the retraining's rules are in the first commit, its results in a later one.

## The vegetation reference

The vegetation figures read NAIP 2022 cores between 2 and 25 m above the ground floor. Recall takes the
lower of two masks, one registered to the lidar and one not, because a rule fixed beforehand could not
choose between them (`scripts/pipeline/naip_mask_choice.py`); false vegetation reads the unregistered
mask. In Golden Gate Park the not-vegetation cores hold {{notveg.sf.ggpark.points}} points, a share of
{{notveg.sf.ggpark.share_of_area}} of the area, {{notveg.sf.ggpark.height_median_m}} m up at the median,
so the park's false vegetation figure rests on little (`scripts/investigations/not_vegetation_heights.py`).

## The building lift, first round against the retraining

Lift is precision against the footprints over the null's precision, so it cannot tell a model that finds
less from one that finds more, less precisely. The first round's M0 (its scores as the first commit
holds them) and the retrained M0 are set side by side, median and range over three seeds each, with the
reading written into `docs/phase4_criteria.md` before the comparison ran: coverage up with precision
down is a trade, both down is worse, anything else is written as measured
(`scripts/investigations/building_breakdown.py`).

{{table:building_change}}

Inference alone separates part of it, on the retrained seed A. In the Financial District its lift is
{{buildings.financial.lift.seed_a.m0_seedA_legacy_iter}} with the first round's block iterator and one
pass, {{buildings.financial.lift.seed_a.m0_seedA_1vote}} with the new iterator and one pass, and
{{buildings.financial.lift.seed_a.four_votes}} with {{votes.n}} passes: the iterator moved it, the voting
did not. The same change of iterator raises accuracy at home from
{{diagnostic.m0_seedA_legacy_iter.surrey.test.accuracy_labelled_voxel_points}} to
{{diagnostic.m0_seedA_1vote.surrey.test.accuracy_labelled_voxel_points}}, and mean IoU over fifteen
classes from {{diagnostic.m0_seedA_legacy_iter.surrey.test.miou_15}} to
{{diagnostic.m0_seedA_1vote.surrey.test.miou_15}}. The rest, from the first round's
{{buildings.financial.lift.old.median}} ({{buildings.financial.lift.old.min}} to
{{buildings.financial.lift.old.max}}) to {{buildings.financial.lift.seed_a.m0_seedA_legacy_iter}}, lies
with the seeds and the retraining, whose class weights now come from {{provenance.dales.train.tiles}}
tiles instead of {{provenance.dales.tiles}} and whose inputs hold no withheld points. Separating those
needs training, and was not done.

## The capacity control

A model with {{train.m0_big_seedA.parameters}} parameters, against the published
{{train.m0_seedA.parameters}}, was trained on the same rented card as the three small seeds, on the same
data and the same number of points seen, with the batch and the learning rate scaled as
`docs/rented_gpu.md` sets out. The rule for adopting it was written before it ran: its building lift
beats the highest small seed in at least 4 of the {{capacity.areas_with_footprints}} areas with
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
after training. M1 learned the fill. Measured returns alone, water holds a median of
{{realwater.zurich2683_1246.source.density_median}} points per square metre on the Swiss tile with the
most of it and {{realwater.zug2680_1225.source.density_median}} at Zug, against
{{realwater.embarcadero.source.density_median}} at Embarcadero and
{{realwater.baybridge.source.density_median}} at Bay Bridge
(`scripts/investigations/real_water_density.py`). With that little real water to learn from, M1 is not
trained again.

At Zug, on a tile it was never trained on, M1 reaches a water IoU of
{{swiss_holdout.m1_seeds.water.iou.median}} ({{swiss_holdout.m1_seeds.water.iou.min}} to
{{swiss_holdout.m1_seeds.water.iou.max}}). That tile's water is {{flags.2680_1225.water.synthetic}} fill.
At Embarcadero M1 finds {{water.m1_seeds.embarcadero.recall.median}}
({{water.m1_seeds.embarcadero.recall.min}} to {{water.m1_seeds.embarcadero.recall.max}}) of the water.
The rule written before training asked for a positive lift over the water null on every seed at both
places; the water null calls every point below the sea level at NOAA gauge 9414290 at the moment of
flight, plus half a metre, water.

{{table:m1_water}}

The water null is not a do-nothing baseline: it knows each point's absolute height in NAVD88 and the sea level at the moment of flight. The model knows neither; it sees height above the floor of a 40 m block.

M1 and M0prime are the old pair, trained on the laptop with class weights from all 40 tiles and on the
Swiss tiles as they were, then predicted again with the same voting as every other model. They are read
only against each other. By the drop rule above, M1 falls against M0prime on ground at Embarcadero, on
buildings and vegetation in the Financial District, on vegetation in the Sunset and on four classes at
home, and gains at Bay Bridge, where less water is called ground, and on two classes at home
(`scripts/pipeline/compare_recipes.py`).
