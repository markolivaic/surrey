# surrey, three layout directions

Direction B was chosen. A is the fallback by the prototype rule; C is not built.

Stills in the A and C mockups come from **ret0**, the four channel checkpoint with its
return ratio forced to 0. The revised B mockup uses **M0 seed A** for its M0 cards and the
old checkpoint for the bug card, and every still carries its label.

## What every direction answers first

The design gate fails unless this is answered before layout.

**The phenomenon.** A model trained on one Canadian suburb labels a real San Francisco
neighbourhood in three dimensions, and it mostly works: the park's trees come out as
trees, the houses as houses, the streets as ground.

**Is it the first thing on screen.** Yes, in all three. The second thing is the bug
switch, the forest becoming buildings when the input is fed as delivered. The third is what
input cannot fix: the bay called ground and the bridge deck called building.

## Required in every direction

What the page must show:

- A real San Francisco neighbourhood as 3D points, filling most of the first screen,
  within ten seconds. Not a map with pins, not a chart.
- The model working, first. The default state is the fixed input on the published model.
- **Input switch**: as delivered (the bug) / as trained. One click, same camera, same
  frame, so the forest becoming buildings and back is the only thing that moves.
- **Model switch**: M0 / M1. M1 is the old M1, learned on synthetic water, and says so.
- **Orthophoto wipe** on the same 3D view: NAIP colour on the lidar points against the
  model's classes, swept by a handle.
- **Class as hue and lightness, confidence as colour strength (chroma)**, with a two-axis legend
  that says so. Water called ground renders as strong ochre: confident and wrong. That stays.
- The water and bridge failure in text above the fold, and reachable in the interaction
  without looking for it.
- Every number on the page read from the registry, with the variant it came from.

How it must feel:

- Like flying low over a place you could walk to.
- Controls that feel physical: a switch you throw, a handle you drag.
- Evidence you can touch, not a report you read.

Not allowed: a grey city, a pooled number, a 2D chart as the hero, anything on the
page implying M1 results before M1 is trained, reusing a claimed skeleton (anneal's
two-column spread, kessler's full-bleed globe with card, suncokret's 3x3 grid, nadir's
blink comparator stack) or the masthead plus disclosure bar pair.

## Direction A. The wipe

**Skeleton:** split-screen with a draggable divider. Unused.

One oblique view of one area fills the page edge to edge. A vertical divider runs
through the scene: left of it the lidar points carry NAIP colour, right of it they
carry the model's classes. Dragging the divider sweeps imagery into classification
over the same trees. The divider handle carries the input switch, so the bug is
thrown from the place the eye already is.

Chrome is a single line above (what this is, where the data is from) and a rail below:
eight area chips, the model switch, the two-axis legend. The disclosure lives in the
legend. Ground warm charcoal `#1c1a17`, so the ground class and the canopy read
against something that is neither kessler's ground nor nadir's grey. Type: Barlow
Condensed for the line and chips, Space Mono for figures.

Strength: the wipe is the argument, "does it match the photograph", made physical.
Risk: one area at a time; the city as a whole is only in the chips.

## Direction B, chosen. Internal name: the flight

**Skeleton:** single-column scrollytelling, the scene pinned while small cards scroll
past it. Unused. "The flight" is a working name only: areas are kilometres apart, a
literal flight between them would need the whole city loaded, so the page moves between
areas by a cut with a short dissolve and never promises a flight. If the prototype shows a
sparse city layer fits the byte budget, this is reopened; until then, a cut.

Direction A stays as the fallback. It is built instead only if the prototype misses the
scroll or area change thresholds written in `scripts/investigations/measure_prototype.py` after
reasonable tuning.

### Cards, in this order

1. **San Francisco, Golden Gate Park, M0 seed A.** It works, the trees are trees. This is
   the first screen: San Francisco, not Surrey.
2. **Surrey, a held-out tile.** Where the model learned, with its accuracy beside the
   do-nothing baseline on the same thinned points.
3. **The same park on the old four channel model, input as delivered.** The forest turns
   into buildings. Same checkpoint, only the input changes; its figures are the only
   `returns_ood` and `ret0` figures on the page.
4. **Embarcadero.** The bay is ground and the bridge deck is a building; the training
   labels have no word for either. Water and bridge deck are scored on Embarcadero and Bay
   Bridge only, and the card says so.
5. **M1, learning water from swissSURFACE3D tiles.** Disabled until trained. Once trained it
   shows the result whatever it is, failure included, with its figures. The text for
   success and for failure is written before the result.

After card 5 the view is free.

### No bug state for M0

M0 has three inputs and no "as delivered" state. No screen may suggest otherwise. In the
free view the model choice comes first (old model, four channels / M0 / M1), and the
input switch is live only on the old model, labelled "old model, four channels".

### Four variants, a hard ceiling

Predictions travel inside standard LAS fields, because PotreeConverter 2.1.3 misreads extra
bytes (`docs/potreeconverter_extra_bytes.md`). Those fields hold four class and confidence
pairs and no more, so the viewer carries exactly four variants, fixed now:

1. `returns_ood`, the old four channel model, input as delivered (card 3 and the bug)
2. `ret0`, the same checkpoint with the return ratio set to 0 (the bug's other side)
3. `m0_seedA`, the published model
4. `m1_seedA`, the model that learns water and bridge deck, once trained

A fifth cannot be added without another format. M0prime, the control, is scored and reported
and never drawn.

### Free view

Direction A lives here: the divider between NAIP colour and classes, the model choice,
the input switch for the old model, all eight areas loaded on demand, and the two-axis
legend.

### Legend

Built from the classes actually present in the loaded area and variant, never typed.
The bridge shows pole and vegetation points on its cables, the water carries scattered
pole points, Surrey has vehicles; each of those gets its row. Rows are classes, columns
are confidence as colour strength, and the legend quotes the map: faintest at the low end
and below, strongest at the high end and above, both read from the registry. When M1
exists, water and bridge deck join the table for the areas where M1 predicts them.

### Colour: what the numbers are, and the rules

The first palette pulled low confidence toward one light grey. The fix tried first, a grey
per theme (0.68 on light, 0.95 on dark), was **HSV value**, not lightness, and it made the
least sure points the lightest thing on the dark ground. Both are gone.

Now each class has, per theme, one **OKLCH lightness** and one hue, and confidence scales
**chroma** only. Contrast against the ground is set by the class's lightness, so an unsure
point neither vanishes nor stands out. The rules, written before any palette was tried, are
in `backend/surrey/palette.py`; in short:

- **Contrast:** every class colour, at full chroma and at the floor, 3:1 against its
  theme's ground, the WCAG 2.1 success criterion 1.4.11 ratio for graphics needed to
  understand the content, in normal vision and in simulated deuteranopia and protanopia.
- **Story pairs** (building and vegetation, ground and vegetation, ground and building,
  water and ground, water and building, bridge deck and building): CIELAB delta E76 at
  least 20 in all three visions; every other pair at least 10.
- **Lightness:** building and vegetation differ by at least 0.15 in OKLab lightness, so the
  bug reads as light against dark when hue fails.
- **Bug frames:** on the committed pair of bug frames, at least 60 percent of the normal
  vision difference survives each simulation (Machado 2009, full severity).
- **Floor:** the smallest chroma floor at which every class at its lowest confidence is
  still nearest its own colour and keeps a visible tint. The current palette has one (0.35),
  so the legend does not need to say that the lowest confidence hides the class.

The confidence map is not a design choice made while looking at San Francisco. It was
fixed on the published model's held-out Surrey points before any San Francisco frame used
it: the 5th percentile of confidence on wrong points to the 95th on right points
(`scripts/pipeline/confidence_mapping.py`). The same map colours every model.

Checked on tokens (`tests/integrity/test_colour_vision.py`), on the committed bug frames in
both themes, and on real frames at the sparsest places, isolated returns over the bay and the
bridge's cables (`scripts/investigations/sparse_contrast_check.py`). The first palette fails the token rules
and its frames fail the bug rule, and the tests check that they do.

B is light by concept, and a dark switch exists; the palette has a lightness per class for
each.

### Halos

The first frames drew black eye-dome halos that turned dark NAIP into a smudge and light
ground into a comic strip. The prototype showed the cause: the pass darkened every point next
to empty background, so it speckled sparse ground and water whatever the strength. The pass
now treats the background as no neighbour at all, and it is on. Each theme takes the strength
`scripts/investigations/edl_sweep.py` chose by a rule written before the sweep, recorded in
`data/artifacts/edl_sweep.json`. In the light theme that strength still draws isolated points
as discs with dark rims; a weaker pass or a smaller point is open.

### Phones

A pinned WebGL scene under scrolling text is painful on a phone. Below 760 px the cards
stack, each with its own still (or a short clip), and the free view is desktop only, with
one sentence that says so.

### How it has to feel

- Like flying low over a place you could walk to: close, oblique, trees with trunks.
- The 3D is the page. Cards are small, sit at the edge and never cover the scene's middle.
- Switches feel like hardware: a toggle you throw, a handle you drag, both usable from the
  keyboard with a visible focus.
- Evidence you touch, not a report you read.
- Not a generic news explainer. Scrollytelling is a known format and the average one reads
  as a template. One type family with its own axes (Recursive, casual for headings, mono
  for figures), no hero headline over the scene, cards that behave like field notes, and
  the scene itself carry the identity.

### Text on the cards

- No typed number. Every figure is a registry key with its variant, resolved by
  `scripts/pipeline/page_figures.py`, which refuses a variant that may not stand on that card.
- No confidence claim ("sure and wrong") before the home against water comparison of hard
  errors exists for every M0 seed.
- Card 5 suggests no M1 result before training.
- Every proper name has a source, or it is not on the page.
- Stills are regenerated from `m0_seedA` before publishing; the bug still comes from the
  old checkpoint and says so.

## Direction C. The city sheet

**Skeleton:** map-first with a bottom sheet that slides over the map. Unused.

The first screen is San Francisco from above: NAIP over the city, the eight scored areas
drawn as squares, each filled by its own 3D render seen straight down. Tapping a square
raises a bottom sheet to sixty percent of the screen: the oblique 3D view of that area,
the input switch, the model switch, the wipe, and three registry figures for that area.
Pulling the sheet down returns to the city.

Surrey is a ninth square in the corner, the home tile, so "trained there, scored here"
is on the map itself. Map chrome is almost none; the sheet is ink navy `#10222e` with
light text. Type: Manrope, Martian Mono for figures.

Strength: the whole city at once, and each area's own failure is one tap away.
Risk: the first screen is a map, so the 3D must be in the squares from the start or the
ten-second test fails.

## What happens next

The technical prototype (`web/prototype.html`) is measured against the thresholds in
`scripts/investigations/measure_prototype.py`, written before measuring. No page code was written before
that report was answered.
