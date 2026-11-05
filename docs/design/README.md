# Design

How the page is laid out and coloured, and why, followed by what this folder holds.

## Layout

The page is one column of short cards that scroll past a 3D scene pinned behind them. Each card moves
the camera to one area. The areas are kilometres apart and only one area's points are loaded at a
time, so the page cuts between them with a short dissolve rather than flying. The cards, in order:

1. Golden Gate Park under the published model: the trees are called vegetation.
2. A held-out Surrey tile, where the model learned, with its accuracy beside the share of ground.
3. The same park under the return-ratio model with its input as delivered: the forest is called
   building. Same checkpoint, only the input changes.
4. The Financial District, taller than any training block.
5. Embarcadero, where the bay is called ground and part of the bridge deck building.
6. M1, the model that learned water from Swiss tiles that were mostly fill.

Each card carries one large figure, one line that sets it against a null or a second figure, and one
line naming the reference; the rest of its figures sit under "more figures".

After the last card the view is free: any area, any of the models, confidence as colour strength, and
a divider dragged between the NAIP photograph on the points and the model's classes. Below 760 px the
cards stack, each with a still, and the free view is left out, because a pinned WebGL scene under
scrolling text does not work on a phone. A desktop that cannot reach the octrees says so, links the
live page and shows the stills the same way; so does the fixture mode the page test runs in, whose
synthetic scene would otherwise sit behind the real figures.

## Four variants in the octrees

Predictions are packed into standard LAS fields, because PotreeConverter 2.1.3 reads extra bytes from
the wrong offset (`docs/potreeconverter_extra_bytes.md`). Those fields hold four class and confidence
pairs, so the octrees carry four variants: the return-ratio model as delivered and with the ratio set
to 0, the published model, and M1.

## Colour

Each class has one OKLCH lightness and one hue per theme, drawn at full chroma. With the free view's
confidence switch on, confidence changes chroma only, so an unsure point keeps its contrast against
the background and neither vanishes nor stands out. It is off by default: at the lowest chroma the
classes fade towards one grey, which a reader who has not asked for confidence takes for a class of
its own. The rules are in `backend/surrey/palette.py`:

- every class colour, at full chroma and at the lowest, has a contrast of at least 3:1 against its
  theme's background, the WCAG 2.1 ratio for graphics, in normal vision and in simulated deuteranopia
  and protanopia;
- the pairs the page's story depends on (building and vegetation, ground and vegetation, ground and
  building, water and ground, water and building, bridge deck and building) are at least 20 apart in
  CIELAB delta E76 in all three visions, and every other pair at least 10;
- building and vegetation differ by at least 0.15 in OKLab lightness, so the forest-as-building frame
  still reads as light against dark where hue fails;
- on the committed pair of bug frames, at least 60% of the difference a normal eye sees survives each
  simulation (Machado 2009, full severity);
- the lowest chroma is the smallest at which every class, at its least confident, is still nearer its
  own colour than any other.

The confidence scale runs from the 5th percentile of confidence on wrong points to the 95th on right
points, measured on the published model's held-out Surrey points
(`scripts/pipeline/confidence_mapping.py`). The same scale colours every model.

`surrey.palette.check_theme` holds the rules for the tokens, `archive/investigations/cvd_bug_frames.py`
measures the frames, and `archive/investigations/sparse_contrast_check.py` checked isolated points over
the bay and on the bridge cables.

## Eye-dome lighting

The shading pass treats empty background as no neighbour, so sparse ground and open water are not
ringed with dark halos. Each theme's strength was chosen by the rule in
`archive/investigations/edl_sweep.py`, recorded in `data/artifacts/edl_sweep.json`.

## Type and numbers

One type family, Recursive, loaded with only the weight, casual and mono axes the page uses: its
casual axis for headings and its mono axis for figures. Body text is 15 px and secondary text 13 px.
Every figure on a card is a registry key with its model variant and its format, resolved by
`scripts/pipeline/page_figures.py`, which refuses a variant that may not stand on that card. A share
prints as a percentage, whole above 10% and to one decimal below; anything else has two decimals
(`web/src/format.js`, checked by `web/tests/format.test.mjs`).

## What is in this folder

- `palette.json`: the class colours in both themes, read by the page and by
  `backend/surrey/palette.py`.
- `cvd_bug_frames.json`: how much of the bug frames' difference survives each simulated colour
  vision, written by `archive/investigations/cvd_bug_frames.py`.
- `sparse_contrast_check.json`: contrast of the sparse returns over the bay and on the bridge cables,
  written by `archive/investigations/sparse_contrast_check.py`.
- `readme/`: the two pictures at the top of the README, shot from the built page over the real
  octrees by `scripts/pipeline/readme_images.py`.
- `frames/`: the four frames `archive/investigations/cvd_bug_frames.py` measures, the return-ratio model's
  forest with the ratio at 0 and as delivered, on both backgrounds, drawn by
  `scripts/pipeline/render_frames.py` and stored as lossless WebP.
