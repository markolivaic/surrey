// The M1 card, the last on the page, written before M1 was trained, for every outcome. The outcome is read from the registry
// (compare_m1_vs_m0prime.json through data/artifacts/quantities.json), never chosen by hand: the rule
// in backend/surrey/m1_criteria.py says M1 succeeds only when its water beats the physical null on
// all three seeds at both Embarcadero and Bay Bridge.

const AREAS = [["embarcadero", "Embarcadero"], ["baybridge", "Bay Bridge"]];

// The same sentence as NULL_DESCRIPTION in backend/surrey/m1_criteria.py; tests/integrity/test_prose.py
// checks that every text naming the water null carries it word for word.
export const NULL_DESCRIPTION = "The water null is not a do-nothing baseline: it knows each point's absolute height in NAVD88 and the sea level at the moment of flight. The model knows neither; it sees height above the floor of a 40 m block.";

function q(figs, key) {
  const f = figs[key];
  return f ? String(f.value) : null;
}

// Water at the scale of single points, and the figures that go with it. The two pictures are static:
// the Swiss points are not in the page's data and do not need to be. scripts/pipeline/water_inset.py chose the
// windows by a rule written before it ran, from labels only, and drew them.
function inset(figs) {
  const v = (key) => q(figs, key) ?? "missing";
  const spread = (key) => `${v(`${key}.median`)} (${v(`${key}.min`)} to ${v(`${key}.max`)})`;
  return `<figure class="inset" aria-label="Water from above at the scale of single points, Zug beside Embarcadero">
      <div class="inset-pair">
        <figure><img class="inset" data-still="water_inset_zug"
          alt="Eight metres of the Zug shoreline from above; M1 calls the points on the water water">
          <figcaption>Zug, never trained on</figcaption></figure>
        <figure><img class="inset" data-still="water_inset_embarcadero"
          alt="Eight metres of the Embarcadero shoreline from above; M1 calls the points on the water ground">
          <figcaption>Embarcadero</figcaption></figure>
      </div>
      <figcaption class="small">An 8 m square from above, the points one crop carries, coloured by M1 seed A:
        <i data-swatch="water"></i> water <i data-swatch="ground"></i> ground
        <i data-swatch="vegetation"></i> vegetation</figcaption>
    </figure>
    <p>M1 learned water: on a Zug tile it was never trained on it finds it with IoU
      ${spread("swiss_holdout.m1_seeds.water.iou")}. At Embarcadero it finds
      ${spread("water.m1_seeds.embarcadero.recall")} of it.</p>
    <p>Spacing is part of the difference. A crop carries the Swiss training water at
      ${v("crop_density.swiss_training.crop_points_per_m2_median")} points per square metre, Zug at
      ${v("crop_density.swiss_held_out.crop_points_per_m2_median")} and San Francisco at
      ${v("crop_density.san_francisco.crop_points_per_m2_median")}, and thinning Embarcadero toward the Swiss
      spacing lifts its water recall from ${v("density.embarcadero.full.water_recall")} to
      ${v("density.embarcadero.thinned.water_recall")} on seed A. Training over a range of densities left it at
      ${v("rented.m1_densityaug_seedA.embarcadero.water_recall")}, inside the band of the three seeds.</p>
    <p>The Swiss water points also sit on a regular 25 cm grid, ${v("lattice.swiss_training.water.share_on_the_lattice")}
      of them against ${v("lattice.swiss_training.water.chance_share")} by chance, and San Francisco's do not
      (${v("lattice.embarcadero.water.share_on_the_lattice")} against ${v("lattice.embarcadero.water.chance_share")}).
      Moving the Zug water off that grid takes seed B from ${v("lattice_test.m1_seedB.as_prepared.water.iou")} to
      ${v("lattice_test.m1_seedB.water_jittered.water.iou")}, and leaves seeds A and C where they were,
      ${v("lattice_test.m1_seedA.as_prepared.water.iou")} to ${v("lattice_test.m1_seedA.water_jittered.water.iou")} and
      ${v("lattice_test.m1_seedC.as_prepared.water.iou")} to ${v("lattice_test.m1_seedC.water_jittered.water.iou")}.
      The rest of the difference is not explained.</p>`;
}

export function m1CardState(figs) {
  const success = figs["compare.m1_vs_m0prime.water.success"];
  if (!success) return "not_trained";
  return success.value === 1 ? "success" : "failure";
}

export function m1CardHtml(figs) {
  const state = m1CardState(figs);
  const intro = "<p>M1 adds swissSURFACE3D tiles from Zurich and Geneva, where water and bridge decks are labelled.</p>";
  if (state === "not_trained") {
    return `${intro}<p>It has not been trained yet. When it has, this card shows what happened, a failure included, with its figures.</p>`;
  }
  const rows = AREAS.map(([key, name]) => {
    const k = q(figs, `compare.m1_vs_m0prime.water.${key}.positive_on`);
    const n = q(figs, `compare.m1_vs_m0prime.water.${key}.seeds`);
    const med = q(figs, `compare.m1_vs_m0prime.water.${key}.lift.median`);
    const lo = q(figs, `compare.m1_vs_m0prime.water.${key}.lift.min`);
    const hi = q(figs, `compare.m1_vs_m0prime.water.${key}.lift.max`);
    return `${name}: lift over the null ${med} (${lo} to ${hi}), positive on ${k} of ${n} seeds`;
  }).join("<br>");
  const verdict = state === "success"
    ? "<p>Against the water null, which calls every point below the measured sea level plus half a metre water, M1's water does better on every seed, at both places the delivery labels water.</p>"
    : "<p>The rule written before training asked M1's water to beat the water null, which calls every point below the measured sea level plus half a metre water, on every seed at both places. It did not, and that is the result.</p>";
  return `${intro}${verdict}<p>${NULL_DESCRIPTION}</p><div class="fig mono">${rows}</div>
    ${inset(figs)}
    <div class="src mono">water is scored on Embarcadero and Bay Bridge only, against delivery class 9; the band with unlabelled water is in the project notes. Zug, swissSURFACE3D: Federal Office of Topography swisstopo (©swisstopo)</div>`;
}
