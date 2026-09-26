// The M1 card, the last on the page. M1 is the old second model, trained before the LAS flags were read, on
// Swiss water that was almost all synthetic; the card says so and reads it only against M0prime. The
// outcome is read from the registry (compare_m1_vs_m0prime.json through data/artifacts/quantities.json),
// never chosen by hand: the rule in backend/surrey/m1_criteria.py says M1 succeeds only when its water
// beats the physical null on all three seeds at both Embarcadero and Bay Bridge.

const AREAS = [["embarcadero", "Embarcadero"], ["baybridge", "Bay Bridge"]];

// The same sentence as NULL_DESCRIPTION in backend/surrey/m1_criteria.py, and in docs/method.md where the
// water null is described.
export const NULL_DESCRIPTION = "The water null is not a do-nothing baseline: it knows each point's absolute height in NAVD88 and the sea level at the moment of flight. The model knows neither; it sees height above the floor of a 40 m block.";

function q(figs, key) {
  const f = figs[key];
  return f ? String(f.value) : null;
}

// What the Swiss water was, and what real water measures on both sides. Every figure here was measured
// on the files, with no model involved (scripts/investigations/las_flags_audit.py and
// real_water_density.py).
function source(figs) {
  const v = (key) => q(figs, key) ?? "missing";
  return `<p>Most of what the Swiss tiles label water was never measured. Where a lake sends no pulse back,
      the survey fills it with points on a regular grid and flags them synthetic: on the four Swiss tiles
      M1 learned from, a share of ${v("flags.swiss_training.water.synthetic.min")} to
      ${v("flags.swiss_training.water.synthetic.max")} of the water. M1 was trained before that flag was read,
      so the water it learned is the fill.</p>
    <p>Measured returns alone, water carries a median of ${v("realwater.zurich2683_1246.source.density_median")}
      points per square metre on the Swiss tile with the most of it and ${v("realwater.zug2680_1225.source.density_median")}
      at Zug, against ${v("realwater.embarcadero.source.density_median")} at Embarcadero and
      ${v("realwater.baybridge.source.density_median")} at Bay Bridge. With that little real water to learn
      from, M1 is not trained again.</p>`;
}

// Water at the scale of single points, and the figures that go with it. The two pictures are static:
// the Swiss points are not in the page's data and do not need to be. scripts/pipeline/water_inset.py chose the
// windows by a rule written before it ran, from labels only, and drew them from the tiles as they were
// prepared then, synthetic water included.
function inset(figs) {
  const v = (key) => q(figs, key) ?? "missing";
  const spread = (key) => `${v(`${key}.median`)} (${v(`${key}.min`)} to ${v(`${key}.max`)})`;
  return `<figure class="inset" aria-label="Water from above at the scale of single points, Zug beside Embarcadero">
      <div class="inset-pair">
        <figure><img class="inset" data-still="water_inset_zug"
          alt="Eight metres of the Zug shoreline from above; the points on the water lie on a regular grid and M1 calls them water">
          <figcaption>Zug, the fill</figcaption></figure>
        <figure><img class="inset" data-still="water_inset_embarcadero"
          alt="Eight metres of the Embarcadero shoreline from above; M1 calls the points on the water ground">
          <figcaption>Embarcadero, measured</figcaption></figure>
      </div>
      <figcaption class="small">An 8 m square from above, the points one crop carries, coloured by the old M1 seed A:
        <i data-swatch="water"></i> water <i data-swatch="ground"></i> ground
        <i data-swatch="vegetation"></i> vegetation</figcaption>
    </figure>
    <p>On the fill it does well: on a Zug tile it was never trained on, water IoU
      ${spread("swiss_holdout.m1_seeds.water.iou")}. Of the Zug water, ${v("flags.2680_1225.water.synthetic")} is fill.
      At Embarcadero it finds ${spread("water.m1_seeds.embarcadero.recall")} of the water.</p>
    <p>The fill lies on a 25 cm grid: ${v("lattice.swiss_training.water.share_on_the_lattice")} of the Swiss
      water points sit on it, against ${v("lattice.swiss_training.water.chance_share")} by chance, and San
      Francisco's do not (${v("lattice.embarcadero.water.share_on_the_lattice")} against
      ${v("lattice.embarcadero.water.chance_share")}). Moving the Zug water off that grid takes seed B from
      ${v("lattice_test.m1_seedB.as_prepared.water.iou")} to ${v("lattice_test.m1_seedB.water_jittered.water.iou")},
      and leaves seeds A and C where they were, ${v("lattice_test.m1_seedA.as_prepared.water.iou")} to
      ${v("lattice_test.m1_seedA.water_jittered.water.iou")} and ${v("lattice_test.m1_seedC.as_prepared.water.iou")} to
      ${v("lattice_test.m1_seedC.water_jittered.water.iou")}: the grid's exact places are not all that A and C took
      from the fill.</p>`;
}

export function m1CardState(figs) {
  const success = figs["compare.m1_vs_m0prime.water.success"];
  if (!success) return "not_trained";
  return success.value === 1 ? "success" : "failure";
}

export function m1CardHtml(figs) {
  const state = m1CardState(figs);
  const intro = `<p>M1 is the old second model: M0prime's recipe with swissSURFACE3D tiles from Zurich and Geneva
      added, where water and bridge decks are labelled. M1 and M0prime are the old pair, predicted again with
      the same voting as every other model, and read only against each other.</p>`;
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
  return `${intro}${source(figs)}${verdict}<p>${NULL_DESCRIPTION}</p><div class="fig mono">${rows}</div>
    ${inset(figs)}
    <div class="src mono">water is scored on Embarcadero and Bay Bridge only, against delivery class 9; the band with unlabelled water is in the project notes. Zug, swissSURFACE3D: Federal Office of Topography swisstopo (©swisstopo)</div>`;
}
