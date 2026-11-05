// The M1 card, the last on the page. M1 is the old second model, trained before the LAS flags were read, on
// Swiss water that was almost all synthetic; the card says so and reads it only against M0prime. The
// outcome is read from the registry (compare_m1_vs_m0prime.json through data/artifacts/quantities.json),
// never chosen by hand: the rule in backend/surrey/m1_criteria.py says M1 succeeds only when its water
// beats the physical null on all three seeds at both Embarcadero and Bay Bridge.

import { count, num, share } from "./format.js";

const AREAS = [["embarcadero", "Embarcadero"], ["baybridge", "Bay Bridge"]];

// The same sentence as NULL_DESCRIPTION in backend/surrey/m1_criteria.py, and in docs/method.md where the
// water null is described.
export const NULL_DESCRIPTION = "The water null is not a do-nothing baseline: it knows each point's absolute height in NAVD88 and the sea level at the moment of flight. The model knows neither; it sees only where points lie relative to each other.";

function q(figs, key, fmt = num) {
  const f = figs[key];
  return f ? fmt(f.value) : "missing";
}

// The two pictures are static: the Swiss points are not in the page's data and do not need to be.
// scripts/pipeline/water_inset.py chose the windows by a rule written before it ran, from labels only,
// and drew them from the tiles as they were prepared then, synthetic water included.
const INSET = `<figure class="inset" aria-label="Water from above at the scale of single points, Zug beside Embarcadero">
      <div class="inset-pair">
        <figure><img class="inset" loading="lazy" width="320" height="320" data-still="water_inset_zug"
          alt="Eight metres of the Zug shoreline from above; the points on the water lie on a regular grid and M1 calls them water">
          <figcaption>Zug, the fill</figcaption></figure>
        <figure><img class="inset" loading="lazy" width="320" height="320" data-still="water_inset_embarcadero"
          alt="Eight metres of the Embarcadero shoreline from above; M1 calls the points on the water ground">
          <figcaption>Embarcadero, measured</figcaption></figure>
      </div>
      <figcaption class="small">An 8 m square from above, coloured by the old M1 seed A:
        <i data-swatch="water"></i> water <i data-swatch="ground"></i> ground
        <i data-swatch="vegetation"></i> vegetation</figcaption>
    </figure>`;

// Everything past the card's one figure: the rule and its outcome, the source, and the grid. Every
// figure on the source was measured on the files, with no model involved
// (archive/investigations/las_flags_audit.py and real_water_density.py).
function more(figs) {
  const v = (key, fmt) => q(figs, key, fmt);
  const spread = (key, fmt) => `${v(`${key}.median`, fmt)} (${v(`${key}.min`, fmt)} to ${v(`${key}.max`, fmt)})`;
  const success = figs["compare.m1_vs_m0prime.water.success"]?.value === 1;
  const verdict = success
    ? "Against the water null, which calls every point below the measured sea level plus half a metre water, M1's water does better on every seed, at both places the delivery labels water."
    : "The rule written before training asked M1's water to beat the water null, which calls every point below the measured sea level plus half a metre water, on every seed at both places. It did not.";
  const rows = AREAS.map(([key, name]) => `${name}: gain over the null ${v(`compare.m1_vs_m0prime.water.${key}.lift.median`)}
      (${v(`compare.m1_vs_m0prime.water.${key}.lift.min`)} to ${v(`compare.m1_vs_m0prime.water.${key}.lift.max`)}),
      positive on ${v(`compare.m1_vs_m0prime.water.${key}.positive_on`, count)} of ${v(`compare.m1_vs_m0prime.water.${key}.seeds`, count)} seeds`).join("<br>");
  return `<details class="more"><summary>more figures</summary>
    <p>M1 is M0prime's recipe with swissSURFACE3D tiles from Zurich and Geneva added, where water and bridge
      decks are labelled; the two are the old pair, predicted again with the same voting as every other model,
      and read only against each other.</p>
    <p>${verdict} ${NULL_DESCRIPTION}</p>
    <p class="mono">${rows}</p>
    <p>On the four Swiss tiles M1 learned from, ${v("flags.swiss_training.water.synthetic.min", share)} to
      ${v("flags.swiss_training.water.synthetic.max", share)} of what is labelled water is flagged synthetic, and
      ${v("flags.2680_1225.water.synthetic", share)} of the Zug water. Measured returns alone, water carries a median of
      ${v("realwater.zurich2683_1246.source.density_median")} points per square metre on the Swiss tile with the most of it and
      ${v("realwater.zug2680_1225.source.density_median")} at Zug, against ${v("realwater.embarcadero.source.density_median")}
      at Embarcadero and ${v("realwater.baybridge.source.density_median")} at Bay Bridge. With that little real water to
      learn from, M1 is not trained again.</p>
    <p>The fill lies on a 25 cm grid: ${v("lattice.swiss_training.water.share_on_the_lattice", share)} of the Swiss water
      points sit on it, against ${v("lattice.swiss_training.water.chance_share", share)} by chance, and San Francisco's do
      not (${v("lattice.embarcadero.water.share_on_the_lattice", share)} against
      ${v("lattice.embarcadero.water.chance_share", share)}). Moving the Zug water off that grid takes seed B's water IoU from
      ${v("lattice_test.m1_seedB.as_prepared.water.iou")} to ${v("lattice_test.m1_seedB.water_jittered.water.iou")}, and leaves
      seeds A and C where they were, ${v("lattice_test.m1_seedA.as_prepared.water.iou")} to
      ${v("lattice_test.m1_seedA.water_jittered.water.iou")} and ${v("lattice_test.m1_seedC.as_prepared.water.iou")} to
      ${v("lattice_test.m1_seedC.water_jittered.water.iou")}. What M1 took from the fill is not settled: on both sides water
      is flat, within ${v("water_input.both.switzerland.z_median")} m of the 1st percentile of height in its 40 m block in
      Switzerland and ${v("water_input.both.san_francisco.z_median")} m in San Francisco, and flat ground has the same local
      shape, which is all the model sees.</p>
    <p>Water IoU on the Zug tile over the seeds: ${spread("swiss_holdout.m1_seeds.water.iou")}. Water found at Embarcadero:
      ${spread("water.m1_seeds.embarcadero.recall", share)}.</p>
  </details>`;
}

export function m1CardHtml(figs) {
  const v = (key, fmt) => q(figs, key, fmt);
  return `<p>M1 is an older model, trained with Swiss tiles added where water is labelled. Most of that water
      was never measured: where a lake sends no pulse back, the survey fills it with points on a regular grid.</p>
    ${INSET}
    <div class="fig">
      <div class="fig-label">water found at Embarcadero by the old M1, median of the seeds</div>
      <div class="fig-big mono">${v("water.m1_seeds.embarcadero.recall.median", share)}</div>
      <div class="fig-cmp">On a Zug tile it never trained on, mostly fill, water IoU
        <span class="mono">${v("swiss_holdout.m1_seeds.water.iou.median")}</span>.</div>
    </div>
    <div class="src mono">reference: water as the delivery labels it; Zug, swissSURFACE3D: Federal Office of Topography swisstopo (©swisstopo)</div>
    ${more(figs)}`;
}
