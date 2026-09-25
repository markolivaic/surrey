// The first screen: the standard's class table as the legend, and a strip that names the data, beside
// the scene as an equal half.
//
// The table is not a legend this project invented. It is Table 17 of the LAS specification, the codes
// a delivery is allowed to use, with the ones the published model has a word for coloured and counted
// and the ones it has no word for struck through. That gap is the project.

// The switches the investigations in scripts/investigations/ compare with exist only in a build made
// with `npm run build:investigate`. The published page reads none of them.
export const INVESTIGATE = import.meta.env.MODE === "investigate";
const params = new URLSearchParams(INVESTIGATE ? location.search : "");
// The first screen needs an area that reads from above. A forest from above is texture, not shape;
// a street grid and a cluster of towers are shape. The park keeps the vegetation card, where it is
// the best thing on the page.
export const HERO_AREA = (params.get("area") || "sunset").toLowerCase();
// Which of the two reveals runs first. 2, chosen: the model's classes in the oblique view from the
// first frame, then the photograph. 1, the alternative: the photograph straight down, then classes,
// then the tilt.
export const ORDER = params.get("order") === "1" ? 1 : 2;
// The opening's oblique view. "mid", chosen: the block runs past both sides and its far edge meets
// the sky. "wide" puts the whole block in the frame, "cards" is the cards' own close framing; both
// stay reachable for comparison.
export const OPENING = ["mid", "wide", "cards"].includes(params.get("open")) ? params.get("open") : "mid";

const fmt = (value) => {
  if (value === null || value === undefined) return "";
  if (typeof value === "number") {
    return Number.isInteger(value) ? value.toLocaleString("en-GB") : String(value);
  }
  return String(value);
};

// One class of this project's taxonomy stands for the standard's code in the legend's swatch: the
// colour is the page's own palette, never a new one.
const SWATCH_CLASS = { 2: "ground", 4: "vegetation", 5: "tree", 6: "habitat", 9: "water", 14: "powerline",
  17: "bridge_deck" };
// The model's own groups with no code in the standard, and one member of each to take the colour from.
const NOCODE = [["pole", "utility-pole"], ["vehicle", "car"], ["fence", "fence"]];
// The standard's codes near a group without covering it, from the document: 15 is a transmission tower
// and 19 an overhead structure such as a traffic light.
const RELATED = { pole: "related: 15, 19" };

export function heroMarkup() {
  return `
    <div class="hero-scene-note mono" aria-hidden="true"></div>
    <img class="still" loading="lazy" alt="San Francisco from the air, lidar points coloured by the published model"
         data-still="hero">
    <div class="hero-text">
      <h1>A model that has only ever seen one city, over another one.</h1>
      <p class="lede">Trained on lidar flown over Surrey, British Columbia. Every point below is San
        Francisco, which it had never seen<span class="wide-only">: ${ORDER === 2
          ? "first as the model calls it, then as the aerial photograph puts it"
          : "first as the aerial photograph puts it, then as the model calls it"}</span><span
        class="phone-only">, coloured as the model calls it</span>. The table is the standard's own list of what a lidar point can be, coloured where
        this model has a word and struck through where it has none.</p>
    </div>
    <figure class="asprs" id="asprs"></figure>
    <div class="strip mono" id="strip"></div>`;
}

export function drawAsprs(table, figs, colourOf) {
  const host = document.getElementById("asprs");
  if (!host || !table) return;
  const rows = table.classes.map((row) => {
    const key = `asprs.${HERO_AREA}.m0_seedA.code${row.code}`;
    const points = figs[`${key}.points`]?.value;
    const share = figs[`${key}.share`]?.value;
    const named = row.m0.length > 0;
    // Code 1 is what a delivery calls a point it has not decided about. No model is asked to predict
    // it, so calling it a gap in the model would be a gap the standard never opened.
    const unclassified = row.code === 1;
    // These two are what the project is about: the model has no word for either, and the delivery
    // labels both, so the comparison can be made and lost.
    const story = !named && !unclassified;
    const swatch = named && SWATCH_CLASS[row.code]
      ? `<i style="background:${colourOf(SWATCH_CLASS[row.code])}"></i>` : '<i class="none"></i>';
    let count = '<span class="dim">no word for it</span>';
    if (named && points !== undefined) count = `${fmt(points)}<span class="dim"> (${fmt(share)})</span>`;
    else if (unclassified) count = '<span class="dim">nothing to predict</span>';
    // The column is headed "in this delivery", so each cell says only what the delivery does.
    const why = row.delivery_labels ? "labelled" : "left blank";
    const kind = named ? "named" : unclassified ? "unclassified" : "missing";
    return `<tr class="${kind}${story ? " story" : ""}">
      <td class="code mono">${row.code}</td>
      <td class="swatch">${swatch}</td>
      <th scope="row">${row.name}</th>
      <td class="count mono">${count}</td>
      <td class="why dim">${why}</td>
    </tr>`;
  }).join("");
  // Drawn on screen in their own colours, so the legend has to name them, but kept apart: they are
  // not the standard's list, they are what the model says that the standard has no code for.
  // One line for the three, each with its mark, its count and its share, so the table stays the
  // standard's list and the model's extra words do not take a row each.
  const extra = NOCODE.map(([group, member]) => {
    const key = `asprs.${HERO_AREA}.m0_seedA.nocode.${group}`;
    const points = figs[`${key}.points`]?.value;
    if (!points) return "";
    const related = RELATED[group] ? ` <span class="dim">${RELATED[group]}</span>` : "";
    return `<span class="nocode-item"><i class="shape-${group}" style="background-color:${colourOf(member)}"></i>
      ${group} <span class="mono">${fmt(points)}<span class="dim"> (${fmt(figs[`${key}.share`]?.value)})</span></span>${related}</span>`;
  }).join("");
  host.innerHTML = `
    <figcaption>What a lidar point is allowed to be</figcaption>
    <table>
      <thead><tr><th class="mono">code</th><th></th><th>the standard's name</th>
        <th>points here</th><th class="why">in this delivery</th></tr></thead>
      <tbody>${rows}</tbody>
      ${extra ? `<tbody class="nocode"><tr><td colspan="5"><div class="nocode-line">
        <span class="dim nocode-head">the model's own words, which the standard has no code for:</span>
        ${extra}</div></td></tr></tbody>` : ""}
    </table>
    <p class="src mono">${table.source}. Counts are the published model's own calls over the area
      drawn behind this; its powerline is counted under 14.</p>`;
}

export function drawStrip(figs) {
  const host = document.getElementById("strip");
  if (!host) return;
  const get = (key) => figs[key]?.value;
  const cells = [
    ["flown", `${get("provenance.sf.flight_date")}`],
    ["imagery", `NAIP ${get("provenance.sf.naip_year")}, ${fmt(get("provenance.sf.days_from_imagery_to_lidar"))} days apart`],
    // Both densities: what was flown, and what the point budget lets the page draw of it here. The
    // second is why the picture is grainy, and saying so costs nothing.
    ["density", `flown at ${fmt(get(`provenance.sf.${HERO_AREA}.points_per_m2`))} points per square metre` +
      (get(`provenance.sf.${HERO_AREA}.drawn_points_per_m2`) === undefined ? ""
        : `, drawn here at about ${Math.round(get(`provenance.sf.${HERO_AREA}.drawn_points_per_m2`))}`)],
    ["sea level", `${fmt(get("waternull.sf.embarcadero.gauge_level_navd88_m.median"))} m NAVD88 at the Embarcadero flight time`],
    // The training split only: the held-out tiles were prepared with the rest and never trained on.
    ["trained on", `${fmt(get("provenance.dales.train.tiles"))} mbendjilali/DALES-2 tiles, ` +
      `${fmt(get("surrey.train.voxel.training_points"))} points`],
    ["held out", `${fmt(get("provenance.dales.test.tiles"))} tiles`],
  ];
  host.innerHTML = cells
    .map(([label, value]) => `<span class="cell"><span class="dim">${label}</span> ${value}</span>`)
    .join("");
}
