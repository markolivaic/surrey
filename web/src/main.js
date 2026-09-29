// The page: a first screen and six cards over a pinned 3D scene, then a free view. Every number comes from
// figures.json, which scripts/pipeline/page_figures.py resolves from the registry and refuses to write when a
// figure's model may not stand where it is placed.

import palette from "../../docs/design/palette.json";
// Eye-dome strength per theme, chosen by the palette's rules in scripts/investigations/edl_sweep.py, never by hand.
import edlSweep from "../../data/artifacts/edl_sweep.json";
import confidenceMap from "../../data/artifacts/confidence_mapping.json";
import { m1CardHtml, m1CardState } from "./m1card.js";
import { HERO_AREA, INVESTIGATE, OPENING, ORDER, drawAsprs, drawStrip, heroMarkup } from "./hero.js";
import { FINE, GROUP_OF } from "./palette.js";
import { Viewer } from "./viewer.js";

const params = new URLSearchParams(location.search);
// Fixture mode is for the CI smoke test only: every area is the synthetic FIXTURE octree, and the
// page says so on screen.
const FIXTURE = params.get("fixture") === "1";
const DATA = FIXTURE ? "fixtures" : (params.get("data") || "data");
const areaName = (name) => (FIXTURE ? "FIXTURE" : name);
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
const narrow = matchMedia("(max-width: 760px)").matches;
const $ = (id) => document.getElementById(id);

const AREAS = [
  ["ggpark", "Golden Gate Park"], ["embarcadero", "Embarcadero"], ["baybridge", "Bay Bridge"],
  ["financial", "Financial District"], ["soma", "SoMa"], ["mission", "Mission"],
  ["sunset", "Sunset"], ["steep", "Russian Hill"],
];
const VARIANT_OF = { old: { delivered: "returns_ood", zero: "ret0" }, m0: "m0_seedA", m1: "m1_seedA" };

// The theme a reader chose on this page is kept for the next visit; until they choose, the system's
// preference decides, and follows it if it changes. index.html sets it before the first paint.
const THEME_KEY = "surrey-theme";
const systemDark = matchMedia("(prefers-color-scheme: dark)");
function chosenTheme() {
  try { return localStorage.getItem(THEME_KEY); } catch { return null; }
}

const state = { theme: document.documentElement.dataset.theme === "dark" ? "dark" : "light", figs: {}, free: false, area: "ggpark", model: "m0", delivered: true, legendVariant: "m0_seedA" };

function oklch(L, C, hDeg) {
  const h = (hDeg * Math.PI) / 180;
  const a = C * Math.cos(h), b = C * Math.sin(h);
  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3;
  const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3;
  const s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3;
  const lin = [4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s];
  const enc = (c) => { c = Math.min(Math.max(c, 0), 1); return c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055; };
  return `rgb(${lin.map((c) => Math.round(enc(c) * 255)).join(",")})`;
}

function drawLegend(manifest, variant) {
  const present = manifest?.classes_present?.[variant];
  if (!present) { $("legend").hidden = true; return; }
  const groups = new Map();
  for (const [id, n] of Object.entries(present)) {
    const g = GROUP_OF[FINE[Number(id)]];
    groups.set(g, (groups.get(g) || 0) + n);
  }
  const tokens = palette.themes[state.theme].tokens;
  const f = palette.floor;
  const rows = [...groups.entries()].sort((a, b) => b[1] - a[1]).map(([g]) => `<tr><td>${g}</td>${[0, 0.5, 1]
    .map((t) => `<td><i style="background:${oklch(tokens[g].L, tokens[g].C * (f + (1 - f) * t), tokens[g].h)}"></i></td>`).join("")}</tr>`).join("");
  $("legend").hidden = false;
  $("legend").innerHTML = `<table><tr><th>class present here</th><th colspan="3">confidence, as colour strength</th></tr>
    <tr><th></th><th class="mono">${confidenceMap.low}</th><th></th><th class="mono">${confidenceMap.high}</th></tr>${rows}</table>
    <div class="note mono">faintest at ${confidenceMap.low} and below, strongest at ${confidenceMap.high} and above; the map is fixed on held-out Surrey points and is the same for every model</div>`;
}

function colourOfClass(name) {
  const group = GROUP_OF[name] || name;
  const token = palette.themes[state.theme].tokens[group];
  return token ? oklch(token.L, token.C, token.h) : "transparent";
}

function fillFigures() {
  document.querySelectorAll("[data-q]").forEach((el) => {
    const f = state.figs[el.dataset.q];
    const v = f?.value;
    // A card's figures carry two decimals, as data-digits says; everything else is shown as the registry holds it.
    const digits = el.dataset.digits;
    const shown = typeof v === "number" && digits !== undefined ? v.toFixed(Number(digits))
      : Number.isInteger(v) && Math.abs(v) >= 10000 ? v.toLocaleString("en-US") : String(v);
    el.textContent = f ? shown : "missing";
    el.title = f ? `registry ${el.dataset.q}, variant ${f.variant}` : "";
  });
  $("m1card-text").innerHTML = m1CardHtml(state.figs);
  const trained = m1CardState(state.figs) !== "not_trained";
  $("m1card").classList.toggle("disabled", !trained);
  $("m1button").disabled = !trained;
  $("m1button").textContent = trained ? "old M1, synthetic water" : "old M1, not trained yet";
}

function stillsFor(theme) {
  const ground = theme === "dark" ? "131815" : "e6e9e6";
  document.querySelectorAll("img.still, img.inset").forEach((img) => {
    // A still stands in for the scene on a phone, so a wider screen never fetches one. The insets are
    // part of the water card on every screen.
    if (img.classList.contains("still") && !narrow) return;
    // The first screen's still follows whichever area it draws; the cards name their own.
    const name = img.dataset.still === "hero" ? `${HERO_AREA}_m0_seedA` : img.dataset.still;
    img.src = `stills/${name}_${ground}.webp`;
  });
}

// Swatches the cards draw from the palette by name, so they change with the theme like the points do.
function paintSwatches() {
  document.querySelectorAll("i[data-swatch]").forEach((i) => { i.style.background = colourOfClass(i.dataset.swatch); });
}

async function main() {
  state.figs = await (await fetch("figures.json")).json();
  // The standard's own class table, written by scripts/pipeline/page_figures.py from backend/surrey/asprs.py,
  // so the page and the registry read one list and not two.
  const asprsTable = await (await fetch("asprs.json")).json().catch(() => null);
  fillFigures();
  stillsFor(state.theme);
  paintSwatches();

  const applyTheme = (theme) => {
    state.theme = theme;
    document.documentElement.dataset.theme = theme;
    $("theme").setAttribute("aria-pressed", String(theme === "dark"));
    $("theme").textContent = theme === "dark" ? "light" : "dark";
    stillsFor(theme);
    paintSwatches();
    drawAsprs(asprsTable, state.figs, colourOfClass);
    if (viewer) {
      viewer.setTheme(theme);
      viewer.setEdl(edlFor(theme));
      if (viewer.current) drawLegend(viewer.current.manifest, state.legendVariant);
      if (state.onHero) $("legend").hidden = true;
    }
  };
  $("theme").setAttribute("aria-pressed", String(state.theme === "dark"));
  $("theme").textContent = state.theme === "dark" ? "light" : "dark";
  $("theme").onclick = () => {
    const next = state.theme === "light" ? "dark" : "light";
    try { localStorage.setItem(THEME_KEY, next); } catch { /* the choice lasts this visit only */ }
    applyTheme(next);
  };
  systemDark.addEventListener("change", (e) => { if (!chosenTheme()) applyTheme(e.matches ? "dark" : "light"); });

  // The first screen: the standard's table and the strip that names the data. Drawn here, before the
  // point where phones stop, because none of it needs WebGL and a phone that stopped first would show
  // a header and nothing else.
  const hero = $("hero");
  if (hero) {
    hero.dataset.area = HERO_AREA;
    hero.innerHTML = heroMarkup();
    drawAsprs(asprsTable, state.figs, colourOfClass);
    drawStrip(state.figs);
    stillsFor(state.theme);
  }

  // Phones get the stills and the text, and no WebGL at all.
  if (narrow) return;

  viewer = new Viewer($("stage"), DATA, reducedMotion);
  if (FIXTURE) {
    const banner = document.createElement("div");
    banner.className = "fixture mono";
    banner.id = "fixture-banner";
    banner.textContent = "FIXTURE: synthetic points and random confidences for the smoke test, not data and not predictions";
    document.body.appendChild(banner);
  }
  // Eye-dome strength: each theme's own, from the sweep. A ?edl= parameter overrides both, for
  // measuring.
  const edlFor = (theme) => (INVESTIGATE && params.get("edl") !== null
    ? Number(params.get("edl")) : edlSweep.chosen[theme].strength);
  if (state.theme !== "light") viewer.setTheme(state.theme);
  viewer.setEdl(edlFor(state.theme));
  window.__surrey = viewer.metrics;
  if (INVESTIGATE) {
    window.__surreyOrbit = (on) => { viewer.orbit = on; };
    window.__surreyEdl = (s) => viewer.setEdl(s);
    window.__surreyViewer = viewer;
  }
  viewer.onFrame = (v) => { $("loading").hidden = !(v.current && v.metrics.areas[v.current.name]?.loading > 0); };

  // Shows an area, or says on screen that it could not be loaded. Returns null when the area failed or
  // a later card has taken the scene in the meantime.
  async function showArea(name, variant, options) {
    try {
      const shown = await viewer.show(name, variant, options);
      return shown.stale ? null : shown;
    } catch (err) {
      console.warn(`area ${name} could not be loaded`, err);
      $("chip").textContent = "this area could not be loaded";
      return null;
    }
  }

  async function showCard(el) {
    const onHero = el.id === "hero";
    const name = areaName(el.dataset.area);
    let variant = el.dataset.variant;
    const shown = await showArea(name, variant);
    if (!shown) return;
    const { entry, variant: v } = shown;
    let chip = el.dataset.chip;
    if (!v && el.dataset.fallbackVariant) {
      variant = el.dataset.fallbackVariant;
      if (!(await showArea(name, variant))) return;
      chip = "M1 not trained: this frame is M0 seed A";
    }
    // A card can ask for one of the opening's framings instead of the low oblique view.
    if (el.dataset.framing) viewer.setPitch(1, el.dataset.framing);
    state.legendVariant = variant;
    $("chip").textContent = chip;
    drawLegend(entry.manifest, variant);
    // Two legends on one screen is one too many: on the first screen the standard's table is the
    // legend, and the confidence key belongs to the cards that talk about confidence.
    state.onHero = onHero;
    if (onHero) $("legend").hidden = true;
    // On the first screen the bottom left belongs to the provenance strip, so the variant label
    // moves across to the side the scene is on.
    document.body.dataset.onHero = onHero ? "1" : "";
  }

  function variantForFree() {
    if (state.model === "old") return state.delivered ? VARIANT_OF.old.delivered : VARIANT_OF.old.zero;
    return VARIANT_OF[state.model];
  }

  async function showFree(keepCamera) {
    const variant = variantForFree();
    const shown = await showArea(areaName(state.area), variant, { keepCamera });
    if (!shown) return;
    const { entry } = shown;
    state.legendVariant = variant;
    const chips = { returns_ood: "old model, four channels, input as delivered", ret0: "old model, four channels, return ratio set to 0",
      m0_seedA: "M0 seed A, the published model", m1_seedA: "old M1 seed A, learned on synthetic water" };
    $("chip").textContent = chips[variant];
    drawLegend(entry.manifest, variant);
  }

  function setFree(on) {
    state.free = on;
    viewer.controls.enabled = on;
    // The page's bottom edge belongs to the free view's panel while it is shown.
    document.body.dataset.free = on ? "1" : "";
    if (on) {
      // A reader can scroll from the first screen straight to the free view; the first screen's reveal
      // must stop there, or it keeps painting the photograph and its label over the free view.
      state.onHero = false;
      document.body.dataset.onHero = "";
    }
    if (!on) setDivider(false);
  }

  // The first screen's own state, kept where the observer can put it back.
  let heroReveal = null;

  // Three reveals on the first screen, in this order and all from data the page already has:
  // the aerial photograph on the points, read straight down, so the place is recognised; then the
  // colour crossing from the photograph to the model's classes, which is the divider run as a
  // reveal rather than as a control; then the tilt that shows it was never flat.
  // With reduced motion there is no reveal: classes, oblique, from the first frame.
  if (hero) {
    await showCard(hero);
    if (reducedMotion) {
      viewer.wipe = 0;
      viewer.setPitch(1, OPENING);
    } else {  // eslint-disable-line no-lonely-if
      const PHOTO_UNTIL = 0.45;   // the first stretch of the scroll carries the crossing
      const PHOTO_CHIP = "the aerial photograph on the points, not the model";
      const reveal = () => {
        // Only while the first screen is the one shown. Once a card has the scene, its colours, its
        // camera and its label are the card's; the scroll that goes on past the first screen must not
        // reach back and paint the photograph over them.
        const box = hero.getBoundingClientRect();
        if (!state.onHero || box.bottom <= 0) return;
        const through = Math.min(Math.max(-box.top / Math.max(box.height * 0.8, 1), 0), 1);
        const crossed = Math.min(through / PHOTO_UNTIL, 1);
        if (ORDER === 2) {
          // Classes in the oblique view from the first frame; the photograph is what the scroll
          // brings, as the proof that the colours sit on a real place.
          viewer.wipe = crossed;
          viewer.setPitch(1, OPENING);
        } else {
          viewer.wipe = 1 - crossed;
          viewer.setPitch(Math.max((through - PHOTO_UNTIL) / (1 - PHOTO_UNTIL), 0), OPENING);
        }
        // Whichever colour holds most of the screen, the label says which one it is.
        const onPhoto = ORDER === 2 ? crossed > 0.5 : crossed < 0.5;
        $("chip").textContent = onPhoto ? PHOTO_CHIP : hero.dataset.chip;
      };
      reveal();
      heroReveal = reveal;
      addEventListener("scroll", reveal, { passive: true });
    }
  }

  const cards = [...document.querySelectorAll(".card")];
  const watch = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      if (e.target.id === "free") {
        setFree(true);
        showFree(false);
        continue;
      }
      setFree(false);
      // Leaving free view puts the divider back to nothing, which would wipe the first screen's
      // photograph away; the reveal owns that state while the first screen is on screen.
      if (e.target.id === "hero") {
        showCard(e.target).then(() => heroReveal && heroReveal());
        continue;
      }
      viewer.wipe = 0;
      showCard(e.target);
      // Preload the next card's area: its hierarchy and coarse root, nothing more.
      const next = cards[cards.indexOf(e.target) + 1];
      if (next) viewer.area(areaName(next.dataset.area));
    }
  }, { rootMargin: "-45% 0px -45% 0px" });
  [...(hero ? [hero] : []), ...cards, $("free")].forEach((el) => watch.observe(el));

  // The credits close the page under the scene's fixed labels; while they are on screen the labels
  // step aside, so nothing is drawn over the credits' text.
  const credits = document.querySelector(".credits");
  if (credits) {
    new IntersectionObserver(([e]) => { document.body.dataset.credits = e.isIntersecting ? "1" : ""; })
      .observe(credits);
  }

  // Free view controls.
  const areasEl = $("areas");
  areasEl.insertAdjacentHTML("afterbegin", "area ");
  for (const [key, label] of AREAS) {
    const b = document.createElement("button");
    b.textContent = label;
    b.dataset.area = key;
    b.setAttribute("aria-pressed", String(key === state.area));
    b.onclick = () => {
      state.area = key;
      areasEl.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      showFree(false);
    };
    areasEl.appendChild(b);
  }
  document.querySelectorAll("#model button[data-model]").forEach((b) => {
    b.onclick = () => {
      state.model = b.dataset.model;
      document.querySelectorAll("#model button[data-model]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      const old = state.model === "old";
      $("input").disabled = !old;
      $("input").setAttribute("aria-pressed", String(old && state.delivered));
      $("inputnote").textContent = old ? "old model, four channels" : "only the old model has this state";
      showFree(true);
    };
  });
  $("input").onclick = () => {
    state.delivered = !state.delivered;
    $("input").setAttribute("aria-pressed", String(state.delivered));
    showFree(true);
  };

  // The divider: NAIP colour left of the handle, classes right of it. Pointer and keyboard.
  let wipe = 0.5;
  // The handle stands over the scene wherever the scene is: the whole window, or the free view's frame.
  function setWipe(x) {
    wipe = Math.min(1, Math.max(0, x));
    const r = $("stage").getBoundingClientRect();
    $("handle").style.left = `${r.left + wipe * r.width}px`;
    $("handle").style.top = `${r.top}px`;
    $("handle").style.height = `${r.height}px`;
    viewer.wipe = $("handle").hidden ? 0 : wipe;
  }
  function setDivider(on) {
    const areaHasPhoto = !String(state.area).match(/^\d/);
    const show = on && state.free && areaHasPhoto;
    $("wipe").setAttribute("aria-pressed", String(show));
    $("handle").hidden = !show;
    setWipe(wipe);
  }
  $("wipe").onclick = () => setDivider($("wipe").getAttribute("aria-pressed") !== "true");
  // The scene's frame changes size between the cards and the free view; the handle follows it.
  new ResizeObserver(() => { if (!$("handle").hidden) setWipe(wipe); }).observe($("stage"));
  const grip = $("grip");
  grip.addEventListener("pointerdown", (e) => {
    grip.setPointerCapture(e.pointerId);
    const move = (ev) => {
      const r = $("stage").getBoundingClientRect();
      setWipe((ev.clientX - r.left) / r.width);
    };
    grip.addEventListener("pointermove", move);
    grip.addEventListener("pointerup", () => grip.removeEventListener("pointermove", move), { once: true });
  });
  grip.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft") { setWipe(wipe - 0.02); e.preventDefault(); }
    if (e.key === "ArrowRight") { setWipe(wipe + 0.02); e.preventDefault(); }
  });
}

let viewer = null;
main();
