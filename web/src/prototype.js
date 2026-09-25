// Technical prototype: can the real clouds load and turn smoothly in a page shaped like
// design B? Instrumented for scripts/investigations/measure_prototype.py; nothing here is the page.

import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { DecoderPool, Octree } from "./potree2/octree.js";
import { EDLPass } from "./render/edl.js";
import { pointMaterial, selectVariant } from "./render/material.js";
import { legendFor } from "./palette.js";
import confidenceMap from "../../data/artifacts/confidence_mapping.json";

const q = new URLSearchParams(location.search);
const DATA = (q.get("data") || "http://127.0.0.1:8772").replace(/\/$/, "");
const BUDGET = Number(q.get("budget") || 1_000_000);
const PRELOAD = q.get("preload") !== "0";
// "loaded" grows the node set a level at a time; "visibility" plans it from the hierarchy (octree.js).
const ORDER = q.get("order") === "loaded" ? "loaded" : "visibility";
const VARIANTS = ["returns_ood", "ret0", "m0_seedA"];
const BG = q.get("bg") || "#e6e9e6";

const metrics = { order: ORDER, budget: BUDGET, firstFrameMs: null, frames: [], switches: [], renderer: null, bytes: 0, errors: [], areas: {} };
window.__surrey = metrics;
window.addEventListener("error", (e) => metrics.errors.push(String(e.message)));

const stage = document.getElementById("stage");
const renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.setClearColor(BG);
stage.appendChild(renderer.domElement);
{
  const gl = renderer.getContext();
  const ext = gl.getExtension("WEBGL_debug_renderer_info");
  metrics.renderer = ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
}

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 1, 5000);
camera.up.set(0, 0, 1);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
// While the cards lead, the wheel scrolls the page; the camera is the reader's only in the free view.
controls.enabled = false;
const edl = new EDLPass(renderer, BG);
edl.material.uniforms.uStrength.value = Number(q.get("edl") ?? 0.5);

const pool = new DecoderPool();
const areas = new Map();

async function area(name) {
  if (areas.has(name)) return areas.get(name).ready;
  const entry = { name, ready: null };
  areas.set(name, entry);
  entry.ready = (async () => {
    // Slot order comes from the area manifest, never from the page.
    const manifest = await (await fetch(`${DATA}/${name}/manifest.json`)).json();
    const material = pointMaterial(manifest.variants, BG === "#131815" ? "dark" : "light", confidenceMap);
    const oct = await new Octree(`${DATA}/${name}`, material, pool, manifest.variants).init();
    entry.preloadStartedAt = performance.now();
    entry.oct = oct;
    entry.material = material;
    entry.manifest = manifest;
    const c = oct.root.box.getCenter(new THREE.Vector3());
    const size = oct.root.box.getSize(new THREE.Vector3());
    entry.target = new THREE.Vector3(c.x, c.y, oct.root.box.min.z + size.z * 0.15);
    entry.eye = new THREE.Vector3(c.x - size.x * 0.55, c.y - size.y * 0.75, oct.root.box.min.z + size.x * 0.55);
    oct.group.visible = false;
    scene.add(oct.group);
    await oct.loadNode(oct.root);
    // What preloading an area costs: hierarchy chunk, metadata-driven root node, nothing more.
    (metrics.areas[name] || (metrics.areas[name] = { idleSince: null })).preloadBytes = oct.bytesFetched;
    return entry;
  })();
  return entry.ready;
}

let current = null;
let switchRecord = null;

async function show(name, variant) {
  if (!name) return;
  const t0 = performance.now();
  const same = current && current.name === name;
  switchRecord = same ? null : { from: current?.name ?? null, to: name, t0, firstFrameMs: null };
  if (!same) renderer.domElement.style.opacity = "0";
  const entry = await area(name);
  selectVariant(entry.material, variant);
  if (!same) {
    if (current) {
      // The link is the thing in short supply at 20 Mbit/s: the area being left stops fetching.
      metrics.stoppedOnSwitch = (metrics.stoppedOnSwitch || 0) + current.oct.stopLoading();
      current.oct.group.visible = false;
    }
    entry.oct.group.visible = true;
    camera.position.copy(entry.eye);
    controls.target.copy(entry.target);
    controls.update();
    current = entry;
    metrics.switches.push(switchRecord);
  }
  document.getElementById("legend").textContent = legendFor(entry.manifest.classes_present[variant])
    .map((g) => g.group).join(", ");
}

// Cards drive the view; after the last card the view is free.
const observer = new IntersectionObserver((entries) => {
  for (const e of entries) {
    if (!e.isIntersecting) continue;
    const el = e.target;
    const free = el.dataset.free === "1";
    document.getElementById("free").classList.toggle("on", free);
    document.getElementById("wipe").classList.toggle("on", free);
    controls.enabled = free;
    if (el.dataset.area) show(el.dataset.area, el.dataset.variant);
    const next = el.nextElementSibling;
    if (PRELOAD && next?.dataset.area) area(next.dataset.area);
  }
}, { rootMargin: "-45% 0px -45% 0px" });
document.querySelectorAll(".card").forEach((c) => observer.observe(c));

const select = document.getElementById("variant");
VARIANTS.forEach((v) => select.add(new Option(v, v)));
select.onchange = () => current && show(current.name, select.value);

// The divider: pointer and keyboard.
const wipe = document.getElementById("wipe");
let wipeX = 0.5;
function setWipe(x) {
  wipeX = Math.min(1, Math.max(0, x));
  wipe.style.left = `${wipeX * 100}%`;
}
setWipe(0.5);
const handle = document.getElementById("handle");
handle.addEventListener("pointerdown", (e) => {
  handle.setPointerCapture(e.pointerId);
  const move = (ev) => setWipe(ev.clientX / innerWidth);
  handle.addEventListener("pointermove", move);
  handle.addEventListener("pointerup", () => handle.removeEventListener("pointermove", move), { once: true });
});
handle.addEventListener("keydown", (e) => {
  if (e.key === "ArrowLeft") setWipe(wipeX - 0.02);
  if (e.key === "ArrowRight") setWipe(wipeX + 0.02);
});

addEventListener("resize", () => {
  renderer.setSize(innerWidth, innerHeight);
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  const s = renderer.getDrawingBufferSize(new THREE.Vector2());
  edl.setSize(s.x, s.y);
});

let orbit = q.get("orbit") === "1";
window.__surreyOrbit = (on) => { orbit = on; };

function frame(now) {
  requestAnimationFrame(frame);
  if (orbit && current) {
    const off = camera.position.clone().sub(controls.target);
    off.applyAxisAngle(new THREE.Vector3(0, 0, 1), 0.006);
    camera.position.copy(controls.target).add(off);
  }
  controls.update();
  let shown = 0;
  if (current) {
    const m = current.material.uniforms;
    m.uResolution.value.set(renderer.domElement.width, renderer.domElement.height);
    m.uProj.value = renderer.domElement.height / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2));
    m.uWipe.value = document.getElementById("wipe").classList.contains("on") ? wipeX : 0;
    const u = current.oct.update(camera, renderer.domElement.height, BUDGET, 4, ORDER);
    shown = u.shown;
    // Settled: nothing loading and nothing more wanted for this camera.
    const rec = metrics.areas[current.name] || (metrics.areas[current.name] = { idleSince: null });
    const idle = u.loading === 0 && u.wanted === 0;
    rec.idleSince = idle ? (rec.idleSince ?? now) : null;
    rec.visibleNames = u.visibleNames;
    rec.loadTimes = current.oct.loadTimes;
    rec.bytes = current.oct.bytesFetched;
    rec.nodeBytes = current.oct.nodeBytes;
    rec.bytesAtLoad = current.oct.bytesAtLoad;
  }
  edl.render(scene, camera);
  metrics.frames.push(now);
  if (metrics.frames.length > 4000) metrics.frames.splice(0, 2000);
  if (shown > 0) {
    if (metrics.firstFrameMs === null) metrics.firstFrameMs = performance.now();
    if (switchRecord && switchRecord.firstFrameMs === null) {
      switchRecord.firstFrameMs = performance.now() - switchRecord.t0;
      renderer.domElement.style.opacity = "1";
    }
  }
  metrics.bytes = [...areas.values()].reduce((s, a) => s + (a.oct?.bytesFetched || 0), 0);
}
requestAnimationFrame(frame);
show("ggpark", "ret0");
