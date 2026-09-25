// The scene: one renderer, one camera, areas loaded on demand, one drawn at a time.
//
// Areas are kilometres apart, so moving between them is a cut with a short dissolve, never a
// flight. With reduced motion there is no dissolve and the camera never moves on its own.

import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import confidenceMap from "../../data/artifacts/confidence_mapping.json";
import { DecoderPool, Octree } from "./potree2/octree.js";
import { EDLPass } from "./render/edl.js";
import { hasVariant, pointMaterial, selectVariant, setTheme } from "./render/material.js";

const GROUNDS = { light: "#e6e9e6", dark: "#131815" };
// Measured, not chosen: at 20 Mbit/s the link is what f and g spend, and 1,000,000 points is the
// largest budget where both pass (prototype_measurements.json, visibility_lever, 2026-09-21).
export const POINT_BUDGET = 1_000_000;

export class Viewer {
  constructor(container, dataBase, reducedMotion) {
    this.dataBase = dataBase.replace(/\/$/, "");
    this.reducedMotion = reducedMotion;
    this.theme = "light";
    this.renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    // The canvas takes its container's size: the whole window behind the cards, a framed figure in
    // the free view.
    this.container = container;
    const [w, h] = this.size();
    this.renderer.setSize(w, h);
    this.renderer.setClearColor(GROUNDS.light);
    container.appendChild(this.renderer.domElement);
    // Eye-dome lighting: edges from depth alone, since points carry no normals. Off until the
    // page sets a strength.
    this.edl = new EDLPass(this.renderer, GROUNDS.light);
    this.edl.material.uniforms.uStrength.value = 0;
    this.orbit = false;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(50, w / h, 1, 5000);
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = !reducedMotion;
    this.controls.enabled = false;
    this.pool = new DecoderPool();
    this.areas = new Map();
    this.current = null;
    this.wipe = 0;
    this.onFrame = null;
    this.metrics = { firstFrameMs: null, switches: [], areas: {}, frames: [] };
    new ResizeObserver(() => this.resize()).observe(container);
    requestAnimationFrame((t) => this.frame(t));
  }

  size() {
    return [Math.max(1, this.container.clientWidth || innerWidth), Math.max(1, this.container.clientHeight || innerHeight)];
  }

  resize() {
    const [w, h] = this.size();
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    const s = this.renderer.getDrawingBufferSize(new THREE.Vector2());
    this.edl.setSize(s.x, s.y);
  }

  setEdl(strength) {
    this.edl.material.uniforms.uStrength.value = Math.max(0, Number(strength) || 0);
  }

  area(name) {
    if (this.areas.has(name)) return this.areas.get(name).ready;
    const entry = { name };
    this.areas.set(name, entry);
    entry.ready = (async () => {
      const manifest = await (await fetch(`${this.dataBase}/${name}/manifest.json`)).json();
      const material = pointMaterial(manifest.variants, this.theme, confidenceMap);
      const oct = await new Octree(`${this.dataBase}/${name}`, material, this.pool, manifest.variants).init();
      Object.assign(entry, { manifest, material, oct });
      oct.group.visible = false;
      this.scene.add(oct.group);
      await oct.loadNode(oct.root);
      // The ground height from the coarse root points, not from the box: a bridge tower or a noise
      // point stretches the box far above or below the street.
      const c = oct.root.box.getCenter(new THREE.Vector3());
      const size = oct.root.box.getSize(new THREE.Vector3());
      const pos = oct.root.points?.geometry.getAttribute("position");
      let ground = oct.root.box.min.z;
      if (pos) {
        const zs = Float32Array.from({ length: pos.count }, (_, i) => pos.getZ(i)).sort();
        ground = zs[Math.floor(zs.length * 0.1)];
      }
      // A low oblique view from the south west, close enough to see trunks and kerbs.
      entry.target = new THREE.Vector3(c.x, c.y, ground);
      entry.eye = new THREE.Vector3(c.x - size.x * 0.45, c.y - size.y * 0.65, ground + size.x * 0.42);
      // The first screen may want more of the block than the kerbs: the same direction on the
      // compass, raised to the angle the stills are drawn at, and pulled back by a share of the
      // distance at which the footprint's circle fits the vertical field. At 1 the whole block
      // floats in the frame; at 0.55 it runs past both sides and the far edge still meets the sky.
      // "still" is the phone still's own framing (scripts/pipeline/render_frames.py draws at 38
      // degrees): the whole block from higher up, for an area whose towers would otherwise stand
      // between the camera and everything behind them.
      {
        const r = 0.5 * Math.hypot(size.x, size.y);
        const fit = r / Math.sin(THREE.MathUtils.degToRad(this.camera.fov) / 2);
        const h = new THREE.Vector2(entry.eye.x - c.x, entry.eye.y - c.y).normalize();
        const at = (k, deg) => {
          const elev = THREE.MathUtils.degToRad(deg);
          return new THREE.Vector3(c.x + h.x * k * fit * Math.cos(elev), c.y + h.y * k * fit * Math.cos(elev),
                                   ground + k * fit * Math.sin(elev));
        };
        entry.openingEyes = { wide: at(1.0, 34), mid: at(0.55, 34), still: at(1.0, 38) };
      }
      return entry;
    })();
    return entry.ready;
  }

  async show(name, variant, { keepCamera = false } = {}) {
    const t0 = performance.now();
    const same = this.current && this.current.name === name;
    if (!same && !this.reducedMotion) this.renderer.domElement.style.opacity = "0";
    const entry = await this.area(name);
    const v = hasVariant(entry.material, variant) ? variant : null;
    if (v) selectVariant(entry.material, v);
    if (!same) {
      if (this.current) {
        // Whatever the last area still had in flight would otherwise share the link with this one.
        this.metrics.stoppedOnSwitch = (this.metrics.stoppedOnSwitch || 0) + this.current.oct.stopLoading();
        this.current.oct.group.visible = false;
      }
      entry.oct.group.visible = true;
      if (!keepCamera) {
        this.camera.position.copy(entry.eye);
        this.controls.target.copy(entry.target);
        this.controls.update();
      }
      this.current = entry;
      this.pendingSwitch = { to: name, t0, firstFrameMs: null };
      this.metrics.switches.push(this.pendingSwitch);
    }
    return { entry, variant: v };
  }

  // The opening: straight down, read as a map, then tilted into the oblique view the cards use.
  // `t` runs 0 (map) to 1 (the view every other card gets). Nothing moves on its own; the page
  // drives this from the reader's own scrolling, and with reduced motion it is never called.
  setPitch(t, framing = null) {
    const entry = this.current;
    if (!entry || !entry.eye) return;
    const target = entry.target;
    const oblique = (framing && entry.openingEyes?.[framing]) || entry.eye;
    const height = oblique.distanceTo(target);
    const above = new THREE.Vector3(target.x, target.y, target.z + height);
    const eased = t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t);
    this.camera.position.lerpVectors(above, oblique, eased);
    this.controls.target.copy(target);
    this.controls.update();
  }

  setTheme(theme) {
    this.theme = theme;
    this.renderer.setClearColor(GROUNDS[theme]);
    this.edl.setBackground(GROUNDS[theme]);
    for (const e of this.areas.values()) if (e.material) setTheme(e.material, theme);
  }

  frame(now) {
    requestAnimationFrame((t) => this.frame(t));
    // The orbit is the frame rate test's, the same turn per frame the prototype was measured with.
    if (this.orbit && this.current) {
      const off = this.camera.position.clone().sub(this.controls.target);
      off.applyAxisAngle(new THREE.Vector3(0, 0, 1), 0.006);
      this.camera.position.copy(this.controls.target).add(off);
    }
    this.controls.update();
    let shown = 0;
    if (this.current) {
      const m = this.current.material.uniforms;
      const el = this.renderer.domElement;
      m.uResolution.value.set(el.width, el.height);
      m.uProj.value = el.height / (2 * Math.tan(THREE.MathUtils.degToRad(this.camera.fov) / 2));
      m.uWipe.value = this.wipe;
      const u = this.current.oct.update(this.camera, el.height, POINT_BUDGET, 4, "visibility");
      shown = u.shown;
      const rec = this.metrics.areas[this.current.name] || (this.metrics.areas[this.current.name] = { idleSince: null });
      rec.idleSince = u.loading === 0 && u.wanted === 0 ? (rec.idleSince ?? now) : null;
      rec.visibleNames = u.visibleNames;
      rec.loadTimes = this.current.oct.loadTimes;
      rec.loading = u.loading;
      rec.shown = u.shown;
    }
    this.edl.render(this.scene, this.camera);
    this.metrics.frames.push(now);
    if (this.metrics.frames.length > 4000) this.metrics.frames.splice(0, 2000);
    if (shown > 0) {
      if (this.metrics.firstFrameMs === null) this.metrics.firstFrameMs = performance.now();
      if (this.pendingSwitch && this.pendingSwitch.firstFrameMs === null) {
        this.pendingSwitch.firstFrameMs = performance.now() - this.pendingSwitch.t0;
        this.renderer.domElement.style.opacity = "1";
      }
    }
    if (this.onFrame) this.onFrame(this);
  }
}
