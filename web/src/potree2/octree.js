// A PotreeConverter 2 octree read over HTTP byte ranges, drawn with three.js.
//
// Hierarchy parsing follows Potree's NodeLoader (BSD-2-Clause, LICENSE.potree): 22 bytes
// per node, proxy nodes pointing at further hierarchy chunks. Everything else here is
// this project's: which nodes to load, the point budget, and the geometry the shader
// reads, which carries every variant's class and confidence so a variant switch is a
// uniform change and never a reload.

import * as THREE from "three";

const BYTES_PER_NODE = 22;

class Node {
  constructor(name, box, level, spacing) {
    this.name = name;
    this.box = box; // THREE.Box3, relative to the octree offset
    this.level = level;
    this.spacing = spacing;
    this.children = new Array(8).fill(null);
    this.type = 2;
    this.numPoints = 0;
    this.byteOffset = 0n;
    this.byteSize = 0n;
    this.hierarchyByteOffset = 0n;
    this.hierarchyByteSize = 0n;
    this.state = "idle"; // idle, loading, loaded, failed
    this.points = null;
  }
}

function childBox(box, index) {
  const min = box.min.clone();
  const max = box.max.clone();
  const size = new THREE.Vector3().subVectors(max, min).multiplyScalar(0.5);
  if (index & 0b001) min.z += size.z; else max.z -= size.z;
  if (index & 0b010) min.y += size.y; else max.y -= size.y;
  if (index & 0b100) min.x += size.x; else max.x -= size.x;
  return new THREE.Box3(min, max);
}

export class DecoderPool {
  constructor(size = Math.min(4, navigator.hardwareConcurrency || 2)) {
    this.workers = [];
    this.idle = [];
    this.queue = [];
    this.pending = new Map();
    this.nextId = 1;
    for (let i = 0; i < size; i++) {
      const w = new Worker(new URL("./decoder.worker.js", import.meta.url), { type: "module" });
      w.onmessage = (e) => {
        const { resolve } = this.pending.get(e.data.id);
        this.pending.delete(e.data.id);
        resolve(e.data.attributes);
        this.idle.push(w);
        this.pump();
      };
      this.workers.push(w);
      this.idle.push(w);
    }
  }

  decode(message) {
    return new Promise((resolve) => {
      this.queue.push({ message, resolve });
      this.pump();
    });
  }

  pump() {
    while (this.idle.length && this.queue.length) {
      const w = this.idle.pop();
      const { message, resolve } = this.queue.shift();
      const id = this.nextId++;
      this.pending.set(id, { resolve });
      w.postMessage({ ...message, id }, [message.buffer]);
    }
  }
}

export class Octree {
  constructor(baseUrl, material, pool, variants) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.variants = variants;
    this.material = material;
    this.pool = pool;
    this.group = new THREE.Group();
    this.loadedPoints = 0;
    this.bytesFetched = 0;
    this.onNodeLoaded = null;
    // performance.now() at which each node became drawable, for the load time measurements.
    this.loadTimes = {};
    // Per node: its compressed bytes, and the octree's running byte total when it became drawable.
    // Together with loadTimes these say whether a load time is the link's throughput or the loader's order.
    this.nodeBytes = {};
    this.bytesAtLoad = {};
    // In flight fetches, so an area the reader has left stops spending the link on itself.
    this.inFlight = new Map();
  }

  async init() {
    const meta = await (await fetch(`${this.baseUrl}/metadata.json`)).json();
    this.meta = meta;
    this.offset = meta.offset;
    const min = new THREE.Vector3(...meta.boundingBox.min).sub(new THREE.Vector3(...meta.offset));
    const max = new THREE.Vector3(...meta.boundingBox.max).sub(new THREE.Vector3(...meta.offset));
    this.root = new Node("r", new THREE.Box3(min, max), 0, meta.spacing);
    this.root.hierarchyByteSize = BigInt(meta.hierarchy.firstChunkSize);
    this.attributes = meta.attributes;
    await this.loadHierarchy(this.root);
    return this;
  }

  async range(file, first, size, signal = undefined) {
    const last = first + size - 1n;
    const r = await fetch(`${this.baseUrl}/${file}`, { headers: { Range: `bytes=${first}-${last}` }, signal });
    if (r.status !== 206) throw new Error(`${file}: expected 206, got ${r.status}`);
    const buf = await r.arrayBuffer();
    this.bytesFetched += buf.byteLength;
    return buf;
  }

  async loadHierarchy(node) {
    const buf = await this.range("hierarchy.bin", node.hierarchyByteOffset, node.hierarchyByteSize);
    const view = new DataView(buf);
    const n = buf.byteLength / BYTES_PER_NODE;
    const nodes = [node];
    for (let i = 0; i < n; i++) {
      const cur = nodes[i];
      const at = i * BYTES_PER_NODE;
      const type = view.getUint8(at);
      const childMask = view.getUint8(at + 1);
      const numPoints = view.getUint32(at + 2, true);
      const byteOffset = view.getBigInt64(at + 6, true);
      const byteSize = view.getBigInt64(at + 14, true);
      if (cur.type === 2 && i === 0) {
        cur.byteOffset = byteOffset;
        cur.byteSize = byteSize;
        cur.numPoints = numPoints;
      } else if (type === 2) {
        cur.hierarchyByteOffset = byteOffset;
        cur.hierarchyByteSize = byteSize;
        cur.numPoints = numPoints;
      } else {
        cur.byteOffset = byteOffset;
        cur.byteSize = byteSize;
        cur.numPoints = numPoints;
      }
      cur.type = type;
      if (cur.type === 2) continue;
      for (let c = 0; c < 8; c++) {
        if (!((1 << c) & childMask)) continue;
        const child = new Node(cur.name + c, childBox(cur.box, c), cur.level + 1, cur.spacing / 2);
        cur.children[c] = child;
        nodes.push(child);
      }
    }
  }

  // Abort every fetch still in flight and put those nodes back, for an area that is no longer shown.
  // Without this the link keeps carrying the last area's nodes while the reader waits for the new one.
  stopLoading() {
    for (const [node, controller] of this.inFlight) {
      controller.abort();
      node.state = "idle";
    }
    const stopped = this.inFlight.size;
    this.inFlight.clear();
    return stopped;
  }

  async loadNode(node) {
    node.state = "loading";
    const controller = new AbortController();
    this.inFlight.set(node, controller);
    try {
      if (node.type === 2) await this.loadHierarchy(node);
      const buffer = node.byteSize > 0n
        ? await this.range("octree.bin", node.byteOffset, node.byteSize, controller.signal) : null;
      // The decoder is given the buffer, which transfers it and leaves byteLength at zero, so its
      // size is taken here. The first reading of this had every node at zero bytes.
      const compressedBytes = buffer ? buffer.byteLength : 0;
      if (!buffer) {
        this.loadTimes[node.name] = performance.now();
        this.nodeBytes[node.name] = 0;
        this.bytesAtLoad[node.name] = this.bytesFetched;
        node.state = "loaded";
        return;
      }
      const attrs = await this.pool.decode({
        buffer, attributes: this.attributes, numPoints: node.numPoints,
        scale: this.meta.scale, offset: this.meta.offset, octreeOffset: this.offset,
      });
      const geom = new THREE.BufferGeometry();
      geom.setAttribute("position", new THREE.BufferAttribute(attrs.position, 3));
      geom.setAttribute("rgb", new THREE.BufferAttribute(attrs.rgb, 3, true));
      // The shader names a variant's pair by its slot, c_0 and p_0 for the first.
      this.variants.forEach((_, i) => {
        geom.setAttribute(`c_${i}`, new THREE.BufferAttribute(attrs[`c_${i}`], 1, false));
        geom.setAttribute(`p_${i}`, new THREE.BufferAttribute(attrs[`p_${i}`], 1, false));
      });
      geom.boundingBox = node.box.clone();
      geom.boundingSphere = node.box.getBoundingSphere(new THREE.Sphere());
      const pts = new THREE.Points(geom, this.material);
      pts.frustumCulled = false;
      pts.userData.spacing = node.spacing;
      pts.onBeforeRender = (_r, _s, _c, _g, material) => {
        material.uniforms.uSpacing.value = node.spacing;
        material.uniformsNeedUpdate = true;
      };
      node.points = pts;
      this.group.add(pts);
      this.loadedPoints += node.numPoints;
      this.loadTimes[node.name] = performance.now();
      this.nodeBytes[node.name] = compressedBytes;
      this.bytesAtLoad[node.name] = this.bytesFetched;
      node.state = "loaded";
      if (this.onNodeLoaded) this.onNodeLoaded(node);
    } catch (err) {
      if (err && err.name === "AbortError") {
        node.state = "idle";
        this.inFlight.delete(node);
        return;
      }
      node.state = "failed";
      console.warn(`node ${node.name} failed`, err);
    } finally {
      if (node.state !== "idle") this.inFlight.delete(node);
    }
  }

  // Pick what to show and what to fetch next: nodes in view, largest on screen first,
  // until the point budget is spent. Visible nodes are drawn; the best few unloaded ones
  // are fetched.
  //
  // order "loaded": a node's children are considered only once it has loaded, so the set
  // grows a level at a time and nodes fetched early can fall out of the budget later.
  // order "visibility" (the second load time lever): the hierarchy already gives every
  // node's box and point count, so the set the budget will show for this camera is planned
  // before anything loads, and only its nodes are fetched, largest on screen first.
  update(camera, viewportHeight, pointBudget, maxConcurrent = 4, order = "loaded") {
    const frustum = new THREE.Frustum().setFromProjectionMatrix(
      new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix,
        new THREE.Matrix4().multiplyMatrices(camera.matrixWorldInverse, this.group.matrixWorld)));
    const camLocal = camera.position.clone().applyMatrix4(new THREE.Matrix4().copy(this.group.matrixWorld).invert());
    const proj = viewportHeight / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2));
    const queue = [{ node: this.root, weight: Infinity }];
    const visible = new Set();
    const wanted = [];
    let budget = 0;
    let loading = 0;
    const walk = (n) => { if (n.state === "loading") loading++; n.children.forEach((c) => c && walk(c)); };
    walk(this.root);
    while (queue.length) {
      queue.sort((a, b) => b.weight - a.weight);
      const { node } = queue.shift();
      if (!frustum.intersectsBox(node.box)) continue;
      if (budget + node.numPoints > pointBudget) break;
      const pushChildren = () => {
        for (const c of node.children) {
          if (!c) continue;
          const sphere = c.box.getBoundingSphere(new THREE.Sphere());
          const dist = Math.max(sphere.center.distanceTo(camLocal) - sphere.radius, 0.1);
          const screen = (sphere.radius / dist) * proj;
          if (screen > 60) queue.push({ node: c, weight: screen });
        }
      };
      if (order === "visibility") {
        budget += node.numPoints;
        if (node.state === "loaded") visible.add(node);
        else if (node.state === "idle") wanted.push(node);
        pushChildren();
      } else if (node.state === "loaded") {
        visible.add(node);
        budget += node.numPoints;
        pushChildren();
      } else if (node.state === "idle") {
        wanted.push(node);
      }
    }
    for (const n of wanted.slice(0, Math.max(0, maxConcurrent - loading))) this.loadNode(n);
    const hide = (n) => {
      if (n.points) n.points.visible = visible.has(n);
      n.children.forEach((c) => c && hide(c));
    };
    hide(this.root);
    let shown = 0;
    visible.forEach((n) => { shown += n.numPoints; });
    return { shown, loading, wanted: wanted.length, visibleNodes: visible.size, visibleNames: [...visible].map((n) => n.name) };
  }
}
