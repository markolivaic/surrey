// The second load time lever plans the node set from the hierarchy. What it must do: fetch exactly the
// nodes the point budget shows once everything has loaded, and nothing outside them. Runs under Node,
// with no WebGL and no network: node loads are recorded, never fetched.
//
//   npm test

import assert from "node:assert/strict";
import test from "node:test";
import * as THREE from "three";
import { Octree } from "../src/potree2/octree.js";

function tree() {
  const oct = new Octree("http://unused", null, null, []);
  const make = (name, box, level) => {
    const n = { name, box, level, spacing: 1, children: new Array(8).fill(null), type: 1, numPoints: 100_000, state: "idle", points: null };
    return n;
  };
  const root = make("r", new THREE.Box3(new THREE.Vector3(0, 0, 0), new THREE.Vector3(100, 100, 100)), 0);
  const split = (node) => {
    for (let c = 0; c < 8; c++) {
      const min = node.box.min.clone();
      const max = node.box.max.clone();
      const size = new THREE.Vector3().subVectors(max, min).multiplyScalar(0.5);
      if (c & 1) min.z += size.z; else max.z -= size.z;
      if (c & 2) min.y += size.y; else max.y -= size.y;
      if (c & 4) min.x += size.x; else max.x -= size.x;
      node.children[c] = make(node.name + c, new THREE.Box3(min, max), node.level + 1);
    }
  };
  split(root);
  root.children.forEach(split);
  oct.root = root;
  // What a decode message is built from, for the tests that let a node get as far as decoding.
  oct.meta = { scale: [0.001, 0.001, 0.001], offset: [0, 0, 0] };
  oct.attributes = [];
  oct.offset = [0, 0, 0];
  oct.requested = [];
  oct.loadNode = (n) => oct.requested.push(n.name);
  return oct;
}

function camera() {
  const cam = new THREE.PerspectiveCamera(60, 1.6, 0.1, 10_000);
  cam.position.set(50, 50, 200);
  cam.lookAt(50, 50, 0);
  cam.updateMatrixWorld();
  return cam;
}

const all = (n, f) => { f(n); n.children.forEach((c) => c && all(c, f)); };
const BUDGET = 2_000_000;

test("planned from the hierarchy, the wanted set is what the budget shows once all has loaded", () => {
  const settled = tree();
  all(settled.root, (n) => { n.state = "loaded"; });
  const shown = new Set(settled.update(camera(), 900, BUDGET).visibleNames);

  const planned = tree();
  planned.root.state = "loaded";
  const u = planned.update(camera(), 900, BUDGET, 1000, "visibility");
  const wantedOrShown = new Set([...u.visibleNames, ...planned.requested]);
  assert.deepEqual([...wantedOrShown].sort(), [...shown].sort());
  assert.ok(planned.requested.some((name) => name.length === 3), "grandchildren are fetched before their parents load");
});

test("self-test: the level by level order cannot see past the loaded frontier", () => {
  const oct = tree();
  oct.root.state = "loaded";
  oct.update(camera(), 900, BUDGET, 1000, "loaded");
  assert.ok(oct.requested.length > 0 && oct.requested.every((name) => name.length === 2));
});

test("the budget still limits the planned set", () => {
  const oct = tree();
  const u = oct.update(camera(), 900, 350_000, 1000, "visibility");
  assert.equal(oct.requested.length + u.visibleNodes, 3);
});

test("an area that is left stops fetching, and its nodes go back to idle", async () => {
  const oct = tree();
  const node = oct.root.children[0];
  let aborted = false;
  // A fetch that never settles until it is aborted, which is what a node in flight looks like.
  oct.range = (file, first, size, signal) => new Promise((_resolve, reject) => {
    signal.addEventListener("abort", () => {
      aborted = true;
      const err = new Error("aborted");
      err.name = "AbortError";
      reject(err);
    });
  });
  node.byteSize = 10n;
  const loading = Object.getPrototypeOf(oct).loadNode.call(oct, node);
  assert.equal(node.state, "loading");
  assert.equal(oct.inFlight.size, 1);
  assert.equal(oct.stopLoading(), 1);
  await loading;
  assert.ok(aborted);
  assert.equal(node.state, "idle");
  assert.equal(oct.inFlight.size, 0);
});

test("self-test: without the stop the node stays in flight", async () => {
  const oct = tree();
  const node = oct.root.children[0];
  oct.range = () => new Promise(() => {});
  node.byteSize = 10n;
  Object.getPrototypeOf(oct).loadNode.call(oct, node);
  await new Promise((r) => setTimeout(r, 10));
  assert.equal(oct.inFlight.size, 1);
  assert.equal(node.state, "loading");
});

test("a decode that finishes after the area was left adds no points", async () => {
  const oct = tree();
  const node = oct.root.children[0];
  let finishDecode;
  oct.range = () => Promise.resolve(new ArrayBuffer(8));
  oct.pool = { decode: () => new Promise((resolve) => { finishDecode = resolve; }) };
  oct.group = { add: () => { throw new Error("points were added after the area was left"); } };
  node.byteSize = 10n;
  const loading = Object.getPrototypeOf(oct).loadNode.call(oct, node);
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(oct.stopLoading(), 1);
  finishDecode({ position: new Float32Array(3), rgb: new Uint16Array(3) });
  await loading;
  assert.equal(node.points, null);
  assert.equal(node.state, "idle");
  assert.equal(oct.inFlight.size, 0);
});

test("a decode that fails marks the node failed and frees its slot", async () => {
  const oct = tree();
  const node = oct.root.children[0];
  oct.range = () => Promise.resolve(new ArrayBuffer(8));
  oct.pool = { decode: () => Promise.reject(new Error("corrupt node")) };
  node.byteSize = 10n;
  const warn = console.warn;
  console.warn = () => {};
  try {
    await Object.getPrototypeOf(oct).loadNode.call(oct, node);
  } finally {
    console.warn = warn;
  }
  assert.equal(node.state, "failed");
  assert.equal(oct.inFlight.size, 0);
});
