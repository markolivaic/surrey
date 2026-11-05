// Decode real nodes of a built octree in Node, with the browser's own worker code, and
// write what came out so scripts/check_octree_decode.py can compare it with the cache.
// No WebGL, so it can run beside training.
//
//   node web/tools/decode_check.mjs data/viewer/embarcadero out.bin 6

import { readFileSync, writeFileSync } from "node:fs";

const [dir, outPath, maxNodesArg] = process.argv.slice(2);
const maxNodes = Number(maxNodesArg || 6);
const meta = JSON.parse(readFileSync(`${dir}/metadata.json`, "utf8"));
const hierarchy = readFileSync(`${dir}/hierarchy.bin`);
const octree = readFileSync(`${dir}/octree.bin`);

let result = null;
globalThis.self = { postMessage: (msg) => { result = msg; } };
await import("../src/potree2/decoder.worker.js");

const view = new DataView(hierarchy.buffer, hierarchy.byteOffset, meta.hierarchy.firstChunkSize);
const nodes = [];
for (let i = 0; i < meta.hierarchy.firstChunkSize / 22 && nodes.length < maxNodes; i++) {
  const type = view.getUint8(i * 22);
  const numPoints = view.getUint32(i * 22 + 2, true);
  const byteOffset = Number(view.getBigInt64(i * 22 + 6, true));
  const byteSize = Number(view.getBigInt64(i * 22 + 14, true));
  if (type !== 2 && byteSize > 0) nodes.push({ i, numPoints, byteOffset, byteSize });
}

const chunks = [];
let total = 0;
for (const n of nodes) {
  const buffer = octree.buffer.slice(octree.byteOffset + n.byteOffset, octree.byteOffset + n.byteOffset + n.byteSize);
  self.onmessage({ data: { id: 1, buffer, attributes: meta.attributes, numPoints: n.numPoints,
    scale: meta.scale, offset: meta.offset, octreeOffset: meta.offset } });
  const a = result.attributes;
  // Per point: x y z as float32 relative to the octree offset, r g b, then c_0 c_1 .. p_0 p_1 ..
  const slots = Object.keys(a).filter((k) => /^[cp]_\d$/.test(k)).sort();
  const rec = 12 + 3 + slots.length;
  const buf = Buffer.alloc(n.numPoints * rec);
  for (let j = 0; j < n.numPoints; j++) {
    const o = j * rec;
    buf.writeFloatLE(a.position[3 * j], o);
    buf.writeFloatLE(a.position[3 * j + 1], o + 4);
    buf.writeFloatLE(a.position[3 * j + 2], o + 8);
    buf[o + 12] = a.rgb[3 * j]; buf[o + 13] = a.rgb[3 * j + 1]; buf[o + 14] = a.rgb[3 * j + 2];
    slots.forEach((name, k) => { buf[o + 15 + k] = a[name][j]; });
  }
  chunks.push(buf);
  globalThis.__slots = slots;
  total += n.numPoints;
}
writeFileSync(outPath, Buffer.concat(chunks));
const slots = globalThis.__slots;
writeFileSync(`${outPath}.json`, JSON.stringify({ nodes: nodes.length, points: total, slots, offset: meta.offset }));
console.log(`decoded ${nodes.length} nodes, ${total} points, slots ${slots.join(" ")}`);
