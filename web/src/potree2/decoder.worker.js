// Decodes one Brotli-encoded node of a PotreeConverter 2 octree.
//
// The Morton decoding of positions and colours follows Potree's
// workers/2.0/DecoderWorker_brotli.js, copyright Markus Schuetz, BSD-2-Clause (see
// LICENSE.potree in this folder). The Brotli decoder is Google's, MIT (vendor/brotli).
//
// Output positions are relative to the octree's own offset, so every node of an area
// shares one transform and float32 keeps sub-millimetre precision over 300 m.

import { BrotliDecode } from "../../vendor/brotli/decode.js";

const SLOT_OF = {
  classification: [0, "c"], "user data": [0, "p"],
  intensity: [1, "cp"], "point source id": [2, "cp"], "scan angle": [3, "cp"],
};

function dealign24b(mortoncode) {
  let x = mortoncode;
  x = ((x & 0b001000001000001000001000) >> 2) | ((x & 0b000001000001000001000001) >> 0);
  x = ((x & 0b000011000000000011000000) >> 4) | ((x & 0b000000000011000000000011) >> 0);
  x = ((x & 0b000000001111000000000000) >> 8) | ((x & 0b000000000000000000001111) >> 0);
  x = ((x & 0b000000000000000000000000) >> 16) | ((x & 0b000000000000000011111111) >> 0);
  return x;
}

self.onmessage = (event) => {
  const { id, buffer, attributes, numPoints, scale, offset, octreeOffset } = event.data;
  const bytes = BrotliDecode(new Int8Array(buffer));
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = {};
  const transfer = [];
  let at = 0;

  for (const attr of attributes) {
    if (attr.name === "position") {
      const pos = new Float32Array(numPoints * 3);
      const dx = offset[0] - octreeOffset[0];
      const dy = offset[1] - octreeOffset[1];
      const dz = offset[2] - octreeOffset[2];
      for (let j = 0; j < numPoints; j++) {
        const mc0 = view.getUint32(at + 4, true);
        const mc1 = view.getUint32(at + 0, true);
        const mc2 = view.getUint32(at + 12, true);
        const mc3 = view.getUint32(at + 8, true);
        at += 16;
        let X = dealign24b((mc3 & 0x00ffffff) >>> 0) | (dealign24b(((mc3 >>> 24) | (mc2 << 8)) >>> 0) << 8);
        let Y = dealign24b((mc3 & 0x00ffffff) >>> 1) | (dealign24b(((mc3 >>> 24) | (mc2 << 8)) >>> 1) << 8);
        let Z = dealign24b((mc3 & 0x00ffffff) >>> 2) | (dealign24b(((mc3 >>> 24) | (mc2 << 8)) >>> 2) << 8);
        if (mc1 !== 0 || mc2 !== 0) {
          X = X | (dealign24b((mc1 & 0x00ffffff) >>> 0) << 16) | (dealign24b(((mc1 >>> 24) | (mc0 << 8)) >>> 0) << 24);
          Y = Y | (dealign24b((mc1 & 0x00ffffff) >>> 1) << 16) | (dealign24b(((mc1 >>> 24) | (mc0 << 8)) >>> 1) << 24);
          Z = Z | (dealign24b((mc1 & 0x00ffffff) >>> 2) << 16) | (dealign24b(((mc1 >>> 24) | (mc0 << 8)) >>> 2) << 24);
        }
        pos[3 * j] = X * scale[0] + dx;
        pos[3 * j + 1] = Y * scale[1] + dy;
        pos[3 * j + 2] = Z * scale[2] + dz;
      }
      out.position = pos;
      transfer.push(pos.buffer);
    } else if (attr.name === "rgb") {
      const rgb = new Uint8Array(numPoints * 3);
      for (let j = 0; j < numPoints; j++) {
        const mc0 = view.getUint32(at + 4, true);
        const mc1 = view.getUint32(at + 0, true);
        at += 8;
        const r = dealign24b((mc1 & 0x00ffffff) >>> 0) | (dealign24b(((mc1 >>> 24) | (mc0 << 8)) >>> 0) << 8);
        const g = dealign24b((mc1 & 0x00ffffff) >>> 1) | (dealign24b(((mc1 >>> 24) | (mc0 << 8)) >>> 1) << 8);
        const b = dealign24b((mc1 & 0x00ffffff) >>> 2) | (dealign24b(((mc1 >>> 24) | (mc0 << 8)) >>> 2) << 8);
        rgb[3 * j] = r > 255 ? r / 256 : r;
        rgb[3 * j + 1] = g > 255 ? g / 256 : g;
        rgb[3 * j + 2] = b > 255 ? b / 256 : b;
      }
      out.rgb = rgb;
      transfer.push(rgb.buffer);
    } else if (attr.name in SLOT_OF) {
      // Predictions travel in standard LAS fields (scripts/pipeline/build_octree.py explains why):
      // slot 0 as classification and user data, slots 1 to 3 as a 16 bit field with the
      // class in the low byte and the confidence in the high byte.
      const [slot, part] = SLOT_OF[attr.name];
      if (attr.size === 1) {
        const a = new Uint8Array(numPoints);
        a.set(new Uint8Array(bytes.buffer, bytes.byteOffset + at, numPoints));
        out[`${part}_${slot}`] = a;
        transfer.push(a.buffer);
      } else {
        const c = new Uint8Array(numPoints);
        const p = new Uint8Array(numPoints);
        for (let j = 0; j < numPoints; j++) {
          c[j] = view.getUint8(at + 2 * j);
          p[j] = view.getUint8(at + 2 * j + 1);
        }
        out[`c_${slot}`] = c;
        out[`p_${slot}`] = p;
        transfer.push(c.buffer, p.buffer);
      }
      at += attr.size * numPoints;
    } else {
      // Attributes the viewer does not read are skipped by their size.
      at += attr.size * numPoints;
    }
  }
  self.postMessage({ id, attributes: out }, transfer);
};
