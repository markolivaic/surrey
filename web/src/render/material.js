// Class as OKLCH hue and lightness, confidence as chroma, NAIP colour on the other side of the
// divider.
//
// The palette and its rules live in docs/design/palette.json and backend/surrey/palette.py; the
// confidence map is fixed on held-out Surrey (data/artifacts/confidence_mapping.json, copied into
// the palette uniforms by the page). Lightness never changes with confidence, so contrast against
// the ground is the class's own and an unsure point neither vanishes nor stands out.
//
// Every variant's class and confidence arrive as their own attributes. The shader reads the pair
// for the selected variant through a one-hot uniform, so switching variant changes a uniform and
// nothing is fetched or rebuilt.

import * as THREE from "three";
import palette from "../../../docs/design/palette.json";
import { FINE, GROUP_OF } from "../palette.js";

export const MAX_VARIANTS = 4;

function tokenArrays(theme) {
  const tokens = palette.themes[theme].tokens;
  const L = new Array(32).fill(0.5);
  const C = new Array(32).fill(0);
  const H = new Array(32).fill(0);
  FINE.forEach((name, i) => {
    const t = tokens[GROUP_OF[name]];
    L[i] = t.L;
    C[i] = t.C;
    H[i] = (t.h * Math.PI) / 180;
  });
  return { L, C, H };
}

export function pointMaterial(variantNames, theme = "light", confidenceMap = { low: 0, high: 1 }) {
  if (variantNames.length > MAX_VARIANTS) throw new Error("too many variants for one material");
  const attrs = variantNames.map((v, i) => `attribute float c_${i};\nattribute float p_${i};`).join("\n");
  const pick = variantNames.map((v, i) => `cls += c_${i} * uSel[${i}]; conf += p_${i} * uSel[${i}];`).join("\n    ");
  const tok = tokenArrays(theme);
  const material = new THREE.ShaderMaterial({
    uniforms: {
      uSel: { value: variantNames.map((_, i) => (i === 0 ? 1 : 0)).concat(Array(MAX_VARIANTS).fill(0)).slice(0, MAX_VARIANTS) },
      uL: { value: tok.L },
      uC: { value: tok.C },
      uH: { value: tok.H },
      uFloor: { value: palette.floor ?? 0.2 },
      uConfLow: { value: confidenceMap.low },
      uConfHigh: { value: confidenceMap.high },
      uWipe: { value: 0.0 },
      uResolution: { value: new THREE.Vector2(1, 1) },
      uPointScale: { value: 1.0 },
      // The drawn node's point spacing in metres, set per node before it renders, so a coarse node
      // draws larger points and a close view has no holes.
      uSpacing: { value: 1.0 },
      uProj: { value: 800.0 },
      uNaipGamma: { value: 0.7 },
      uOpacity: { value: 1.0 },
    },
    vertexShader: /* glsl */ `
    ${attrs}
    attribute vec3 rgb;
    uniform float uSel[${MAX_VARIANTS}];
    uniform float uL[32];
    uniform float uC[32];
    uniform float uH[32];
    uniform float uFloor;
    uniform float uConfLow;
    uniform float uConfHigh;
    uniform float uPointScale;
    uniform float uSpacing;
    uniform float uProj;
    uniform float uNaipGamma;
    varying vec3 vClass;
    varying vec3 vPhoto;

    vec3 oklchToSrgb(float L, float C, float h) {
      float a = C * cos(h), b = C * sin(h);
      float l_ = pow(L + 0.3963377774 * a + 0.2158037573 * b, 3.0);
      float m_ = pow(L - 0.1055613458 * a - 0.0638541728 * b, 3.0);
      float s_ = pow(L - 0.0894841775 * a - 1.2914855480 * b, 3.0);
      vec3 lin = vec3(4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_,
                      -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_,
                      -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_);
      lin = clamp(lin, 0.0, 1.0);
      return mix(12.92 * lin, 1.055 * pow(lin, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, lin));
    }

    void main() {
      float cls = 0.0;
      float conf = 0.0;
    ${pick}
      int k = int(cls + 0.5);
      float t = clamp((conf / 255.0 - uConfLow) / (uConfHigh - uConfLow), 0.0, 1.0);
      vClass = oklchToSrgb(uL[k], uC[k] * (uFloor + (1.0 - uFloor) * t), uH[k]);
      vPhoto = pow(rgb, vec3(uNaipGamma));
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      gl_Position = projectionMatrix * mv;
      gl_PointSize = clamp(uPointScale * uSpacing * 0.6 * uProj / max(-mv.z, 0.1), 1.0, 8.0);
    }`,
    fragmentShader: /* glsl */ `
    uniform float uWipe;
    uniform vec2 uResolution;
    uniform float uOpacity;
    varying vec3 vClass;
    varying vec3 vPhoto;
    void main() {
      vec2 d = gl_PointCoord - 0.5;
      if (dot(d, d) > 0.25) discard;
      bool photo = gl_FragCoord.x / uResolution.x < uWipe;
      gl_FragColor = vec4(photo ? vPhoto : vClass, uOpacity);
    }`,
    transparent: false,
  });
  material.userData.variants = variantNames;
  return material;
}

export function selectVariant(material, name) {
  const i = material.userData.variants.indexOf(name);
  if (i < 0) throw new Error(`variant ${name} is not in this octree`);
  material.uniforms.uSel.value = material.uniforms.uSel.value.map((_, k) => (k === i ? 1 : 0));
}

export function setTheme(material, theme) {
  const tok = tokenArrays(theme);
  material.uniforms.uL.value = tok.L;
  material.uniforms.uC.value = tok.C;
  material.uniforms.uH.value = tok.H;
}

export function hasVariant(material, name) {
  return material.userData.variants.includes(name);
}
