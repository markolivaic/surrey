// Eye-dome lighting as a screen pass: darken a pixel by how much nearer its neighbours
// are, in log depth. Points have no normals, and this is what gives them edges.
//
// The design frames drew black halos that read as a smudge on dark NAIP and as a comic
// strip on light ground, so strength is a uniform and the prototype shoots three of them.
//
// The first version counted an empty background pixel as a neighbour far behind the point,
// which darkened every point that bordered a gap, and a lidar cloud at a million points is
// mostly points bordering gaps: black specks everywhere. A background neighbour now counts
// for nothing, and the depth differences are averaged over the neighbours that are points,
// so a point beside a hole is shaded by the points it does have and not by the hole.

import * as THREE from "three";

// The page ground as the numbers the shader writes. A THREE.Color would convert the hex into the
// linear working space, and the pass writes its uniform raw, so the ground came out darker than the
// page around it: grey on the light theme, black on the dark one.
const groundVector = (hex) => {
  const n = parseInt(String(hex).replace("#", ""), 16);
  return new THREE.Vector3(((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255);
};

export class EDLPass {
  constructor(renderer, background) {
    this.renderer = renderer;
    const size = renderer.getDrawingBufferSize(new THREE.Vector2());
    this.target = new THREE.WebGLRenderTarget(size.x, size.y, {
      depthTexture: new THREE.DepthTexture(size.x, size.y),
      samples: 0,
    });
    this.material = new THREE.ShaderMaterial({
      uniforms: {
        tColor: { value: this.target.texture },
        tDepth: { value: this.target.depthTexture },
        uTexel: { value: new THREE.Vector2(1 / size.x, 1 / size.y) },
        uNear: { value: 1 },
        uFar: { value: 5000 },
        uStrength: { value: 0.5 },
        uRadius: { value: 1.4 },
        uBackground: { value: groundVector(background) },
      },
      vertexShader: /* glsl */ `varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`,
      fragmentShader: /* glsl */ `
      uniform sampler2D tColor;
      uniform sampler2D tDepth;
      uniform vec2 uTexel;
      uniform float uNear, uFar, uStrength, uRadius;
      uniform vec3 uBackground;
      varying vec2 vUv;
      float linear(float d) { float z = d * 2.0 - 1.0; return (2.0 * uNear * uFar) / (uFar + uNear - z * (uFar - uNear)); }
      float logDepth(vec2 uv) { float d = texture2D(tDepth, uv).r; return d >= 1.0 ? 0.0 : log2(linear(d)); }
      void main() {
        float d = texture2D(tDepth, vUv).r;
        float centre = logDepth(vUv);
        vec2 offs[8];
        offs[0] = vec2(1, 0); offs[1] = vec2(-1, 0); offs[2] = vec2(0, 1); offs[3] = vec2(0, -1);
        offs[4] = vec2(0.707, 0.707); offs[5] = vec2(-0.707, 0.707); offs[6] = vec2(0.707, -0.707); offs[7] = vec2(-0.707, -0.707);
        if (d >= 1.0) {
          // Background pixels next to points get no halo at all.
          gl_FragColor = vec4(uBackground, 1.0);
          return;
        }
        float sum = 0.0;
        float covered = 0.0;
        for (int i = 0; i < 8; i++) {
          float n = logDepth(vUv + offs[i] * uTexel * uRadius);
          if (n == 0.0) continue;   // the background is not a neighbour
          covered += 1.0;
          sum += max(0.0, centre - n);
        }
        // Scaled to eight neighbours, so a point with all of them is shaded as before.
        float response = covered > 0.0 ? sum * 8.0 / covered : 0.0;
        float shade = exp(-response * 60.0 * uStrength);
        gl_FragColor = vec4(texture2D(tColor, vUv).rgb * shade, 1.0);
      }`,
      depthTest: false,
      depthWrite: false,
    });
    this.quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.material);
    this.scene = new THREE.Scene();
    this.scene.add(this.quad);
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  }

  setBackground(hex) {
    this.material.uniforms.uBackground.value.copy(groundVector(hex));
  }

  setSize(w, h) {
    this.target.setSize(w, h);
    this.material.uniforms.uTexel.value.set(1 / w, 1 / h);
  }

  render(scene, camera) {
    const u = this.material.uniforms;
    u.uNear.value = camera.near;
    u.uFar.value = camera.far;
    if (u.uStrength.value <= 0) {
      this.renderer.setRenderTarget(null);
      this.renderer.render(scene, camera);
      return;
    }
    this.renderer.setRenderTarget(this.target);
    this.renderer.render(scene, camera);
    this.renderer.setRenderTarget(null);
    this.renderer.render(this.scene, this.camera);
  }
}
