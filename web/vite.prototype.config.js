// The technical prototype, built on its own for scripts/investigations/measure_prototype.py.
// The page does not ship it.
import { defineConfig } from "vite";

export default defineConfig({
  base: "./",
  publicDir: "public",
  server: { fs: { allow: [".."] } },
  build: {
    outDir: "dist-prototype",
    rollupOptions: { input: { prototype: "prototype.html" } },
  },
  worker: { format: "es" },
});
