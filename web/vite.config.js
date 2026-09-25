import { defineConfig } from "vite";

// `vite build` is the page as published. `vite build --mode investigate` adds the switches the scripts in
// scripts/investigations/ compare with, into dist-investigate, which is never served to a reader.
export default defineConfig(({ mode }) => ({
  base: "./",
  publicDir: "public",
  server: { fs: { allow: [".."] } },
  build: {
    outDir: mode === "investigate" ? "dist-investigate" : "dist",
    rollupOptions: { input: { index: "index.html" } },
  },
  worker: { format: "es" },
}));
