import { defineConfig } from "vite";

// Where the hosted page finds the octrees, which are too large for the code repository.
const HOSTED_OCTREES = "https://huggingface.co/datasets/Maki101/surrey-viewer/resolve/main";
const OUT_DIR = { investigate: "dist-investigate", pages: "dist-pages" };

// `vite build` is the page served locally, with the octrees in data/ beside it. `vite build --mode pages`
// is the same page for GitHub Pages, reading them from Hugging Face. `vite build --mode investigate` adds
// the switches the scripts in archive/investigations/ compare with, and is never served to a reader.
export default defineConfig(({ mode }) => ({
  base: "./",
  publicDir: "public",
  define: { __OCTREES__: JSON.stringify(mode === "pages" ? HOSTED_OCTREES : "data") },
  server: { fs: { allow: [".."] } },
  build: {
    outDir: OUT_DIR[mode] || "dist",
    rollupOptions: { input: { index: "index.html" } },
  },
  worker: { format: "es" },
}));
