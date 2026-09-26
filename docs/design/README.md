# The design record

What is in this folder and why. A file stays here only if the page, the README or a test reads it;
the sheets each design round was chosen from are kept outside the repository.

- `directions.md`: the design rules the page is held to.
- `palette.json`: the class colours in both themes, read by the page and by `backend/surrey/palette.py`.
- `cvd_bug_frames.json`: how much of the bug frame difference survives each simulated colour vision,
  written by `scripts/investigations/cvd_bug_frames.py`.
- `sparse_contrast_check.json`: contrast of the sparse returns over the bay and the bridge cables,
  written by `scripts/investigations/sparse_contrast_check.py`.
- `readme/`: the two pictures at the top of the README, the first screen and cards 1 and 3 side by
  side, shot from the built page over the real octrees by `scripts/pipeline/readme_images.py`.
- `frames/`: the six frames `tests/integrity/test_colour_vision.py` reads, the old model's forest
  before and after the return ratio fix, on both grounds and in the first palette, stored as lossless
  WebP so every pixel is the one the renderer drew:
  `frames/ggpark_ret0_oblique_e6e9e6_edl00_p2.webp`, `frames/ggpark_returns_ood_oblique_e6e9e6_edl00_p2.webp`,
  `frames/ggpark_ret0_oblique_131815_edl00_p2.webp`, `frames/ggpark_returns_ood_oblique_131815_edl00_p2.webp`,
  `frames/ggpark_ret0_oblique_e6e9e6.webp`, `frames/ggpark_returns_ood_oblique_e6e9e6.webp`.
  `scripts/pipeline/render_frames.py` draws frames into `data/cache/frames`; one becomes evidence here only by
  being copied in by hand and named in this list.
