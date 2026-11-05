# Investigations

One-off measurements, each written to answer one question. They are kept runnable from the
repository root (`python archive/investigations/<name>.py`) and each writes the artefact named beside
it under `data/artifacts/`. Several were run on models that are no longer the published one, mostly
the return-ratio model, and say so in their docstrings. Four feed figures in the README or the method
notes: `precision_audit`, `high_points_class`, `water_input_test` and `water_lattice_test`, with
`real_water_density`, `las_flags_audit` and `not_vegetation_heights` behind the method notes.

| Script | Question | Artefact | Answer |
|---|---|---|---|
| `auroc_check` | Does confidence still separate the model's right ground calls from its wrong ones away from home? | `auroc.json` | Less well in San Francisco than at home, on the return-ratio model. |
| `auroc_control` | Is that a lost signal, or is water just hard to separate? | `auroc_control.json` | Split into easy errors (more than 3 m up) and hard ones (within 0.5 m of the ground); the README's water AUROC comes from this split. |
| `bench_model` | What does a training step cost on the laptop card? | `bench_model.json` | Sets the training budget; measured on the four-channel configuration. |
| `block_floor_check` | Do the easy errors come from blocks that lie entirely on a roof? | `block_floor_check.json` | Measured on the return-ratio model; written on the wrong belief that the model reads height above the block floor. |
| `check_resume_gpu` | Does a run stopped and resumed on the card draw the same batches? | `resume_check_gpu.json` | Yes, and the weights match bit for bit. |
| `compare_rented` | How do the earlier rented-card runs compare? | `archive/artifacts/compare_rented.json` | Superseded; see `archive/artifacts/README.md`. |
| `confidence_check` | How sure is the model when it is wrong? | `confidence.json` | Less sure over water than on its right calls. |
| `crop_density` | What point density does the model see once a crop is drawn? | `crop_density.json` | A fixed-length crop thins dense sources and pads sparse ones with repeats. |
| `cvd_bug_frames` | How much of the forest-as-building difference survives colour vision deficiency? | `docs/design/cvd_bug_frames.json` | Enough to pass the palette's rule on the committed frames. |
| `density_classes_test` | Does point spacing move only water, or every class? | `density_classes_test.json` | Measured with a control spacing; see the artefact's reading. |
| `edl_candidates` | How does eye-dome lighting look on the live page at candidate strengths? | `edl_candidates.json` | Fed the sweep below. |
| `edl_sweep` | Which eye-dome strength does each theme take? | `edl_sweep.json` | The strength each theme uses, chosen by a rule fixed before the sweep. |
| `evergreen_candidates` | Where could an evergreen, leaf-on vegetation source come from? | `evergreen_candidates.json` | A list of French tiles; not used further. |
| `four_class_eval` | How does the model score on four classes shared with other national sets? | `four_class_eval.json` | Accuracy and mean IoU on held-out Surrey for the four shared classes. |
| `ground_band_no_water` | How does the ground band change with water left out? | `ground_band_no_water.json` | Both bands per area. |
| `ground_cell_choice` | Which label-free ground surface keeps Surrey's buildings and San Francisco's hills apart? | `ground_cell_choice.json` | None did by the rule set for it, so height is never a model input. |
| `ground_reference_check` | Does the delivery's unassigned class mean "not ground"? | `ground_reference_check.json` | No: part of it is ground, so ground precision is given as a band. |
| `hero_emptiness` | How much of the first screen is background? | `hero_emptiness.json` | Used to frame the first screen. |
| `high_points_class` | What does the model call the highest points in each area? | `high_points_class*.json` | Building for the published model, pole for the fp16-search model, above 150 m in the Financial District. |
| `hosting_check` | Can a static page read byte ranges from each candidate host? | `hosting_check.json` | Hugging Face serves them across origins. |
| `iter_full_agreement` | How much do predictions move when inference draws tensors the way training draws crops? | `iter_full_agreement.json` | Measured against a second run of the same iterator. |
| `las_flags_audit` | Which LAS flags does each source set, per class? | `las_flags_audit.json` | Most Swiss water is flagged synthetic; preparation drops flagged points. |
| `legend_check` | Is the first screen's legend readable at the size it is drawn? | `legend_check.json` | Swatch colours measured as drawn. |
| `national_survey` | What do other national lidar sets look like across many tiles? | `national_survey.json` | Density, dates and labelling per set. |
| `national_tile_check` | How contaminated is each national classification on one urban tile? | `national_tile_check.json` | Per set, with the caveat that one tile is not a country. |
| `not_vegetation_heights` | Where do the NAIP not-vegetation cores stand? | `not_vegetation_heights.json` | In Golden Gate Park they are few and high, so false vegetation there is not read. |
| `palette_search` | Which class palette passes the rules in `backend/surrey/palette.py`? | `docs/design/palette.json` | The palette the page uses. |
| `points_per_pixel` | Why does one framing look grainier than another? | `points_per_pixel.json` | Points drawn per covered pixel, per framing. |
| `precision_audit` | Where does half precision lose the geometry, layer by layer? | `precision_audit.json` | In the neighbour search: the fp16-search model kept a minority of the true neighbours above 100 m. |
| `real_water_density` | How densely is real water measured once synthetic points are removed? | `real_water_density.json` | Far more sparsely on the Swiss tiles than in San Francisco. |
| `rented_speedup` | How much faster is the rented card than the laptop? | `rented_speedup.json` | About 25 times on the same configuration. |
| `repro_potreeconverter_extra_bytes` | Does PotreeConverter 2.1.3 read LAS extra bytes correctly? | `docs/potreeconverter_extra_bytes.json` | No: it reads them from the wrong offset, so predictions go in standard fields. |
| `return_channel_check` | What happens when the return ratio is set back to its training value? | `return_channel_check.json` | San Francisco vegetation comes back: the dead channel was a major cause. |
| `return_ratio_check` | Does the return ratio separate classes at home and away? | `return_ratio_check.json` | It is 0 on every DALES-2 point, so it carries nothing at home. |
| `score_transfer` | What can be scored when the Surrey model runs on San Francisco? | `transfer_sf.json` | The first scoring, on the return-ratio model. |
| `score_vegetation` | What does the model call the trees NAIP sees? | `vegetation_sf.json` | The first vegetation scoring, on the return-ratio model. |
| `shoot_hero` | How do the first screen's candidate arrangements look? | frames only | Used to choose the first screen. |
| `sparse_contrast_check` | Do isolated points reach 3:1 contrast on real frames? | `docs/design/sparse_contrast_check.json` | Yes, in both themes. |
| `tower_colour_check` | What does the model call the Financial District's towers, and what does a frame show? | `tower_colour_check.json` | Read with `high_points_class`. |
| `viewer_budget` | How much of San Francisco fits in a repository with no backend? | `viewer_budget.json` | Bytes per point by encoding, so the octrees are hosted elsewhere. |
| `water_density_test` | Does point density explain the water failure? | `water_density_test.json` | Measured by thinning San Francisco to the Swiss spacing. |
| `water_input_test` | What does the model see on open water? | `water_input_test.json` | Water is flat on both sides, as flat ground is; written on the wrong belief that the model reads height above the block floor. |
| `water_lattice` | Do water points sit on a regular grid? | `water_lattice.json` | Swiss water does, on 25 cm; San Francisco's does not. |
| `water_lattice_test` | Does M1 read water by that grid? | `water_lattice_test.json` | On one seed of three. |
| `where_is_dales` | Where was DALES flown? | `where_is_dales.json` | Surrey, British Columbia, from the tiles' coordinates. |
