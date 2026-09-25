# Licence for the model weights and the predictions

The MIT licence in `LICENSE` covers the code in this repository and nothing else. This file covers
what the code produced by training:

- the trained weights, every checkpoint of every recipe and seed (M0, M0prime, M1), wherever they
  are passed on;
- the predictions those weights make, as saved by `scripts/pipeline/predict_cache.py`, and the class and
  confidence fields in the viewer octrees and on the page.

These are released under **Creative Commons Attribution-NonCommercial 4.0 International
(CC BY-NC 4.0)**, https://creativecommons.org/licenses/by-nc/4.0/legalcode.

## Why non-commercial

Every model here learned from the labels of the Hugging Face mirror mbendjilali/DALES-2. Those
labels are a revision of the DALES annotations, whose project page gives them "Creative Commons 3.0"
with a separate sentence that the data may not be used for commercial purposes
(THIRD_PARTY_NOTICES.md sets out all three layers). Whether weights trained on labels carry the
labels' terms is an open legal question. This project takes the restrictive reading and passes the
restriction on.

CC BY-NC 4.0 here grants only what this project can grant. It cannot lift the upstream restriction:
commercial use of the weights or predictions needs that question settled with the University of
Dayton first, and the DALES page says they consider such requests case by case.

## Attribution

Credit this project, and carry the upstream attributions of what the models were trained on:

- the Surrey lidar, all recipes:
  > Contains information licensed under the Open Government License – City of Surrey.
- the DALES, DALES Objects and DALES 2 papers, all recipes (citations in THIRD_PARTY_NOTICES.md);
- the Swiss training tiles, M0prime and M1: Federal Office of Topography swisstopo (©swisstopo);
- the water mask that left DALES water points out of M0prime and M1 training:
  > Contains information licensed under the Open Government Licence – British Columbia.

Predictions over San Francisco are drawn on USGS 3DEP points and NAIP imagery, both United States
Government public domain; the points and imagery keep those terms, and only the class and confidence
fall under this file.

## What the predictions are

Predictions of a model trained on lidar from Surrey, British Columbia, and from Zurich and Geneva for
M0prime and M1. They are not measurements, not ground truth and not part of any USGS deliverable.
