# Hashes of the two rented-card runs

- `fp16_search_inputs.sha256`, `fp16_search_outputs.sha256`: every file sent to the card for the run
  that trained the fp16-search model and re-predicted the older models, checked on the card before
  anything ran, and every checkpoint and prediction it produced, checked on the laptop after the
  download.
- `fp32_search_inputs.sha256`, `fp32_search_outputs.sha256`: the same for the run that trained and
  predicted the published models. Its outputs came back over the card's direct SSH port
  (`scripts/pipeline/rented_pull.sh --ssh`), and every hash was checked on the laptop.

The chains' own logs stayed on the machines that ran them. The criteria each run was read by
(`docs/criteria/`) were written before the run; the commit history that showed that order was not
kept when this repository was published.
