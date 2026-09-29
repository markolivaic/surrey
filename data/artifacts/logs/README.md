# Hashes of the rented rounds

- `phase4_inputs.sha256`: the SHA-256 of every file sent to the rented card for Phase 4, checked on the
  card before anything ran.
- `phase4_outputs.sha256`: the SHA-256 of every checkpoint and prediction the card produced, checked on
  the laptop after the download.
- `phase6_inputs.sha256`, `phase6_outputs.sha256`: the same for Phase 6. No Hugging Face login could be
  put on the card, so its outputs came back over the card's direct SSH port, not the vast.ai proxy
  (`scripts/pipeline/phase6_pull.sh --ssh`), and every hash was checked on the laptop.

The chains' own logs stay on the machines that ran them and are not kept here. Before this
repository's first commit, the order in which rules were written and results appeared is self-reported,
in the dates the project wrote into its own artefacts. From the first commit on, the git history shows
it: the retraining's rules are in the first commit, its results in a later one.
