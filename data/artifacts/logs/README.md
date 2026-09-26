# GPU chain logs

Each long run on a card appends to a chain log: when a rule was written, when a run started and
finished, and what it wrote. Read against the rules in `docs/phase4_criteria.md` and the git history,
they show what was decided before which result existed.

- `gpu_chain_laptop.log`: the chain on the laptop's RTX 3050 and the rented RTX 4090 up to
  2026-09-21, the runs whose artefacts are in `data/artifacts` today. It is the local log with two
  changes and no others: the working copy's absolute path is removed from the lines that name a
  file written, and Windows line endings are made plain newlines.
- `gpu_chain_phase4.log`: written by the Phase 4 chain as it runs, and committed as it stands.
- `phase4_inputs.sha256` and `phase4_outputs.sha256`: the SHA-256 of every file sent to the rented
  card and of every checkpoint and prediction brought back, checked on the other side each time.

Before the first commit of this repository, the order of rules and results rests on these logs and
the dates inside the artefacts, which the project wrote itself. From the first commit on, git records
the order.
