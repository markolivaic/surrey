# Archive

Code and records the published results do not depend on, kept so the measurements behind them can be
read and rerun.

- `investigations/`: one-off measurements, each with its question and answer in `investigations/README.md`.
- `rented/`: the rented-card scripts that trained and predicted the fp16-search model and re-predicted
  the older models. The code they ran used the fp16 neighbour search, which this repository no longer
  has, so they document that run rather than reproduce it; `data/artifacts/logs/fp16_search_*.sha256`
  hold the hashes of what went in and came out.
- `artifacts/`: records of five earlier rented-card runs that are not comparable to the published
  figures, with the reason in `artifacts/README.md`.
