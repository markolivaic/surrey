# The capacity control, on a rented card

The model has 726,225 parameters because the laptop has 4.29 GB, not because anyone decided that was
the right size. This is the one objection to the project that has no answer in the repository, so one
large M0 and one large M1 are trained on a rented card and reported as a control. Everything else
stays as it is: the 5 s load time threshold, the water criterion, the seed rules, the published model.

Written before any of it runs.

## Who does what

The machine is rented, paid for and destroyed by hand, outside this repository. No script here creates
an account or reads a card or a token. The scripts need one line: the host and port of the instance,
the one vast.ai prints as `ssh -p <port> root@<host>`. A public key goes on the instance so that line
works; the private half stays on the local machine.

When the work is done the exact destroy command and the hours the box was up are printed, because a
stopped instance still bills its disk.

## What the instance is

An RTX 4090, 24 GB, one GPU, at least 8 CPU cores and 32 GB of RAM, 100 GB of disk, on the
`vastai/pytorch:cuda-12.8.1-auto` image, which matches the CUDA build this project runs locally
(torch 2.11.0+cu128).

`requirements.txt` pins a CPU wheel on purpose, so that CI runs without a GPU. Installing it on the
instance would replace a working CUDA torch with a CPU one and every run would be quietly slow, so
`scripts/pipeline/rented_setup.sh` installs every pin except torch.

## The data

The instance rebuilds its own data from the public sources with this project's own scripts: the
mbendjilali/DALES-2 mirror from Hugging Face, the Swiss sub-tiles from swisstopo's STAC service, both anonymous. That
is deliberate: those scripts have only ever run on this laptop, so this is also the clean machine test
of them, and `rented_setup.json` records how long it took. If the preparation costs more than two
hours of billed time it is stopped and the prepared cache is sent instead
(`scripts/pipeline/rented_push.sh --with-cache`), and that is recorded too. San Francisco stays here: the
scoring runs on the laptop, so the target data never needs to travel.

## Before any long run, two measurements and a stop

1. 200 steps with the configuration the small runs used (width 128, batch 4), for the real speed-up
   against the RTX 3050 and an estimate of what a full run will cost. The laptop's own figure to read
   it against: 12,000 steps in 13,853.9 s, 0.48 steps a second.
2. A memory probe: the largest configuration that fits in 24 GB with room to spare, by a procedure
   fixed before it runs. Width, depth and batch are recorded with the reason.

Nothing long starts until the speed-up and the chosen configuration have been reviewed.

## What is trained, and how the comparison is read

One large M0 (seed A) and one large M1 (seed A), same architecture, wider model and larger batch, same
data preparation. Length is set by **points seen**, not by steps, so capacity is not confused with how
much data the model saw; the learning rate is scaled linearly with the batch and that is stated in the
artefact. The comparison is not called "capacity alone": it is a larger model under the standard
scaling recipe, and it carries the batch and the learning rate with it.

One seed per recipe. A difference counts only when the large model falls **outside the min to max
range of the three small seeds** for that measurement and that area. Registry variants are
`m0_big_seedA` and `m1_big_seedA`, with a device field saying the figure came from a rented card.

## The upgrade decision, written before the results

- If the large M0 improves the building lift beyond the range of the three small seeds in **at least 4
  of the 6 areas that have footprints**, the recommendation is B+: the page moves to the large model,
  M0 is retrained on three seeds and everything is scored again. The rule recommends; the decision
  is made on the figures, and it is reported with them.
- If it does not, the large model stays a control in the README: one paragraph and one row in a table,
  and nothing else is rewritten.
- If the large M1 gets a **positive** water lift, that is a new story and needs three seeds before
  anything is claimed. It is reported immediately and not written up.
- In domain and every San Francisco class are measured, not only buildings.
