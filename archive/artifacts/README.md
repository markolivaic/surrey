# Earlier runs, kept out of the results

Five runs trained on a rented card before the retraining that produced the published model, and the
two records that compared and timed them:

- `m0_rented_small_seedA`, the published small configuration on the rented card, to see whether the
  machine itself changed anything;
- `m0_big_2x_seedA` and `m0_big_lr2_seedA`, two variations of the large model;
- `m1_big_seedA`, the large configuration on the Swiss water data;
- `m1_densityaug_seedA`, the small water model with each crop's density drawn at random;
- `compare_rented.json`, those runs read against each other, and `bench_rented_200.json`, a timing.

All five used the fp16 neighbour search, were predicted on an earlier preparation of the San Francisco
tiles (the Bay Bridge tile held 526 more points then), and were scored by an earlier version of the
scoring. They cannot be scored again on today's tiles and are not comparable to any figure in the
README. Paths inside the records are as they were on the rented machine.
