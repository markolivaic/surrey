# Labelled aerial point cloud datasets: can a second city be added?

Gate D0, checked 2026-09-13. The question was whether at least two labelled aerial
LiDAR datasets besides DALES meet three conditions: a data licence that permits the
use, a download needing no registration, institutional email or click-through, and
an aerial rather than a mobile sensor.

Every answer below comes from the data provider's own page or record. A paper licence
and a hub tag are not a data licence. DALES taught that, and so did this survey:
the SensatUrban repository carries an MIT file that covers its code and says nothing
about its data.

Nothing here was verified from the data itself. Checking geography from coordinates,
the way DALES was shown not to be Dayton, needs the files downloaded, and the two
datasets that pass would mean 16.0 GB and 42.6 GB. That waits for a decision.

## The eight candidates

| dataset | 1 data licence | 2 download without registration | 3 aerial | sensor |
|---|---|---|---|---|
| DublinCity | not verifiable, see below | not verifiable | yes, helicopter | LiDAR |
| Hessigheim H3D | not stated on the access page | **no**, registration form with name, institution and email | yes, UAV | LiDAR, plus mesh |
| LASDU | CC BY-NC-SA 4.0 per the repository README, no LICENSE file | **no**, the README says to email the authors | yes | LiDAR |
| US3D | not stated on the dataset page | **no**, an IEEE account login is required | yes | LiDAR, plus satellite |
| DFC2019 track 4 | **no**, data owned by DigitalGlobe, IARPA and JHU with dissemination restrictions | **no**, IEEE DataPort subscription | yes | LiDAR |
| Vaihingen 3D | research use only, no redistribution to third parties, DGPF acknowledgement required | yes on the letter, a public download link protected by a password shown on the page as an image | yes | LiDAR |
| STPLS3D | none stated | **no**, Google Forms | yes, UAV | photogrammetry, plus synthetic |
| SensatUrban | not stated for the data | **no**, Google Form | yes, UAV | photogrammetry |
| Swiss3DCities | CC BY-NC-SA 4.0, from the Zenodo record metadata | yes, Zenodo open access | yes, multirotor drone | photogrammetry |

Two meet conditions 1 to 3 as written: **Vaihingen 3D** and **Swiss3DCities**.

## What each passing set actually is

**Vaihingen 3D.** Airborne laser scanning over Vaihingen, Germany. Nine classes:
powerline, low vegetation, impervious surfaces, car, fence or hedge, roof, facade,
shrub, tree. Papers using it report about 1.16 million labelled points and 4 to 7
points per square metre; the provider page does not state either. The terms of use
come from the provider's own document and forbid use other than research and any
distribution to third parties.

**Swiss3DCities.** Drone photogrammetry over Zurich, Zug and Davos, 2.7 square
kilometres, five classes: terrain, construction, urban asset, vegetation, vehicle.
One archive of 42.6 GB. Licence CC BY-NC-SA 4.0.

## Why a literal pass is a weak one

Neither set changes only the city.

Swiss3DCities changes the sensor. Photogrammetry reconstructs surfaces from images;
it has no multiple returns and no canopy penetration. A model trained on Surrey plus
Swiss3DCities that does better in San Francisco could be benefiting from a second
sensor rather than a second city, and with one photogrammetric set there is no way to
separate the two.

Vaihingen changes the density by an order of magnitude and the size by more than two.
Its roughly 1.16 million points are a quarter of one percent of DALES before thinning.
Its access password is an image, so a clean-clone script cannot fetch it, and writing
the password into the repository sits badly with terms that forbid distribution. It
would have to be downloaded by hand.

Neither is American, and neither labels water or a bridge deck, the two classes the
target delivery labels and the model cannot name.

## DublinCity could not be checked

The paper states helicopter ALS at about 300 m, an average density of 250 to 348
points per square metre, and about 260 million labelled points out of 1.4 billion, in
13 classes across three levels: building, ground, vegetation and undefined; then
roof, facade, trees, bushes, street, sidewalk and grass; then doors and windows. The
paper points to `v-sense.scss.tcd.ie/DublinCity/` for access. That host refused
connections from three independent routes on 2026-09-13, so its licence and access
terms are unknown and it is counted as not passing rather than guessed.

## Sources

- DublinCity: [paper](https://arxiv.org/abs/1909.03613)
- Hessigheim H3D: [access page](https://ifpwww.ifp.uni-stuttgart.de/benchmark/hessigheim/subscribe.aspx), [ISPRS benchmark page](https://www2.isprs.org/commissions/comm2/wg2/benchmark/)
- LASDU: [repository](https://github.com/Yusheng-Xu/LASDU-Semantic-Labeling-Benchmark)
- US3D: [IEEE DataPort](https://ieee-dataport.org/open-access/urban-semantic-3d-dataset)
- DFC2019: [IEEE DataPort](https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019)
- Vaihingen: [download page](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/default.aspx), [terms of use](https://www2.isprs.org/media/komfssn5/complexscenes_revision_v4.pdf), [class list](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/3d-semantic-labeling.aspx)
- STPLS3D: [repository](https://github.com/meidachen/STPLS3D)
- SensatUrban: [repository](https://github.com/QingyongHu/SensatUrban)
- Swiss3DCities: [Zenodo record](https://zenodo.org/records/4390295)
