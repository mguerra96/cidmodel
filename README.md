# cidmodel

[![Software DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22961716.svg)](https://doi.org/10.5281/zenodo.22961716)
[![Data DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22958846.svg)](https://doi.org/10.5281/zenodo.22958846)

Forward model of the coseismic ionospheric disturbance (CID) of the
28 March 2025 Mw 7.7 Myanmar (Mandalay) earthquake, and the code that produces
every figure of the accompanying paper:

> Guerra, M., Cesaroni, C., Astafyeva, E., Ouar, I. D., & Spogli, L. *The 2025
> Mw 7.7 Myanmar Earthquake: Finite Fault Versus Point Source in Modeling
> Coseismic Ionospheric Disturbances.* Submitted to Journal of Geophysical
> Research: Space Physics.

The model radiates an acoustic pulse from the USGS finite-fault model (or from
a point source at the epicentre or at the moment centroid), ray-traces it through a windy, absorbing
atmosphere (NRLMSISE-00, HWM14), couples it into the ionosphere along the
geomagnetic field (IRI-2020, IGRF), and integrates the resulting electron-density
perturbation along each GNSS line of sight into synthetic slant TEC, which is
compared with the observed TEC arcs.

## Repository layout

```
src/cidmodel/     the model, as an installable Python package
pipeline/         the build steps, run in order (01 ... 05; 06 for the weighting tests)
plots/            one script per figure
make_figures.py   regenerates every paper and SI figure
tools/repick/     the interactive picker and the blind re-pick package (SI Text S9)
data/inputs/      inputs: small ones tracked here, the rest from Zenodo
data/cubes/       model cubes, from Zenodo (or rebuilt by pipeline/)
```

## Installation

With conda (recommended: cartopy's GEOS/PROJ libraries install cleanly from
conda-forge):

```
conda env create -f environment.yml
conda activate cidmodel
```

This installs the package in editable mode (`pip install -e .`). The package
locates `data/` and `figures/` relative to the repository, so keep it installed
in editable mode rather than as a copy.

Without conda: `pip install -e ".[figures]"` in a Python >= 3.11 environment.

## Data

The observation tables and the model cubes are archived on Zenodo:
[doi:10.5281/zenodo.22958846](https://doi.org/10.5281/zenodo.22958846).
This DOI resolves to the latest version of the record; use the version that
matches the release of this code (see Versions below). Download the files and
place each as below; the small inputs are already in this repository.

| File | Place in | Size | Source |
|---|---|---|---|
| `tec_observations.parquet` | `data/inputs/` | 13 MB | Zenodo |
| `arrival_picks.parquet` | `data/inputs/` | 35 kB | Zenodo |
| `ne0.npy` | `data/inputs/` | 45 MB | Zenodo, or `pipeline/01_background.py` |
| `B_field.npy` | `data/inputs/` | 134 MB | Zenodo, or `pipeline/01_background.py` |
| `wind_profile_mandalay.npz` | `data/inputs/` | 8 kB | this repository |
| `myanmar_m7.7_usgs_finitefault.fsp` | `data/inputs/` | 56 kB | this repository |
| `ShakeMaps/pgv.shp` `.shx` `.dbf` `.prj` | `data/inputs/ShakeMaps/` | 7 MB | this repository |
| `ray_cube.npz` | `data/cubes/` | 289 MB | Zenodo, or pipeline steps 2-3 |
| `dne_binned.npz` + `dne_binned.npy` | `data/cubes/` | 3.2 GB | Zenodo, or pipeline step 4 |
| `dne_point.npz` + `dne_point.npy` | `data/cubes/` | 3.2 GB | Zenodo, or pipeline step 4 |
| `dne_centroid.npz` + `dne_centroid.npy` | `data/cubes/` | 3.2 GB | Zenodo (v1.1), or pipeline step 4 |
| `stec_basis.parquet` + `stec_basis_sources.npz` | `data/cubes/` | 27 MB | Zenodo (v1.1), or pipeline step 6 |
| `repick_pica_picks.csv`, `repick_pica_key.csv` | `data/inputs/` | 7 kB | this repository |

Each dNe cube is a pair: the `.npy` holds the frames (float32, read as a memory
map) and the small `.npz` beside it holds the axes, the frame times and the
fitted source scale. Keep the two together.

**The observations.** `tec_observations.parquet` holds one row per sample of
each of the 249 satellite-receiver arcs: time, uncalibrated slant TEC (GFLC),
pierce-point position and height, satellite elevation and azimuth, receiver
position, and the synthetic slant TEC of the source models (`synth_stec` for
the finite fault, `synth_stec_point` for the epicentre and, from v1.1,
`synth_stec_centroid` and `synth_stec_centroid_t0` for the moment centroid at
its centroid time and at the origin time). `arrival_picks.parquet` holds one row per arc with its
hand-picked arrival, period and amplitude. Both tables are built from GNSS data
that are partly not publicly redistributable, so the raw data and the
preprocessing that turns them into these tables are not part of this
repository; see the Open Research section of the paper.

## Reproducing the figures

With the data in place:

```
python make_figures.py            # all figures -> figures/paper/Figure_<n>.png
python make_figures.py 5 7 S1     # selected figures
```

| Figure | Script |
|---|---|
| 1 | `plots/plot_geomap.py` |
| 2 | `plots/plot_model_figure.py` |
| 3 | `plots/plot_arrival_map.py` |
| 4 | `plots/plot_travel_time.py` |
| 5 | `plots/plot_ns_coupling_chain.py` |
| 6 | `plots/plot_dne_slices_grid.py` |
| 7 | `plots/plot_synth_arcs.py` (arguments in `make_figures.py`) |
| 8 | `plots/plot_peak_time_and_ptp.py` |
| S1 | `plots/plot_si_fault_distance.py` |
| S2 | `plots/plot_si_secondary_population.py` |

Only Figure 5 does any modelling (it recomputes the neutral velocity field from
the ray cube); the others read the tables and cubes.

## Rebuilding the model

Each version of the data record on Zenodo is the output of the pipeline below
at the matching release of this code (see Versions); rerunning that release
reproduces it. Every step reads and writes the canonical files in
`data/inputs/` and `data/cubes/` by default, overwriting them.

| Step | Script | Output |
|---|---|---|
| 1 | `pipeline/01_background.py` | `ne0.npy` (IRI-2020), `B_field.npy` (IGRF) |
| 2 | `pipeline/02_deposit.py` | `deposit_raw.npz`, then runs step 3 |
| 3 | `pipeline/03_interpolate.py` | `ray_cube.npz` |
| 4 | `pipeline/04_cube.py --sources binned`, `point`, `centroid` (and `centroid_t0`) | `dne_binned`, `dne_point`, `dne_centroid` (`dne_centroid_t0`) |
| 5 | `pipeline/05_synth_stec.py --sources <same>` | synthetic sTEC columns in `tec_observations.parquet`, fitted source scales |
| 6 | `pipeline/06_source_basis.py` (optional) | `stec_basis.parquet`: the sTEC of each of the 30 merged sources alone, for the weighting and polarity tests |

Steps 2 and 4 are the expensive ones and run in parallel over all CPU cores.
`01_background.py` refuses to overwrite existing files unless given `--force`.

**HWM14.** The winds come from `data/inputs/wind_profile_mandalay.npz`, a cached
HWM14 profile at the epicentre. HWM14 has no working pip distribution, so it is
needed only to rebuild that cache (e.g. for another event): install `pyhwm2014`
in a separate conda environment and point the `HWM14_PY` environment variable at
its Python interpreter.

## Blind re-picking

`tools/repick/` holds the interactive picker used for the arrival picks, with
the model synthetics removed, and the scripts of the blind re-pick described in
the Supporting Information (Text S9): `make_package.py` draws the stratified,
anonymised subset (no synthetics, original picks, station, satellite, range or
azimuth), `repick.py` is the picker, and `compare.py` compares the new picks
with the originals. With the defaults they rebuild and compare the paper's
48-arc re-pick (`data/inputs/repick_pica_*.csv`). See
`tools/repick/PICKER_README.md`.

## Versions

| Release | Data record | Paper |
|---|---|---|
| v1.0.1 | Zenodo data version 1.0 | original submission |
| v1.1.0 | Zenodo data version 1.1 | revised manuscript (revision 1) |

**v1.1.0** (revision 1). The model physics is unchanged; what changed:

- Origin time 06:20:52 UTC (USGS PAGER solution) instead of 06:20:56; the
  finite-fault rupture times are counted from it, and the cached HWM14 wind
  profile is re-evaluated at that time.
- Ray-cube range axis: values computed at the range-cell centres (5, 15, 25 ...
  km) were labelled with the cell edges (0, 10, 20 ... km) in v1.0.1, so every
  lookup read a point 5 km further out (at 300 km altitude, 5-6 s of extra delay
  between 200 and 1000 km and about 1.5% in amplitude). Fixed in step 3.
- New point source at the moment centroid of the finite-fault model
  (`--sources centroid`, and `centroid_t0` radiating at the origin time).
- New step 6 and `04_cube.py --only K`: the sTEC basis of the single sources,
  for re-weighting tests (uniform, slip, random polarity) without a rebuild.
- Arcs whose synthetic has no positive peak are a separate category: counted,
  not drawn, in Figure 8; marked in Figure S2, which now uses the same arcs as
  the timing statistics.
- Figure 7 shows eleven arcs from several satellites, receivers and azimuths,
  with the hand picks; Figure S1 gives the cluster offset with a 95% interval
  from a bootstrap over receivers.
- `tools/repick/`: the picker and the blind re-pick.

All cubes and synthetic columns were rebuilt with v1.1.0 (Zenodo data version
1.1), so the v1.0 data do not match this code.

## Data sources and credits

The following third-party products are redistributed in `data/inputs/` for
convenience, unmodified:

- **USGS ShakeMap, peak ground velocity** (`data/inputs/ShakeMaps/pgv.shp`,
  `.dbf`, `.shx`, `.prj`), used for the PGV contours in Figure 1.
  U.S. Geological Survey (2025). *M 7.7 – 2025 Mandalay, Burma (Myanmar)
  earthquake*, event ID us7000pn9s, ShakeMap product.
  https://earthquake.usgs.gov/earthquakes/eventpage/us7000pn9s
- **USGS finite-fault model** (`data/inputs/myanmar_m7.7_usgs_finitefault.fsp`),
  the source description of the model.
  U.S. Geological Survey (2025), event ID us7000pn9s, finite-fault product
  (same event page); described in Goldberg, D. E., et al. (2025). Ultralong,
  supershear rupture of the 2025 Mw 7.7 Mandalay earthquake reveals unaccounted
  risk. *Science*, 390(6772), 458–462. https://doi.org/10.1126/science.ady3581

USGS-authored data are in the U.S. public domain; please cite the sources above
when reusing them.

## License and citation

The code is released under the MIT License (see `LICENSE`). If you use it,
please cite the paper above and this software (see `CITATION.cff`). The
software is archived on Zenodo:
[doi:10.5281/zenodo.22961716](https://doi.org/10.5281/zenodo.22961716). This
DOI always resolves to the latest release; each release also has its own
version DOI, listed on that page.
