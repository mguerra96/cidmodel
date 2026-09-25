"""The acoustic source: a finite supershear rupture.

This is the one substantive departure from IonoSeis (Mikesell et al. 2019),
which uses a point source.  The rupture is read from the USGS finite-fault
model and turned into a set of surface point sources, each with a position, a
rupture time, and a moment weight:

    usgs_fault_sources    parse the SRCMOD .fsp, collapse down-dip -> surface
    binned_fault_sources  snap+merge onto grid nodes (build-time default)
    n_wave                the N-wave source-time function (Mikesell eq. 8)

Every source dict has the schema {x, y, t_rupture_s, weight}, so the point
source (a single dict at the epicentre), the raw fault, and the binned fault
are all drop-in interchangeable in the build.
"""
import math

import numpy as np

from .geometry import geo_to_km
from .paths import FSP


def usgs_fault_sources(fsp_path=FSP, min_moment_frac=1e-4, verbose=True):
    """
    Acoustic sources from the USGS finite-fault model (SRCMOD .fsp format).

    The FSP model has 530 subfaults on 4 segments (106 surface positions x 5
    down-dip rows).  Down-dip rows are **collapsed onto the surface**, summing
    moment:

      * the fault dips 82 deg, so the deepest row (18.4 km) lies only 2.6 km
        laterally from the surface trace -- 6% of one 40 km grid cell;
      * the acoustic wave launches at the ground surface; the upward path
        through the crust is not modelled and at ~3.5 km/s would contribute
        <7 s against a ~600 s acoustic travel time;
      * IonoSeis makes the same assumption (a source "located at the Earth's
        surface", Mikesell et al. 2019 sec. 2.3).

    Rupture time per surface element is the MOMENT-WEIGHTED mean of its
    down-dip column, not the shallowest value (the surface row carries only 5%
    of the moment; the 5.0 and 9.5 km rows together carry 75%).  Weights are
    SF_MOMENT normalised to sum to 1.

    Discarded by the collapse: rise-time structure and down-dip rupture
    propagation, so each element is a single impulse -- consistent with the
    N-wave source-time function (sigma ~ 20 s) being much longer than the
    ~9 s average rise time.

    Returns a list of dict(x, y, t_rupture_s, weight, slip_m); drop-in for
    binned_fault_sources / the point source (same schema).
    """
    rows = []
    with open(fsp_path) as fh:
        for line in fh:
            if line.startswith('%') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 10:
                rows.append([float(v) for v in parts[:10]])

    if not rows:
        raise ValueError(f"no subfault rows parsed from {fsp_path}")

    arr = np.array(rows)
    lat, lon, _, _, z, slip, _, trup, _, moment = arr.T

    x, y = geo_to_km(lat, lon)

    # Group subfaults into down-dip columns.  In this format X == EW and
    # Y == NS (map coordinates, per the FSP header) -- NOT along-strike/down-
    # dip.  The fault strikes ~N-6E, so the along-strike coordinate is Y, which
    # steps by the declared Dx = 5 km.  Depth rows of one column share Y to a
    # few metres while the dip makes their lat/lon differ by ~0.3 km (which is
    # why grouping on position fails).  Key on Y rounded to 1 km.
    y_strike = arr[:, 3]                       # Y == NS (km), along strike
    key = np.round(y_strike, 0)
    _, inverse = np.unique(key, return_inverse=True)
    n_cols = int(inverse.max()) + 1

    sources = []
    total_moment = moment.sum()
    for c in range(n_cols):
        m = inverse == c
        w = moment[m].sum()
        if w <= min_moment_frac * total_moment:
            continue
        wts = moment[m]
        if wts.sum() <= 0:
            continue
        sources.append(dict(
            x=float(np.average(x[m], weights=wts)),
            y=float(np.average(y[m], weights=wts)),
            t_rupture_s=float(np.average(trup[m], weights=wts)),
            weight=float(w / total_moment),
            slip_m=float(np.average(slip[m], weights=wts)),
        ))

    wsum = sum(s['weight'] for s in sources)   # renormalise after dropping
    for s in sources:
        s['weight'] /= wsum

    if verbose:
        ys = np.array([s['y'] for s in sources])
        ts = np.array([s['t_rupture_s'] for s in sources])
        ws = np.array([s['weight'] for s in sources])
        print(f"  USGS FFM: {len(arr)} subfaults -> {len(sources)} surface elements")
        print(f"    extent {ys.min():+.0f} .. {ys.max():+.0f} km N, "
              f"rupture 0 .. {ts.max():.0f} s")
        print(f"    moment-weighted centroid {np.average(ys, weights=ws):+.0f} km N")
    return sources


def binned_fault_sources(grid, sources=None, verbose=True):
    """
    Bin the finite-fault surface elements onto the model's horizontal grid.

    Each source snaps to the nearest (x, y) node at z = 0 (all sources are
    surface point sources), and every source landing on the same node merges
    into ONE effective source:

        x, y         the grid node coordinates (exactly on-grid)
        weight       SUM of the merged moment weights
        t_rupture_s  moment-WEIGHTED MEAN of the merged rupture times, kept at
                     full precision (NOT quantised to the time step, which
                     would re-introduce arrival-time staircasing)

    Why: the fault has ~124 elements over a ~520 km line; at 20 km grid spacing
    many fall in one cell, so sub-cell separation is unresolved anyway.
    Collapsing co-located elements (124 -> ~30) cuts the per-frame source loop
    proportionally and puts every source on a node, so the superposition is
    smoother.  A source-geometry approximation, not just an optimisation:
    sub-20-km structure is discarded (but was unresolved).

    Merged weights sum to 1 exactly (binning is a partition of the input).
    Returns dict(x, y, t_rupture_s, weight) -- same schema as
    usgs_fault_sources / the point source, so a drop-in for the build.
    """
    if sources is None:
        sources = usgs_fault_sources(verbose=False)

    sx = np.array([s['x'] for s in sources])
    sy = np.array([s['y'] for s in sources])
    sw = np.array([s['weight'] for s in sources])
    st = np.array([s['t_rupture_s'] for s in sources])

    # nearest grid node in x and y independently (rectilinear grid)
    ix = np.argmin(np.abs(sx[:, None] - grid.x[None, :]), axis=1)
    iy = np.argmin(np.abs(sy[:, None] - grid.y[None, :]), axis=1)

    merged = {}
    for k in range(len(sources)):
        key = (int(ix[k]), int(iy[k]))
        acc = merged.setdefault(key, {'w': 0.0, 'wt': 0.0})
        acc['w'] += sw[k]
        acc['wt'] += sw[k] * st[k]             # for the weighted-mean rupture time

    out = []
    for (i, j), acc in merged.items():
        w = acc['w']
        out.append(dict(x=float(grid.x[i]), y=float(grid.y[j]),
                        t_rupture_s=float(acc['wt'] / w), weight=float(w)))

    wsum = sum(s['weight'] for s in out)
    assert abs(wsum - 1.0) < 1e-9, f"binned weights sum to {wsum}, not 1"

    if verbose:
        disp = np.hypot(sx - grid.x[ix], sy - grid.y[iy])
        ys = np.array([s['y'] for s in out])
        ws = np.array([s['weight'] for s in out])
        print(f"  binned {len(sources)} elements -> {len(out)} grid nodes "
              f"({len(sources)/len(out):.1f}x fewer)")
        print(f"    snap displacement max {disp.max():.1f} km, "
              f"mean {disp.mean():.1f} km (half-cell "
              f"{0.5*(grid.x[1]-grid.x[0]):.0f} km)")
        print(f"    moment-weighted centroid {np.average(ys, weights=ws):+.0f} km N")
    return out


def n_wave(t, t0, sigma):
    """
    N-wave source-time function: first derivative of a Gaussian.

    Mikesell et al. (2019) eq. 8, without the A_o / A_z amplitude factors
    (applied separately).  Vectorised over `t`.

    Polarity: COMPRESSION-first.  A regular earthquake N-wave leads with
    compression (upward gas push -> TEC increase on arrival), then rarefaction.
    The bare first derivative of a Gaussian leads with its NEGATIVE lobe, so it
    is negated here to make the source compression-first (Heki & Ping 2005;
    Heki & Fujimoto 2022 Fig. 8).  This pairs with the physical -div continuity
    sign applied in 04_cube.py: both sign choices are consistent, so
    the calibrated sTEC is unchanged, but each piece is now textbook-correct.

    Width convention: Mikesell eq. 8 writes exp(-tau^2 / sigma^2); this uses
    the standard Gaussian exp(-tau^2 / 2 sigma^2), so our sigma is sqrt(2)
    times theirs for the same pulse -- sigma is NOT directly comparable between
    the two.  An N-wave of width sigma has dominant period ~ 4.44 sigma.
    """
    tau = np.asarray(t, dtype=float) - t0
    return -(math.sqrt(2.0) / (sigma ** 1.5 * math.pi ** 0.25)) * \
           tau * np.exp(-tau ** 2 / (2.0 * sigma ** 2))
