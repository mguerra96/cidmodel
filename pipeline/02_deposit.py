"""Step 2 -- trace the ray fan and deposit it onto the (range, altitude) grid (then runs step 3).

Why this exists
---------------
The crossing-table build (since removed) recorded, per ray and per grid
altitude, where the ray CROSSES
that altitude.  Near apogee a ray is nearly horizontal -- one measured ray
spent 592 km of range within 20 km of its turning height -- yet it contributes
only two crossings there.  The cube was therefore blank across the whole
apogee band above the F region, and no interpolation could honestly fill it
because the ray leaves an altitude on the way up and returns to it hundreds of
km downrange, with nothing in between.

Here each ray is walked in 1 km arc-length steps and every sample is deposited
into the (range, altitude) cell it lands in.  Cell values are means over the
samples inside, which -- the step being uniform -- are path-length weighted.

Rules applied (see cidmodel/deposit.py for the full argument):
  * landing-range fold: past the take-off where the z=0 landing range turns
    over, the descending branch is dropped.  Those rays arc above the F region,
    are absorbed to ~1e-120 and return through the far field, and recording
    them left 130 decades of amplitude between adjacent cells.  Measured at
    az 90 the dynamic range goes 130 -> 35 decades (az 270: 145 -> 35).
  * branches averaged where both reach a cell (~0.01% of cells, travel times
    within ~2 s once the fold rule is in force).
  * interpolation cut to the ALPHA SHAPE of the samples, so the near-field
    dome -- which the convex hull spans by joining one arc back to itself --
    is not invented.  See `deposit.alpha_mask`.

Outputs
-------
  deposit_raw{TAG}.npz    the accumulators, NOT interpolated.  Large (~1 GB)
                          but it makes the interpolation a post-hoc choice:
                          re-run `03_interpolate.py` with a different
                          region rule in minutes instead of re-tracing.
  ray_cube{TAG}.npz       interpolated, written in `RayCube` layout so
                          04_cube.py reads it unchanged.  A non-default
                          --alpha-km adds an `_a<km>` / `_hull` suffix; the
                          default alpha leaves the name bare.

The interpolation is chained on by default (alpha 100 km, az/az+180 paired);
`--no-interpolate` stops after the accumulators.  It is a separate pass rather
than done in the tracing workers because pairing needs both halves of a plane
at once, and a worker holds one azimuth.

Run from a real terminal (ProcessPool double-spawn fails under tool wrappers):
    python pipeline/02_deposit.py
    python pipeline/02_deposit.py --tag _trial --n-takeoff 300
"""
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from cidmodel import paths
from cidmodel.geometry import model_grid
from cidmodel.atmosphere import build_atmosphere
from cidmodel.config import RAY_CEILING_KM
from cidmodel import deposit as D
from cidmodel.interpolation import interpolate_deposit

warnings.filterwarnings('ignore')

# Absorption frequency.  Enters through `absorption_profile` inside
# `deposit_azimuth`, so it is baked into the ACCUMULATORS -- re-interpolating
# cannot change it, only a re-trace can.
#
# 4 mHz, tested against 10 mHz on 2026-08-04 and kept.  The two are not related
# by a scale factor: alpha goes roughly as f^2 and enters as exp(-tau)
# accumulated ALONG THE PATH, so they diverge with range.  Measured against the
# old crossing cube at z = 350 km, 10 mHz gives a flat ~240x offset (200-800 km)
# while 4 mHz runs 73x -> 0.9x, i.e. 10 mHz agrees with the crossing cube's
# decay law and 4 mHz does not.
#
# 4 mHz was kept anyway because it fits the ARCS better: at 10 mHz absorption
# extinguishes the far field, and illuminated arcs drop 82 -> 34 with the
# >400 km share falling from a median 47% to 11%.  Timing was no better either.
# Note both comparisons are per-arc amplitude-normalised, so neither settles
# the ABSOLUTE amplitude -- that needs A0 fitted, still open.
# 2026-09-17: 0.004 -> 0.002.  Not a tuning choice -- 2 mHz is the MEASURED
# median of the 249 picked periods (481.2 s -> 2.08 mHz), so the cube is now
# absorbed at the frequency the disturbance is actually observed to carry.
# The picks run 114-1105 s (IQR 323-605 s, i.e. 1.65-3.09 mHz) and broaden with
# range -- median 166 s inside 300 km against 548 s beyond 900 km -- and 79% of
# arcs sit past 600 km, so a single value is necessarily a far-field compromise.
# 4 mHz (250 s) suited only the 10 nearest arcs and over-absorbed everything
# else: refitting A0 on the 2 mHz cube cut the log10(model/obs) scatter from
# 0.379 to 0.255 dex and flattened the residual range slope from r^-1.57 to
# r^-0.93 on an identical 235-arc set (observed r^-0.11).
#
# The real wave sweeps ~14 -> ~2 mHz along the path as the N-wave broadens
# (see the sigma_eff law in coupling.py), which one constant cannot represent;
# frequency-dependent absorption would be the principled fix.
FREQ_HZ  = 0.002
# 10 km range, 1 deg azimuth -- matched to what the dNe grid can carry rather
# than to the old crossing cube.  The dNe grid is 20 km horizontally, so the
# 2.5 km range step was 8x finer than anything downstream could use, while
# 3 deg azimuth was COARSER than the dNe grid past ~380 km (3 deg = 52 km at
# 1000 km range).  The cube was over-resolved radially and under-resolved
# azimuthally; this fixes both.  Coarser cells also collect more samples per
# cell, so real deposit coverage roughly doubles (25% -> 47%) for free.
RNG_STEP = 10.0
AZ_STEP  = 1.0
RNG_MAX  = 1980.0
# '' = the canonical generation: ray_cube.npz / deposit_raw.npz.  This was
# '_dep' from 2026-09-18 to 2026-09-21, while the fan-axis A_w fix (AW-8)
# was built alongside the generation it replaced; that comparison is over
# and the fix is canonical, so a plain run must rebuild the real cubes
# rather than quietly make a parallel set nothing downstream reads.
TAG      = ''

_P = {}


def _init(interps, rng_edges, z_edges, n_takeoff, freq_hz):
    """Seed each worker once.

    The atmosphere is built ONCE in the parent and shipped through initargs.
    Rebuilding it per worker costs a full
    NRLMSISE-00 + HWM14 evaluation each (the repeated 'Computing sound speed
    profile' banners) for an object that is identical every time.

    `freq_hz` travels through initargs rather than being read from the module
    global in the worker: the parent setting FREQ_HZ would not reach a spawned
    process, so an override would silently trace at the default instead.
    """
    _P.update(interps=interps, rng_edges=rng_edges, z_edges=z_edges,
              n_takeoff=n_takeoff, freq_hz=freq_hz,
              rng=0.5 * (rng_edges[:-1] + rng_edges[1:]),
              z=0.5 * (z_edges[:-1] + z_edges[1:]))


def _one(az):
    """Deposit one azimuth inside the worker.

    Deposit ONLY -- the interpolation is no longer done here.  It used to be,
    to move it into the pool for free, but the settled region rule pairs each
    azimuth with az+180 as one plane and a worker holding a single azimuth
    cannot do that.  `interpolation.interpolate_deposit` runs it afterwards over the
    saved accumulators, in its own pool, for ~2.7 s per azimuth.
    """
    return az, D.deposit_azimuth(az, _P['interps'], _P['rng_edges'],
                                 _P['z_edges'], freq_hz=_P['freq_hz'],
                                 n_takeoff=_P['n_takeoff'])


def main(tag=TAG, n_takeoff=D.N_TAKEOFF, n_jobs=None,
         az_step=AZ_STEP, rng_step=RNG_STEP, interpolate=True,
         freq_hz=FREQ_HZ):
    t_start = time.time()
    grid = model_grid()
    az_all = np.arange(0.0, 360.0, az_step)
    rng_edges = np.arange(0.0, RNG_MAX + rng_step, rng_step)
    z_edges = np.arange(grid.z[0], grid.z[-1] + grid.z[1] - grid.z[0],
                        grid.z[1] - grid.z[0])
    rng = 0.5 * (rng_edges[:-1] + rng_edges[1:])
    zc = 0.5 * (z_edges[:-1] + z_edges[1:])
    n_r, n_z = len(rng), len(zc)

    raw_path = paths.cube(f'deposit_raw{tag}.npz')
    print(f'deposit build: {len(az_all)} azimuths x {n_takeoff} rays, '
          f'grid {n_r} x {n_z} @ {rng_step} km range, {az_step} deg az',
          flush=True)
    print(f'  arc step {D.DS_KM} km, {freq_hz*1e3:.0f} mHz', flush=True)
    print(f'  -> {os.path.basename(raw_path)}', flush=True)

    keys = [f'{k}_{b}' for k in ('n', 'rays', 't', 'az_', 'aw', 'px', 'pz')
            for b in ('up', 'dn')]
    store = {k: np.zeros((len(az_all), n_r, n_z), np.float32) for k in keys}
    folds = np.full(len(az_all), np.nan)
    dropped = np.zeros(len(az_all), int)

    print('  building atmosphere once in the parent ...', flush=True)
    _, _, _, _, interps = build_atmosphere(alt_max=RAY_CEILING_KM, alt_step=2)

    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(
            max_workers=n_jobs, initializer=_init,
            initargs=(interps, rng_edges, z_edges, n_takeoff,
                      freq_hz)) as ex:
        for az, acc in ex.map(_one, [float(a) for a in az_all], chunksize=1):
            ia = int(np.argmin(np.abs(az_all - az)))
            for k in keys:
                store[k][ia] = acc[k].astype(np.float32)
            folds[ia] = float(acc['fold'])
            dropped[ia] = int(acc['n_dropped'])
            done += 1
            if done % 20 == 0:
                rate = (time.time() - t0) / done
                print(f'    {done}/{len(az_all)} azimuths '
                      f'({(time.time()-t0)/60:.1f} min, '
                      f'eta {rate*(len(az_all)-done)/60:.1f} min)', flush=True)
    print(f'  traced in {(time.time()-t0)/60:.1f} min', flush=True)
    print(f'  fold take-off: {np.nanmin(folds):.2f}..{np.nanmax(folds):.2f} deg, '
          f'{dropped.sum()} descending branches dropped', flush=True)

    tmp = raw_path + '.tmp.npz'
    np.savez_compressed(tmp, az=az_all, rng=rng, z=zc, fold=folds,
                        dropped=dropped, freq_hz=freq_hz,
                        n_takeoff=n_takeoff, ds_km=D.DS_KM, **store)
    os.replace(tmp, raw_path)
    print(f'  saved raw accumulators -> {os.path.basename(raw_path)} '
          f'({os.path.getsize(raw_path)/1e9:.2f} GB)', flush=True)

    if interpolate:
        print(flush=True)
        interpolate_deposit(src_tag=tag, n_jobs=n_jobs)
    print(f'\nDONE total {(time.time()-t_start)/60:.1f} min', flush=True)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--tag', default=TAG)
    ap.add_argument('--n-takeoff', type=int, default=D.N_TAKEOFF)
    ap.add_argument('--n-jobs', type=int, default=None)
    ap.add_argument('--az-step', type=float, default=AZ_STEP)
    ap.add_argument('--rng-step', type=float, default=RNG_STEP)
    ap.add_argument('--freq-hz', type=float, default=FREQ_HZ,
                    help=f'absorption frequency (default {FREQ_HZ}, i.e. '
                         f'{FREQ_HZ*1e3:.0f} mHz).  Baked into the '
                         f'accumulators; changing it needs a re-trace.')
    ap.add_argument('--no-interpolate', dest='interpolate',
                    action='store_false',
                    help='stop after the accumulators; run '
                         '03_interpolate.py separately')
    ap.set_defaults(interpolate=True)
    a = ap.parse_args()
    main(tag=a.tag, n_takeoff=a.n_takeoff, n_jobs=a.n_jobs,
         az_step=a.az_step, rng_step=a.rng_step, interpolate=a.interpolate,
         freq_hz=a.freq_hz)
