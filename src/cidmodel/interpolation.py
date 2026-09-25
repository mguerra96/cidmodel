"""Interpolate a saved ray deposit into a ray cube.

The expensive half of the deposit build is tracing; interpolation is seconds
per azimuth.  `pipeline/02_deposit.py` therefore saves the raw accumulators,
and `interpolate_deposit` turns them into a cube -- so the region rule stays
open after the fact.  `pipeline/03_interpolate.py` is its command line.

Needs no atmosphere and no ray tracing: the alpha shape is built from the
deposited samples already in the file.

Defaults are the settled configuration:

  alpha shape, 100 km    The interpolation region.  `griddata` fills the CONVEX
        hull, which spans the near-field dome -- a region no ray enters, where
        the interpolant joins one arc back to itself.  The alpha shape is the
        non-convex hull and excludes it by construction.

  az/az+180 paired       Each azimuth is interpolated together with its
        opposite as one signed-range plane through the epicentre.  Values are
        identical wherever both are defined; what it fixes is the seam at
        range 0, which is ragged when each half is cut off at the grid edge.

  RayCube layout         Output is written in the format `RayCube.load` reads
        (node-centred axes, ENU k) so pipeline/04_cube.py consumes it unchanged.
        See `to_raycube` -- the mismatch would otherwise be SILENT.
"""
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import paths
from . import deposit as D
from .geometry import model_grid

warnings.filterwarnings('ignore')

KEYS = [f'{k}_{b}' for k in ('n', 'rays', 't', 'az_', 'aw', 'px', 'pz')
        for b in ('up', 'dn')]

# Innermost range bins replaced by their azimuthal median -- see
# `axisymmetrise`.  1 because the defect is exactly one bin wide: bin 0 is the
# only bin with unsampled azimuths, and the excess scatter over the pre-weight
# cube drops from 4-21x there to 1.2-3.3x at bin 1.  0 disables it.
AXISYM_BINS = 1

_W = {}


def _init_interp(raw_path, rng, zc, alpha_km):
    """Seed each worker with the raw deposit, opened once per process.

    np.load on an .npz is lazy, so a worker reads only the azimuth slices it is
    handed rather than holding a 1.1 GB copy.
    """
    _W.update(d=np.load(raw_path), rng=rng, zc=zc, alpha_km=alpha_km)


def _interp_pair(job):
    """Interpolate one az/az+180 plane (or a lone azimuth) inside a worker."""
    ia, ib = job
    d, rng, zc, a_km = _W['d'], _W['rng'], _W['zc'], _W['alpha_km']
    ca = D.collapse({k: d[k][ia].astype(float) for k in KEYS})

    if ib is None:
        return (ia, None,
                D.interpolate(ca['amp'], ca['have'], rng, zc, log=True,
                              alpha_km=a_km),
                D.interpolate(ca['t'], ca['have'], rng, zc, alpha_km=a_km),
                D.interpolate(ca['px'], ca['have'], rng, zc, alpha_km=a_km),
                D.interpolate(ca['pz'], ca['have'], rng, zc, alpha_km=a_km))

    cb = D.collapse({k: d[k][ib].astype(float) for k in KEYS})
    return (ia, ib,
            D.interpolate_plane(ca['amp'], ca['have'], cb['amp'], cb['have'],
                                rng, zc, log=True, alpha_km=a_km),
            D.interpolate_plane(ca['t'], ca['have'], cb['t'], cb['have'],
                                rng, zc, alpha_km=a_km),
            D.interpolate_plane(ca['px'], ca['have'], cb['px'], cb['have'],
                                rng, zc, alpha_km=a_km),
            D.interpolate_plane(ca['pz'], ca['have'], cb['pz'], cb['have'],
                                rng, zc, alpha_km=a_km))


def to_raycube(az, rng_c, z_c, amp, t_w, px, pz, verbose=True):
    """Recast the deposit's own layout into the one `RayCube` reads.

    Downstream (04_cube.py -> coupling.py) loads cubes through
    `RayCube.load`, which wants node-centred axes and a 3-D ENU unit vector.
    The deposit has neither, and the mismatch is SILENT rather than fatal:

      axes    The deposit's rng and z are CELL CENTRES -- rng from 5.0 at 10 km,
              z from 101.25 at 2.5 km (280 of them, against the model grid's
              281 from 100.0).  `RayCube._lookup_nearest` indexes with
              rint(rng / d_r), which assumes an axis based at 0, and `lookup`
              hands back whole altitude columns WITHOUT ever indexing z -- it
              trusts them to line up with the model grid.  Left alone, every
              column would be misregistered by 1.25 km and the grid's top
              altitude never filled, with nothing raised.

      k       The deposit stores (px, pz) in its azimuth's vertical plane,
              already normalised.  ENU follows the convention used throughout
              raytracing.py (sin = east, cos = north; see ray_odes_spherical W_along):
              kx = px sin(az), ky = px cos(az), kz = pz.

    Range resolution is deliberately NOT changed: 10 km against the old cube's
    2.5 km is coarser t_w quantisation under nearest-neighbour lookup, accepted
    in exchange for the deposit's 1 deg azimuth (3x finer than the old 3 deg).
    """
    z_out = model_grid().z
    d_r = float(rng_c[1] - rng_c[0])
    rng_out = np.arange(len(rng_c), dtype=float) * d_r
    n_az, n_r = len(az), len(rng_c)
    dz_c = float(z_c[1] - z_c[0])
    # nearest source cell per target altitude, used to re-mask: np.interp would
    # otherwise carry the nearest finite value across a gap in the fan.
    src_i = np.clip(np.rint((z_out - z_c[0]) / dz_c).astype(int),
                    0, len(z_c) - 1)

    if verbose:
        print(f'  recasting to RayCube layout: rng {rng_c[0]:.1f}..'
              f'{rng_c[-1]:.1f} -> 0..{rng_out[-1]:.0f} km, '
              f'z {len(z_c)} -> {len(z_out)} on the model grid', flush=True)

    out = {}
    for name, a in (('t_w', t_w), ('amp', amp), ('px', px), ('pz', pz)):
        res = np.empty((n_az, n_r, len(z_out)), np.float32)
        for ia in range(n_az):
            col = np.asarray(a[ia], float)
            good = np.isfinite(col)
            r = np.full((n_r, len(z_out)), np.nan)
            for ir in range(n_r):
                g = good[ir]
                if g.sum() >= 2:
                    r[ir] = np.interp(z_out, z_c[g], col[ir][g],
                                      left=np.nan, right=np.nan)
            r[~good[:, src_i]] = np.nan
            res[ia] = r
        out[name] = res

    azr = np.radians(az)[:, None, None]
    return dict(az=az, rng=rng_out, z=z_out, t_w=out['t_w'], amp=out['amp'],
                kx=(out['px'] * np.sin(azr)).astype(np.float32),
                ky=(out['px'] * np.cos(azr)).astype(np.float32),
                kz=out['pz'].astype(np.float32))


def axisymmetrise(out, n_bins=AXISYM_BINS, verbose=True):
    """Replace the innermost range bins with their azimuthal median.

    The range = 0 bin is a SINGLE physical point -- the column directly over the
    epicentre -- stored once per azimuth.  Its 360 copies should agree, and in
    a well-sampled cube they do (1.1x across azimuth in the pre-weight cube).
    In the sin(theta0)-weighted cube they disagree by up to 25x, which prints as
    a pinwheel at the origin.  See the note in `deposit._fields` on the weight.

    The cause is sampling, not the weight (verified 2026-09-16: A_w against
    ARRIVAL RANGE is smooth and single-valued from 0.02 to 12 deg, so rays
    reaching the same place agree -- the formula is not double-counting).  The
    take-off fan is uniform in angle, so only 5 of 200 rays sit below 1 deg,
    while it takes theta0 < ~0.5 deg to land in the first range bin at F-region
    heights.  That bin is therefore served by ~7 rays, 93-120 of its 360
    azimuths receive NO sample at all, and which steep ray happens to fall in a
    given azimuth's cell is arbitrary.  Since 1/sqrt(sin theta0) runs 15.1 at
    0.25 deg to 1.34 at 34 deg, that jitter is amplified into the pinwheel.

    Bin 0 is CATEGORICALLY different, not merely worst: it is the only bin with
    empty azimuths.  Every other bin has full azimuthal coverage, and the excess
    scatter over the clean cube falls from 4-21x at bin 0 to 1.2-3.3x at bin 1
    and decays smoothly outward with no second feature -- so the defect is one
    bin wide and `AXISYM_BINS = 1` is the honest default.

    What it costs: in the clean cube the TRUE azimuthal variation of amplitude
    is 1.4-3% inside 20 km, so collapsing these bins discards a few percent of
    real signal to remove an order-of-magnitude artefact.  That margin does NOT
    hold far out -- real wind-driven anisotropy reaches 8% by 150 km and keeps
    growing -- which is why this must stay confined to the inner bins and is not
    a general smoother.

    Amplitude is medianed in LOG space (it is log-distributed and spans
    decades); t_w and the k components are medianed linearly.  The MEDIAN, not
    the mean, because the cells filled by interpolation where no ray landed are
    exactly the outliers -- a mean would let them set the answer.

    Modifies `out` in place and returns it.
    """
    if not n_bins:
        return out

    for name in ('t_w', 'amp', 'kx', 'ky', 'kz'):
        a = out[name]
        for ir in range(min(n_bins, a.shape[1])):
            v = a[:, ir, :].astype(float)              # (n_az, n_z)
            if name == 'amp':
                with np.errstate(divide='ignore', invalid='ignore'):
                    lv = np.log10(np.where(v > 0, v, np.nan))
                m = np.nanmedian(lv, axis=0)
                rep = 10.0 ** m
            else:
                rep = np.nanmedian(v, axis=0)
            # Keep the fan's own reach: a column no azimuth reached stays NaN,
            # and an azimuth that legitimately has no arrival here is not
            # handed one.
            filled = np.broadcast_to(rep, v.shape).copy()
            filled[~np.isfinite(v)] = np.nan
            a[:, ir, :] = filled.astype(a.dtype)

    # Re-normalise k: the three components were medianed independently, so the
    # vector they form is no longer a unit vector.
    mag = np.sqrt(out['kx'].astype(float) ** 2 + out['ky'].astype(float) ** 2
                  + out['kz'].astype(float) ** 2)
    with np.errstate(invalid='ignore', divide='ignore'):
        for name in ('kx', 'ky', 'kz'):
            out[name] = (out[name] / np.where(mag > 0, mag, np.nan)
                         ).astype(np.float32)

    if verbose:
        print(f'  axisymmetrised the innermost {n_bins} range bin(s): '
              'azimuthal median per altitude', flush=True)
    return out


def interpolate_deposit(src_tag='', out_tag=None, n_jobs=None, alpha_km=D.ALPHA_KM,
         paired=True, axisym_bins=AXISYM_BINS):
    raw_path = paths.cube(f'deposit_raw{src_tag}.npz')
    if out_tag is None:
        # The default alpha leaves the tag bare: `ray_cube_v3.npz`, not
        # `ray_cube_v3_a100.npz`.  Suffixing every cube with a parameter that
        # never varies just made the canonical names longer.  A NON-default
        # alpha still marks itself, so the two cannot collide on disk.
        if alpha_km == D.ALPHA_KM:
            out_tag = src_tag
        else:
            out_tag = f'{src_tag}_a{alpha_km:.0f}' if alpha_km else f'{src_tag}_hull'
    cube_path = paths.cube(f'ray_cube{out_tag}.npz')

    d = np.load(raw_path)
    az, rng, zc = d['az'], d['rng'], d['z']
    n_az, n_r, n_z = len(az), len(rng), len(zc)
    region = f'alpha shape {alpha_km:.0f} km' if alpha_km else 'full convex hull'
    if paired:
        region += ', az/az+180 paired'
    print(f'{os.path.basename(raw_path)}: {n_az} az x {n_r} x {n_z}')
    print(f'  region: {region} -> {os.path.basename(cube_path)}', flush=True)

    amp = np.full((n_az, n_r, n_z), np.nan, np.float32)
    t_w = np.full_like(amp, np.nan)
    px = np.full_like(amp, np.nan)
    pz = np.full_like(amp, np.nan)

    # Pair each azimuth with az+180: the two halves of one vertical plane
    # through the epicentre.  The partner has to exist on the grid, which it
    # does at 1 deg over 360; anything else falls back to per-azimuth.
    jobs = []
    if paired:
        seen = set()
        for ia, a in enumerate(az):
            if ia in seen:
                continue
            j = int(np.argmin(np.abs(az - (a + 180.0) % 360.0)))
            if j != ia and abs((az[j] - (a + 180.0)) % 360.0) < 1e-6:
                jobs.append((ia, j))
                seen.update((ia, j))
            else:
                jobs.append((ia, None))
                seen.add(ia)
        n_pair = sum(1 for _, b in jobs if b is not None)
        print(f'  {n_pair} azimuth pairs'
              + (f', {len(jobs)-n_pair} unpaired' if len(jobs) > n_pair else ''),
              flush=True)
    else:
        jobs = [(ia, None) for ia in range(n_az)]

    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(
            max_workers=n_jobs, initializer=_init_interp,
            initargs=(raw_path, rng, zc, alpha_km)) as ex:
        for ia, ib, a_, t_, x_, z_ in ex.map(_interp_pair, jobs, chunksize=1):
            if ib is None:
                amp[ia], t_w[ia], px[ia], pz[ia] = a_, t_, x_, z_
            else:
                amp[ia], amp[ib] = a_
                t_w[ia], t_w[ib] = t_
                px[ia], px[ib] = x_
                pz[ia], pz[ib] = z_
            done += 1
            if done % 20 == 0:
                print(f'    {done}/{len(jobs)} ({time.time()-t0:.0f}s)',
                      flush=True)
    print(f'  interpolated in {(time.time()-t0)/60:.1f} min', flush=True)

    mag = np.hypot(px, pz)
    with np.errstate(invalid='ignore', divide='ignore'):
        px, pz = px / mag, pz / mag

    out = to_raycube(az, rng, zc, amp, t_w, px, pz)
    axisymmetrise(out, axisym_bins)
    tmp = cube_path + '.tmp.npz'
    np.savez_compressed(tmp, **out)
    os.replace(tmp, cube_path)

    f = np.isfinite(out['amp'])
    k = np.sqrt(out['kx'][f] ** 2 + out['ky'][f] ** 2 + out['kz'][f] ** 2)
    print(f'  saved {os.path.basename(cube_path)} in '
          f'{(time.time()-t0)/60:.1f} min, coverage {100*f.mean():.1f}%, '
          f'amp max {np.nanmax(out["amp"][f]):.2e}')
    print(f'  |k| over filled cells: {np.nanmin(k):.4f} .. {np.nanmax(k):.4f} '
          f'(should be 1)')
