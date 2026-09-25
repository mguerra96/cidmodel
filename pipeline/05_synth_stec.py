"""Step 5 -- add synthetic sTEC columns to tec_observations.parquet, on the observation times.

The LOS integration in `stec_los.run_scan` produces synthetic sTEC on the
CUBE's frame times (141 frames, 15 s, 0-2100 s).  The observations live on
their own per-arc sampling (30 s, -1256 to 3244 s).  Comparing them therefore
means interpolating one onto the other, which every plotting script has so far
done for itself.  This does it once and stores the result beside the
observation it belongs to, so a plot becomes a column subtraction.

    python pipeline/05_synth_stec.py                       # fault + point
    python pipeline/05_synth_stec.py --sources binned      # just the fault
    python pipeline/05_synth_stec.py --tag _v3 --suffix _v3

Written columns (one per source model, float64 TECU, SCALED by that model's
own fitted A0):

    synth_stec          finite fault   (--sources binned)
    synth_stec_point    point source   (--sources point)

Source scale A0 -- fitted here, per model
-----------------------------------------
A0 cannot be a build input: it is derived by comparing this very output against
the observed peak-to-peak, so it does not exist until the integration is done.
The dNe cubes therefore hold UNSCALED physics, and the chain from dNe to sTEC
is exactly linear in the scale (verified to 3.5e-08 relative), so one A0=1
integration is rescaled afterwards rather than re-integrated.

`--fit-a0` (the default) integrates at 1.0, then sets each model's A0 to the
value centring log10(model/obs) peak-to-peak on zero over the picked arcs, and
writes the SCALED column.  Each source model gets its OWN A0: what is under
test is how a source distributes energy in space, so each is shown at its best
scale and the comparison rests on shape, timing and scatter rather than on a
shared normalisation.  `--a0 X Y` overrides with explicit values.

The fitted scale is also written back into that source's cube sidecar (`A0`),
which 04_cube.py leaves at 1.0 because the value does not exist until this
runs.  That is what lets the dNe plots draw a scaled field without being handed
the number: they read A0 from the sidecar, so the cuts and the sTEC columns are
always shown at the same scale.  Only an unsuffixed run writes it -- a
`--suffix` run is a side-by-side generation, and its scale must not redefine
what the canonical figures are drawn at.

NaN, NOT zero, outside the model
--------------------------------
The cube covers 0-2100 s, but 43% of observation rows fall outside that -- the
whole pre-event baseline, and everything after 35 min.  Those rows get NaN,
because zero would be a claim (the model predicts no disturbance here) rather
than an absence (the model says nothing here).  The distinction matters for
exactly the things this column exists to make easy: a detrend or an RMS over a
window that runs off the end of the cube would silently average model zeros
into the statistic.  Arcs the scan cannot illuminate ARE a real zero and are
stored as such -- `run_scan` already distinguishes the two cases.

Additive, not a rebuild
-----------------------
This reads the table, adds columns, and writes it back.  It does NOT rebuild
the observations: those come from a preprocessing step that is not part of this
repository and that rebuilds the table from scratch, dropping these columns --
so rerun this after any rebuild of the table.  Re-running this alone is safe and idempotent: an existing column of the
same name is overwritten.
"""
import argparse
import os

import numpy as np
import pandas as pd

from cidmodel import paths
from cidmodel.stec_los import (load_cube, stec_series_tvar, _read_arc_timeseries)
from cidmodel.geometry import geo_to_km

# Column name per source model.  `binned` is the finite fault -- the default
# model -- so it gets the unqualified name.
COLUMN = {'binned': 'synth_stec', 'point': 'synth_stec_point'}


def fit_a0(col, obs, picks):
    """Scale centring log10(model/obs) peak-to-peak on zero.

    Peak-to-peak per arc on both sides; the median of the log ratio is the
    single multiplicative offset, so A0 = 1 / 10**median.  Median rather than
    mean because the ratio distribution has a heavy tail at long range.
    """
    amp = {a: g for a, g in obs.groupby('ArcID')}
    r = []
    for _, p in picks.iterrows():
        g = amp.get(p['ArcID'])
        if g is None:
            continue
        v = g[col].to_numpy(float)
        v = v[np.isfinite(v)]
        if v.size < 20 or not np.isfinite(p['amp_ptp']) or p['amp_ptp'] <= 0:
            continue
        m = float(v.max() - v.min())
        if m > 1e-12:
            r.append(np.log10(m / float(p['amp_ptp'])))
    if not r:
        return 1.0, 0
    return float(10.0 ** -np.median(r)), len(r)


def synth_for_source(obs, source, tag, suffix=''):
    """Model sTEC for one source model, interpolated onto every obs row.

    Integrated UNSCALED (a0=1); the caller applies the fitted A0.
    Returns a float64 Series aligned to `obs.index`.
    """
    meta_path = paths.require_cube(tag, f'dne_{source}')

    cube_dne, x, y, z, times = load_cube(meta_path, a0=1.0)
    t0, t1 = float(times[0]), float(times[-1])
    arcs = sorted(obs['ArcID'].unique())
    tseries = _read_arc_timeseries(arcs)

    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    out = pd.Series(np.nan, index=obs.index, dtype=float)

    n_flat = n_done = 0
    for i, arc in enumerate(arcs, 1):
        rows = obs.index[obs['ArcID'] == arc]
        ts = tseries.get(arc)
        if ts is None:
            continue
        ipx, ipy = geo_to_km(float(np.median(ts['lat'])),
                             float(np.median(ts['lon'])))
        if not (xmin < ipx < xmax and ymin < ipy < ymax):
            continue                       # leave NaN: the model has no opinion

        stec_t = stec_series_tvar(cube_dne, x, y, z, times, ts)
        if np.max(np.abs(stec_t)) < 1e-4:
            n_flat += 1                    # illuminated-but-flat is a REAL zero

        # Interpolate onto this arc's own sample times, and blank anything
        # outside the cube window -- np.interp would otherwise clamp to the end
        # values and manufacture a flat model tail.
        t_obs = obs.loc[rows, 't_s'].to_numpy(float)
        v = np.interp(t_obs, times, stec_t)
        v[(t_obs < t0) | (t_obs > t1)] = np.nan
        out.loc[rows] = v
        n_done += 1

        if i % 50 == 0:
            print(f'    {i}/{len(arcs)} arcs', flush=True)

    inside = np.isfinite(out).sum()
    print(f'  {source:6s}: {n_done}/{len(arcs)} arcs integrated '
          f'({n_flat} illuminated but flat), '
          f'{inside:,}/{len(out):,} rows inside the {t0:.0f}-{t1:.0f} s window')
    return out


def store_a0(source, tag, scale):
    """Write the fitted source scale back into that cube's sidecar.

    A0 is the one number in the sidecar that cannot be known when the cube is
    built: it is fitted by comparing this very cube's sTEC against the observed
    peak-to-peak.  04_cube.py therefore writes A0 = 1.0 (true of the raw
    frames), and this puts the fitted value there once it exists, so the plots
    that read the sidecar draw the scaled field without being told the number.

    Rewritten through a temp file and os.replace, like every other cube write,
    so an interrupted run cannot leave a half-written sidecar.  Only A0 changes;
    every other key is copied through untouched.
    """
    path = paths.require_cube(tag, f'dne_{source}')
    with np.load(path, allow_pickle=True) as d:
        arrays = {k: d[k].copy() for k in d.files}
    was = float(arrays['A0']) if 'A0' in arrays else None
    arrays['A0'] = np.array(float(scale))
    tmp = path + '.tmp.npz'
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    print(f'  A0 -> {os.path.basename(path)}'
          + (f' (was {was:g})' if was is not None else ' (field added)'))


def main(sources, tag, suffix='', obs_path=None, a0=None, do_fit=True):
    obs_path = obs_path or paths.TEC_OBS
    obs = pd.read_parquet(obs_path)
    print(f'{os.path.basename(obs_path)}: {len(obs):,} rows, '
          f'{obs["ArcID"].nunique()} arcs')
    picks = pd.read_parquet(paths.ARRIVAL_PICKS)
    picks = picks[np.isfinite(picks['period_s'])]

    scales = {}
    for i, source in enumerate(sources):
        col = COLUMN[source] + suffix
        print(f'\nintegrating {source} -> column "{col}" (unscaled)', flush=True)
        obs[col] = synth_for_source(obs, source, tag, suffix)

        if a0 is not None:
            scale = float(a0[i] if len(a0) > 1 else a0[0])
            print(f'  A0 = {scale:.4f} (given)')
        elif do_fit:
            scale, n = fit_a0(col, obs, picks)
            print(f'  A0 = {scale:.4f} (fitted on {n} picked arcs, '
                  f'centring log10(model/obs) on zero)')
        else:
            scale = 1.0
            print('  A0 = 1 (unscaled)')
        obs[col] = obs[col] * scale
        scales[source] = scale

        # Put the scale where the dNe plots will find it.  Only for the
        # canonical column set: a `--suffix` run is a side-by-side generation
        # whose columns are not what the cube's own name promises, so it must
        # not redefine the scale the unsuffixed figures are drawn at.
        if not suffix:
            store_a0(source, tag, scale)

    obs.to_parquet(obs_path, index=False)
    print(f'\nwrote {obs_path}')
    print('  source scales: ' +
          '  '.join(f'{k}={v:.4f}' for k, v in scales.items()))
    cols = [COLUMN[s] + suffix for s in sources]
    print(obs[['t_s'] + cols].describe().to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--sources', nargs='+', default=['binned', 'point'],
                    choices=['binned', 'point'],
                    help='source models to integrate (default both)')
    ap.add_argument('--tag', default='',
                    help='dNe cube tag, i.e. dne_<source><tag>.npz (e.g. "_v3"; '
                         'default: the canonical dne_binned / dne_point)')
    ap.add_argument('--suffix', default='',
                    help='appended to the column names, to keep two model '
                         'generations side by side (e.g. "_v3")')
    ap.add_argument('--obs', default=None, help='observations table (.parquet)')
    ap.add_argument('--a0', type=float, nargs='+', default=None,
                    help='explicit source scale(s), one per --sources entry '
                         '(or a single value for all).  Omit to FIT one per '
                         'model against the picked peak-to-peak')
    ap.add_argument('--no-fit', action='store_true',
                    help='write unscaled columns (A0 = 1) instead of fitting')
    a = ap.parse_args()
    if a.a0 is not None and len(a.a0) not in (1, len(a.sources)):
        ap.error(f'--a0 takes 1 or {len(a.sources)} values, got {len(a.a0)}')
    main(sources=tuple(a.sources), tag=a.tag, suffix=a.suffix, obs_path=a.obs,
         a0=a.a0, do_fit=not a.no_fit)
