"""Time of the peak: picked vs synthetic, one panel per source model.

Observed   `t_max_s` from arrival_picks.parquet -- the time of the POSITIVE peak of the
           detrended arc.  (`t_min_s`, the trough, always follows it; `t_arr_s`
           is the onset, a median 116 s earlier.)
Synthetic  the time at which the integrated dNe column reaches its maximum.

The maximum, not the onset, because an onset needs a threshold and the model's
is arbitrarily small where the wave is weak -- whereas the peak is well defined
in both series and is what `t_max_s` records.

Left panel is the finite fault, right the point source, on identical axes with
the 1:1 line drawn, so the two source models are compared like with like.  A
point above 1:1 means the MODEL IS LATE.  Colour is IPP range, because the
interesting question is whether the error grows with distance -- a constant
offset would be a source-time error, a growing one is a propagation-SPEED
error.

    python plots/plot_peak_time.py
    python plots/plot_peak_time.py --sat G11 E15
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cidmodel import paths
# Arcs whose synthetic never really lights up have a meaningless argmax, so
# they are dropped rather than plotted as noise.  0.05 TECU peak-to-peak is
# well below every picked amplitude (min 0.23) and well above the numerical
# floor the unilluminated arcs sit at (1e-4 .. 1e-3).
MIN_PTP = 0.05


# Regime tints for the two sides of the 1:1 line.  Muted fills (the data are
# the figure, not the background) with a darker twin for the corner labels.
C_LATE,  C_LATE_T  = '#c0392b', '#8e2b20'      # above 1:1: model arrives late
C_EARLY, C_EARLY_T = '#2471a3', '#1a5276'      # below 1:1: model arrives early


def peak_time(t, v):
    """Time of the maximum of `v`, over its finite samples."""
    m = np.isfinite(v)
    if not m.any():
        return np.nan, 0.0
    tt, vv = t[m], v[m]
    return float(tt[np.argmax(vv)]), float(vv.max() - vv.min())


def main(gen, sats=None, out=None):
    obs = pd.read_parquet(paths.TEC_OBS)
    _f, _p = paths.synth_columns(gen, obs)
    COL = {'finite fault': _f, 'point source': _p}
    picks = pd.read_parquet(paths.ARRIVAL_PICKS)

    rows = []
    for arc, g in obs.groupby('ArcID'):
        t = g['t_s'].to_numpy(float)
        r = {'ArcID': arc}
        for lab, col in COL.items():
            tp, ptp = peak_time(t, g[col].to_numpy(float))
            r[lab] = tp if ptp >= MIN_PTP else np.nan
        rows.append(r)
    syn = pd.DataFrame(rows)

    d = picks[['ArcID', 't_max_s', 'dist_km', 'sat']].merge(syn, on='ArcID')
    if sats:
        d = d[d['sat'].isin(sats)]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.9), sharex=True,
                             sharey=True)
    lim = (500, 2150)

    for ax, lab in zip(axes, COL):
        m = np.isfinite(d[lab]) & np.isfinite(d['t_max_s'])
        x, y = d.loc[m, 't_max_s'], d.loc[m, lab]
        resid = y - x

        # Shade the two regimes the 1:1 line separates, so which side a
        # point falls on reads at a glance instead of from a legend entry.
        ax.fill_between(lim, lim, lim[1], color=C_LATE, alpha=0.13,
                        lw=0, zorder=0)
        ax.fill_between(lim, lim[0], lim, color=C_EARLY, alpha=0.13,
                        lw=0, zorder=0)
        ax.plot(lim, lim, color='0.4', lw=1.2, ls='--', zorder=1)
        ax.annotate('model late', xy=(0.045, 0.955), xycoords='axes fraction',
                    ha='left', va='top', fontsize=9, color=C_LATE_T)
        ax.annotate('model early', xy=(0.955, 0.045), xycoords='axes fraction',
                    ha='right', va='bottom', fontsize=9, color=C_EARLY_T)
        sc = ax.scatter(x, y, c=d.loc[m, 'dist_km'], cmap='viridis', s=34,
                        edgecolor='k', lw=0.4, zorder=3)
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)
        ax.set_xlabel('picked time of peak  $t_{max}$ (s)')
        ax.set_title(f'{lab}   ({int(m.sum())} arcs)', fontsize=11)
        print(f'{lab:14s} n={int(m.sum()):3d}  median {np.median(resid):+7.1f} s  '
              f'mean {resid.mean():+7.1f} s  rms {np.sqrt((resid**2).mean()):6.1f} s')

    axes[0].set_ylabel('synthetic time of peak (s)')
    cb = fig.colorbar(sc, ax=axes, fraction=0.030, pad=0.02)
    cb.set_label('IPP range from epicentre (km)')
    fig.suptitle('Time of peak: synthetic vs picked', fontsize=12)

    _g = paths.gen_suffix(gen)
    dest = out or paths.fig('peak_time' + _g
                            + ('_' + '_'.join(sats) if sats else '') + '.png')
    fig.savefig(dest, dpi=150, bbox_inches='tight')
    print('wrote', dest)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--gen', default='',
                    help='synthetic-sTEC column suffix; default is the '
                         'final published columns (pass e.g. "_v3" only while '
                         'comparing generations side by side)')
    ap.add_argument('--sat', nargs='+', default=None)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    main(a.gen, sats=a.sat, out=a.out)
