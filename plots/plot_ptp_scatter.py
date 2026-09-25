"""Peak-to-peak amplitude: picked vs synthetic, one panel per source model.

The amplitude twin of plot_peak_time.py.  Observed is `amp_ptp` from
arrival_picks.parquet; synthetic is max - min of the integrated dNe column.  Log-log
with the 1:1 line, coloured by IPP range, so a point ABOVE 1:1 means the model
is TOO LARGE.

Read it differently from the timing figure, though.  Timing has no free
parameter, so its 1:1 line is an absolute test.  Amplitude carries the cube's
A0, a single global multiplier fitted to remove the MEDIAN bias -- so
vertical position relative to 1:1 is partly a calibration choice and only the
SPREAD along the line, and its trend with range, is parameter-free.  That is
why the range colouring matters more here than in the timing plot: the model
decays much faster with range than the observed amplitude does, so
the cloud is expected to swing from above 1:1 at short range to below it at
long range, and no choice of A0 can remove that rotation.

`--min-synth` drops arcs whose SYNTHETIC peak-to-peak falls below a threshold
(0.05 TECU by default).  Those are arcs the ray fan barely illuminates: their
synthetic is a numerical residue rather than a modelled pulse, so they sit as a
floor-hugging streak along the bottom of the panel, stretch the axis by two
decades and drag the median ratio down without carrying any information about
how well the model reproduces a disturbance it actually predicts.  The cut is
on the MODEL only -- cutting on the observation would be selecting on the
quantity under test.  Pass `--min-synth 0` to keep everything.

    python plots/plot_ptp_scatter.py
    python plots/plot_ptp_scatter.py --min-synth 0.05
    python plots/plot_ptp_scatter.py --sat G11 E15
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cidmodel import paths
# Regime tints for the two sides of the 1:1 line.  Muted fills (the data are
# the figure, not the background) with a darker twin for the corner labels.
C_BIG,   C_BIG_T   = '#c0392b', '#8e2b20'      # above 1:1: model too large
C_SMALL, C_SMALL_T = '#2471a3', '#1a5276'      # below 1:1: model too small


def ptp(v):
    v = v[np.isfinite(v)]
    return float(v.max() - v.min()) if v.size else np.nan


def main(gen, sats=None, out=None, min_synth=0.05):
    obs = pd.read_parquet(paths.TEC_OBS)
    _f, _p = paths.synth_columns(gen, obs)
    COL = {'finite fault': _f, 'point source': _p}
    picks = pd.read_parquet(paths.ARRIVAL_PICKS)

    rows = []
    for arc, g in obs.groupby('ArcID'):
        r = {'ArcID': arc}
        for lab, col in COL.items():
            r[lab] = ptp(g[col].to_numpy(float))
        rows.append(r)
    syn = pd.DataFrame(rows)

    d = picks[['ArcID', 'amp_ptp', 'dist_km', 'sat']].merge(syn, on='ArcID')
    if sats:
        d = d[d['sat'].isin(sats)]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.9), sharex=True,
                             sharey=True)
    lim = (5e-2, 3e1)

    for ax, lab in zip(axes, COL):
        finite = (np.isfinite(d[lab]) & np.isfinite(d['amp_ptp'])
                  & (d[lab] > 0) & (d['amp_ptp'] > 0))
        # Cut on the MODEL, per panel: each source model is judged on the arcs
        # it actually illuminates.  Cutting on `amp_ptp` instead would select
        # on the observation, i.e. on the quantity under test.
        m = finite & (d[lab] >= min_synth)
        n_cut = int(finite.sum() - m.sum())
        x, y = d.loc[m, 'amp_ptp'], d.loc[m, lab]
        # ratio in LOG space: the distribution spans decades, so the median of
        # log10 is the honest centre and its spread the honest scatter.
        lr = np.log10(y.to_numpy(float) / x.to_numpy(float))

        # Shade the two regimes the 1:1 line separates, so which side a
        # point falls on reads at a glance instead of from a legend entry.
        ax.fill_between(lim, lim, lim[1], color=C_BIG, alpha=0.13,
                        lw=0, zorder=0)
        ax.fill_between(lim, lim[0], lim, color=C_SMALL, alpha=0.13,
                        lw=0, zorder=0)
        ax.plot(lim, lim, color='0.4', lw=1.2, ls='--', zorder=1)
        if min_synth > 0:
            ax.axhline(min_synth, color='crimson', lw=1.0, ls=':', zorder=2)
        ax.annotate('model too large', xy=(0.045, 0.955),
                    xycoords='axes fraction', ha='left', va='top',
                    fontsize=9, color=C_BIG_T)
        ax.annotate('model too small', xy=(0.955, 0.045),
                    xycoords='axes fraction', ha='right', va='bottom',
                    fontsize=9, color=C_SMALL_T)
        sc = ax.scatter(x, y, c=d.loc[m, 'dist_km'], cmap='viridis', s=34,
                        edgecolor='k', lw=0.4, zorder=3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_aspect('equal')
        ax.grid(alpha=0.3, which='both')
        ax.set_xlabel('picked peak-to-peak $\\delta$TEC (TECU)')
        ax.set_title(f'{lab}   ({int(m.sum())} arcs)', fontsize=11)
        print(f'{lab:14s} n={int(m.sum()):3d} (cut {n_cut})  median model/obs '
              f'{10**np.median(lr):5.2f}x  '
              f'(IQR {10**np.percentile(lr, 25):.2f}-'
              f'{10**np.percentile(lr, 75):.2f}x)  '
              f'within 2x: {100*np.mean(np.abs(lr) < np.log10(2)):4.1f}%')

    axes[0].set_ylabel('synthetic peak-to-peak $\\delta$TEC (TECU)')
    cb = fig.colorbar(sc, ax=axes, fraction=0.030, pad=0.02)
    cb.set_label('IPP range from epicentre (km)')
    sub = (f'   arcs with synthetic $\\geq$ {min_synth:g} TECU'
           if min_synth > 0 else '')
    fig.suptitle('Peak-to-peak amplitude: synthetic vs picked' + sub,
                 fontsize=12)

    dest = out or paths.fig(
        'ptp_scatter' + paths.gen_suffix(gen)
        + ('_' + '_'.join(sats) if sats else '')
        + (f'_min{min_synth:g}'.replace('.', 'p') if min_synth > 0 else '')
        + '.png')
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
    ap.add_argument('--min-synth', type=float, default=0.05,
                    help='drop arcs whose SYNTHETIC ptp is below this (TECU); '
                         '0 keeps everything (default 0.05)')
    a = ap.parse_args()
    main(a.gen, sats=a.sat, out=a.out, min_synth=a.min_synth)
