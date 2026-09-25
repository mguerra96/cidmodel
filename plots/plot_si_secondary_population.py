"""SI Figure S2: the secondary population of the timing error, and its period.

Figure 8c-d shows a bimodal timing error.  This figure backs the reading of
Section 3.3 that the secondary lobe is Rayleigh-induced:

    (a)  finite-fault peak lag (synthetic - picked) against epicentral range;
         the arcs above SEC_LAG form the secondary population, and they sit
         almost all in the far field.
    (b)  picked period against range, same split; over the range band the
         secondary arcs occupy (shaded, 10th-90th percentile of their range)
         their periods are about half those of the main population.

The lag is computed exactly as in Figure 8 (plot_peak_time_and_ptp.collect),
so the two figures agree arc by arc.

    python plots/plot_si_secondary_population.py
    python plots/plot_si_secondary_population.py --gen _v4
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cidmodel import paths
from plot_peak_time_and_ptp import collect

C_MAIN, C_SEC = '0.55', '#c0392b'
SEC_LAG = 400.0          # s, the trough between the two lobes of Figure 8c


def style(ax):
    ax.grid(alpha=0.25, lw=0.6)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def main(gen='', out=None):
    picks = pd.read_parquet(paths.ARRIVAL_PICKS)
    d = collect(gen, None, 0.05).merge(picks[['ArcID', 'period_s']], on='ArcID')
    d['lag'] = d['t|finite fault'] - d['t_max_s']
    d = d[np.isfinite(d['lag'])]
    sec = (d['lag'] > SEC_LAG).to_numpy()
    r = d['dist_km'].to_numpy(float)
    lo, hi = np.percentile(r[sec], [10, 90])
    band = (r >= lo) & (r <= hi)
    med_s, med_m = d['period_s'][sec & band].median(), d['period_s'][~sec & band].median()
    print(f'{len(d)} arcs, secondary {sec.sum()}; band {lo:.0f}-{hi:.0f} km: '
          f'period {med_s:.0f} s (n={(sec & band).sum()}) vs {med_m:.0f} s (n={(~sec & band).sum()})')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True, gridspec_kw={'wspace': 0.22})
    ax = axes[0]
    for m, c, lab in [(~sec, C_MAIN, f'main (n={(~sec).sum()})'),
                      (sec, C_SEC, f'secondary, lag > {SEC_LAG:.0f} s (n={sec.sum()})')]:
        ax.scatter(r[m], d['lag'][m], s=30, color=c, edgecolor='white', lw=0.7,
                   alpha=0.9, zorder=3, label=lab)
    ax.axhline(SEC_LAG, color='0.3', ls='--', lw=1)
    ax.axhline(0, color='0.3', lw=0.8)
    ax.set_ylabel('finite-fault peak lag, synthetic − picked (s)')
    ax.text(0.03, 0.97, '(a)', transform=ax.transAxes, va='top', fontsize=10.5)
    ax.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9), frameon=False, fontsize=9)

    ax = axes[1]
    ax.axvspan(lo, hi, color='0.94', lw=0, zorder=0)
    for m, c in [(~sec, C_MAIN), (sec, C_SEC)]:
        ax.scatter(r[m], d['period_s'][m], s=30, color=c, edgecolor='white', lw=0.7,
                   alpha=0.9, zorder=3)
    for med, c in [(med_m, '0.25'), (med_s, C_SEC)]:
        ax.plot([lo, hi], [med, med], color=c, lw=2, zorder=4)
    ax.text(0.03, 0.97, f'(b)  median period, {lo:.0f}–{hi:.0f} km:\n'
                        f'      main {med_m:.0f} s, secondary {med_s:.0f} s',
            transform=ax.transAxes, va='top', fontsize=10.5)
    ax.set_ylabel('picked period (s)')
    ax.set_ylim(80, 1300)
    for ax in axes:
        ax.set_xlabel('distance from the epicentre (km)')
        style(ax)

    dest = out or paths.fig('si_secondary_population' + paths.gen_suffix(gen) + '.png')
    fig.savefig(dest, dpi=200, bbox_inches='tight')
    print(dest)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--gen', default='', help='model generation suffix (default: published)')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    main(a.gen, a.out)
