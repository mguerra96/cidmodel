"""SI Figure S1: travel time and amplitude against epicentral vs fault distance.

Backs two statements of Section 3.1 that no main-text figure shows:

    top row      the two azimuthal clusters of the travel-time diagram (along
                 the rupture, 100-220 deg, and off it, W + NE) are offset by
                 about two minutes at equal EPICENTRAL range, and fall onto
                 one curve when range is measured from the nearest point of
                 the rupture.  Each panel carries a common-slope fit with a
                 per-cluster offset, over the range both clusters share.
    bottom row   picked peak-to-peak amplitude barely decays with epicentral
                 range but decays clearly with distance from the fault; at
                 equal distance from the fault the off-rupture arcs are still
                 the weaker ones (the geomagnetic control).

Observations only, so there is no model generation to choose.

    python plots/plot_si_fault_distance.py
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter
from cidmodel import paths
C_ALONG, C_OFF = '#2a78d6', '#eb6834'      # validated categorical pair
ALONG_AZ = (100, 220)                      # deg, the along-rupture sector of Section 3.1
DISTANCES = [('dist_km', 'distance from the epicentre (km)'),
             ('fault_dist_km', 'distance from the nearest point of the rupture (km)')]


def common_slope(x, t, off):
    """t = a + b x + c off: one slope, a per-cluster offset c.  Returns coefs, 1-sigma."""
    X = np.column_stack([np.ones(len(x)), x, off.astype(float)])
    b, *_ = np.linalg.lstsq(X, t, rcond=None)
    r = t - X @ b
    se = np.sqrt(np.diag(r @ r / (len(t) - 3) * np.linalg.inv(X.T @ X)))
    return b, se


def style(ax):
    ax.grid(alpha=0.25, lw=0.6)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def clusters(ax, x, y, along):
    for m, c, mk, lab in [(along, C_ALONG, 'o', f'along rupture, {ALONG_AZ[0]}–{ALONG_AZ[1]}° (n={along.sum()})'),
                          (~along, C_OFF, '^', f'off rupture, W + NE (n={(~along).sum()})')]:
        ax.scatter(x[m], y[m], s=30, marker=mk, color=c, edgecolor='white', lw=0.7,
                   alpha=0.9, zorder=3, label=lab)


def main(out=None):
    p = pd.read_parquet(paths.ARRIVAL_PICKS)
    along = p.az_deg.between(*ALONG_AZ).to_numpy()
    t = p.t_arr_s.to_numpy(float)
    amp = p.amp_ptp.to_numpy(float)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex='col',
                             gridspec_kw={'wspace': 0.08, 'hspace': 0.08})
    for j, (xk, xl) in enumerate(DISTANCES):
        x = p[xk].to_numpy(float)

        # --- travel time: common slope + offset over the shared range
        ax = axes[0, j]
        lim = x[~along].max() * 1.05
        q = x <= lim
        b, se = common_slope(x[q], t[q], ~along[q])
        ax.axvspan(0, lim, color='0.94', lw=0, zorder=0)
        clusters(ax, x, t / 60, along)
        rs = np.linspace(0, lim, 50)
        ax.plot(rs, (b[0] + b[1] * rs) / 60, color=C_ALONG, lw=2, zorder=4)
        ax.plot(rs, (b[0] + b[2] + b[1] * rs) / 60, color=C_OFF, lw=2, ls='--', zorder=4)
        ax.text(0.03, 0.97, f'({"ab"[j]})  off-rupture offset {b[2]:+.0f} ± {se[2]:.0f} s\n'
                            f'       common slope {1 / b[1]:.1f} km/s',
                transform=ax.transAxes, va='top', fontsize=10.5)
        ax.set_ylim(8, 22.5)
        style(ax)
        print(f'{xk}: offset {b[2]:+.0f} ± {se[2]:.0f} s, slope {1 / b[1]:.2f} km/s, fit to {lim:.0f} km')

        # --- amplitude: log-log power law and correlation of log amplitude
        ax = axes[1, j]
        la = np.log10(amp)
        r = np.corrcoef(x, la)[0, 1]
        k, c0 = np.polyfit(np.log10(x), la, 1)
        clusters(ax, x, amp, along)
        xs = np.linspace(max(x.min(), 10), x.max(), 200)
        ax.plot(xs, 10 ** (c0 + k * np.log10(xs)), color='0.2', lw=1.6, zorder=4)
        ax.text(0.03, 0.97, f'({"cd"[j]})  r = {r:+.2f}, amplitude ∝ distance$^{{{k:.2f}}}$',
                transform=ax.transAxes, va='top', fontsize=10.5)
        ax.set_yscale('log')
        ax.set_ylim(0.15, 12)
        ax.yaxis.set_major_locator(FixedLocator([0.2, 0.5, 1, 2, 5, 10]))
        ax.yaxis.set_major_formatter(lambda v, _: f'{v:g}')
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(xl)
        ax.set_xlim(0, 1700)
        style(ax)
        print(f'{xk}: r(log amp) {r:+.2f}, power {k:.2f}')

    axes[0, 1].set_yticklabels([])
    axes[1, 1].set_yticklabels([])
    axes[0, 0].set_ylabel('picked arrival (min after origin)')
    axes[1, 0].set_ylabel('picked peak-to-peak slant TEC (TECU)')
    axes[1, 1].legend(loc='lower left', frameon=False, fontsize=9)

    dest = out or paths.fig('si_fault_distance.png')
    fig.savefig(dest, dpi=200, bbox_inches='tight')
    print(dest)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', default=None)
    main(ap.parse_args().out)
