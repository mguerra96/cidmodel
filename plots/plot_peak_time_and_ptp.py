"""Timing AND amplitude against the 1:1 line, in one figure.

The combination of plot_peak_time.py and plot_ptp_scatter.py.  Both ask the
same question of the same 249 arcs -- how far off the bisectrix does the model
sit, and does that depend on range -- so they belong on one sheet where the
two answers can be read against each other.  A model can be right about WHEN
and wrong about HOW BIG, and only the pair distinguishes that from a model
that is wrong about the propagation itself.

Layout: 2 columns (finite fault | point source) x 4 rows.

    row 1   time of peak, synthetic vs picked          (linear, s)
    row 2   distribution of the distance from 1:1 for row 1
    row 3   peak-to-peak amplitude, synthetic vs picked (log-log, TECU)
    row 4   distribution of the distance from 1:1 for row 3

The distribution rows measure the PERPENDICULAR signed distance from the
bisectrix, d = (y - x)/sqrt(2), taken in the plot's own coordinates -- so in
seconds for the timing panels and in decades of the model/obs ratio for the
log-log amplitude panels, which is the only frame where the 1:1 line of a log
plot is a straight bisectrix and a factor of 2 too large and a factor of 2 too
small are the same distance from it.  d = 0 is the line; the sign keeps the
scatter panel's own convention, positive being the upper-left regime (model
late, model too large), and the panel is tinted with the same two colours so a
lobe reads without a legend.  Its axis RUNS RIGHT TO LEFT for the same reason:
the positive regime is the upper-LEFT half of the scatter above, so a lobe
drawn on the left of the histogram sits under the half of the scatter it
summarises instead of mirroring it.  A kernel density estimate is drawn over the
histogram, because the question being asked of these panels is the SHAPE --
whether the cloud is one centred lobe, an offset one, or two -- and a KDE
answers that without the answer depending on the bin edges.

Read the two pairs differently.  Timing has no free parameter, so the position
of its lobe is an absolute test.  Amplitude carries the cube's A0, a single
global multiplier fitted to remove the median bias, so the amplitude lobe is
CENTRED BY CONSTRUCTION and only its width and skew, and the range trend in
the scatter above it, carry information.  Both docstrings of the parent
scripts say this at greater length.

    python plots/plot_peak_time_and_ptp.py
    python plots/plot_peak_time_and_ptp.py --min-synth 0
    python plots/plot_peak_time_and_ptp.py --sat G11 E15
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cidmodel import paths
from plot_peak_time import MIN_PTP, peak_time
from plot_ptp_scatter import ptp

# Regime tints, shared by both metrics: above the 1:1 line is the RED regime
# (model late / model too large), below it the BLUE one.  Taken from the two
# parent scripts so the combined figure keeps their colour convention.
C_HI, C_HI_T = '#c0392b', '#8e2b20'
C_LO, C_LO_T = '#2471a3', '#1a5276'

SQRT2 = np.sqrt(2.0)

# Vertical gap opened between the timing section and the amplitude section,
# in figure fractions.  Applied by sliding the lower section down; see main().
# Wide enough to carry the amplitude section's own heading, which is drawn
# INTO this gap: at 0.022 the heading sat against the x-label of the timing
# histogram above it and read as belonging to that panel.
SECTION_GAP = 0.040


def perp(x, y):
    """Signed perpendicular distance from the bisectrix, in plot units.

    `x`/`y` are already in the axis' own coordinates (seconds, or log10 TECU),
    so the 1:1 line is y = x and the distance to it is (y - x)/sqrt(2).  The
    sign is kept: positive is above the line, the same regime the scatter
    panel tints red.
    """
    return (np.asarray(y, float) - np.asarray(x, float)) / SQRT2


def kde(d, grid):
    """Gaussian KDE of `d` on `grid`, Silverman bandwidth.  None if degenerate.

    Hand-rolled rather than scipy.stats.gaussian_kde only so that a degenerate
    sample (n < 3, or zero spread) returns None instead of raising -- a
    satellite subset passed via `--sat` can be that small.
    """
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if d.size < 3:
        return None
    sd = d.std(ddof=1)
    iqr = np.subtract(*np.percentile(d, [75, 25]))
    spread = min(sd, iqr / 1.349) if iqr > 0 else sd
    if not spread > 0:
        return None
    h = 0.9 * spread * d.size ** (-0.2)
    z = (grid[:, None] - d[None, :]) / h
    return np.exp(-0.5 * z ** 2).sum(axis=1) / (d.size * h * np.sqrt(2 * np.pi))


def dist_panel(ax, d, unit, log_ratio=False, lim=None):
    """Histogram + KDE of the distance from the bisectrix, on `ax`.

    `lim` fixes the symmetric half-range.  The caller passes ONE value for
    both panels of a row: the two source models are being compared, and a lobe
    that is narrower only because its own panel is wider would be a figure
    that answers the question wrongly.
    """
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if not d.size:
        ax.axis('off')
        return

    lim = float(lim) if lim else float(np.abs(d).max()) * 1.15
    # A symmetric axis on purpose: the panel is about whether the lobe sits
    # off zero, which is only legible if equal distances either side of the
    # line occupy equal space.
    edges = np.linspace(-lim, lim, 41)

    ax.axvspan(0, lim, color=C_HI, alpha=0.13, lw=0, zorder=0)
    ax.axvspan(-lim, 0, color=C_LO, alpha=0.13, lw=0, zorder=0)
    ax.hist(d, bins=edges, color='0.45', alpha=0.55, lw=0.4,
            edgecolor='white', density=True, zorder=2)

    grid = np.linspace(-lim, lim, 400)
    k = kde(d, grid)
    if k is not None:
        ax.plot(grid, k, color='0.15', lw=1.6, zorder=4)

    med = float(np.median(d))
    ax.axvline(0, color='0.4', lw=1.2, ls='--', zorder=3)
    ax.axvline(med, color='#8e44ad', lw=1.4, zorder=5)

    # The median is annotated in the READABLE quantity rather than in the
    # perpendicular distance itself: a reader wants "the model is 250 s late"
    # or "1.4x too large", not the projection onto the normal of the line.
    if log_ratio:
        txt = f'median {10 ** (med * SQRT2):.2f}$\\times$'
    else:
        txt = f'median {med * SQRT2:+.0f} s'
    # Annotated in the NEGATIVE corner, which the reversed axis puts on the
    # right: the lobe of both metrics sits on the positive side, so the label
    # goes opposite it rather than on top of it.
    ax.annotate(txt, xy=(0.98, 0.92), xycoords='axes fraction',
                ha='right', va='top', fontsize=8.5, color='#5b2c6f')

    # Positive (model late / too large) on the LEFT, so the axis runs
    # +lim -> -lim.  In the scatter above, that regime is the upper-LEFT half
    # tinted red; an axis increasing rightwards would put the red lobe on the
    # opposite side from the red half it summarises, and the two rows would
    # have to be read in mirror image of each other.  The AXIS is reversed
    # rather than the data negated, so `d`, the median and its annotation all
    # keep the sign convention of `perp`.
    ax.set_xlim(lim, -lim)
    ax.set_yticks([])
    ax.grid(alpha=0.25, axis='x')
    ax.set_xlabel(f'distance from 1:1   {unit}', fontsize=9)


def collect(gen, sats, min_synth):
    """Per-arc picked and synthetic timing/amplitude, merged on ArcID."""
    obs = pd.read_parquet(paths.TEC_OBS)
    _f, _p = paths.synth_columns(gen, obs)
    cols = {'finite fault': _f, 'point source': _p}
    picks = pd.read_parquet(paths.ARRIVAL_PICKS)

    rows = []
    for arc, g in obs.groupby('ArcID'):
        t = g['t_s'].to_numpy(float)
        r = {'ArcID': arc}
        for lab, col in cols.items():
            v = g[col].to_numpy(float)
            tp, _ = peak_time(t, v)
            a = ptp(v)
            # The timing cut is the parent script's: an arc the fan barely
            # lights up has a meaningless argmax.  Amplitude keeps its own
            # `min_synth`, applied per panel below.
            r[f't|{lab}'] = tp if a >= MIN_PTP else np.nan
            r[f'a|{lab}'] = a
        rows.append(r)

    d = picks[['ArcID', 't_max_s', 'amp_ptp', 'dist_km', 'sat']].merge(
        pd.DataFrame(rows), on='ArcID')
    return d[d['sat'].isin(sats)] if sats else d


def main(gen, sats=None, out=None, min_synth=0.05):
    d = collect(gen, sats, min_synth)
    labels = ['finite fault', 'point source']

    fig, axes = plt.subplots(
        4, 2, figsize=(12.4, 15.4),
        gridspec_kw={'height_ratios': [3, 1, 3, 1], 'hspace': 0.30,
                     'wspace': 0.12})

    t_lim = (500, 2150)
    a_lim = (5e-2, 3e1)

    # The distance-from-1:1 half-range, computed over BOTH source models
    # before anything is drawn, so the two distribution panels of a row share
    # one axis and their widths are directly comparable.
    t_perp, a_perp = {}, {}
    for lab in labels:
        m = np.isfinite(d[f't|{lab}']) & np.isfinite(d['t_max_s'])
        t_perp[lab] = perp(d.loc[m, 't_max_s'], d.loc[m, f't|{lab}'])
        ma = (np.isfinite(d[f'a|{lab}']) & np.isfinite(d['amp_ptp'])
              & (d[f'a|{lab}'] > 0) & (d['amp_ptp'] > 0)
              & (d[f'a|{lab}'] >= min_synth))
        a_perp[lab] = perp(np.log10(d.loc[ma, 'amp_ptp']),
                           np.log10(d.loc[ma, f'a|{lab}']))
    t_half = max(np.abs(v).max() for v in t_perp.values()) * 1.10
    a_half = max(np.abs(v).max() for v in a_perp.values()) * 1.10

    for j, lab in enumerate(labels):
        # --- row 1/2: time of peak ------------------------------------
        ax = axes[0, j]
        m = np.isfinite(d[f't|{lab}']) & np.isfinite(d['t_max_s'])
        x, y = d.loc[m, 't_max_s'].to_numpy(float), d.loc[m, f't|{lab}'].to_numpy(float)

        ax.fill_between(t_lim, t_lim, t_lim[1], color=C_HI, alpha=0.13,
                        lw=0, zorder=0)
        ax.fill_between(t_lim, t_lim[0], t_lim, color=C_LO, alpha=0.13,
                        lw=0, zorder=0)
        ax.plot(t_lim, t_lim, color='0.4', lw=1.2, ls='--', zorder=1)
        ax.annotate('model late', xy=(0.045, 0.955), xycoords='axes fraction',
                    ha='left', va='top', fontsize=9, color=C_HI_T)
        ax.annotate('model early', xy=(0.955, 0.045),
                    xycoords='axes fraction', ha='right', va='bottom',
                    fontsize=9, color=C_LO_T)
        sc_t = ax.scatter(x, y, c=d.loc[m, 'dist_km'], cmap='viridis', s=34,
                          edgecolor='k', lw=0.4, zorder=3)
        ax.set_xlim(*t_lim)
        ax.set_ylim(*t_lim)
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)
        ax.set_xlabel('picked time of peak  $t_{max}$ (s)', fontsize=9)
        ax.set_title(f'{lab}   ({int(m.sum())} arcs)', fontsize=11)

        dist_panel(axes[1, j], perp(x, y), '(s)', lim=t_half)
        resid = y - x
        print(f'time  {lab:14s} n={int(m.sum()):3d}  '
              f'median {np.median(resid):+7.1f} s  mean {resid.mean():+7.1f} s'
              f'  rms {np.sqrt((resid ** 2).mean()):6.1f} s')

        # --- row 3/4: peak-to-peak amplitude --------------------------
        ax = axes[2, j]
        finite = (np.isfinite(d[f'a|{lab}']) & np.isfinite(d['amp_ptp'])
                  & (d[f'a|{lab}'] > 0) & (d['amp_ptp'] > 0))
        m = finite & (d[f'a|{lab}'] >= min_synth)
        n_cut = int(finite.sum() - m.sum())
        xa = d.loc[m, 'amp_ptp'].to_numpy(float)
        ya = d.loc[m, f'a|{lab}'].to_numpy(float)

        ax.fill_between(a_lim, a_lim, a_lim[1], color=C_HI, alpha=0.13,
                        lw=0, zorder=0)
        ax.fill_between(a_lim, a_lim[0], a_lim, color=C_LO, alpha=0.13,
                        lw=0, zorder=0)
        ax.plot(a_lim, a_lim, color='0.4', lw=1.2, ls='--', zorder=1)
        if min_synth > 0:
            ax.axhline(min_synth, color='crimson', lw=1.0, ls=':', zorder=2)
        ax.annotate('model too large', xy=(0.045, 0.955),
                    xycoords='axes fraction', ha='left', va='top',
                    fontsize=9, color=C_HI_T)
        ax.annotate('model too small', xy=(0.955, 0.045),
                    xycoords='axes fraction', ha='right', va='bottom',
                    fontsize=9, color=C_LO_T)
        sc_a = ax.scatter(xa, ya, c=d.loc[m, 'dist_km'], cmap='viridis', s=34,
                          edgecolor='k', lw=0.4, zorder=3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(*a_lim)
        ax.set_ylim(*a_lim)
        ax.set_aspect('equal')
        ax.grid(alpha=0.3, which='both')
        ax.set_xlabel('picked peak-to-peak $\\delta$TEC (TECU)', fontsize=9)
        ax.set_title(f'{lab}   ({int(m.sum())} arcs)', fontsize=11)

        # In LOG space: on a log-log plot the bisectrix is straight only in
        # log coordinates, so that is where the perpendicular distance means
        # what it looks like.
        dist_panel(axes[3, j], perp(np.log10(xa), np.log10(ya)),
                   '(decades of model/obs)', log_ratio=True, lim=a_half)
        lr = np.log10(ya / xa)
        print(f'amp   {lab:14s} n={int(m.sum()):3d} (cut {n_cut})  '
              f'median model/obs {10 ** np.median(lr):5.2f}x  '
              f'(IQR {10 ** np.percentile(lr, 25):.2f}-'
              f'{10 ** np.percentile(lr, 75):.2f}x)  '
              f'within 2x: {100 * np.mean(np.abs(lr) < np.log10(2)):4.1f}%')

    axes[0, 0].set_ylabel('synthetic time of peak (s)')
    axes[2, 0].set_ylabel('synthetic peak-to-peak $\\delta$TEC (TECU)')
    for r in (1, 3):
        axes[r, 0].set_ylabel('density', fontsize=9)

    # One colourbar per SCATTER row, each attached to its own two panels, so
    # the distribution rows are not squeezed by a bar that means nothing to
    # them.  Both show the same quantity and the same scale.
    cbars = {}
    for row, sc in ((0, sc_t), (2, sc_a)):
        cb = fig.colorbar(sc, ax=list(axes[row, :]), fraction=0.030, pad=0.02)
        cb.set_label('IPP range from epicentre (km)', fontsize=9)
        cbars[row] = cb

    # Put each distribution panel on the x-extent of the scatter above it.
    # The scatters are set_aspect('equal'), so matplotlib shrinks them inside
    # their gridspec cell to whatever width makes the box square -- narrower,
    # and differently offset, than the cell the histogram below fills.  The
    # result is a histogram wider than the panel it summarises, whose zero
    # does not fall under that panel's 1:1 line.  The square extent is only
    # known once the aspect and the colourbars have been applied, so the
    # positions are copied HERE, after both, and a draw is forced to settle
    # them.  Height and vertical position are left alone.
    fig.canvas.draw()
    for row in (0, 2):
        for j in range(2):
            src = axes[row, j].get_position()
            dst = axes[row + 1, j].get_position()
            axes[row + 1, j].set_position(
                [src.x0, dst.y0, src.width, dst.height])

    # Separate the two SECTIONS.  A single gridspec hspace cannot do this: it
    # is uniform, so widening it to open a gap between the timing block and
    # the amplitude block would equally push each histogram away from the
    # scatter it belongs to -- which is the pairing the figure depends on.
    # The amplitude section (its scatter, its histogram, and the colourbar
    # placed against that scatter) is therefore slid down bodily instead, so
    # the gap falls only between the two blocks.
    for ax in list(axes[2, :]) + list(axes[3, :]) + [cbars[2].ax]:
        p = ax.get_position()
        ax.set_position([p.x0, p.y0 - SECTION_GAP, p.width, p.height])

    # Panel letters, for the manuscript and its caption.  One per AXES rather
    # than one per section, so a caption can address a scatter and its own
    # distribution separately ("the far-field arcs of (c) are the negative
    # tail of (d)").  They run in reading order down the sheet:
    #
    #     (a) (b)   timing scatter        fault | point
    #     (c) (d)   timing distribution
    #     (e) (f)   amplitude scatter
    #     (g) (h)   amplitude distribution
    #
    # Drawn in FIGURE coordinates off each axes' settled rectangle, after the
    # width alignment and the section shift, so a letter cannot drift from the
    # panel it labels.  Placed outside the top-left corner, where no panel has
    # data: inside, they would land on the red regime tint or on the KDE.
    for k, ax in enumerate(axes.ravel()):
        p = ax.get_position()
        fig.text(p.x0 - 0.026, p.y1 + 0.004, f'({chr(ord("a") + k)})',
                 fontsize=12, fontweight='bold', ha='left', va='bottom')

    # A heading per SECTION rather than one over the whole sheet: the figure
    # is two independent tests that happen to share a page, and the arcs cut
    # from one are not cut from the other.  Each names the quantity its block
    # measures the error in; what that error is measured AGAINST is already on
    # the axis labels, and the per-panel n and the `min_synth` cut are on the
    # panel titles and in the stdout summary, so none of it repeats here.
    #
    # NOT via subplots_adjust: the colourbars are placed against the current
    # axes rectangles, and moving those afterwards slides the panels out from
    # under them.  Each heading instead goes in FIGURE coordinates, just above
    # its own scatter row, read off that row's settled position -- which is
    # also why this runs after the section shift, not before it.
    heads = {0: 'Arrival time error',
             2: 'Peak to peak amplitude error'}
    for row, txt in heads.items():
        p = axes[row, 0].get_position()
        # Centred on the two scatter columns, not on the figure: the
        # colourbar sits outside them and would pull the text off-centre.
        x_mid = 0.5 * (p.x0 + axes[row, 1].get_position().x1)
        fig.text(x_mid, p.y1 + 0.028, txt, fontsize=13,
                 ha='center', va='bottom')

    dest = out or paths.fig(
        'peak_time_and_ptp' + paths.gen_suffix(gen)
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
