"""Observed vs synthetic sTEC per arc, straight off tec_observations.parquet.

Reads the `synth_stec` (finite fault) and `synth_stec_point` columns that
pipeline/05_synth_stec.py stores beside each observation, so no cube, no LOS
integration and no ray tracing happen here -- a panel is three columns of one
dataframe plotted against t_s.

One panel per arc, four curves, all in raw sTEC units
-----------------------------------------------------
    GFLC (raw)        the measurement, untouched
    SG trend          its Savitzky-Golay slow background alone
    trend + fault     that background PLUS the finite-fault synthetic
    trend + point     that background plus the point-source synthetic

The synthetic is laid ON the observed background rather than compared to a
detrended residual, so each coloured curve is a full RECONSTRUCTION of the
measurement under that source model: if a model were right, its curve would lie
on the black one.  Nothing is rescaled or offset to make that work, so the
comparison is direct -- a model that is too big, too small or mistimed departs
from the black trace by exactly its error, in TECU.

This also keeps the detrend honest without a second panel.  The trend is drawn,
so a reader can see that what the model is being added to is genuinely smooth,
and that the CID in the black trace is the part the trend does not explain.

The window is 1500 s, from stec_los, and is wide ON PURPOSE.  Measured over the
25 G11 arcs, narrowing it costs a median 13% of the peak amplitude at 1200 s,
36% at 900 s and 65% at 600 s -- and the loss is range-dependent (0.92 of the
1500 s amplitude at 360 km against 0.82 at 1192 km, because the far-field pulse
is broader and so closer to the window width).  A narrow window therefore
STEEPENS the apparent observed decay with range, which is exactly the quantity
under test.

The MODEL is NOT detrended.  dNe is driven by the earthquake alone: it starts
from zero, has no ionospheric background in it, and nothing in the integration
introduces a slow trend.  There is nothing there to remove, so filtering it
would only attenuate the signal being tested.  The usual "filter both
identically" rule applies to a shared contaminant; here only one side has one.

Amplitude
---------
The model carries the cube's A0 (=10), a scale fitted to remove the MEDIAN
amplitude bias over all arcs -- not a per-arc prediction.  A factor offset on
any single panel is therefore expected; what is meaningful is the shape, the
timing, and whether the model runs systematically high or low with range.

    python plots/plot_synth_arcs.py                       # 12 arcs by range
    python plots/plot_synth_arcs.py --arcs BNEU_C05_1000 CHMA_C30_1000
    python plots/plot_synth_arcs.py --n 20
    python plots/plot_synth_arcs.py --by-sat                # EVERY arc, one
                                                            # figure per
                                                            # satellite

`--by-sat` drops the range-spread selection entirely: it plots all 249 arcs,
grouped by satellite, into figures/synth_arcs_all/synth_arcs_<SAT>.png.  A
satellite is the natural unit because every arc under it shares one orbit, so
a panel-to-panel change across the sheet is a change in RECEIVER geometry --
range and azimuth -- with the source-to-IPP sampling held as fixed as the data
allow.  Panels stay sorted by IPP range within the sheet, so the amplitude
decay with distance reads down the page.
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cidmodel import paths
from cidmodel.stec_los import DETREND_WIN_S

C_OBS, C_FAULT, C_POINT, C_RAW = '#111111', '#c0392b', '#2471a3', '#b0b0b0'

XLIM_S = (0, 2500)      # common time axis on every panel, in s after origin


def sg_trend(t, v, win_s=DETREND_WIN_S):
    """The Savitzky-Golay slow trend of `v`, over the FINITE samples only.

    Returned rather than subtracted so the caller can DRAW it against the raw
    series -- the point of the upper panel is to show what the detrend removes,
    which is only checkable if the trend itself is visible.  NaN is left where
    `v` was NaN, so the filter never smears across a gap.
    """
    from scipy.signal import savgol_filter

    out = np.full(len(v), np.nan)
    m = np.isfinite(v)
    if m.sum() < 5:
        return out
    vv = v[m]
    dt = np.median(np.diff(t[m])) or 30.0
    win = int(round(win_s / dt))
    win = win + 1 if win % 2 == 0 else win
    n = int(m.sum())
    win = min(win, n if n % 2 else n - 1)
    out[m] = np.median(vv) if win < 5 else savgol_filter(vv, win, polyorder=2)
    return out


def pick_arcs(df, n, col_fault):
    """`n` arcs spread evenly over the observed IPP range."""
    g = (df.groupby('ArcID')
           .agg(dist=('ipp_dist_km', 'median'),
                pk=(col_fault, lambda v: np.nanmax(np.abs(v))))
           .dropna(subset=['pk']))
    g = g[g['pk'] > 1e-3].sort_values('dist')
    if len(g) <= n:
        return g.index.tolist()
    take = np.linspace(0, len(g) - 1, n).round().astype(int)
    return g.index[np.unique(take)].tolist()


def draw_panel(ax, a, col_fault, col_point, tick_fs=7):
    """One arc's four curves on `ax`; returns (dist_km, az_deg) for the title.

    Split out of `main` so the single-sheet and per-satellite modes draw a
    panel the SAME way -- the comparison being made is identical in both, and
    two copies of it would be free to drift apart.
    """
    t = a['t_s'].to_numpy(float)
    g = a['GFLC'].to_numpy(float)
    trend = sg_trend(t, g)             # the slow background alone
    f = a[col_fault].to_numpy(float)
    p = a[col_point].to_numpy(float)

    # Centre the panel on its own arc: subtract the mean sTEC over the
    # displayed window, the SAME constant from every curve, so the comparison
    # between them is untouched and only the absolute level goes.  That level
    # is the whole reason the y ticks were wide -- an arc sitting at -386.5
    # TECU spends four characters per label saying nothing the figure tests,
    # since the model carries one global A0 and is not a per-arc prediction.
    _w = (t >= XLIM_S[0]) & (t <= XLIM_S[1])
    _ref = g[_w & np.isfinite(g)]
    if _ref.size:
        _off = float(_ref.mean())
        g, trend = g - _off, trend - _off

    # Everything is in RAW sTEC units on one axis: the synthetic is laid on
    # the observed background, so each coloured curve is a full
    # reconstruction of the measurement under that source model.
    ax.plot(t, g, color=C_OBS, lw=1.6, zorder=4, label='GFLC (raw)')
    ax.plot(t, trend, color=C_RAW, lw=1.0, zorder=1, label='SG trend')
    ax.plot(t, trend + f, color=C_FAULT, lw=1.3, zorder=3,
            label='trend + fault')
    ax.plot(t, trend + p, color=C_POINT, lw=1.3, ls='--', zorder=3,
            label='trend + point')

    mw = np.isfinite(f)
    if mw.any():
        # A FIXED 0-2500 s window on every panel, rather than one cropped to
        # each arc's synthetic: with a common axis the arrival time can be
        # read ACROSS panels, which is the point of a per-satellite sheet.
        ax.set_xlim(*XLIM_S)
        # y-scale over the displayed window only: across the whole arc the
        # background can drift tens of TECU and would flatten the CID.  The
        # window is NOT intersected with the synthetic's own span -- that ends
        # at ~2075 s against observations running to ~2495 s, so masking on it
        # would leave the last ~400 s of the black trace out of the y-scale
        # and free to run off the top of a panel it is still drawn in.
        w = (t >= XLIM_S[0]) & (t <= XLIM_S[1])
        span = np.concatenate([g[w], (trend + f)[w], (trend + p)[w]])
        span = span[np.isfinite(span)]
        if span.size:
            lo, hi = float(span.min()), float(span.max())
            pad = 0.10 * max(hi - lo, 1e-6)
            ax.set_ylim(lo - pad, hi + pad)

    # Visible tick MARKS on both axes, drawn inward so they cost no width,
    # with unlabelled minors between them: the labels are sparse (3 y, 4 x) to
    # keep the panel mostly data, and the minors put the intermediate values
    # back without spending a single character on them.
    ax.tick_params(labelsize=tick_fs, direction='in', top=True, right=True,
                   which='both')
    ax.tick_params(which='major', length=3.2, width=0.6)
    ax.tick_params(which='minor', length=1.8, width=0.5)
    ax.locator_params(axis='y', nbins=3)
    ax.locator_params(axis='x', nbins=4)
    ax.minorticks_on()
    ax.grid(alpha=0.25)
    return float(a['ipp_dist_km'].median()), float(a['ipp_az'].median())


# A4 portrait text block at ~20 mm margins, in inches.  A figure built to
# THIS size is placed at 1:1 in the manuscript -- nothing is scaled down
# afterwards, which is the only way a stated point size survives to the page.
#
# WIDTH is the constraint; height only has to stay under the page.  Filling
# the full 9.6 in with 3 rows gives tall, slender panels -- the wave is a
# horizontal feature, so a panel wider than it is tall reads far better, and
# a figure that stops short of the bottom margin looks better on the page
# than one straining to fill it.  The height therefore FOLLOWS from the panel
# aspect instead of being imposed.
# Width of the `--page` sheet, in inches.  No longer pinned to a paper size:
# the figure is simply drawn large and the HEIGHT follows from the panel
# aspect and the row count, so nothing is squashed to meet a page.  Scale
# this one number to make the whole figure bigger or smaller -- the type
# scales with it, so the proportions hold.
PAGE_W_IN = 13.4
PANEL_AR = 0.68           # panel height / width; the pulse is horizontal
ROW_EXTRA_IN = 0.074      # per row, as a FRACTION of width: title + ticks
PAGE_PAD_IN = 0.045       # suptitle strip + bottom x-label, same units

# Type, in points at PAGE_W_IN.  What matters is the ratio of text to panel,
# so these are scaled by the width in `draw_sheet` rather than being fixed.
# The legend runs 50% over the tick size: it sits in an empty grid cell with
# room to spare and is the first thing a reader consults.
_FS_REF_W = 13.4
PRINT_FS = {'title': 16.8, 'label': 16.8, 'tick': 14.4, 'legend': 21.6,
            'suptitle': 26.0}


def draw_sheet(df, arcs, col_fault, col_point, title, dest, ncol=4,
               page=False):  # noqa: E501
    """A grid of `arcs` panels -> `dest`.  Returns the path written.

    `page` fits the sheet to an A4 text block for the manuscript: the figure
    takes the page's size rather than the panel count's, and the type is set
    in absolute points for printing at 1:1.  Off (the default), the panel
    keeps a fixed generous size and the sheet grows with the arc count --
    right for a screen sheet, wrong for a page.
    """
    ncol = min(ncol, len(arcs))
    nrow = int(np.ceil(len(arcs) / ncol))
    fs = PRINT_FS if page else {'title': 8, 'label': 8, 'tick': 7, 'legend': 6}

    if page:
        w = PAGE_W_IN
        pw = w / ncol
        # Height FOLLOWS the content -- no cap, so a tall figure is simply
        # tall rather than compressed to fit something.
        figsize = (w, nrow * (pw * PANEL_AR + ROW_EXTRA_IN * w)
                   + PAGE_PAD_IN * w)
        k = w / _FS_REF_W
        fs = {kk: vv * k for kk, vv in fs.items()}
    else:
        figsize = (4.2 * ncol, 2.7 * nrow)
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)
    axes = axes.ravel()

    for ax, arc in zip(axes, arcs):
        a = df[df['ArcID'] == arc].sort_values('t_s')
        d, az = draw_panel(ax, a, col_fault, col_point, tick_fs=fs['tick'])
        # On a page the panel is ~42 mm wide, which will not hold the ArcID,
        # the range and the azimuth on one line -- so the station keeps its
        # own line and the geometry goes beneath it.
        ax.set_title(
            (f'{arc.split("_")[0]}\n{d:.0f} km, az {az:.0f}$\\degree$' if page
             else rf'{arc}   {d:.0f} km, az {az:.0f}$\degree$'),
            fontsize=fs['title'], linespacing=1.15)

    spare = list(axes[len(arcs):])
    for ax in spare:
        ax.axis('off')

    # EVERY panel keeps its own tick labels.  The x window is shared, but the
    # y scale is not -- centring removed the absolute level, not the spread,
    # and that still runs from +/-5 TECU near the source to +/-1 far out.  A
    # panel without its own numbers would be read against its neighbour's.
    # Only the axis NAMES are edge-only, since those really do repeat.
    for k in range(0, len(arcs), ncol):
        axes[k].set_ylabel(r'$\delta$sTEC (TECU)', fontsize=fs['label'])
    for col in range(ncol):
        last = max((k for k in range(len(arcs)) if k % ncol == col),
                   default=None)
        if last is not None:
            axes[last].set_xlabel('t after origin (s)', fontsize=fs['label'])

    # The legend goes in a SPARE cell when the arcs do not fill the grid --
    # it is the one piece of furniture that costs a panel nothing there, and
    # inside a panel it covers the very trace it is labelling (on the page
    # figure the first panel's pulse sits exactly under 'upper left').  Falls
    # back to the first panel when every cell is used.
    handles, labels = axes[0].get_legend_handles_labels()
    if spare:
        spare[0].legend(handles, labels, fontsize=fs['legend'],
                        loc='center', frameon=False)
    else:
        axes[0].legend(fontsize=fs['legend'], loc='upper left',
                       framealpha=0.9)

    # The title is one long sentence, so on a narrow sheet (a satellite with
    # one or two arcs is only 4-8 in wide) it must be WRAPPED and shrunk or it
    # runs off both edges.  Both limits are absolute, not fractional: the
    # reserved strip is 0.32 in per wrapped line, because a one-row sheet is
    # ~3 in tall and a fraction that suits a 12-row sheet is under a line of
    # text here.
    import textwrap
    w_in = fig.get_figwidth()
    # The suptitle scales with the sheet like every other size (it used to be
    # pinned at 11 pt, so it shrank away as the figure grew), and it must NOT
    # be called `fs` -- that name holds the panel sizes, and shadowing it here
    # is what silently un-scaled the rest of the text.
    sup_fs = (fs['suptitle'] if page
              else (11 if w_in >= 12 else (10 if w_in >= 8 else 8.5)))
    # Wrap width in CHARACTERS that actually fit: a character is about
    # 0.55 * the point size wide, and 72 pt to the inch, so the line holds
    # ~ w_in * 72 / (0.55 * sup_fs) of them.  (The earlier form divided by
    # the font size without the inch conversion, which shrank the wrap as
    # the type grew -- the opposite of what is needed.)
    nch = int(w_in * 72.0 / (0.55 * sup_fs))
    wrapped = textwrap.fill(title, max(20, nch))
    nline = wrapped.count('\n') + 1
    fig.suptitle(wrapped, fontsize=sup_fs)
    strip = (0.032 * w_in if page else 0.32)
    fig.tight_layout(rect=(0, 0, 1, 1 - strip * nline / fig.get_figheight()))
    # 300 dpi for a page figure: it is already drawn at full size, so the
    # raster only has to carry that size honestly -- 600 on a 13 in sheet is
    # an 8000 px file for no visible gain.
    fig.savefig(dest, dpi=(300 if page else 140))
    plt.close(fig)
    print('wrote', dest)
    return dest


def by_sat(df, col_fault, col_point, outdir=None):
    """EVERY arc, one figure per satellite, panels sorted by IPP range."""
    outdir = outdir or paths.fig('synth_arcs_all')
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    order = (df.groupby(['sat', 'ArcID'])['ipp_dist_km'].median()
               .reset_index().sort_values(['sat', 'ipp_dist_km']))

    for sat, grp in order.groupby('sat'):
        arcs = grp['ArcID'].tolist()
        draw_sheet(df, arcs, col_fault, col_point,
                   f'{sat}: observed sTEC against the synthetic '
                   r'$\delta$TEC laid on the observed background   '
                   f'({len(arcs)} arc{"s" * (len(arcs) != 1)})',
                   os.path.join(outdir, f'synth_arcs_{sat}.png'))
    print(f'{len(order)} arcs over {order["sat"].nunique()} satellites -> {outdir}')


def main(gen, arcs=None, n=12, out=None, obs_path=None, all_by_sat=False,
         sat=None, page=False, ncol=4):
    df = pd.read_parquet(obs_path or paths.TEC_OBS)
    COL_FAULT, COL_POINT = paths.synth_columns(gen, df)

    if all_by_sat:
        return by_sat(df, COL_FAULT, COL_POINT, outdir=out)

    # `--sat` narrows the pool BEFORE the range spread is taken, so the arcs
    # span that satellite's own range coverage rather than the whole
    # dataset's -- the point of a single-satellite figure.
    pool = df[df['sat'] == sat] if sat else df
    if sat and pool.empty:
        raise SystemExit(f'no arcs for satellite {sat!r}')
    arcs = arcs or pick_arcs(pool, n, COL_FAULT)
    draw_sheet(pool, arcs, COL_FAULT, COL_POINT,
               (f'{sat}: observed' if sat else 'Observed')
               + r' sTEC against the synthetic $\delta$TEC laid on the '
               'observed background',
               out or paths.fig(
                   'synth_arcs' + paths.gen_suffix(gen)
                   + (f'_{sat}' if sat else '')
                   + ('_page' if page else '') + '.png'),
               ncol=ncol, page=page)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--gen', default='',
                    help='synthetic-sTEC column suffix; default is the '
                         'final published columns (pass e.g. "_v3" only while '
                         'comparing generations side by side)')
    ap.add_argument('--arcs', nargs='+', default=None)
    ap.add_argument('--n', type=int, default=12,
                    help='how many arcs to pick across the range span')
    ap.add_argument('--by-sat', action='store_true',
                    help='plot EVERY arc, one figure per satellite, into '
                         'figures/synth_arcs_all/ (--out sets the directory)')
    ap.add_argument('--sat', default=None,
                    help='restrict to ONE satellite (e.g. G11); the --n arcs '
                         'are then spread over that satellite range')
    ap.add_argument('--ncol', type=int, default=4,
                    help='columns in the panel grid (default 4); fewer '
                         'columns means wider panels')
    ap.add_argument('--page', action='store_true',
                    help='fit the sheet to an A4 text block at print type '
                         'sizes, 600 dpi, for placing at 1:1 in a manuscript')
    ap.add_argument('--out', default=None)
    ap.add_argument('--obs', default=None)
    a = ap.parse_args()
    main(a.gen, arcs=a.arcs, n=a.n, out=a.out, obs_path=a.obs,
         all_by_sat=a.by_sat, sat=a.sat, page=a.page, ncol=a.ncol)
