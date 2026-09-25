"""
Grid of 2-D dNe horizontal slices: finite fault vs point source, over time.

  rows    : the timestamps in TIMES
  columns : finite-fault {250, 300, 350} km  |  point-source {250, 300, 350} km

All panels share one colour scale.  dNe is the time-integrated end product
already stored in the cube, so each panel is a slice read straight off the
memmap -- unlike v_n.B_hat, which has to be recomputed from the ray cube for
every (source, time).  That makes this script cheap (no ray cube, no
atmosphere, no neutral-velocity call) and it means the panels show exactly the field the
sTEC integration consumes rather than a re-derivation of it.

Values are scaled by A0 (from the sidecar, or --a0) and drawn in 1e8 m^-3.

Two things worth reading carefully in the result:
  * the polarity flip across hmF2 is imposed by the vertical Ne0 gradient in
    the dNe = -div(Ne0 v_i) step, NOT by a propagating +/- wave -- so the 250
    and 350 km columns can be opposite in sign at the same instant.
  * unlike v_n.B_hat, dNe carries no "non-illuminated" flag: an exactly-zero
    node can mean unlit OR genuinely zero disturbance.  Zeros are left as the
    colour-map midpoint rather than masked, so absence of colour here does not
    by itself mean absence of ray coverage.

  python plots/plot_dne_slices_grid.py              # Figure 6
  python plots/plot_dne_slices_grid.py --tag _v3 [--times 600 690 780 870]
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm, colors

from cidmodel import paths
from cidmodel.geometry import model_grid

TIMES     = [660.0, 750.0, 840.0]     # rows (s after origin)
Z_SLICES  = [250.0, 300.0, 350.0]            # altitudes per source (km)
SMOOTH_DZ = 0                                # +/- nodes averaged along z
HALF_KM   = 1000.0                            # horizontal data crop half-width (km)
XLIM      = (-500.0, 500.0)                  # display x range (km)
YLIM      = (-750.0, 500.0)                  # display y range (km)
PCT       = 99.5                             # robust shared colour-limit pct
NQ        = 4                                # B quiver arrows across a panel
OUT_FILE  = paths.fig('dne_slices_grid_fault_vs_point.png')


def _slice(field, z, zt, mx, my):
    """z-averaged horizontal slice at altitude zt."""
    iz = int(np.argmin(np.abs(z - zt)))
    z0 = max(0, iz - SMOOTH_DZ)
    z1 = min(len(z) - 1, iz + SMOOTH_DZ)
    return field[:, :, z0:z1 + 1].mean(axis=2)[np.ix_(mx, my)]


def main(tag, times=None, out_file=None, a0=None):
    g = model_grid()
    times = list(times or TIMES)
    NT = len(times)
    mx = np.abs(g.x) <= HALF_KM
    my = np.abs(g.y) <= HALF_KM
    xc, yc = g.x[mx], g.y[my]
    extent = [xc[0], xc[-1], yc[0], yc[-1]]

    # B_hat for the quiver: sampled on the SAME crop as the panels and at each
    # slice altitude, not a single epicentre vector -- the inclination swings
    # ~34 deg across this grid, so one arrow direction would be wrong at the
    # edges.  Only the horizontal (E,N) part is drawable in a map view; the
    # arrows therefore shorten where the field dips more steeply into the page.
    Bfull = np.load(paths.B_FIELD)
    # Lattice spans the DISPLAY window, not the data crop: HALF_KM (1000) is
    # much wider than XLIM/YLIM, so sampling the crop put most arrows outside
    # the visible area.  Inset from the edges so the long arrows (pivot='mid')
    # do not overhang the frame.
    _xq = np.linspace(XLIM[0], XLIM[1], NQ + 2)[1:-1]
    _yq = np.linspace(YLIM[0], YLIM[1], NQ + 2)[1:-1]
    iq = np.array([int(np.argmin(np.abs(xc - v))) for v in _xq])
    jq = np.array([int(np.argmin(np.abs(yc - v))) for v in _yq])
    Xq, Yq = np.meshgrid(xc[iq], yc[jq], indexing='ij')
    Bq = {}
    for zt in Z_SLICES:
        izq = int(np.argmin(np.abs(g.z - zt)))
        Bs = Bfull[np.ix_(mx, my)][:, :, izq, :]
        Bq[zt] = (Bs[np.ix_(iq, jq)][:, :, 0], Bs[np.ix_(iq, jq)][:, :, 1])

    # ---- dNe straight off the cubes (sidecar for times + A0, memmap for data)
    cubes = {}
    for kind in ('binned', 'point'):
        meta = np.load(paths.cube(f'dne_{kind}{tag}.npz'), allow_pickle=True)
        # The source scale is a FITTED result: 04_cube.py writes A0 = 1.0
        # and 05_synth_stec.py replaces it with the fitted value once that
        # exists, so reading it here draws these slices at the same scale as
        # the synth_stec columns.  A cube not yet fitted still reads 1.0 (the
        # raw field).  `a0` overrides either way.
        # Each source model has its OWN fitted A0 (2026-09-18), so --a0 takes
        # one value per model in the order (binned, point); a single value is
        # applied to both, which is only meaningful for --a0 1 (raw field).
        _stored = float(meta['A0']) if 'A0' in meta.files else 1.0
        if a0 is None:
            _scale = _stored
        else:
            _scale = float(a0[0] if len(a0) == 1
                           else a0[('binned', 'point').index(kind)])
        cubes[kind] = (
            np.asarray(meta['times'], float),
            _scale,
            np.lib.format.open_memmap(
                paths.cube(str(meta['dne_memmap'])), mode='r'))

    # panels[(row, col)] = slice ; cols 0-2 fault z, cols 3-5 point z
    panels = {}
    for r, t in enumerate(times):
        for kind, base in (('binned', 0), ('point', 3)):
            tt, A0, arr = cubes[kind]
            it = int(np.argmin(np.abs(tt - t)))
            if abs(tt[it] - t) > 1e-6:
                print(f'  note: t={t:.0f}s -> nearest stored frame '
                      f'{tt[it]:.0f}s ({kind})')
            field = np.asarray(arr[it], float) * A0 / 1e8
            for c, zt in enumerate(Z_SLICES):
                panels[(r, base + c)] = _slice(field, g.z, zt, mx, my)
            pk = max(np.abs(panels[(r, base + c)]).max()
                     for c in range(len(Z_SLICES)))
            print(f'  t={t:.0f}s  {kind:6s} |dNe|max={pk:.3g}e8')

    # one shared symmetric colour limit across every panel
    allv = np.concatenate([np.abs(v[np.isfinite(v)]) for v in panels.values()])
    allv = allv[allv > 0]
    lim = np.nanpercentile(allv, PCT) if allv.size else 1.0
    if not np.isfinite(lim) or lim <= 0:
        lim = 1.0
    norm = colors.Normalize(-lim, lim)
    cmap = cm.RdBu_r

    nz = len(Z_SLICES)                                   # 3 per block
    _t = np.array([-300.0, 0.0, 300.0])
    xt = _t[(_t > XLIM[0] + 1e-6) & (_t < XLIM[1] - 1e-6)]
    yt = _t[(_t > YLIM[0] + 1e-6) & (_t < YLIM[1] - 1e-6)]
    # GridSpec: [3 fault cols][spacer][3 point cols]; panels within a block
    # touch (wspace=hspace=0).  Cell physical size is set to the data aspect so
    # equal-aspect panels fill each cell exactly -> rows AND columns touch with
    # the TRUE aspect preserved.
    FS = 1.44                                            # font-size scale (+44%)
    xspan = XLIM[1] - XLIM[0]
    yspan = YLIM[1] - YLIM[0]
    pw = 1.9                                             # panel width (in)
    ph = pw * yspan / xspan                              # true-aspect height
    # horizontal colorbar underneath: the right margin drops back to a small
    # gutter (the B arrow still lives there) and the extra room goes into
    # height for the bar + its labels below the bottom row.
    fig = plt.figure(figsize=(pw * (2 * nz) + 1.1, ph * NT + 2.0))
    # reserve a bottom strip for the colorbar.  It has to hold the bar, its
    # tick numbers AND the axis label, plus the row of x tick labels already
    # under the bottom panels -- 1.05 in was not enough and clipped the lot.
    _figh = ph * NT + 2.0
    _bot = 1.62 / _figh
    gs = fig.add_gridspec(NT, 2 * nz + 1,
                          width_ratios=[1] * nz + [0.28] + [1] * nz,
                          wspace=0.0, hspace=0.0,
                          bottom=_bot, top=0.94, left=0.085, right=0.955)

    def _col_to_gs(c):
        """map data column 0..5 to GridSpec column (skip the spacer at nz)."""
        return c if c < nz else c + 1

    im = None
    axs = {}
    for r in range(NT):
        for c in range(2 * nz):
            ax = fig.add_subplot(gs[r, _col_to_gs(c)])
            axs[(r, c)] = ax
            im = ax.imshow(panels[(r, c)].T, origin='lower', extent=extent,
                           cmap=cmap, norm=norm, aspect='equal',
                           interpolation='nearest')
            # B_hat horizontal projection at this panel's altitude
            _bx, _by = Bq[Z_SLICES[c % nz]]
            # smaller `scale` = LONGER arrows (it is units-per-width, not a
            # length multiplier); width/headwidth thicken them.
            ax.quiver(Xq, Yq, _bx, _by, angles='xy', pivot='mid', color='k',
                      alpha=0.5, width=0.005, headwidth=7.0, headlength=8.0,
                      headaxislength=6.5, scale=6.5, scale_units='width')
            ax.set_xticks(xt); ax.set_yticks(yt)
            ax.grid(True, color='0.6', lw=0.4, alpha=0.5)
            ax.tick_params(labelbottom=(r == NT - 1), labelleft=(c == 0),
                           labelsize=7 * FS, length=2)
            ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
            for s in ax.spines.values():
                s.set_edgecolor('0.6'); s.set_linewidth(0.6)
            if r == 0:                                  # column headers
                ax.set_title(f'{int(Z_SLICES[c % nz])} km', fontsize=10 * FS,
                             pad=4)
            if c == 0:                                  # row time labels
                ax.set_ylabel(f't = {int(times[r])} s\ny [km]', fontsize=9 * FS)
            if r == NT - 1:                             # every bottom panel
                ax.set_xlabel('x [km]', fontsize=9 * FS)

    # note: NO subplots_adjust -- rescaling the grid region independently in
    # x/y would break the equal-aspect cell tiling.
    #
    # Anchor the colorbar to the MEASURED right edge of the last panel, not to
    # an estimate: aspect='equal' shrinks each axes inside its GridSpec slot,
    # so the drawn panels do not span the nominal pw*(2*nz) width and any
    # guess either overlaps the 350 km column or leaves a gap.  A draw is
    # needed first to make the renderer positions real.
    figh = fig.get_size_inches()[1]
    fig.canvas.draw()
    right = max(axs[(r, 2 * nz - 1)].get_position().x1 for r in range(NT))
    left = min(axs[(r, 0)].get_position().x0 for r in range(NT))
    bot = min(axs[(NT - 1, c)].get_position().y0 for c in range(2 * nz))
    # horizontal bar under the panels, centred on their measured span and
    # inset so it reads as a legend rather than a seventh column
    cw = 0.46 * (right - left)
    # sit the bar inside the reserved strip: 0.45 in below the panel bottoms,
    # which clears the x tick labels and still leaves room underneath for the
    # bar's own numbers and its axis label.
    cax = fig.add_axes([0.5 * (left + right) - 0.5 * cw,
                        bot - 0.78 / figh, cw, 0.16 / figh])

    # group headers, centred on the MEASURED extent of each 3-column block and
    # placed just above the tallest panel top in row 0
    _top = max(axs[(0, c)].get_position().y1 for c in range(2 * nz))
    for _c0, _c1, _lab in ((0, nz - 1, 'FINITE FAULT'),
                           (nz, 2 * nz - 1, 'POINT SOURCE')):
        _xa = axs[(0, _c0)].get_position().x0
        _xb = axs[(0, _c1)].get_position().x1
        fig.text(0.5 * (_xa + _xb), _top + 0.028, _lab, ha='center',
                 va='bottom', fontsize=13 * FS, fontweight='bold')
    cb = fig.colorbar(im, cax=cax, ticks=[-lim, 0, lim],
                      orientation='horizontal')
    cb.ax.set_xticklabels([f'$-${lim:.1f}', '0', f'$+${lim:.1f}'])
    cb.ax.tick_params(labelsize=9 * FS, length=2, pad=2)
    cb.set_label('$\\delta N_e$   ($10^{8}$ m$^{-3}$)', fontsize=10 * FS,
                 labelpad=4)
    cb.outline.set_linewidth(0.5)

    # (the standalone B sketch is gone -- B_hat is now drawn as a quiver on
    # every panel, which also shows how the field varies across the grid)

    os.makedirs(paths.FIGURES, exist_ok=True)
    dest = out_file or paths.fig(f'dne_slices_grid_fault_vs_point{tag}.png')
    fig.savefig(dest, dpi=150)
    print('saved', dest)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--tag', default='',
                    help='dNe cube tag, i.e. dne_<binned|point><tag>.npz '
                         '(e.g. "_v3"; default: the canonical cubes)')
    ap.add_argument('--times', type=float, nargs='+', default=None,
                    help='row timestamps [s] (default 660 750 840)')
    ap.add_argument('--out', default=None, help='output PNG')
    ap.add_argument('--a0', type=float, nargs='+', default=None,
                    help='source scale(s): one value for both, or TWO in the '
                         'order (binned, point) since each model now has its '
                         'own fitted A0 (e.g. --a0 1 for the raw '
                         'unscaled field)')
    _a = ap.parse_args()
    if _a.a0 is not None and len(_a.a0) not in (1, 2):
        ap.error(f'--a0 takes 1 or 2 values, got {len(_a.a0)}')
    main(tag=_a.tag, times=_a.times, out_file=_a.out, a0=_a.a0)
