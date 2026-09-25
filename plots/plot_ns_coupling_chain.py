"""N-S vertical cut of |v_n|, v_i and dNe -- finite fault beside point source.

Three rows, the coupling chain in order:

    |v_n|        neutral velocity MAGNITUDE from the ray cube (the N-wave
                 carried along k_hat), a signed-free scalar
    v_i . B_hat  the FIELD-PARALLEL part, the only component that couples
    dNe          = -div(Ne0 v_i) integrated in time -- the observable

Columns are the two source models, so a feature can be traced down the chain
within a model and across the pair at the same instant.

WHY THE NORM FOR v_n, BUT A SIGNED SCALAR FOR v_i.  v_n is a vector with no
preferred axis, and its ENU components flip sign either side of the source by
pure convention (a radially outward wave is northward on the north flank and
southward on the south).  The norm is convention-free and shows where the
acoustic energy IS, which is what the comparison needs.  v_i, by contrast, is
parallel to B_hat by construction, so v_i . B_hat is its full signed magnitude
-- the actual coupling coefficient (`alpha` from ion_velocity), whose SIGN
matters because it sets the polarity of the dNe that follows.  Positive is
along B_hat, which at the epicentre points north and DOWNWARD (declination
-1.0 deg, inclination -33.2 deg), so the quiver on that row is B_hat itself.

The dNe polarity flip at hmF2 is imposed by the vertical Ne0 gradient in the
last step (the v . grad Ne0 term of the divergence), not by v_n, which is a
clean bipolar N-wave throughout.

SCALES.  Each source model carries its OWN fitted A0, read from that cube's
sidecar (written there by pipeline/05_synth_stec.py), so the dNe row is drawn at
each model's own calibrated scale and matches the synth_stec columns without
the number being passed in.  --a0 overrides with two values in the order
(binned, point); --a0 1 1 shows the raw unscaled field.

A0 scales ALL THREE rows, not just dNe.  The chain from v_n through v_i to
dNe and on to the synthetic sTEC is linear in the source scale (coupling.py),
so the value fitted against the observed sTEC is the same one that turns v_n
into m s^-1 -- there is no separate velocity calibration.  With the fitted
scale the velocity panels read in m/s and are labelled as such; on an unfitted
cube (A0 = 1) or under --a0 1 1 they fall back to arbitrary units.

v_n and v_i are NOT stored in the dNe cube (only the time-integrated dNe is),
so they are recomputed here from the ray cube, one
neutral_velocity_field_numba call per source model.

    python plots/plot_ns_coupling_chain.py                # Figure 5 (720 s)
    python plots/plot_ns_coupling_chain.py --time 1200
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cidmodel import paths
from cidmodel.geometry import model_grid
from cidmodel.raytracing import RayCube
from cidmodel.coupling import neutral_velocity_field_numba, B_LIN
from cidmodel.ionosphere import ion_velocity
from cidmodel.sources import binned_fault_sources

KINDS = ('binned', 'point')
LABEL = {'binned': 'finite fault', 'point': 'point source'}


def _sources(kind, g):
    if kind == 'point':
        return [dict(x=0.0, y=0.0, t_rupture_s=0.0, weight=1.0)]
    return binned_fault_sources(g, verbose=False)


def _dne_cube(kind, gen):
    """(sidecar, memmap) for one source model's dNe cube."""
    meta = np.load(paths.require_cube(gen, f'dne_{kind}'), allow_pickle=True)
    mm = np.lib.format.open_memmap(
        paths.cube(str(meta['dne_memmap'])), mode='r')
    return meta, mm


def _ray_cube_for(gen, given):
    """Which ray cube to recompute v_n from.

    The dNe sidecars record the ray cube they were integrated against, so the
    generation already determines it: the velocity rows and the dNe row have
    to come from the same physics or the figure is comparing two models.
    `given` overrides (for a deliberate cross-check) but is reported when it
    disagrees, because silently mixing generations is exactly the failure the
    `ray_cube` field was added to prevent.
    """
    declared = set()
    for kind in KINDS:
        meta = np.load(paths.require_cube(gen, f'dne_{kind}'), allow_pickle=True)
        if 'ray_cube' in meta.files:
            declared.add(str(meta['ray_cube']))
    if len(declared) > 1:
        raise SystemExit('the two dNe cubes were built against different ray '
                         'cubes: ' + ', '.join(sorted(declared)))
    if given:
        name = os.path.basename(paths.require_cube(given, 'ray'))
        if declared and name not in declared:
            print(f'  ! --cube {name} but the dNe cubes declare '
                  f'{declared.pop()}: velocities and dNe are from DIFFERENT '
                  f'generations', flush=True)
        return given
    if not declared:
        raise SystemExit('the dNe cubes record no ray_cube; pass --cube')
    return declared.pop()


def main(cube_path, gen, t=720.0, x_km=0.0, a0=None, pct=99.5, out=None):
    g = model_grid()
    cube_path = _ray_cube_for(gen, cube_path)
    cube = RayCube.load(paths.require_cube(cube_path, 'ray'))
    B = np.load(paths.B_FIELD)
    ix = int(np.argmin(np.abs(g.x - x_km)))
    ext = [g.y[0], g.y[-1], g.z[0], g.z[-1]]

    # Figure sized to the DATA aspect: each panel is 2000 km wide by 700 km
    # tall (2.86:1) and drawn aspect='equal', so three stacked rows need only
    # ~5.9 in of panel height against 11.2 in of width.  A taller figure does
    # not stretch the panels -- imshow keeps them true -- it just letterboxes
    # them in whitespace.  The extra inch is the title, tick labels and the
    # column names under the bottom row.
    fig, axes = plt.subplots(3, len(KINDS), figsize=(11.2, 6.9),
                             sharex=True, sharey=True, squeeze=False)

    # dNe shares ONE colour scale across the two models: the whole point of the
    # comparison is which source puts more disturbance where, and a per-panel
    # scale would normalise exactly that difference away.  v_n / v_i carry the
    # same A0 (so they read in m/s) but keep a per-panel colour scale, since
    # their job here is to show the SHAPE of the wave feeding each dNe row.
    dne_panels = {}
    scales = {}
    for kind in KINDS:
        meta, mm = _dne_cube(kind, gen)
        it = int(np.argmin(np.abs(np.asarray(meta['times'], float) - t)))
        # A0 comes from the sidecar, where 05_synth_stec.py put the fitted
        # value -- so this row is drawn at the same scale as the synth_stec
        # columns and the dNe cuts, without the number being retyped here.
        # A cube that has not been fitted still carries 1.0 (the raw field).
        scale = (float(meta['A0']) if 'A0' in meta.files else 1.0) \
            if a0 is None else float(a0[KINDS.index(kind)])
        scales[kind] = scale
        # 1e10, not 1e8: the calibrated peak is a few times 1e11, which reads
        # as a four-digit 2800 in 1e8 units.  In 1e10 the same field is ~280.
        dne_panels[kind] = np.asarray(mm[it][ix, :, :], float) * scale / 1e10
    vis = np.concatenate([d[np.abs(g.y) <= 1000.0].ravel()
                          for d in dne_panels.values()])
    nz = vis[vis != 0]
    dne_lim = np.percentile(np.abs(nz), pct) if nz.size else 1.0

    for c, kind in enumerate(KINDS):
        vn = neutral_velocity_field_numba(g, cube, _sources(kind, g), float(t),
                                          bilinear=False)
        vi, _ = ion_velocity(vn, B)
        Bp = B[ix, :, :, :]
        # A0 scales the velocities into m/s, not just the dNe row: the chain
        # from v_n to sTEC is linear in it (coupling.py), so the same scale
        # that calibrates dNe against the observed sTEC calibrates the
        # velocity that produced it.  A cube that has not been fitted carries
        # A0 = 1 and these stay in the raw units the field is computed in.
        vn_norm = np.linalg.norm(vn[ix], axis=-1) * scales[kind]
        vi_par = np.einsum('yzc,yzc->yz', vi[ix], Bp) * scales[kind]
        d = dne_panels[kind]

        _u = r'  (m s$^{-1}$)' if scales[kind] != 1.0 else '  (arb.)'
        rows = [
            (vn_norm, 'Reds',   False, None,     r'$|v_n|$' + _u),
            (vi_par,  'RdBu_r', True,  None,     r'$v_i\cdot\hat{B}$' + _u),
            (d,       'RdBu_r', False, dne_lim,  r'$\delta N_e$  ($10^{10}$ m$^{-3}$)'),
        ]
        for ax, (v, cmap, quiv, lim, lab) in zip(axes[:, c], rows):
            if lim is None:
                # scale on the VISIBLE strip only -- panels are clipped to
                # +-1000 km, so a percentile over the full +-1400 km row would
                # let off-screen peaks set a limit nothing on screen reaches.
                vv = v[np.abs(g.y) <= 1000.0]
                nzv = vv[vv != 0]
                lim = np.percentile(np.abs(nzv), pct) if nzv.size else 1.0
            vmin = 0.0 if cmap == 'Reds' else -lim
            im = ax.imshow(v.T, origin='lower', extent=ext, aspect='equal',
                           cmap=cmap, vmin=vmin, vmax=lim)
            cb = plt.colorbar(im, ax=ax, pad=0.01, fraction=0.018,
                              shrink=0.70, aspect=20)
            cb.ax.tick_params(labelsize=6.5, length=2, pad=1.5)
            if quiv:
                sy = max(len(g.y) // 7, 1)
                sz = max(len(g.z) // 4, 1)
                Y, Z = np.meshgrid(g.y[::sy], g.z[::sz], indexing='ij')
                ax.quiver(Y, Z, Bp[::sy, ::sz, 1], Bp[::sy, ::sz, 2],
                          angles='xy', pivot='mid', color='k', alpha=.45,
                          width=0.0022, headwidth=9.0, headlength=11.0,
                          headaxislength=9.5, scale=14, scale_units='width')
            ax.set_title(lab, fontsize=9, pad=3)

        # Column label under the bottom panel, below its x-axis label: the
        # two columns are the comparison, so the name sits with the finished
        # chain rather than over the row that starts it.
        axes[-1][c].annotate(
            LABEL[kind], xy=(0.5, 0.0), xytext=(0, -38),
            xycoords='axes fraction', textcoords='offset points',
            ha='center', va='top', fontsize=15, fontweight='bold')
        print(f'{kind:7s} t={t:6.0f}s  '
              f'|vn|max={vn_norm.max():.4g}'
              f'{" m/s" if scales[kind] != 1.0 else " (arb.)"}  '
              f'|vi.B|max={np.abs(vi_par).max():.4g}  '
              f'|dNe|max={np.abs(d).max():.4g}e10', flush=True)

    for ax in axes[:, 0]:
        ax.set_ylabel('altitude (km)')
    axes[0][0].set_xlim(-1000, 1000)
    for ax in axes[-1]:
        ax.set_xlabel('north (km)')

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.06, top=0.91, bottom=0.10)

    fig.suptitle(f'N-S cut at x={g.x[ix]:.0f} km, t={t:.0f} s',
                 fontsize=14, y=0.975)
    # No provenance on the figure itself -- it is for reading, not auditing.
    # The run reports it instead, so what produced a PNG is still recoverable
    # from the terminal that made it.
    _sc = [scales[k] for k in KINDS]
    print(f'  {os.path.basename(paths.require_cube(cube_path, "ray"))}, '
          f'sig = {B_LIN:g}*t_w, '
          + ('A0 1 (raw)' if all(v == 1.0 for v in _sc)
             else 'A0 %g / %g%s' % (_sc[0], _sc[1],
                                    ' (given)' if a0 is not None else '')))
    out = out or paths.fig(
        f'ns_coupling_chain{paths.gen_suffix(gen)}_t{t:.0f}.png')
    fig.savefig(out, dpi=125, bbox_inches='tight')
    print('wrote', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--cube', default=None,
                    help='ray cube to recompute v_n from: a path, a filename '
                         'or a tag.  Optional -- the dNe cubes record which '
                         'ray cube they were built against, and that is used '
                         'by default so both halves of the figure come from '
                         'one generation')
    ap.add_argument('--gen', default='',
                    help='dNe generation tag such as "_v3" (default: the '
                         'canonical cubes)')
    ap.add_argument('--time', type=float, default=720.0,
                    help='frame to draw [s after origin]')
    ap.add_argument('--x', type=float, default=0.0,
                    help='east coordinate of the N-S cut plane [km]')
    ap.add_argument('--a0', type=float, nargs=2, default=None,
                    metavar=('BINNED', 'POINT'),
                    help='the two fitted source scales, in the order '
                         '(binned, point).  Omit for the raw unscaled field')
    ap.add_argument('--pct', type=float, default=99.5)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    main(a.cube, gen=a.gen, t=a.time, x_km=a.x, a0=a.a0, pct=a.pct, out=a.out)
