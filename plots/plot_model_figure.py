"""
Single composite figure for the paper's model section.

Landscape, two columns:

  LEFT  (full height) : the finite-fault source -- grid-binned sub-sources over
                        a cartopy map of the Sagaing rupture with the USGS
                        surface trace, epicentre and beach ball, plus a narrow
                        rupture space-time panel (rupture time vs latitude)
                        sharing the map's latitude axis.
  RIGHT top           : background state at the epicentre, 0-500 km --
                        (b) HWM14 winds U (E+) and V (N+) in ONE panel;
                        (c) NRLMSISE-00 sound speed c and IRI-2020 Ne in one
                        panel on TWO x axes (c bottom, Ne top).
  RIGHT bottom        : (d) the imposed neutral-velocity N-wave pulse and its
                        broadening, sig_eff = B_LIN * t_w, one panel with the
                        pulse family for several t_w.

The broadening coefficient is imported from cidmodel/coupling.py rather than
restated here, so this panel cannot drift from the law the cubes were built
with.  It previously hardcoded the superseded sqrt "old-age" form
(sigma 16.3, b 2.94, t_w0 631), which PERIOD-1 replaced with the Mikesell
linear law -- the figure went on showing the old one.

  python plot_model_figure.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from obspy.imaging.beachball import beach

from cidmodel import paths
from cidmodel.config import EQ_LAT, EQ_LON, EQ_TIME
from cidmodel.geometry import model_grid, km_to_geo
from cidmodel.sources import binned_fault_sources, n_wave
from cidmodel.atmosphere import build_atmosphere
from cidmodel.coupling import B_LIN
from cidmodel.ionosphere import _iri_profile

OUT_FILE = paths.fig('model_figure.png')

# One knob for every text size in the figure.  The sizes below are the
# original design; FS scales them together, so their RATIOS -- panel letter
# over title over tick label -- survive any change to the overall size.
FS = 1.2


def _f(size):
    """Original design size, scaled by FS."""
    return round(size * FS, 1)



# USGS finite-fault mechanism (.fsp header): near-vertical right-lateral
# strike-slip on the Sagaing Fault.
FOCAL = (358, 82, -175)          # strike, dip, rake (deg)

# background profiles capped at 500 km: above this HWM14 winds flatline into a
# top-boundary continuation and the dNe disturbance is already dead.
ALT_TOP_KM = 500.0

PULSE_TW = (700.0, 1100.0, 1500.0, 1900.0)      # wave-travel times to draw [s]

C_SOUND = '#c0392b'          # warm red   -- sound speed
C_ZONAL = '#1f77b4'          # blue       -- zonal wind U (E+)
C_MERID = '#2ca02c'          # green      -- meridional wind V (N+)
C_NE    = '#7d3c98'          # purple     -- electron density

PANEL_KW = dict(fontsize=_f(13), fontweight='bold', va='top', ha='left')


def _read_fsp():
    """Per-subfault (lat, lon, depth_km, slip_m, trup_s) from the .fsp rows.

    Columns: LAT LON X Y Z SLIP RAKE TRUP RISE SF_MOMENT (TRUP = rupture time).
    """
    lat, lon, dep, slip, trup = [], [], [], [], []
    for line in open(paths.FSP):
        if line.lstrip().startswith('%') or not line.strip():
            continue
        p = line.split()
        if len(p) < 8:
            continue
        try:
            vals = [float(x) for x in p[:8]]
        except ValueError:
            continue
        lat.append(vals[0]); lon.append(vals[1])
        dep.append(vals[4]); slip.append(vals[5]); trup.append(vals[7])
    return (np.array(lat), np.array(lon), np.array(dep),
            np.array(slip), np.array(trup))


# ======================================================================
#  LEFT column: finite-fault source (map + rupture space-time)
# ======================================================================
def _panel_source(fig, gs_cell):
    g = model_grid()
    src = binned_fault_sources(g, verbose=False)
    xs = np.array([s['x'] for s in src])
    ys = np.array([s['y'] for s in src])
    w  = np.array([s['weight'] for s in src])
    trup = np.array([s['t_rupture_s'] for s in src])
    lat, lon = km_to_geo(xs, ys)

    flat, flon, fdep, fslip, ftrup = _read_fsp()      # real fault plane

    # true surface trace = shallowest (top-edge) row of subfaults, sorted N->S
    top = fdep <= (fdep.min() + 0.5)
    to = np.argsort(flat[top])[::-1]
    trace_lat, trace_lon = flat[top][to], flon[top][to]

    proj = ccrs.PlateCarree()
    pad = 2.6
    ext = [lon.min() - pad, lon.max() + pad, lat.min() - pad, lat.max() + pad]
    LAT0, LAT1 = ext[2], ext[3]        # latitude axis shared map <-> space-time

    sub = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_cell,
                                           width_ratios=[3.0, 1.05],
                                           wspace=0.08)

    # ---------------- map ------------------------------------------------
    ax = fig.add_subplot(sub[0, 0], projection=proj)
    ax.set_extent([ext[0], ext[1], LAT0, LAT1], crs=proj)
    # auto aspect so the map's latitude axis fills the cell and coincides with
    # the space-time panel's shared latitude axis (no cartopy equal-aspect pad)
    ax.set_aspect('auto')
    ax.add_feature(cfeature.LAND,      facecolor='#f0ede8', zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor='#d6eaf8', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0, zorder=1)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.7, linestyle=':', zorder=1)
    ax.add_feature(cfeature.RIVERS,    edgecolor='#aed6f1', linewidth=0.4,
                   zorder=1)
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='0.6',
                      alpha=0.5, linestyle=':')
    gl.top_labels = False; gl.right_labels = False
    gl.xlabel_style = {'size': _f(8)}; gl.ylabel_style = {'size': _f(8)}

    ax.plot(trace_lon, trace_lat, '-', color='0.2', lw=1.8, alpha=0.9,
            transform=proj, zorder=3, label='USGS fault trace')

    # uniform small markers; colour (not size) carries the moment weight
    order = np.argsort(w)
    sc = ax.scatter(lon[order], lat[order], c=w[order], s=34, cmap='inferno',
                    edgecolors='k', linewidths=0.4, alpha=0.95,
                    transform=proj, zorder=4)

    ax.plot(EQ_LON, EQ_LAT, marker='*', color='cyan', ms=13,
            markeredgecolor='k', markeredgewidth=0.7, transform=proj,
            zorder=5, label='epicentre')

    # beach ball offset NW into open space, with a leader line.  obspy's
    # beach() collection does not honour cartopy's geographic transform, so it
    # is drawn on a small plain inset axes anchored at (bb_lon, bb_lat) below.
    bb_lon = EQ_LON - 1.3
    bb_lat = EQ_LAT + 1.2
    ax.plot([EQ_LON, bb_lon], [EQ_LAT, bb_lat], color='0.4', lw=0.7,
            transform=proj, zorder=5)

    ax.set_title('Finite-fault source over the Sagaing rupture', fontsize=_f(11))
    ax.legend(loc='upper right', fontsize=_f(8), framealpha=0.9)

    # ---------------- rupture space-time --------------------------------
    # x = rupture time, y = latitude (space along fault, shared with the map);
    # the rupture front sweeps S in space-time -> slope = rupture speed.
    # These are the BINNED sources' moment-weighted mean t_rupture_s, i.e. the
    # timing the forward model actually uses; the raw .fsp subfault cloud is
    # drawn faintly behind as the distribution they condense from.
    axf = fig.add_subplot(sub[0, 1])
    axf.scatter(ftrup, flat, s=4, color='0.72', edgecolors='none', alpha=0.7,
                zorder=1, label='USGS subfaults')
    order_t = np.argsort(w)
    axf.scatter(trup[order_t], lat[order_t], c=w[order_t], s=34, cmap='inferno',
                vmin=w.min(), vmax=w.max(), edgecolors='k', linewidths=0.4,
                alpha=0.95, zorder=4, label='binned sources')
    axf.axhline(EQ_LAT, color='c', lw=1.4, ls='--', zorder=3)
    axf.legend(loc='lower right', fontsize=_f(6), framealpha=0.9,
               handletextpad=0.3, borderpad=0.3)
    axf.set_ylim(LAT0, LAT1)                            # shared latitude axis
    axf.set_xlim(-3, np.nanmax(ftrup) * 1.05)
    axf.set_xlabel('rupture time [s]', fontsize=_f(9))
    axf.tick_params(labelsize=_f(8), labelleft=False)      # lat labels are on map
    axf.yaxis.set_major_locator(plt.MultipleLocator(1.5))
    axf.xaxis.set_major_locator(plt.MultipleLocator(50))
    axf.grid(True, which='major', lw=0.5, alpha=0.5, ls=':', color='0.6')

    return ax, axf, sc, ext, LAT0, LAT1, bb_lon, bb_lat, len(src), len(flat)


def _add_beachball(fig, ax, ext, LAT0, LAT1, bb_lon, bb_lat):
    """Beach ball on a plain inset axes (cartopy-transform-proof)."""
    BW = 1.6                                            # diameter (deg)
    box = ax.get_position()
    x0m, x1m = ext[0], ext[1]
    fx = box.x0 + box.width  * (bb_lon - x0m) / (x1m - x0m)
    fy = box.y0 + box.height * (bb_lat - LAT0) / (LAT1 - LAT0)
    wfrac = box.width  * BW / (x1m - x0m)
    hfrac = box.height * BW / (LAT1 - LAT0)
    axb = fig.add_axes([fx - wfrac / 2, fy - hfrac / 2, wfrac, hfrac], zorder=6)
    axb.set_xlim(-1, 1); axb.set_ylim(-1, 1)
    axb.set_aspect('equal'); axb.axis('off')
    # NB: no axes= arg -> width is in data units (with axes= it scales by the
    # axes' pixel size and collapses to a dot on a small inset).
    axb.add_collection(beach(FOCAL, xy=(0, 0), width=1.9, linewidth=0.9,
                             facecolor='k', bgcolor='white'))


# ======================================================================
#  RIGHT top: background state (winds | sound speed + Ne on twin x)
# ======================================================================
def _panel_background(fig, gs_cell):
    print("Building atmosphere profiles (NRLMSISE-00 + HWM14) to "
          f"{ALT_TOP_KM:.0f} km ...")
    z_atm, c, U, V = build_atmosphere(alt_max=ALT_TOP_KM, alt_step=5.0)[:4]

    print("Querying IRI-2020 Ne profile ...")
    z_ne = np.arange(0.0, ALT_TOP_KM + 1e-6, 5.0)
    ne = _iri_profile(EQ_TIME, EQ_LAT, EQ_LON, z_ne)

    finite = np.isfinite(ne)
    hmf2 = z_ne[finite][np.argmax(ne[finite])] if finite.any() else np.nan

    sub = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_cell,
                                           wspace=0.10)

    # --- winds: U and V in ONE panel -------------------------------------
    ax_w = fig.add_subplot(sub[0, 0])
    ax_w.axvline(0.0, color='gray', lw=0.6, zorder=1)
    ax_w.plot(U, z_atm, color=C_ZONAL, lw=1.8, label='U  zonal (E+)')
    ax_w.plot(V, z_atm, color=C_MERID, lw=1.8, label='V  meridional (N+)')
    ax_w.set_xlabel('horizontal wind  [m s$^{-1}$]', fontsize=_f(9))
    ax_w.set_ylabel('altitude  [km]', fontsize=_f(9))
    # title drawn later in figure coords (see main) -- keeps (b)/(c) aligned
    ax_w.legend(loc='upper left', fontsize=_f(7.5), framealpha=0.9)

    # --- sound speed + Ne on TWO x axes ----------------------------------
    ax_c = fig.add_subplot(sub[0, 1], sharey=ax_w)
    ax_c.plot(c, z_atm, color=C_SOUND, lw=1.8, label='c')
    ax_c.set_xlabel('sound speed  c  [m s$^{-1}$]', fontsize=_f(9), color=C_SOUND)
    ax_c.tick_params(axis='x', colors=C_SOUND, labelsize=_f(8))
    ax_c.spines['bottom'].set_color(C_SOUND)

    ax_ne = ax_c.twiny()
    ax_ne.plot(ne, z_ne, color=C_NE, lw=1.8, label='N$_e$')
    ax_ne.set_xlabel('electron density  N$_e$  [m$^{-3}$]', fontsize=_f(9),
                     color=C_NE)
    ax_ne.tick_params(axis='x', colors=C_NE, labelsize=_f(8))
    ax_ne.spines['top'].set_color(C_NE)
    ax_ne.set_xlim(left=0.0)
    if np.isfinite(hmf2):
        ax_ne.axhline(hmf2, color=C_NE, lw=0.7, ls=':', alpha=0.8)
        ax_ne.annotate(f'hmF2 $\\approx$ {hmf2:.0f} km', xy=(0.97, hmf2),
                       xycoords=('axes fraction', 'data'), ha='right',
                       va='bottom', fontsize=_f(7.5), color=C_NE)
    # title drawn later in figure coords (see main), above the Ne x-label

    for a in (ax_w, ax_c):
        a.set_ylim(0, ALT_TOP_KM)
        a.grid(True, lw=0.4, alpha=0.4, ls=':')
    ax_c.tick_params(axis='y', labelleft=False, labelsize=_f(8))
    ax_w.tick_params(axis='y', labelsize=_f(8))

    return ax_w, ax_c, ax_ne, hmf2


# ======================================================================
#  RIGHT bottom: the imposed N-wave pulse + its broadening with travel time
# ======================================================================
def _panel_pulse(fig, gs_cell):
    ax = fig.add_subplot(gs_cell)

    sigs = [B_LIN * tw for tw in PULSE_TW]
    span = 3.5 * max(sigs)
    tau = np.linspace(-span, span, 2001)
    cmap = plt.get_cmap('viridis')
    colors = [cmap(f) for f in np.linspace(0.05, 0.85, len(PULSE_TW))]

    for tw, sig, col in zip(PULSE_TW, sigs, colors):
        v = n_wave(tau, 0.0, sig)
        ax.plot(tau, v, color=col, lw=1.8,
                label=f'$t_w$={tw:.0f} s   $\\sigma$={sig:.0f} s')
        # mark the compression peak of each curve
        ip = int(np.argmax(v))
        ax.plot(tau[ip], v[ip], 'o', color=col, ms=5, zorder=5)

    # annotate the NARROWEST pulse: compression head, rarefaction tail, and the
    # zero crossing that defines t_w (the pulse centre, travelling at c).
    sig0 = sigs[0]
    v0 = n_wave(tau, 0.0, sig0)
    ax.axvline(0.0, color='0.35', lw=0.9, ls='--', zorder=2)
    ax.axhline(0.0, color='0.6', lw=0.6, zorder=1)
    ax.annotate('compression\nHEAD', xy=(-sig0, v0.max()),
                xytext=(-2.9 * max(sigs), 0.72 * v0.max()),
                fontsize=_f(8), color='#b03a2e', fontweight='bold',
                ha='left', va='center',
                arrowprops=dict(arrowstyle='->', color='#b03a2e', lw=0.9))
    ax.annotate('rarefaction\nTAIL', xy=(sig0, v0.min()),
                xytext=(1.4 * max(sigs), 0.80 * v0.min()),
                fontsize=_f(8), color='#1a5276', fontweight='bold',
                ha='left', va='center',
                arrowprops=dict(arrowstyle='->', color='#1a5276', lw=0.9))
    ax.annotate('zero crossing = $t_w$\n(centre, travels at $c$)',
                xy=(0.0, 0.10 * v0.max()),
                xytext=(0.20 * max(sigs), 0.52 * v0.max()),
                fontsize=_f(8), color='0.25', ha='left', va='center',
                arrowprops=dict(arrowstyle='->', color='0.35', lw=0.9))

    ax.set_xlabel('$\\tau = t - (t_{rup} + t_w)$   [s]', fontsize=_f(9))
    ax.set_ylabel('$v_n$ pulse  (arb.)', fontsize=_f(9))
    ax.set_title('Imposed neutral-velocity N-wave: pulse broadening'
                 f'   ($\\sigma = {B_LIN:g}\\,t_w$)',
                 fontsize=_f(10))
    ax.set_xlim(-span, span)
    ax.legend(loc='upper right', fontsize=_f(7.5), framealpha=0.9)
    ax.tick_params(labelsize=_f(8))
    ax.grid(True, lw=0.4, alpha=0.35, ls=':')
    return ax


def main():
    fig = plt.figure(figsize=(16.0, 9.0))
    gs = gridspec.GridSpec(2, 2, width_ratios=[0.92, 1.0],
                           height_ratios=[1.0, 0.82],
                           wspace=0.18, hspace=0.42,
                           left=0.045, right=0.985, top=0.915, bottom=0.10,   # top: no suptitle, so only the (b)/(c)
                           figure=fig)
    gs_left = gs[:, 0]                                  # map spans both rows

    ax_map, ax_st, sc, ext, LAT0, LAT1, bb_lon, bb_lat, n_src, n_sub = \
        _panel_source(fig, gs_left)
    ax_w, ax_c, ax_ne, hmf2 = _panel_background(fig, gs[0, 1])
    ax_p = _panel_pulse(fig, gs[1, 1])

    # No overall title: the event and epicentre belong in the caption, and
    # the panels carry their own headings.  The run still reports them, so
    # the figure subject stays recoverable from the terminal that made it.
    print(f'  {EQ_TIME:%Y-%m-%d %H:%M} UT, epicentre '
          f'{EQ_LAT:.2f}N {EQ_LON:.2f}E')

    # let layout settle, then (a) align the space-time latitude axis exactly
    # with the map's vertical extent and (b) place the beach ball inset.
    fig.canvas.draw()
    mp = ax_map.get_position()
    fp = ax_st.get_position()
    ax_st.set_position([fp.x0, mp.y0, fp.width, mp.height])
    _add_beachball(fig, ax_map, ext, LAT0, LAT1, bb_lon, bb_lat)

    # colorbar in its own axes under the map (does not resize the map box)
    mb = ax_map.get_position()
    cax = fig.add_axes([mb.x0 + 0.16 * mb.width, mb.y0 - 0.052,
                        0.68 * mb.width, 0.016])
    cb = fig.colorbar(sc, cax=cax, orientation='horizontal')
    cb.set_label('binned moment weight', fontsize=_f(9))
    cb.ax.tick_params(labelsize=_f(8))
    cb.outline.set_linewidth(0.5)

    # (b) and (c) titles in figure coords, at a common height above their boxes
    bw, bc = ax_w.get_position(), ax_c.get_position()
    y_ttl = max(bw.y1, bc.y1) + 0.048
    fig.text(bw.x0 + bw.width / 2, y_ttl, 'Neutral wind (HWM14)',
             ha='center', va='bottom', fontsize=_f(10))
    # (c)'s title is longer than its panel is wide, and its panel is the
    # rightmost, so centring it on the axes pushes it off the page at FS > 1.
    # Right-align it to the panel's right edge instead: it then grows leftward
    # into the gap between (b) and (c) rather than past the figure border.
    fig.text(bc.x1, y_ttl,
             'Sound speed (NRLMSISE-00)  &  N$_e$ (IRI-2020)',
             ha='right', va='bottom', fontsize=_f(10))

    # panel letters, in figure coords so the cartopy axes cannot shift them
    for ax, letter in ((ax_map, '(a)'), (ax_w, '(b)'), (ax_c, '(c)'),
                       (ax_p, '(d)')):
        box = ax.get_position()
        fig.text(box.x0 - 0.012, box.y1 + 0.028, letter, **PANEL_KW)

    os.makedirs(paths.FIGURES, exist_ok=True)
    fig.savefig(OUT_FILE, dpi=170)      # no bbox='tight' (would undo align)
    print('saved', OUT_FILE, ' n binned:', n_src, ' n subfaults:', n_sub,
          f' hmF2 {hmf2:.0f} km')


if __name__ == '__main__':
    main()
