"""Travel-time curve of the hand-picked CID arrivals, marker size = period.

Arrival time against the epicentral range of the IPP, for every usable pick in
arrival_picks.parquet.  Marker AREA scales with the picked period (crest-to-trough doubled),
so the dispersion is visible on the same axes as the moveout: if the wave is
dispersive the large markers separate from the small ones rather than mixing.

A least-squares line is drawn (dashed) but should be argued with, not trusted:
the picks do not lie on one straight line -- the curve is concave -- so a single
fit through all of it returns an apparent velocity that belongs to no branch.
Constant-speed reference slopes were drawn alongside it until 2026-09-17 and
have been removed: a physical acoustic-vs-Rayleigh discriminant has to remove
each arc's OWN modelled climb instead of a single constant, or it is biased by
range.

Abscissa (--x):
  epicentre  great-circle range from the hypocentre (default)
  fault      distance to the nearest MOMENT-WEIGHTED binned source.  The rupture
             is ~520 km long, so for a southern IPP the epicentre overstates the
             path by up to ~440 km.  Weighted, because the bare nearest segment
             is the y=-440 km tip -- 0.2% of the moment -- for ~80% of these
             arcs, which would let a patch that radiates almost nothing define
             the reference.

  python plots/plot_travel_time.py
  python plots/plot_travel_time.py --x fault
  python plots/plot_travel_time.py --by-constellation
"""
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from cidmodel import paths
# marker area (pt^2) at the smallest and largest period in the data
S_MIN, S_MAX = 18.0, 320.0


def size_from_period(T, T_lo, T_hi):
    """Marker AREA linear in period, so the visual radius goes as sqrt(T)."""
    f = np.clip((T - T_lo) / max(T_hi - T_lo, 1e-9), 0.0, 1.0)
    return S_MIN + f * (S_MAX - S_MIN)


def main():
    args = sys.argv[1:]
    p = pd.read_parquet(paths.ARRIVAL_PICKS)
    d = p[p['t_arr_s'].notna() & p['dist_km'].notna()].copy()
    d = d[d['period_s'].notna()]
    if not len(d):
        sys.exit('arrival_picks.parquet holds no usable picks with a period')

    xkey = 'dist_km'
    xlabel = 'IPP epicentral range [km]'
    xtag = ''
    if '--x' in args:
        w = args[args.index('--x') + 1].lower()
        if w.startswith('f'):
            xkey = 'src_dist_km'
            xlabel = 'IPP distance to nearest moment-weighted fault segment [km]'
            xtag = '_fault'
        elif w.startswith('c'):
            xkey = 'centroid_dist_km'
            xlabel = ('moment-weighted distance to the nearest 30% of the '
                      'radiating fault [km]')
            xtag = '_centroid'
        if xkey != 'dist_km':
            if xkey not in d.columns:
                sys.exit('arrival_picks.parquet has no %s column' % xkey)
            d = d[d[xkey].notna()]

    # which picked time to plot: onset (default), the positive crest, or the
    # trough.  The crest is the most repeatable marker (an extremum is
    # unambiguous, an onset is a judgement call) but it lags the onset by more
    # at longer range as the N-wave broadens, so it reads a SLOWER apparent
    # velocity -- the difference between the two is itself the broadening.
    tkey, tlab, ttag = 't_arr_s', 'picked arrival', ''
    if '--t' in args:
        w = args[args.index('--t') + 1].lower()
        if w.startswith('max') or w.startswith('c'):
            tkey, tlab, ttag = 't_max_s', 'time of max positive dTEC', '_tmax'
        elif w.startswith('min') or w.startswith('tr'):
            tkey, tlab, ttag = 't_min_s', 'time of min (trough) dTEC', '_tmin'
    d = d[d[tkey].notna()]

    T_lo, T_hi = d['period_s'].min(), d['period_s'].max()
    d['msize'] = size_from_period(d['period_s'].values, T_lo, T_hi)
    x, y = d[xkey].values, d[tkey].values / 60.0

    by_con = '--by-constellation' in args
    fig, ax = plt.subplots(figsize=(13, 9))

    if by_con:
        key, cmap_name = 'constellation', 'tab10'
        groups = sorted(d[key].dropna().unique())
        cols = plt.get_cmap(cmap_name)(np.linspace(0, 1, max(len(groups), 3)))
        cmap = {g: cols[i] for i, g in enumerate(groups)}
        for g in groups:
            s = d[d[key] == g]
            ax.scatter(s[tkey] / 60.0, s[xkey], s=s['msize'],
                       color=cmap[g], alpha=.75, edgecolor='k', lw=.4,
                       label='%s (%d)' % (g, len(s)), zorder=3)
        leg1 = ax.legend(title='constellation', fontsize=9, loc='upper left',
                         framealpha=.92)
    else:
        # one marker shape for every pick, so marker size reads as period only
        sc = ax.scatter(d[tkey] / 60.0, d[xkey], s=d['msize'],
                        c=d['az_deg'], cmap='twilight', vmin=0, vmax=360,
                        marker='o', alpha=.85, edgecolor='k', lw=.4, zorder=3)
        cb = fig.colorbar(sc, ax=ax, pad=.015)
        cb.set_label('IPP azimuth from epicentre [deg]', fontsize=10)
        # No pick-count legend: the arc count is already in the title.
        print(f'  {len(d)} picks')
        leg1 = None
    if leg1 is not None:
        ax.add_artist(leg1)

    # straight-line fit over everything -- shown to be argued with, not trusted
    m = np.isfinite(x) & np.isfinite(y)
    pf = np.polyfit(x[m], y[m] * 60.0, 1)
    r = np.corrcoef(x[m], y[m])[0, 1]
    # The fit is still done as t(r) -- distance is the controlled quantity and
    # time the measured one -- so v, the intercept and r are unchanged by the
    # axis flip; only the drawing is transposed.  With time on x the slope of
    # each line IS the propagation velocity, read straight off the plot.
    xs = np.linspace(x[m].min(), x[m].max(), 50)
    ax.plot(np.polyval(pf, xs) / 60.0, xs, 'k--', lw=1.5, zorder=4,
            label='single-line fit: %.0f m/s (r=%.2f)' % (1000.0 / pf[0], r))

    # t0, the fitted time offset, is kept as a marker on its own: it is where
    # the moveout extrapolates back to zero horizontal range, i.e. the time the
    # disturbance spent climbing to IPP height before covering any ground.
    # Constant-horizontal-speed reference lines were drawn from it until
    # 2026-09-17 and have been removed -- on these axes they invited the picks
    # to be read as a single propagation speed, which the concave curve is not.
    t0_min = pf[1] / 60.0
    ax.axvline(t0_min, color='0.45', lw=1.0, ls='-', alpha=.7, zorder=2)
    # placed low, where the references sweep the corner empty -- the upper-left
    # is taken by the fit legend
    ax.annotate('t$_0$ = %.1f min (fitted):\nclimb to IPP height' % t0_min,
                xy=(t0_min + 0.12, x[m].max() * 0.055), fontsize=8.5,
                color='0.35', va='bottom', ha='left', rotation=90)

    handles = [Line2D([], [], color='k', ls='--', lw=1.5,
                      label='single-line fit: %.0f m/s (r=%.2f)'
                            % (1000.0 / pf[0], r))]
    ax.add_artist(ax.legend(handles=handles, fontsize=9, loc='upper left',
                            framealpha=.92))

    # period -> size key, along the bottom where the data is sparse (the
    # constant-velocity references sweep the lower-left corner empty)
    ticks = [t for t in (150, 300, 500, 700, 900, 1100) if T_lo <= t <= T_hi]
    sh = [Line2D([], [], ls='none', marker='o', markerfacecolor='0.6',
                 markeredgecolor='k', markeredgewidth=.4,
                 markersize=np.sqrt(size_from_period(t, T_lo, T_hi)),
                 label='%d s' % t) for t in ticks]
    ax.add_artist(ax.legend(handles=sh, title='picked period (marker area)',
                            fontsize=9, loc='lower right', ncol=2,
                            framealpha=.92, labelspacing=1.5, borderpad=0.9,
                            handletextpad=1.2, columnspacing=1.4))

    ax.set_ylabel(xlabel, fontsize=12)
    ax.set_xlabel('%s [min after origin]' % tlab, fontsize=12)
    # Start the time axis at 8 min: nothing is picked before ~9 min (onset) or
    # ~10.3 min (crest), so the band below that is empty and only compresses the
    # data.  8 rather than 10 keeps the earliest onsets on-axis, so the same
    # limit works for every timing marker.  The constant-velocity references
    # still emanate from (0, 0) -- they are simply drawn from off-axis, so their
    # slopes are unchanged.
    t_lo = float(args[args.index('--tmin') + 1]) if '--tmin' in args else 8.0
    ax.set_xlim(t_lo, max(y[m].max() * 1.04, 24))
    ax.set_ylim(0, x[m].max() * 1.05)
    ax.grid(alpha=.25)
    ax.set_axisbelow(True)
    ax.set_title('%s vs range  (%d arcs; marker area proportional to period)'
                 % (tlab, len(d)), fontsize=13)

    out = paths.fig('travel_time%s%s%s.png'
                    % (xtag, ttag, '_by_constellation' if by_con else ''))
    fig.savefig(out, dpi=160, bbox_inches='tight')
    print('saved', out)

    # slope in range bins: where does the apparent velocity actually change?
    print('\napparent velocity in range bins:')
    edges = [0, 400, 700, 1000, 1300, 2000]
    for a, b in zip(edges[:-1], edges[1:]):
        s = d[(d[xkey] >= a) & (d[xkey] < b)]
        if len(s) < 5:
            continue
        q = np.polyfit(s[xkey], s[tkey], 1)
        print('  %4d-%4d km  n=%3d   v = %5.0f m/s   median T = %3.0f s'
              % (a, b, len(s), 1000.0 / q[0] if q[0] else np.nan,
                 s['period_s'].median()))

    print('\nperiod vs range (dispersion check):')
    ok = np.isfinite(d[xkey]) & np.isfinite(d['period_s'])
    pc = np.corrcoef(d[xkey][ok], d['period_s'][ok])[0, 1]
    lp = np.polyfit(d[xkey][ok], d['period_s'][ok], 1)
    print('  T = %.3f * r + %.0f s   (r = %+.2f)' % (lp[0], lp[1], pc))


if __name__ == '__main__':
    main()
