"""Arrival time and verticalized amplitude on a geographic map.

One dot per picked arc, drawn at the ionospheric pierce point it had AT THE
MOMENT OF THE ARRIVAL.  Colour is the picked arrival time, size the
peak-to-peak amplitude after removing line-of-sight obliquity.  The USGS
rupture trace and isodistance rings from the epicentre are drawn underneath, so
the figure asks one question: does the disturbance leave the fault as a
circular front from a point, or as an elongated front from a 500 km line?

The IPP moves along its arc, so both the position and the elevation are
interpolated to `t_arr_s` rather than taken as arc means -- the pierce point
travels hundreds of km over an arc and elevation moves by 6 deg over a median
arc (45 deg over the worst), so an arc-average would misplace the dot and
mis-scale it at once.

Verticalization.  A slant measurement through a tilted line of sight crosses
more of the disturbed shell than a vertical one, by a factor that depends only
on elevation under the thin-shell assumption:

    sin(chi) = R_E cos(Ele) / (R_E + h_ipp),   dTEC_vert = dTEC_slant * cos(chi)

with h_ipp = 350 km, the pierce height the arc geometry already assumes.

Read the sizes as first-order only.  The thin-shell factor is derived for
BACKGROUND TEC through a horizontal layer; a CID is a tilted, propagating
wavefront, so cos(chi) removes the bulk geometric obliquity but not the angle
between the wave normal and the line of sight.  It is the conventional
correction, not an exact one, and the residual is largest where the wavefront
is steepest -- close in.

    python plots/plot_arrival_map.py
    python plots/plot_arrival_map.py --no-verticalize
    python plots/plot_arrival_map.py --stations
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from cidmodel import paths
from cidmodel.config import EQ_LAT, EQ_LON, R_EARTH_KM
from plot_geomap import load_fault_trace, load_observed_stations


IPP_H_KM = 350.0          # pierce height the arc geometry assumes

MAP_EXTENT = [87, 110, 8.5, 26]     # lon_min, lon_max, lat_min, lat_max

RING_STEP_KM = 250
RING_MAX_KM  = 1750

# Dot area (pt^2) spans this range over the amplitude range present, so the
# smallest pick stays visible and the largest does not swamp its neighbours.
S_MIN, S_MAX = 12.0, 300.0


def obliquity(ele_deg, h_km=IPP_H_KM):
    """Thin-shell slant -> vertical factor cos(chi) for elevation `ele_deg`."""
    sin_chi = R_EARTH_KM * np.cos(np.radians(ele_deg)) / (R_EARTH_KM + h_km)
    return np.cos(np.arcsin(sin_chi))


def sample_at_pick(obs, picks, cols=('Lat', 'Lon', 'Ele')):
    """Interpolate `cols` to each arc's own pick time.

    The pierce point and the elevation both move through an arc, so the dot's
    position and its obliquity factor belong to the instant of the arrival.
    """
    want = dict(zip(picks['ArcID'], picks['t_arr_s']))
    out = {}
    for arc, g in obs.groupby('ArcID'):
        if arc not in want:
            continue
        g = g.sort_values('t_s')
        ts = g['t_s'].to_numpy(float)
        out[arc] = [float(np.interp(want[arc], ts, g[c].to_numpy(float)))
                    for c in cols]
    return pd.DataFrame.from_dict(out, orient='index', columns=list(cols))


def size_from_amp(a, lo=None, hi=None):
    """Map amplitude to dot AREA, linearly between S_MIN and S_MAX.

    Area-linear-in-amplitude is the usual bubble convention: apparent size then
    scales as sqrt(amplitude), so the largest pick reads as biggest without
    covering its neighbours.  `lo`/`hi` pin the mapping to the data range so
    the size legend is drawn against the same scale as the dots.
    """
    lo = np.nanmin(a) if lo is None else lo
    hi = np.nanmax(a) if hi is None else hi
    frac = (np.asarray(a, float) - lo) / (hi - lo) if hi > lo else np.zeros_like(a)
    return S_MIN + frac * (S_MAX - S_MIN)


def _draw_rings(ax, f_lat, f_lon, proj):
    """Isodistance rings from the epicentre, labelled along their NW flank.

    Great-circle rings, not flat-Earth circles: at 1750 km the two differ by
    enough to matter against the arrival-time colouring the rings exist to be
    read against.
    """
    az = np.radians(np.linspace(0, 360, 721))
    lat0, lon0 = np.radians(EQ_LAT), np.radians(EQ_LON)
    for d in range(RING_STEP_KM, RING_MAX_KM + 1, RING_STEP_KM):
        ang = d / R_EARTH_KM
        lat = np.arcsin(np.sin(lat0) * np.cos(ang) +
                        np.cos(lat0) * np.sin(ang) * np.cos(az))
        lon = lon0 + np.arctan2(np.sin(az) * np.sin(ang) * np.cos(lat0),
                                np.cos(ang) - np.sin(lat0) * np.sin(lat))
        ax.plot(np.degrees(lon), np.degrees(lat), transform=proj,
                color='0.45', lw=0.6, ls=(0, (4, 4)), alpha=0.75, zorder=2)
        # Label on the SW flank, which is empty of data on this event -- but
        # only when that anchor actually falls inside the map.  The outermost
        # rings clip the corner, so their SW point lies off-canvas and
        # matplotlib would park the label in the margin beside the axes.
        i = np.argmin(np.abs(az - np.radians(225)))
        la, lo = np.degrees(lat[i]), np.degrees(lon[i])
        if not (MAP_EXTENT[0] <= lo <= MAP_EXTENT[1]
                and MAP_EXTENT[2] <= la <= MAP_EXTENT[3]):
            continue
        ax.text(lo, la, f'{d}', transform=proj, fontsize=7.5, color='0.35',
                ha='center', va='center', zorder=3,
                bbox=dict(fc='white', ec='none', alpha=0.72, pad=0.9))


def main(out=None, verticalize=True, stations=False):
    picks = pd.read_parquet(paths.ARRIVAL_PICKS)
    obs = pd.read_parquet(paths.TEC_OBS)

    at = sample_at_pick(obs, picks)
    picks = picks.join(at, on='ArcID')

    if verticalize:
        picks['obliq'] = obliquity(picks['Ele'].to_numpy(float))
        picks['amp'] = picks['amp_ptp'] * picks['obliq']
        amp_lab = 'vertical $\\delta$TEC$_{p-p}$ (TECU)'
    else:
        picks['amp'] = picks['amp_ptp']
        amp_lab = 'slant $\\delta$TEC$_{p-p}$ (TECU)'

    d = picks.dropna(subset=['Lat', 'Lon', 't_arr_s', 'amp'])
    amp = d['amp'].to_numpy(float)
    a_lo, a_hi = float(np.nanmin(amp)), float(np.nanmax(amp))

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(11.4, 9.0))
    ax = fig.add_subplot(111, projection=proj)
    ax.set_extent(MAP_EXTENT, crs=proj)

    ax.add_feature(cfeature.LAND, facecolor='#f2efe9', zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='#dce9f2', zorder=0)
    ax.add_feature(cfeature.COASTLINE, lw=0.6, edgecolor='0.35', zorder=1)
    ax.add_feature(cfeature.BORDERS, lw=0.45, edgecolor='0.55',
                   linestyle=':', zorder=1)

    gl = ax.gridlines(draw_labels=True, lw=0.4, color='0.75', alpha=0.5,
                      linestyle=':', zorder=1)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {'size': 9}

    f_lat, f_lon, _slip = load_fault_trace()
    _draw_rings(ax, f_lat, f_lon, proj)

    if stations:
        st = load_observed_stations()
        ax.plot(st['lon'], st['lat'], transform=proj, ls='none', marker='^',
                ms=4.2, color='0.25', mec='white', mew=0.4, alpha=0.75,
                zorder=4, label='GNSS receiver')

    ax.plot(f_lon, f_lat, transform=proj, color='#111111', lw=3.6,
            solid_capstyle='round', zorder=6, label='USGS rupture trace')
    ax.plot([EQ_LON], [EQ_LAT], transform=proj, ls='none', marker='*',
            ms=18, color='#111111', mec='white', mew=1.0, zorder=7,
            label='epicentre')

    sc = ax.scatter(d['Lon'], d['Lat'], transform=proj,
                    c=d['t_arr_s'].to_numpy(float),
                    s=size_from_amp(amp, a_lo, a_hi), cmap='plasma',
                    edgecolor='k', lw=0.4, alpha=0.9, zorder=5)

    ax.set_title('Ionospheric arrivals, Myanmar M7.7 (28 Mar 2025)\n'
                 f'{len(d)} picked arcs at their pierce point '
                 f'({IPP_H_KM:.0f} km) at arrival',
                 fontsize=12.5, pad=12)

    cb = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.78)
    cb.set_label('arrival time after origin (s)', fontsize=10)

    # Size legend: three round values spanning the data, drawn through the same
    # mapping as the dots so the reader can measure rather than guess.
    ticks = [a_lo, 0.5 * (a_lo + a_hi), a_hi]
    sizes = size_from_amp(np.array(ticks), a_lo, a_hi)
    handles = [Line2D([], [], ls='none', marker='o', color='0.55',
                      mec='k', mew=0.4, markersize=np.sqrt(sizes[i]),
                      label=f'{t:.1f}') for i, t in enumerate(ticks)]
    leg = ax.legend(handles=handles, title=amp_lab, loc='lower left',
                    frameon=True, framealpha=0.9, labelspacing=1.4,
                    handletextpad=1.2, fontsize=9, borderpad=0.9)
    leg.get_title().set_fontsize(9)
    leg.set_zorder(20)
    ax.add_artist(leg)
    l2 = ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=9)
    l2.set_zorder(20)

    if verticalize:
        ob = picks['obliq'].to_numpy(float)
        print(f'obliquity cos(chi): min {np.nanmin(ob):.3f}  '
              f'median {np.nanmedian(ob):.3f}  max {np.nanmax(ob):.3f}')
        print(f'amplitude: slant median {picks["amp_ptp"].median():.2f} -> '
              f'vertical median {picks["amp"].median():.2f} TECU')
    print(f'{len(d)} arcs plotted; IPP lat {d["Lat"].min():.1f}-'
          f'{d["Lat"].max():.1f}, lon {d["Lon"].min():.1f}-{d["Lon"].max():.1f}; '
          f'arrival {d["t_arr_s"].min():.0f}-{d["t_arr_s"].max():.0f} s')

    out = out or paths.fig('arrival_map%s.png' % ('' if verticalize else '_slant'))
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print('wrote', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out')
    ap.add_argument('--no-verticalize', action='store_true',
                    help='plot raw slant amplitude instead')
    ap.add_argument('--stations', action='store_true',
                    help='also mark the contributing GNSS receivers')
    a = ap.parse_args()
    main(out=a.out, verticalize=not a.no_verticalize, stations=a.stations)
