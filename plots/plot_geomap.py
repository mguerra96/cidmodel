"""
plot_geomap.py
==============
Standalone geographic overview map for the Myanmar M7.7 earthquake
(28 March 2025, 06:20:56 UTC):

  * USGS ShakeMap PGV contours (filled polygons, cm/s)
  * the USGS finite-fault surface trace, coloured by slip
  * every GNSS receiver contributing an arc in tec_observations.parquet
  * epicentre + isodistance rings from the epicentre

The receiver list comes straight from the observations table, so the map shows
exactly the sites whose arcs enter the analysis (one marker per site, however
many arcs it carries).  All receivers are drawn identically and unlabelled --
which arcs a site ends up contributing is a matter of the satellite geometry on
the day, not a property of the site.

Usage
-----
    python plots/plot_geomap.py
"""

import os
import struct

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from obspy.imaging.beachball import beach
from cidmodel import paths as P_
# ---------------------------------------------------------------------------
# Shapefile reader (pure Python — no geopandas required)
# ---------------------------------------------------------------------------

def _read_dbf(path):
    with open(path, 'rb') as f:
        data = f.read()
    n_records   = struct.unpack_from('<I', data, 4)[0]
    header_size = struct.unpack_from('<H', data, 8)[0]
    record_size = struct.unpack_from('<H', data, 10)[0]
    fields = []
    offset = 32
    while data[offset] != 0x0D:
        name  = data[offset:offset + 11].rstrip(b'\x00').decode()
        ftype = chr(data[offset + 11])
        flen  = data[offset + 16]
        fields.append((name, ftype, flen))
        offset += 32
    col_offsets = [1]
    for f in fields[:-1]:
        col_offsets.append(col_offsets[-1] + f[2])
    records = []
    rec_start = header_size
    for i in range(n_records):
        rec = data[rec_start + i * record_size: rec_start + (i + 1) * record_size]
        row = {}
        for j, (name, ftype, flen) in enumerate(fields):
            raw = rec[col_offsets[j]: col_offsets[j] + flen].strip()
            try:
                row[name] = float(raw) if ftype == 'N' else raw.decode()
            except (ValueError, UnicodeDecodeError):
                row[name] = None
        records.append(row)
    return records


def _read_shp_polygons(path):
    with open(path, 'rb') as f:
        data = f.read()
    file_size = len(data)
    offset    = 100
    polygons  = []
    while offset < file_size - 8:
        offset += 8
        shape_type = struct.unpack_from('<i', data, offset)[0]; offset += 4
        if shape_type == 0:
            polygons.append([]); continue
        offset += 32
        num_parts  = struct.unpack_from('<i', data, offset)[0]; offset += 4
        num_points = struct.unpack_from('<i', data, offset)[0]; offset += 4
        parts = [struct.unpack_from('<i', data, offset + k * 4)[0]
                 for k in range(num_parts)]
        offset += num_parts * 4
        pts = np.frombuffer(data, dtype='<f8', count=num_points * 2, offset=offset)
        xs  = pts[0::2]; ys = pts[1::2]
        offset += num_points * 16
        rings = []
        for k, start in enumerate(parts):
            end = parts[k + 1] if k + 1 < num_parts else num_points
            rings.append((xs[start:end], ys[start:end]))
        polygons.append(rings)
    return polygons


def load_pgv_shapefile(shp_dir):
    base     = os.path.join(shp_dir, 'pgv')
    dbf_recs = _read_dbf(base + '.dbf')
    polys    = _read_shp_polygons(base + '.shp')
    values   = [r['PARAMVALUE'] for r in dbf_recs]
    return polys, values


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EQ_LAT, EQ_LON = 22.001, 95.925

MAP_EXTENT = [86, 112, 8, 30]       # lon_min, lon_max, lat_min, lat_max
PGV_MIN    = 1.0                    # cm/s, lowest contour drawn (all 90 levels)
PGV_CMAP   = 'YlOrRd'

PGV_ALPHA  = 0.55                   # let the basemap read through the PGV field

RING_STEP_KM  = 250                 # isodistance rings from the FAULT TRACE
RING_MAX_KM   = 2000

# Zoom inset (bottom left): fault region, drawn with individual subfaults.
# Anchored flush to the main map's bottom-left corner, so it shares the
# parent's bottom and left spines.
INSET_PAD_DEG = 0.9                 # padding around the fault trace bbox
INSET_W       = 0.2304              # fraction of the map axes width  (0.192 x1.2)
INSET_H       = 0.4464              # fraction of the map axes height (0.372 x1.2)
BEACHBALL_W   = 0.085               # beach-ball diameter, fraction of map width

STATION_MS    = 7                   # receiver triangle marker size (pt)

OUT_FILE = P_.fig('geomap.png')

R_EARTH_KM = 6371.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_fault_trace(fsp_path=P_.FSP):
    """
    Surface trace of the USGS finite-fault model.

    The .fsp lists 200 subfaults per segment as (LAT LON X Y Z SLIP RAKE TRUP
    RISE SF_MOMENT).  Down-dip rows are collapsed onto the surface exactly as
    `sources.usgs_fault_sources` does -- group by the along-strike coordinate
    Y (NS, km) rounded to 1 km, and take the moment-weighted mean position and
    slip of each column.  Returns (lat, lon, slip_m) sorted south -> north.
    """
    rows = []
    with open(fsp_path) as fh:
        for line in fh:
            if line.startswith('%') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 10:
                rows.append([float(v) for v in parts[:10]])
    arr = np.array(rows)
    lat, lon, _, y_ns, _, slip, _, _, _, moment = arr.T

    key = np.round(y_ns, 0)
    _, inverse = np.unique(key, return_inverse=True)

    out = []
    for c in range(int(inverse.max()) + 1):
        m = inverse == c
        w = moment[m]
        if w.sum() <= 0:
            continue
        out.append((np.average(lat[m], weights=w),
                    np.average(lon[m], weights=w),
                    np.average(slip[m], weights=w)))
    out.sort(key=lambda r: r[0])
    a = np.array(out)
    return a[:, 0], a[:, 1], a[:, 2]


def load_mechanism(fsp_path=P_.FSP):
    """
    Focal mechanism (strike, dip, rake) from the .fsp header.

    The header carries a single line "% Mech : STRK = 358 DIP = 82 RAKE = -175"
    -- the USGS preferred nodal plane for the event.  Returns a (3,) tuple.
    """
    with open(fsp_path) as fh:
        for line in fh:
            if line.startswith('%') and 'STRK' in line and 'DIP' in line:
                p = line.replace('=', ' ').split()
                return (float(p[p.index('STRK') + 1]),
                        float(p[p.index('DIP') + 1]),
                        float(p[p.index('RAKE') + 1]))
    raise ValueError(f"no '% Mech :' line found in {fsp_path}")


def load_subfaults(fsp_path=P_.FSP):
    """
    Individual subfault centres from the .fsp, for the zoom inset.

    Returns a record array with lat, lon, z (km depth) and slip (m) for every
    subfault row -- the full 2-D rupture plane, NOT collapsed down-dip the way
    `load_fault_trace` does.  Also returns the along-strike/down-dip cell sizes
    (Dx, Dz) parsed from the header so the patches can be drawn to scale.
    """
    rows = []
    dx = dz = strike = None
    with open(fsp_path) as fh:
        for line in fh:
            if line.startswith('%'):
                if 'Dx' in line and 'Dz' in line and 'Invs' in line:
                    for tok, name in (('Dx', 'dx'), ('Dz', 'dz')):
                        parts = line.split()
                        i = parts.index(tok)
                        val = float(parts[i + 2])
                        if name == 'dx':
                            dx = val
                        else:
                            dz = val
                if strike is None and 'STRK' in line:
                    parts = line.replace('=', ' ').split()
                    strike = float(parts[parts.index('STRK') + 1])
                continue
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 10:
                rows.append([float(v) for v in parts[:10]])

    arr = np.array(rows)
    return dict(lat=arr[:, 0], lon=arr[:, 1], z=arr[:, 4], slip=arr[:, 5],
                dx=dx or 5.0, dz=dz or 4.5, strike=strike or 358.0)


def load_observed_stations(obs_path=P_.TEC_OBS):
    """
    Receivers carrying at least one arc in the observations table.

    `tec_observations.parquet` holds one row per sample and already carries the
    receiver position (`sta_lat`, `sta_lon`), so no station table is needed --
    and no site can appear that contributes nothing.  A handful of receivers
    (e.g. huev) exist in both networks; they are collapsed to one marker at
    the mean position, since the duplicate coordinates agree to <0.01 deg.

    Returns a DataFrame with columns name, lat, lon, n_arcs, sorted by name.
    """
    obs = pd.read_parquet(obs_path)
    obs = obs.assign(name=obs['station'].astype(str).str.strip().str.lower())
    g = obs.groupby('name')
    out = pd.DataFrame({
        'lat': g['sta_lat'].mean(),
        'lon': g['sta_lon'].mean(),
        'n_arcs': g['ArcID'].nunique(),
    }).reset_index()
    return out.sort_values('name').reset_index(drop=True)


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _draw_beachball(fig, ax, mech):
    """
    Focal-mechanism beach ball tucked under the legend (upper right).

    Two obspy gotchas:

      * beach() returns a collection that does NOT honour cartopy's geographic
        transform (it lands off-canvas on a GeoAxes), so it goes on a plain
        matplotlib axes positioned in figure coordinates;
      * passing axes= makes `width` scale by the axes' pixel size and collapse
        to a dot on a small inset -- omit it so width stays in data units.
    """
    fig.draw_without_rendering()
    inv = fig.transFigure.inverted()
    bb = ax.patch.get_window_extent()
    (mx0, my0), (mx1, my1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])

    w = (mx1 - mx0) * BEACHBALL_W
    h = w * fig.get_size_inches()[0] / fig.get_size_inches()[1]   # keep it round

    # Sit directly below the legend box: measure the legend's rendered extent
    # rather than guessing a corner offset, so the two stay stacked whatever
    # the legend ends up sizing to.  The caption below the ball needs room, so
    # the gap is generous.
    leg = ax.get_legend()
    lb = inv.transform(leg.get_window_extent())
    (lx0, ly0), (lx1, ly1) = lb
    bx = 0.5 * (lx0 + lx1) - w / 2               # centred under the legend
    by = ly0 - h - (my1 - my0) * 0.055           # clear of the legend's bottom

    axb = fig.add_axes([bx, by, w, h], zorder=21)
    axb.set_xlim(-1, 1)
    axb.set_ylim(-1, 1)
    axb.set_aspect('equal')
    axb.axis('off')
    axb.add_collection(beach(mech, xy=(0, 0), width=1.9, linewidth=0.9,
                             facecolor='k', bgcolor='white'))
    axb.text(0, -1.15, f'strike {mech[0]:.0f}°  dip {mech[1]:.0f}°  '
                       f'rake {mech[2]:.0f}°',
             ha='center', va='top', fontsize=8, transform=axb.transData)
    return axb


def _snap_inset_to_map(fig, ax, axi):
    """
    Place the inset flush into the map's bottom-left corner.

    Must run AFTER a draw: a cartopy GeoAxes with set_extent shrinks its patch
    to honour the data aspect ratio, so the visible spines sit inside the axes
    rectangle and axes-fraction coordinates do not land on them.  Measure the
    rendered patch and set the inset position in figure coordinates instead.
    """
    fig.draw_without_rendering()
    inv = fig.transFigure.inverted()
    bb = ax.patch.get_window_extent()
    (mx0, my0), (mx1, my1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])

    w = (mx1 - mx0) * INSET_W
    h = (my1 - my0) * INSET_H
    axi.set_position([mx0, my0, w, h])           # left & bottom spines shared

    # Widen the inset's LONGITUDE span so the data aspect matches the box.
    # With 'equal' aspect the patch would otherwise shrink inside the box and
    # its left edge would fall short of the map's (and forcing aspect='auto'
    # visibly stretches the rupture).  Padding lon instead keeps the geography
    # undistorted and lets the patch fill the box exactly.
    fig_w, fig_h = fig.get_size_inches()
    lat0, lat1 = axi.get_ylim()
    want_dlon = (lat1 - lat0) * (w * fig_w) / (h * fig_h)
    lon_c = 0.5 * sum(axi.get_xlim())
    axi.set_xlim(lon_c - want_dlon / 2, lon_c + want_dlon / 2)


def _draw_fault_inset(fig, ax, f_lat, f_lon, sub, proj):
    """
    Bottom-left zoom on the rupture, overlapping the main map.

    Shows every subfault of the USGS model as a small rectangle coloured by its
    own slip -- no PGV here, the inset is about the slip distribution alone.
    Because the fault dips 82 deg, the down-dip rows project almost on top of
    each other in map view, so the rectangles are offset perpendicular to
    strike BY DEPTH and the fan is exaggerated to be legible: this is a
    schematic of the down-dip structure, not a true projection of the plane.
    """
    # extra head/foot room so the in-panel title and colourbar clear the fault
    lat0 = f_lat.min() - INSET_PAD_DEG
    lat1 = f_lat.max() + INSET_PAD_DEG * 2.2
    lon0, lon1 = f_lon.min() - INSET_PAD_DEG, f_lon.max() + INSET_PAD_DEG

    # Bottom-left corner.  Position is set in FIGURE coords from the map's
    # actually-rendered patch, not via inset_axes fractions: a cartopy axes
    # with set_extent shrinks its patch to the data aspect, so axes-fraction
    # 0.0 is NOT where the visible left spine is, and the inset would stop
    # short of it.  Snapped exactly in _snap_inset_to_map after layout.
    axi = fig.add_axes([0.5, 0.1, 0.3, 0.4], projection=proj, zorder=20)
    axi.set_extent([lon0, lon1, lat0, lat1], crs=proj)
    axi.patch.set_alpha(1.0)

    axi.add_feature(cfeature.LAND,      facecolor='#f2efe9', zorder=0)
    axi.add_feature(cfeature.OCEAN,     facecolor='#d9eaf7', zorder=0)
    axi.add_feature(cfeature.COASTLINE, linewidth=0.7, zorder=2)
    axi.add_feature(cfeature.BORDERS,   linewidth=0.5, linestyle=':', zorder=2)

    # (no PGV in the inset -- it is about the slip distribution alone)

    # --- subfault patches -------------------------------------------------
    slip = sub['slip']
    snorm = mcolors.Normalize(0, slip.max())
    scmap = plt.colormaps['magma']

    # local deg-per-km, and the strike / perpendicular unit vectors in degrees
    km_lat = 1.0 / 110.574
    km_lon = 1.0 / (111.320 * np.cos(np.radians(f_lat.mean())))
    th = np.radians(sub['strike'])
    s_lat, s_lon = np.cos(th) * km_lat, np.sin(th) * km_lon   # along strike
    p_lat, p_lon = -np.sin(th) * km_lat, np.cos(th) * km_lon  # perpendicular

    half_l = sub['dx'] / 2.0                     # along-strike half-length, km
    # The true down-dip extent is 22.5 km, which at this zoom is a hairline.
    # Exaggerate the fan so the 5 depth rows are actually legible: give the
    # whole plane ~18% of the inset's E-W span.  This is a schematic of the
    # down-dip structure, hence the "fanned out" note in the docstring.
    zr = np.unique(np.round(sub['z'], 3))
    fan_km = 0.26 * (lon1 - lon0) / km_lon
    width = fan_km / max(len(zr), 1)             # drawn width per depth row, km
    zidx = {z: k for k, z in enumerate(zr)}
    off = np.array([zidx[round(z, 3)] for z in sub['z']]) * width

    sq, sc = [], []
    for k in range(len(slip)):
        cl, cn = sub['lat'][k], sub['lon'][k]
        cl += p_lat * (off[k] - fan_km / 2)      # centre the fan on the trace
        cn += p_lon * (off[k] - fan_km / 2)
        corners = []
        for a, b in ((+1, 0), (+1, +1), (-1, +1), (-1, 0)):
            corners.append((cn + s_lon * a * half_l + p_lon * b * width,
                            cl + s_lat * a * half_l + p_lat * b * width))
        sq.append(MplPolygon(np.array(corners), closed=True))
        sc.append(scmap(snorm(slip[k])))
    axi.add_collection(PatchCollection(sq, facecolors=sc, edgecolors='k',
                                       linewidths=0.15, zorder=6,
                                       transform=proj))

    # surface trace + epicentre for reference
    axi.plot(f_lon, f_lat, '-', color='k', lw=1.2, transform=proj, zorder=7)
    axi.plot(EQ_LON, EQ_LAT, '*', color='white', ms=15, markeredgecolor='k',
             markeredgewidth=1.0, transform=proj, zorder=8)

    # colourbar sits inside the panel on a backing patch -- flush with the
    # figure edge there is no room below it for the label
    axi.add_patch(plt.Rectangle((0.045, 0.035), 0.50, 0.105,
                                transform=axi.transAxes, facecolor='white',
                                edgecolor='0.6', linewidth=0.5, alpha=0.92,
                                zorder=9))
    cs = axi.inset_axes([0.085, 0.095, 0.42, 0.028])
    cb = fig.colorbar(cm.ScalarMappable(cmap=scmap, norm=snorm), cax=cs,
                      orientation='horizontal')
    cb.set_label('slip (m)', fontsize=8, labelpad=1)
    cb.ax.tick_params(labelsize=7, length=2)
    cs.set_zorder(10)

    # title inside the panel -- outside it collides with the station labels
    axi.text(0.5, 0.985, f'rupture detail\n{len(slip)} subfaults\n'
                         f'(width exaggerated)',
             transform=axi.transAxes, ha='center', va='top',
             fontsize=7.5, fontweight='bold', zorder=11, linespacing=1.25,
             bbox=dict(boxstyle='round,pad=0.28', facecolor='white',
                       edgecolor='0.6', linewidth=0.5, alpha=0.92))
    for sp in axi.spines.values():
        sp.set_linewidth(1.4)
        sp.set_edgecolor('0.15')

    # tie the inset back to the region it magnifies
    ax.add_patch(plt.Rectangle((lon0, lat0), lon1 - lon0, lat1 - lat0,
                               transform=proj, facecolor='none',
                               edgecolor='0.15', linewidth=1.4, zorder=9))
    return axi


def plot_geomap(pgv_polys, pgv_vals, f_lat, f_lon, f_slip, sub, mech,
                stations_df, save_path=OUT_FILE):
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(12, 11))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent(MAP_EXTENT, crs=proj)

    # --- basemap ----------------------------------------------------------
    ax.add_feature(cfeature.LAND,      facecolor='#f2efe9', zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor='#d9eaf7', zorder=0)
    ax.add_feature(cfeature.LAKES,     facecolor='#d9eaf7', zorder=1)
    ax.add_feature(cfeature.RIVERS,    edgecolor='#b8d8ee', linewidth=0.4, zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.9, zorder=2)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.6, linestyle=':', zorder=2)

    # --- PGV --------------------------------------------------------------
    norm = mcolors.LogNorm(vmin=PGV_MIN, vmax=max(pgv_vals))
    cmap = plt.colormaps[PGV_CMAP]
    # The ShakeMap contours are NESTED: one entry per 2 cm/s level, each
    # enclosing all the higher ones.  Two things matter here:
    #
    #  * each entry is a MULTI-ring shape -- the low levels carry 700-1500
    #    rings (tiles of the contouring mesh), and rings[0] is a 9-point
    #    fragment.  Drawing only rings[0] (as the PGV>=50 figure could get away
    #    with) throws away essentially the whole footprint, so draw every ring.
    #  * they must be painted weakest-first, since interior holes are not cut
    #    and a low outer contour would otherwise bury the strong core.
    order = np.argsort(pgv_vals)                     # ascending -> strong last
    patches, colors = [], []
    for i in order:
        rings, val = pgv_polys[i], pgv_vals[i]
        if not rings or val < PGV_MIN:
            continue
        c = cmap(norm(val))
        for ox, oy in rings:
            if len(ox) < 3:
                continue
            patches.append(MplPolygon(np.column_stack([ox, oy]), closed=True))
            colors.append(c)
    print(f"  drawing {len(patches)} PGV rings (>= {PGV_MIN:g} cm/s)")
    ax.add_collection(PatchCollection(patches, facecolors=colors,
                                      edgecolors='none', alpha=PGV_ALPHA,
                                      match_original=False,
                                      zorder=3, transform=proj))

    # opaque backing panel so the colourbar never collides with a station
    # (upper left: the bottom-right corner now belongs to the zoom inset)
    ax.add_patch(plt.Rectangle((0.022, 0.885), 0.315, 0.082,
                               transform=ax.transAxes, facecolor='white',
                               edgecolor='0.6', linewidth=0.6, alpha=0.9,
                               zorder=11))

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = ax.inset_axes([0.045, 0.912, 0.26, 0.020])
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_label('PGV (cm s$^{-1}$)', fontsize=10, labelpad=2)
    cbar.ax.tick_params(labelsize=9, length=2)
    cax.set_zorder(12)

    # --- isodistance rings from the FAULT TRACE ---------------------------
    # Shortest distance to the rupture, not to the epicentre: for a 200 km
    # fault the two differ by up to ~100 km near the ends, and the acoustic
    # source is the whole trace.  Densify the trace first so the min-distance
    # is taken against a continuous line rather than 128 discrete nodes.
    t = np.linspace(0, 1, len(f_lat))
    td = np.linspace(0, 1, 600)
    tr_lat = np.interp(td, t, f_lat)
    tr_lon = np.interp(td, t, f_lon)

    glon = np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 400)
    glat = np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 400)
    GLON, GLAT = np.meshgrid(glon, glat)
    DIST = haversine_km(tr_lat[None, None, :], tr_lon[None, None, :],
                        GLAT[:, :, None], GLON[:, :, None]).min(axis=2)
    levels = np.arange(RING_STEP_KM, RING_MAX_KM + 1, RING_STEP_KM)
    cs = ax.contour(GLON, GLAT, DIST, levels=levels, colors='0.35',
                    linewidths=0.7, linestyles='--', alpha=0.7,
                    transform=proj, zorder=4)
    ax.clabel(cs, fmt='%d km', fontsize=9, inline=True, inline_spacing=3)

    # --- fault trace: plain line on the main map (slip lives in the inset) --
    ax.plot(f_lon, f_lat, '-', color='k', lw=2.4,
            transform=proj, zorder=6, solid_capstyle='round')

    # --- GNSS receivers ---------------------------------------------------
    # Every receiver contributing an arc is drawn the same way and left
    # unlabelled: how many arcs a site carries depends on the satellite
    # geometry of the day, not on the site.
    n_off = 0
    for _, row in stations_df.iterrows():
        lat, lon = row['lat'], row['lon']
        if not (MAP_EXTENT[0] <= lon <= MAP_EXTENT[1]
                and MAP_EXTENT[2] <= lat <= MAP_EXTENT[3]):
            n_off += 1
            continue
        ax.plot(lon, lat, '^', color='#1f4e9c', ms=STATION_MS,
                markeredgecolor='k', markeredgewidth=0.7,
                transform=proj, zorder=8)
    if n_off:
        print(f"  note: {n_off} receiver(s) fall outside MAP_EXTENT "
              f"and are not drawn")

    # --- epicentre --------------------------------------------------------
    ax.plot(EQ_LON, EQ_LAT, '*', color='white', ms=22,
            markeredgecolor='k', markeredgewidth=1.3,
            transform=proj, zorder=10)

    # --- zoom inset: the rupture, drawn subfault by subfault ---------------
    axi = _draw_fault_inset(fig, ax, f_lat, f_lon, sub, proj)

    # --- legend -----------------------------------------------------------
    handles = [
        Line2D([], [], marker='*', color='none', markerfacecolor='white',
               markeredgecolor='k', markersize=17, label='epicentre'),
        Line2D([], [], color='k', lw=2.4,
               label='finite-fault surface trace'),
        Line2D([], [], marker='^', color='none', markerfacecolor='#1f4e9c',
               markeredgecolor='k', markersize=STATION_MS,
               label='GNSS receiver'),
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=10,
              framealpha=0.92).set_zorder(12)

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                      alpha=0.5, linestyle=':')
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {'size': 11}

    ax.set_title('Myanmar M7.7, 28 March 2025 06:20:56 UTC\n'
                 'USGS ShakeMap PGV, finite-fault trace and GNSS receivers',
                 fontsize=14, fontweight='bold')

    # Snap the inset to the map's rendered patch so its right/bottom edges sit
    # exactly on the map's left/bottom spines (see _draw_fault_inset).
    _snap_inset_to_map(fig, ax, axi)

    # beach ball goes in after layout settles, for the same reason as the inset
    _draw_beachball(fig, ax, mech)

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("Loading PGV shapefile...")
    pgv_polys, pgv_vals = load_pgv_shapefile(P_.SHAKEMAPS)
    print(f"  {len(pgv_polys)} polygons, PGV {min(pgv_vals):.1f}-{max(pgv_vals):.0f} cm/s")

    print("Loading finite-fault model...")
    f_lat, f_lon, f_slip = load_fault_trace()
    print(f"  {len(f_lat)} surface elements, "
          f"{f_lat.min():.2f}-{f_lat.max():.2f} N, slip max {f_slip.max():.1f} m")
    sub = load_subfaults()
    print(f"  {len(sub['slip'])} subfaults, Dx={sub['dx']} Dz={sub['dz']} "
          f"strike={sub['strike']}, depth {sub['z'].min():.1f}-{sub['z'].max():.1f} km")
    mech = load_mechanism()
    print(f"  mechanism: strike {mech[0]:.0f} dip {mech[1]:.0f} rake {mech[2]:.0f}")

    print("Loading stations...")
    stations_df = load_observed_stations()
    print(f"  {len(stations_df)} receivers carrying an arc in "
          f"{os.path.basename(P_.TEC_OBS)}, "
          f"{int(stations_df['n_arcs'].sum())} arcs total")

    print("Plotting...")
    plot_geomap(pgv_polys, pgv_vals, f_lat, f_lon, f_slip, sub, mech,
                stations_df)
