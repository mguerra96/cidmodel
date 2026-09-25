"""Synthetic slant-TEC by integrating a dNe cube along a GNSS line of sight.

The observable is the LOS integral of the electron-density perturbation:

    sTEC(t) = integral  dNe(r, t) ds     along the receiver -> satellite ray

Integrating straight from the 4-D dNe cube is the ground truth: consistent with
the vertical-plane dNe figures and independent of the earlier analytic
Bagiya-2023 surrogate (whose SGF = |k x los| discarded the LOS sign and could
not reproduce polarity -- the reason the 3-D cube model replaced it).

Geometry
--------
Straight LOS in the local flat-Earth frame, sampled at the cube's vertical
spacing.  The LOS is pinned at the arc's IPP position (lat/lon) at its IPP shell
height (z_ref = ipp_height_km from the observations table, NOT the F-peak);
horizontal offset grows as (z - z_ref)/tan(el).  Satellite az/el and the IPP
position/height are TIME-VARYING -- interpolated per cube frame from the arc's
time series in tec_observations.parquet, so
the model LOS follows the same moving geometry as the observation (drift is up
to ~150 km IPP motion / ~8 deg elevation over the 35 min window).

CLI
---
  # one arc, full time series + altitude decomposition + figure
  python -m cidmodel.stec_los --arc BNEU_C05_1000 --cube dne_binned.npz

  # scan every arc for the >400 km contribution share
  python -m cidmodel.stec_los --cube dne_binned.npz --scan-all
"""
import argparse
import math
import os

import numpy as np
# pandas is imported lazily, inside the table readers: a cold `import pandas`
# can stall for minutes on some machines (AV scanning its extension modules),
# and loading a cube does not need it.

from .geometry import geo_to_km
from . import paths
# Arc geometry and observed GFLC, one row per sample.
GEOM_COLS = ['t_s', 'Lat', 'Lon', 'Azi', 'Ele', 'ipp_height_km']
DETREND_WIN_S = 1500.0         # SG highpass window; WIDE on purpose (see below)


class _ScaledCube:
    """Lazy (m, nx, ny, nz) view that multiplies frames by `a0` on access.

    Wraps either an in-.npz `dne` array or a disk-backed memmap.  Downstream
    code only indexes single frames (cube[k]) or the leading length, so we
    avoid materialising the whole a0*dne array (2.3 GB at hi-res).

    a0 defaults to 1.0 -- the cubes hold UNSCALED physics and the source scale
    is a fitted result applied downstream.  The whole chain is linear in it
    (verified to 3.5e-08 relative), so scaling here and scaling the finished
    sTEC column are identical.
    """
    def __init__(self, dne, a0=1.0):
        self._dne = dne
        self._a0 = a0

    def __getitem__(self, k):
        return np.asarray(self._dne[k], dtype=np.float32) * self._a0

    def __len__(self):
        return self._dne.shape[0]

    @property
    def shape(self):
        return self._dne.shape


def load_cube(path, a0=1.0):
    """Return (dne * a0, x, y, z, times) from a dNe cube .npz.

    Two on-disk forms are supported:
      * standard: the `dne` array lives inside the .npz.
      * streamed: the .npz is a metadata sidecar carrying `dne_memmap` (the
        .npy filename); the heavy dne is a disk memmap loaded lazily.

    `a0` is the SOURCE SCALE and defaults to 1.0 (unscaled).  Cubes built after
    2026-09-18 carry no A0 at all -- it is a fitted result, derived by comparing
    this very output against the observations, so it cannot be known at build
    time.  Older cubes stored one; it is ignored unless the caller asks for it
    by passing a0 explicitly.
    """
    d = np.load(path)
    if 'dne' in d.files:
        dne = d['dne']
    else:
        mm = str(d['dne_memmap'])
        mm_path = mm if os.path.isabs(mm) else os.path.join(
            os.path.dirname(os.path.abspath(path)), mm)
        dne = np.load(mm_path, mmap_mode='r')
    return _ScaledCube(dne, a0), d['x'], d['y'], d['z'], d['times']


def los_geometry(ipp_lat, ipp_lon, az_deg, el_deg, z, z_ref):
    """LOS sample points (n_z, 3) and slant path element ds (n_z,), in m."""
    ipx, ipy = geo_to_km(ipp_lat, ipp_lon)
    az, el = math.radians(az_deg), math.radians(el_deg)
    horiz = (z - z_ref) / math.tan(el)          # km horizontal offset per alt
    xs = ipx + horiz * math.sin(az)
    ys = ipy + horiz * math.cos(az)
    ds = np.gradient(z * 1e3) / math.sin(el)    # vertical -> slant (m)
    return np.stack([xs, ys, z], axis=-1), ds


def _bilinear_weights(x, y, pts):
    """
    Precompute x-y bilinear corner indices and weights for LOS points.

    The LOS is sampled at the cube's own z nodes, so z needs no interpolation
    (each point's altitude IS a grid level iz); only the horizontal (x, y)
    position is interpolated.  Returns index/weight arrays that gather a whole
    (nt,) time series in ONE vectorised op per corner, instead of rebuilding a
    RegularGridInterpolator per timestep (~100x faster).  Points outside the
    horizontal grid get zero weight.
    """
    xs, ys = pts[:, 0], pts[:, 1]
    iz = np.arange(pts.shape[0])                 # LOS point k <-> grid level k
    dx = x[1] - x[0]; dy = y[1] - y[0]
    fx = (xs - x[0]) / dx; fy = (ys - y[0]) / dy
    ix0 = np.floor(fx).astype(int); iy0 = np.floor(fy).astype(int)
    wx = fx - ix0; wy = fy - iy0
    inside = (ix0 >= 0) & (ix0 + 1 < len(x)) & (iy0 >= 0) & (iy0 + 1 < len(y))
    ix0 = np.clip(ix0, 0, len(x) - 2); iy0 = np.clip(iy0, 0, len(y) - 2)
    w = np.where(inside, 1.0, 0.0)
    corners = [(ix0,     iy0,     (1 - wx) * (1 - wy) * w),
               (ix0 + 1, iy0,     wx * (1 - wy) * w),
               (ix0,     iy0 + 1, (1 - wx) * wy * w),
               (ix0 + 1, iy0 + 1, wx * wy * w)]
    return iz, corners


def sample_los(cube_dne_frame, iz, corners):
    """dNe along the LOS for one time frame, via the precomputed weights."""
    val = np.zeros(len(iz))
    for ix, iy, w in corners:
        val += cube_dne_frame[ix, iy, iz] * w
    return val


def stec_series(cube_dne, x, y, z, times, ipp_lat, ipp_lon, az_deg, el_deg, z_ref):
    """
    sTEC(t) in TECU for every cube time by LOS integration, FIXED geometry.

    Vectorised: the x-y bilinear weights are built once, then every timestep is
    a cheap gather.  Returns (stec_t, pts, ds, (iz, corners)) so callers can
    reuse the geometry for an altitude-band decomposition.  Used for the
    single-frame diagnostic panels; the arc scan uses stec_series_tvar.
    """
    pts, ds = los_geometry(ipp_lat, ipp_lon, az_deg, el_deg, z, z_ref)
    iz, corners = _bilinear_weights(x, y, pts)
    stec_t = np.array([np.sum(sample_los(cube_dne[k], iz, corners) * ds) / 1e16
                       for k in range(len(times))])
    return stec_t, pts, ds, (iz, corners)


def stec_series_tvar(cube_dne, x, y, z, times, ts):
    """
    sTEC(t) with TIME-VARYING geometry: each cube frame uses the arc's own
    interpolated (lat, lon, az, el, ipp_h) at that frame's time.

    `ts` is the arc's geometry time series (from _read_arc_timeseries).  The
    LOS is rebuilt per frame -- a bit slower than the frozen path, but the
    geometry drift (up to ~150 km IPP motion / ~8 deg elevation) is too large
    to ignore for a fair obs match.
    """
    lat_t, lon_t, az_t, el_t, ipph_t = _interp_geom(ts, times)
    stec_t = np.empty(len(times))
    for k in range(len(times)):
        pts, ds = los_geometry(lat_t[k], lon_t[k], az_t[k], el_t[k],
                               z, ipph_t[k])
        iz, corners = _bilinear_weights(x, y, pts)
        stec_t[k] = np.sum(sample_los(cube_dne[k], iz, corners) * ds) / 1e16
    return stec_t


def altitude_bands(cube_dne_frame, z, ds, geom,
                   bands=((100, 300), (300, 350), (350, 400),
                          (400, 450), (450, 500))):
    """Signed sTEC (TECU) and |integrand| share (%) per altitude band."""
    iz, corners = geom
    integ = sample_los(cube_dne_frame, iz, corners) * ds / 1e16
    absum = np.abs(integ).sum()
    out = []
    for a, b in bands:
        m = (z >= a) & (z < b)
        out.append((a, b, float(integ[m].sum()),
                    100 * np.abs(integ[m]).sum() / absum if absum else float('nan')))
    return integ, out


def _read_arc_timeseries(arc_ids=None):
    """
    Per-arc TIME SERIES of (t, lat, lon, az, el, ipp_h) from the observations
    table (paths.TEC_OBS).

    The GNSS satellite moves during the ~35 min CID window, so its az/el and
    the IPP (lat/lon and shell height) drift.  The observed dTEC is measured
    along the true moving LOS, so the model LOS must follow the SAME time-
    varying geometry (interpolated to the cube frame times).

    Returns {ArcID: dict(t, lat, lon, az, el, ipp_h)}, each sorted by t
    (seconds after EQ).
    """
    import pandas as pd

    obs = pd.read_parquet(paths.TEC_OBS, columns=['ArcID'] + GEOM_COLS)
    if arc_ids is not None:
        obs = obs[obs['ArcID'].isin(set(arc_ids))]
    obs = obs.dropna(subset=GEOM_COLS)
    out = {}
    for aid, g in obs.groupby('ArcID', sort=False):
        arr = np.array(sorted(g[GEOM_COLS].itertuples(index=False, name=None)))
        out[aid] = dict(t=arr[:, 0], lat=arr[:, 1], lon=arr[:, 2],
                        az=arr[:, 3], el=arr[:, 4], ipp_h=arr[:, 5])
    return out


def _interp_geom(ts, target_t):
    """Interpolate an arc's geometry time series onto `target_t` (cube frames).

    Linear in lat/lon/el/ipp_h; azimuth via its sin/cos so the 0/360 deg wrap
    is handled (a naive linear interp jumps ~360 across the seam).  Times
    outside coverage clamp to the ends (np.interp default) -- fine, those
    frames are pre/post the CID and contribute ~0.
    """
    t = ts['t']
    lat = np.interp(target_t, t, ts['lat'])
    lon = np.interp(target_t, t, ts['lon'])
    el = np.interp(target_t, t, ts['el'])
    ipp_h = np.interp(target_t, t, ts['ipp_h'])
    araz = np.radians(ts['az'])
    az = np.degrees(np.arctan2(np.interp(target_t, t, np.sin(araz)),
                               np.interp(target_t, t, np.cos(araz)))) % 360.0
    return lat, lon, az, el, ipp_h


def _read_arc_geometry(arc_ids=None):
    """Per-arc MEDIAN geometry (lat, lon, az, el, ipp_h) -- for the single-
    frame diagnostic panels and the grid-bounds check.  The arc SCAN uses the
    time-resolved path (_read_arc_timeseries)."""
    out = {}
    for aid, ts in _read_arc_timeseries(arc_ids).items():
        araz = np.radians(ts['az'])
        az = np.degrees(np.arctan2(np.median(np.sin(araz)),
                                   np.median(np.cos(araz)))) % 360.0
        out[aid] = dict(lat=float(np.median(ts['lat'])),
                        lon=float(np.median(ts['lon'])),
                        az=float(az), el=float(np.median(ts['el'])),
                        ipp_h=float(np.median(ts['ipp_h'])))
    return out


def _arc_row(arc_id):
    geo = _read_arc_geometry([arc_id])
    if arc_id not in geo:
        raise SystemExit(f"arc {arc_id} not found in {paths.TEC_OBS}")
    return geo[arc_id]


def observed_gflc(arc_id):
    """
    Raw observed slant TEC (GFLC) for an arc: (t_after_eq_s, gflc_TECU).

    The measured sTEC as stored in the observations table, WITHOUT detrending
    -- it carries the slow ionospheric background (tens of TECU) as well as the
    CID wiggle (~1 TECU).  Empty arrays if the arc is absent.
    """
    import pandas as pd

    obs = pd.read_parquet(paths.TEC_OBS, columns=['ArcID', 't_s', 'GFLC'],
                          filters=[('ArcID', '==', arc_id)])
    obs = obs.dropna(subset=['t_s', 'GFLC'])
    if len(obs) < 2:
        return np.array([]), np.array([])
    t = obs['t_s'].to_numpy(float); g = obs['GFLC'].to_numpy(float)
    order = np.argsort(t)
    return t[order], g[order]


def observed_dtec(arc_id, detrend_win_s=DETREND_WIN_S):
    """
    Observed dTEC: raw GFLC minus a 2nd-order Savitzky-Golay slow trend.

    The detrend window is 1500 s ON PURPOSE.  A narrow (~600 s) window eats
    ~45% of the CID and manufactures spurious side-lobes; the same wide window
    must be used on obs AND any model curve compared to it.

    Returns (t_after_eq_s, dtec_TECU), empty arrays if the arc is absent.
    """
    from scipy.signal import savgol_filter

    t, g = observed_gflc(arc_id)
    if len(g) < 5:
        return np.array([]), np.array([])
    dt_s = np.median(np.diff(t)) or 30.0

    win = int(round(detrend_win_s / dt_s))
    win = win + 1 if win % 2 == 0 else win           # odd
    win = min(win, len(g) if len(g) % 2 else len(g) - 1)
    if win < 5:
        return t, g - np.median(g)
    trend = savgol_filter(g, win, polyorder=2)
    return t, g - trend


def run_one(arc_id, cube_path, make_fig=True, show_obs=True, panel_time_s=None,
            obs_mode='stec'):
    cube_dne, x, y, z, times = load_cube(cube_path)
    a = _arc_row(arc_id)
    z_ref = a['ipp_h']          # pin the LOS at THIS arc's IPP shell height
    stec_t, pts, ds, geom = stec_series(cube_dne, x, y, z, times,
                                        a['lat'], a['lon'], a['az'], a['el'], z_ref)
    # Frame for the dNe-on-LOS / running-integral panels: peak-|sTEC| by
    # default, or a caller-fixed instant (to compare two cubes at the SAME
    # time, since their peaks can differ).
    if panel_time_s is None:
        kpk = int(np.argmax(np.abs(stec_t)))
    else:
        kpk = int(np.argmin(np.abs(times - panel_time_s)))
    integ, bands = altitude_bands(cube_dne[kpk], z, ds, geom)

    kmax = int(np.argmax(np.abs(stec_t)))
    when = 'peak' if panel_time_s is None else 'requested'
    print(f"\n{arc_id}   az={a['az']:.1f} el={a['el']:.1f}   cube={cube_path}")
    print(f"  peak |sTEC| {stec_t[kmax]:+.4f} TECU at t={times[kmax]:.0f}s")
    print(f"  panel frame ({when}): t={times[kpk]:.0f}s, sTEC={stec_t[kpk]:+.4f} TECU")
    tot = stec_t[kpk]
    print(f"  altitude decomposition at t={times[kpk]:.0f}s:")
    for lo, hi, net, share in bands:
        print(f"    {lo:3d}-{hi:3d} km:  net {net:+.4f} TECU   |{share:4.0f}%| of integrand")
    above = sum(net for lo, hi, net, _ in bands if lo >= 400)
    print(f"  >400 km net: {above:+.4f} TECU ({100*above/tot:.0f}% of sTEC)" if tot else "")

    if make_fig:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        ax[0].plot(times / 60, stec_t, 'k-o', ms=3, label='synthetic')
        ax[0].axhline(0, color='gray', lw=.6)
        ax[0].set_xlabel('min after EQ'); ax[0].set_ylabel('synthetic sTEC [TECU]')
        ax[0].set_title(f'{arc_id}  LOS integral of cube')
        # Observed TEC on a twin axis: amplitude scales differ (synthetic
        # carries the arbitrary A0), so the overlay compares SHAPE/TIMING.
        if show_obs:
            if obs_mode == 'stec':
                t_obs, y_obs = observed_gflc(arc_id)
                obs_lbl = 'observed sTEC'
            else:
                t_obs, y_obs = observed_dtec(arc_id)
                obs_lbl = 'observed dTEC'
            if len(t_obs):
                m = (t_obs >= 0) & (t_obs <= times.max())
                axo = ax[0].twinx()
                axo.plot(t_obs[m] / 60, y_obs[m], color='crimson', lw=1.2,
                         label=obs_lbl)
                axo.set_ylabel(obs_lbl + ' [TECU]', color='crimson')
                axo.tick_params(axis='y', labelcolor='crimson')
                l0, la0 = ax[0].get_legend_handles_labels()
                l1, la1 = axo.get_legend_handles_labels()
                ax[0].legend(l0 + l1, la0 + la1, fontsize=8, loc='upper right')
        ax[1].plot(sample_los(cube_dne[kpk], *geom), z, 'b-')
        ax[1].axvline(0, color='gray', lw=.6)
        ax[1].axhline(z_ref, color='k', ls=':'); ax[1].set_xlabel('dNe on LOS [m^-3]')
        ax[1].set_ylabel('alt [km]'); ax[1].set_title(f't={times[kpk]:.0f}s')
        ax[2].plot(np.cumsum(integ), z, 'r-'); ax[2].axvline(0, color='gray', lw=.6)
        ax[2].axhline(z_ref, color='k', ls=':'); ax[2].set_xlabel('running sTEC [TECU]')
        ax[2].set_ylabel('alt [km]'); ax[2].set_title('cumulative bottom->top')
        plt.tight_layout()
        base = os.path.splitext(os.path.basename(cube_path))[0]
        tag = base.replace('dne_', '') or 'cube'
        tstr = '' if panel_time_s is None else f'_t{int(times[kpk]):04d}'
        out = paths.fig(f'los_integral_{arc_id}_{tag}{tstr}.png')
        plt.savefig(out, dpi=110)
        print(f"  saved {out}")
    return times, stec_t


def run_scan(cube_path, arc_ids, save_path=None):
    """
    LOS-integrate every arc in `arc_ids`, print the >400 km share, and
    (optionally) save each arc's synthetic sTEC(t) to `save_path` (.npz).

    Saved file: times (s after EQ) + one array per ArcID (sTEC in TECU, still
    carrying the cube's A0 scale).  Arcs outside the grid or not illuminated
    are saved as all-zeros so the plotting side can label them flat.
    """
    cube_dne, x, y, z, times = load_cube(cube_path)
    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    tseries = _read_arc_timeseries(arc_ids)
    print(f"\nscan {cube_path}:  arc            sTEC     >400km%")
    shares = []
    saved = {}
    for arc in arc_ids:
        if arc not in tseries:
            print(f"  {arc:16s}  (no geometry)")
            saved[arc] = np.zeros_like(times)
            continue
        ts = tseries[arc]
        ipx, ipy = geo_to_km(float(np.median(ts['lat'])),
                             float(np.median(ts['lon'])))
        if not (xmin < ipx < xmax and ymin < ipy < ymax):
            print(f"  {arc:16s}  (IPP outside grid)")
            saved[arc] = np.zeros_like(times)
            continue
        stec_t = stec_series_tvar(cube_dne, x, y, z, times, ts)
        saved[arc] = stec_t
        if np.max(np.abs(stec_t)) < 1e-4:
            print(f"  {arc:16s}  (not illuminated)")
            continue
        kpk = int(np.argmax(np.abs(stec_t)))
        la, lo, azd, eld, ipph = _interp_geom(ts, np.array([times[kpk]]))
        pts, ds = los_geometry(la[0], lo[0], azd[0], eld[0], z, ipph[0])
        geom = _bilinear_weights(x, y, pts)
        integ, _ = altitude_bands(cube_dne[kpk], z, ds, geom)
        share = 100 * np.abs(integ[z >= 400]).sum() / np.abs(integ).sum()
        shares.append(share)
        print(f"  {arc:16s}  {stec_t[kpk]:+.4f}   {share:4.0f}%")
    if shares:
        shares = np.array(shares)
        print(f"\n  arcs: {len(shares)}  median >400km share: {np.median(shares):.0f}%  "
              f"(>50%: {(shares > 50).sum()}/{len(shares)})")
    if save_path:
        np.savez_compressed(save_path, times=times,
                            **{a: saved[a] for a in saved})
        print(f"  saved {len(saved)} arc synthetics -> {save_path}")


def _all_arc_ids():
    """Every ArcID in the observations table, sorted."""
    import pandas as pd
    return sorted(pd.read_parquet(paths.TEC_OBS, columns=['ArcID'])['ArcID']
                  .unique().tolist())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cube', default='',
                    help='dNe cube .npz sidecar: a path, a filename, or a tag '
                         'such as "_v3" (default: the canonical dne_binned.npz)')
    ap.add_argument('--arc', default=None, help='single ArcID, e.g. BNEU_C05_1000')
    ap.add_argument('--scan-all', action='store_true',
                    help='scan EVERY arc in the observations table for the '
                         '>400 km share')
    ap.add_argument('--no-fig', action='store_true')
    ap.add_argument('--no-obs', action='store_true',
                    help='do not overlay the observed GFLC')
    ap.add_argument('--panel-time', type=float, default=None,
                    help='fix the dNe/running panels to this time (s after EQ); '
                         'default is the peak-|sTEC| frame')
    ap.add_argument('--obs-mode', choices=['stec', 'dtec'], default='stec',
                    help="overlay raw measured sTEC (GFLC, default) or the "
                         "detrended dTEC perturbation")
    ap.add_argument('--save', default=None,
                    help="with --scan-all: save per-arc synthetic sTEC to this .npz")
    args = ap.parse_args()
    args.cube = paths.require_cube(args.cube, 'dne_binned')

    if args.scan_all:
        run_scan(args.cube, _all_arc_ids(), save_path=args.save)
    elif args.arc:
        run_one(args.arc, args.cube, make_fig=not args.no_fig,
                show_obs=not args.no_obs, panel_time_s=args.panel_time,
                obs_mode=args.obs_mode)
    else:
        ap.error('give --arc ARCID or --scan-all')
