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

pipeline/05_synth_stec.py drives this over every arc.
"""
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
# Savitzky-Golay detrend window for observed dTEC.  WIDE on purpose: a narrow
# (~600 s) window eats ~45% of the CID and manufactures spurious side-lobes,
# and the same window must be used on obs AND any model curve compared to it.
DETREND_WIN_S = 1500.0


class _ScaledCube:
    """Lazy (m, nx, ny, nz) view that multiplies frames by `a0` on access.

    Wraps either an in-.npz `dne` array or a disk-backed memmap.  Indexing is
    passed straight through, so `cube[k, ix, iy, iz]` reads only those points
    from disk instead of materialising the frame (22 MB on the model grid).

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


def open_dne(path):
    """(sidecar, frames, A0) of a streamed dNe cube.

    The .npz sidecar names its frames file in `dne_memmap`; it is resolved next
    to the sidecar, so a cube moved or passed by path reads its OWN frames.
    `frames` is a read-only memmap.  A0 is the fitted source scale that
    05_synth_stec.py writes into the sidecar (1.0 until the fit has run).
    """
    meta = np.load(path)
    mm = str(meta['dne_memmap'])
    if not os.path.isabs(mm):
        mm = os.path.join(os.path.dirname(os.path.abspath(path)), mm)
    a0 = float(meta['A0']) if 'A0' in meta.files else 1.0
    return meta, np.load(mm, mmap_mode='r'), a0


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
    dne = d['dne'] if 'dne' in d.files else open_dne(path)[1]
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


def sample_los(cube_dne, k, iz, corners):
    """dNe along the LOS in frame `k`, via the precomputed weights.

    Gathers only the 4 x n_z corner points of the frame, not the whole frame.
    """
    val = np.zeros(len(iz))
    for ix, iy, w in corners:
        val += cube_dne[k, ix, iy, iz] * w
    return val


def stec_series_tvar(cube_dne, x, y, z, times, ts):
    """
    sTEC(t) with TIME-VARYING geometry: each cube frame uses the arc's own
    interpolated (lat, lon, az, el, ipp_h) at that frame's time.

    `ts` is the arc's geometry time series (from _read_arc_timeseries).  The
    LOS is rebuilt per frame -- a bit slower than a frozen path, but the
    geometry drift (up to ~150 km IPP motion / ~8 deg elevation) is too large
    to ignore for a fair obs match.
    """
    lat_t, lon_t, az_t, el_t, ipph_t = _interp_geom(ts, times)
    stec_t = np.empty(len(times))
    for k in range(len(times)):
        pts, ds = los_geometry(lat_t[k], lon_t[k], az_t[k], el_t[k],
                               z, ipph_t[k])
        iz, corners = _bilinear_weights(x, y, pts)
        stec_t[k] = np.sum(sample_los(cube_dne, k, iz, corners) * ds) / 1e16
    return stec_t


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
