"""Background ionosphere: IRI-2020 electron density and the IGRF field.

The plasma state the coupling step needs:

    background_ne        Ne0(x,y,z) from IRI-2020 (coarse subgrid + interp)
    profile_summary      NmF2 / hmF2 / vTEC of the epicentre column
    magnetic_field_grid  IGRF unit field B_hat(x,y,z)
    ion_velocity         v_i = (v_n . B_hat) B_hat  (Hooke 1970)

Ne0 and B_hat are expensive and static, so builds cache them to disk
(paths.NE0, paths.B_FIELD) and only recompute on demand.
"""
import math
import warnings

import numpy as np

from .config import EQ_TIME
from .geometry import km_to_geo


# ---------------------------------------------------------------------------
# Background electron density (IRI-2020)
# ---------------------------------------------------------------------------

def _iri_profile(time, lat, lon, alt_km):
    """
    Electron density profile from IRI-2020 at one (lat, lon).

    IRI returns Ne in cm^-3; converted here to m^-3.  IRI's -1 no-data
    sentinel (and any non-positive value) becomes NaN.
    """
    import iri20py

    model = iri20py.Iri2020()
    _, ds = model.evaluate(time, float(lat), float(lon),
                           np.asarray(alt_km, dtype=float))
    ne = np.asarray(ds['Ne'].values, dtype=float).ravel() * 1e6  # -> m^-3
    ne[~np.isfinite(ne) | (ne <= 0)] = np.nan
    return ne


def background_ne(grid, time=EQ_TIME, lat_step_deg=1.0, verbose=True):
    """
    Background electron density Ne0 on the grid, from IRI-2020.

    IRI is expensive per call, so it is evaluated on a coarse (lat, lon)
    subgrid and interpolated horizontally onto the model columns.  Ne0 varies
    smoothly in the horizontal (the sharp structure is vertical), so this is a
    good approximation -- but the equatorial anomaly imposes a real
    latitudinal gradient, so the subgrid must be fine enough to capture it.

    Returns ne0 (nx, ny, nz) in m^-3.
    """
    from scipy.interpolate import RegularGridInterpolator

    lat_c, lon_c = km_to_geo(grid.x, grid.y)
    lats = np.arange(float(np.min(lat_c)), float(np.max(lat_c)) + lat_step_deg,
                     lat_step_deg)
    lons = np.arange(float(np.min(lon_c)), float(np.max(lon_c)) + lat_step_deg,
                     lat_step_deg)

    if verbose:
        print(f"  IRI-2020 on {len(lats)} x {len(lons)} = "
              f"{len(lats)*len(lons)} columns, {len(grid.z)} altitudes ...")

    prof = np.empty((len(lats), len(lons), len(grid.z)), dtype=float)
    for i, la in enumerate(lats):
        for j, lo in enumerate(lons):
            prof[i, j, :] = _iri_profile(time, la, lo, grid.z)

    n_bad = int(np.count_nonzero(~np.isfinite(prof)))
    if n_bad:
        warnings.warn(f"IRI returned {n_bad} non-finite Ne values; "
                      f"filling by nearest-altitude interpolation")
        for i in range(prof.shape[0]):
            for j in range(prof.shape[1]):
                col = prof[i, j]
                m = np.isfinite(col)
                if m.sum() >= 2:
                    col[~m] = np.interp(grid.z[~m], grid.z[m], col[m])
                elif m.sum() == 0:
                    col[:] = 0.0

    interp = RegularGridInterpolator(
        (lats, lons, grid.z), prof, bounds_error=False, fill_value=None)

    XX, YY, ZZ = grid.meshgrid()
    LAT, LON = km_to_geo(XX, YY)
    pts = np.stack([LAT.ravel(), LON.ravel(), ZZ.ravel()], axis=-1)
    ne0 = interp(pts).reshape(grid.shape)
    return np.clip(ne0, 0.0, None)


def profile_summary(ne0, grid):
    """NmF2 / hmF2 / vertical TEC of the column nearest the epicentre."""
    ix = int(np.argmin(np.abs(grid.x)))
    iy = int(np.argmin(np.abs(grid.y)))
    col = ne0[ix, iy, :]
    k = int(np.nanargmax(col))
    tec = float(np.trapezoid(col, grid.z * 1e3) / 1e16)
    return dict(NmF2=float(col[k]), hmF2=float(grid.z[k]), vTEC_TECU=tec)


# ---------------------------------------------------------------------------
# Geomagnetic field and ion coupling
# ---------------------------------------------------------------------------

def magnetic_field_grid(grid, time=EQ_TIME, verbose=True):
    """
    IGRF unit magnetic field vector B_hat (ENU) at every grid node.

    Returns (nx, ny, nz, 3).
    """
    import ppigrf
    import pandas as pd

    nx, ny, nz = grid.shape
    B = np.zeros((nx, ny, nz, 3), dtype=float)
    date = pd.Timestamp(time)

    XX, YY = np.meshgrid(grid.x, grid.y, indexing='ij')
    LAT, LON = km_to_geo(XX, YY)

    for iz, z in enumerate(grid.z):
        Be, Bn, Bu = ppigrf.igrf(LON, LAT, float(z), date)
        Be = np.asarray(Be).reshape(nx, ny)
        Bn = np.asarray(Bn).reshape(nx, ny)
        Bu = np.asarray(Bu).reshape(nx, ny)
        norm = np.sqrt(Be**2 + Bn**2 + Bu**2)
        norm[norm == 0] = 1.0
        B[:, :, iz, 0] = Be / norm
        B[:, :, iz, 1] = Bn / norm
        B[:, :, iz, 2] = Bu / norm

    if verbose:
        ix, iy = nx // 2, ny // 2
        b = B[ix, iy, nz // 2]
        incl = math.degrees(math.atan2(-b[2], math.hypot(b[0], b[1])))
        print(f"  B_hat at grid centre: [{b[0]:+.3f}, {b[1]:+.3f}, {b[2]:+.3f}], "
              f"inclination {incl:.1f} deg")
    return B


def ion_velocity(vn, B_hat):
    """
    Project the neutral velocity onto the field line:
    v_i = (v_n . B_hat) B_hat.

    Mikesell et al. (2019) eq. 4, after Hooke (1970): ions are constrained to
    move along B, so only the field-parallel component of the neutral motion
    couples into the plasma.

    Returns (v_i, alpha): v_i is (nx, ny, nz, 3) and alpha = v_n . B_hat is the
    signed coupling coefficient (nx, ny, nz).
    """
    alpha = np.einsum('xyzc,xyzc->xyz', vn, B_hat)
    vi = alpha[..., None] * B_hat
    return vi, alpha
