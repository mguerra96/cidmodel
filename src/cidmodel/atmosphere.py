"""Neutral atmosphere: sound speed, HWM14 winds, and acoustic absorption.

Everything the ray tracer and the amplitude model need about the background
neutral atmosphere, from NRLMSISE-00 (density, temperature, composition) and
HWM14 (horizontal winds):

    sound_speed          c(z) from composition-weighted mean molecular mass
    query_hwm14_profile  zonal/meridional winds via the HWM14 conda env
    cached_wind_profile  those winds from data/inputs/, re-querying on a miss
    build_atmosphere     gridded c, U, V + interpolators for the ray ODEs
    absorption_profile   viscous+thermal alpha(z) and density rho(z)

The absorption/amplitude pair follows Mikesell et al. (2019), appendices A3/C.
"""
import os
import math
import subprocess

import numpy as np
from nrlmsise00 import msise_flat
from scipy.interpolate import interp1d

from . import paths
from .config import (EQ_LAT, EQ_LON, EQ_TIME, GAMMA, R_GAS, F107, AP, MOL_MASS,
                    HWM14_PY, RAY_ATM_MAX_KM, RAY_ATM_STEP_KM,
                    WIND_CACHE)


# ---------------------------------------------------------------------------
# Sound speed and winds
# ---------------------------------------------------------------------------

def sound_speed(alt_km, lat=EQ_LAT, lon=EQ_LON):
    """
    Acoustic sound speed at altitude `alt_km` (km) from NRLMSISE-00.

    c = sqrt(gamma R T / M), with M the composition-weighted mean molecular
    mass.  Returns m/s.
    """
    o = msise_flat(EQ_TIME, float(alt_km), lat, lon, F107, F107, AP)
    T = o[10]                                       # temperature (K)
    n = np.array([o[0], o[1], o[2], o[3], o[4], 0, o[6], o[7]])  # number dens.
    n_tot = n.sum()
    M = (MOL_MASS * n).sum() / n_tot if n_tot > 0 else 0.029
    return np.sqrt(GAMMA * R_GAS * T / M)


def query_hwm14_profile(alts, lat=EQ_LAT, lon=EQ_LON,
                        time=None, ap=AP):
    """
    Batch-query the HWM14 horizontal wind model.  NEEDS THE CONDA ENV.

    HWM14 has no usable pip distribution (see config.HWM14_PY), so it is driven
    by a subprocess with MPLBACKEND=Agg to dodge a matplotlib_inline import
    clash.  Callers should normally use `build_atmosphere`, which caches the
    result -- this runs the model itself.

    `time` is a datetime; year, day-of-year and UT hours are DERIVED from it.
    They used to be hardcoded to Myanmar's values (2025, day 87, 6.3498 h),
    which silently produced Myanmar winds for any other event.

    Returns (U, V) in m/s -- U zonal (E+), V meridional (N+).
    """
    if not HWM14_PY:
        raise RuntimeError(
            "HWM14 is needed to (re)build the wind cache "
            f"data/inputs/{WIND_CACHE}.  Set the HWM14_PY environment variable "
            "to the python of a conda env with pyhwm2014 installed.")
    if time is None:
        time = EQ_TIME
    year = time.year
    day = time.timetuple().tm_yday
    ut = time.hour + time.minute / 60.0 + time.second / 3600.0
    alt_str = ','.join(f'{a:.1f}' for a in alts)
    script = f"""
import os
import pyhwm2014
os.chdir(os.path.dirname(pyhwm2014.__file__))   # HWM14 reads its .dat files from cwd
from pyhwm2014 import HWM14
alts = [{alt_str}]
for alt in alts:
    hwm = HWM14(alt=float(alt), altlim=[float(alt),float(alt)], altstp=1,
                year={year}, day={day}, ut={ut},
                glat={lat}, glon={lon},
                ap=[-1, {ap}], option=1, verbose=False)
    print(hwm.Uwind[0], hwm.Vwind[0])
"""
    env = os.environ.copy()
    env['MPLBACKEND'] = 'Agg'
    result = subprocess.run([HWM14_PY, '-c', script],
                            capture_output=True, text=True,
                            timeout=120, env=env)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    rows = [list(map(float, l.split()))
            for l in result.stdout.strip().split('\n')]
    return np.array([r[0] for r in rows]), np.array([r[1] for r in rows])


def cached_wind_profile(z_grid, lat=EQ_LAT, lon=EQ_LON, time=None, ap=AP,
                        refresh=False):
    """
    HWM14 winds on `z_grid`, from the on-disk cache when it matches.

    The whole pipeline needs HWM14 exactly once, for this single 1-D profile
    (~6 KB), yet HWM14 is the one background model with no pip distribution.
    Caching it is what lets every other script run on a machine that has only
    `pip install -r requirements.txt`; the conda env is needed only to create
    the cache, once per event.

    The cache stores its own provenance -- lat, lon, time, ap and the altitude
    grid.  A mismatch on any of them is a MISS, not a silent reuse: a cache
    made for one earthquake must never supply winds for another.  Altitudes are
    interpolated if the cached grid merely differs in spacing but covers the
    request, since the profile is smooth and this is the common case.

    `refresh=True` forces a re-query (needs conda).
    """
    if time is None:
        time = EQ_TIME
    path = paths.inp(WIND_CACHE)

    if not refresh and os.path.exists(path):
        d = np.load(path, allow_pickle=False)
        ok = (abs(float(d['lat']) - lat) < 1e-6
              and abs(float(d['lon']) - lon) < 1e-6
              and abs(float(d['ap']) - ap) < 1e-6
              and str(d['time']) == time.isoformat()
              and float(d['z'][0]) <= z_grid[0]
              and float(d['z'][-1]) >= z_grid[-1])
        if ok:
            zc, Uc, Vc = d['z'], d['U'], d['V']
            if np.array_equal(zc, z_grid):
                print(f"  HWM14 winds from cache ({WIND_CACHE})")
                return Uc, Vc
            print(f"  HWM14 winds from cache ({WIND_CACHE}), "
                  f"interpolated {len(zc)} -> {len(z_grid)} levels")
            return (np.interp(z_grid, zc, Uc), np.interp(z_grid, zc, Vc))
        print("  cached winds do not match this event -- re-querying HWM14")

    print("  Querying HWM14 wind profile (needs the conda env)...")
    U, V = query_hwm14_profile(z_grid, lat=lat, lon=lon, time=time, ap=ap)
    np.savez_compressed(path, z=z_grid, U=U, V=V, lat=lat, lon=lon, ap=ap,
                        time=time.isoformat())
    print(f"  cached -> {WIND_CACHE} ({os.path.getsize(path)/1024:.1f} KB)")
    return U, V


def build_atmosphere(alt_max=RAY_ATM_MAX_KM, alt_step=RAY_ATM_STEP_KM):
    """
    Gridded neutral-atmosphere profiles and the interpolators the ray ODEs
    consume:

      c(z)          sound speed (m/s)
      dc/dz(z)      sound-speed gradient (s^-1)
      U(z), V(z)    HWM14 zonal/meridional winds (m/s)
      dU/dz, dV/dz  wind shear (s^-1)

    Returns (z_grid, c_grid, U_grid, V_grid, interps).
    """
    z_grid = np.arange(0, alt_max + 1, alt_step, dtype=float)

    print("  Computing NRLMSISE-00 sound speed profile...")
    c_grid = np.array([sound_speed(z) for z in z_grid])

    U_grid, V_grid = cached_wind_profile(z_grid)

    dc_dz = np.gradient(c_grid, z_grid * 1e3)
    dU_dz = np.gradient(U_grid, z_grid * 1e3)
    dV_dz = np.gradient(V_grid, z_grid * 1e3)

    # Second derivatives, for the paraxial (variational) ray system: dF/dy
    # differentiates the refraction term, which carries dc/dz and dW/dz, so it
    # needs d2c/dz2 and d2W/dz2.  Built once here rather than differenced
    # per step -- see raytracing.ray_odes_spherical(with_jacobian=True).
    d2c_dz2 = np.gradient(dc_dz, z_grid * 1e3)
    d2U_dz2 = np.gradient(dU_dz, z_grid * 1e3)
    d2V_dz2 = np.gradient(dV_dz, z_grid * 1e3)

    kw = dict(kind='linear', bounds_error=False)
    interps = dict(
        c   =interp1d(z_grid, c_grid, fill_value='extrapolate', **kw),
        dcdz=interp1d(z_grid, dc_dz,  fill_value=0.0,           **kw),
        U   =interp1d(z_grid, U_grid, fill_value='extrapolate', **kw),
        V   =interp1d(z_grid, V_grid, fill_value='extrapolate', **kw),
        dUdz=interp1d(z_grid, dU_dz,  fill_value=0.0,           **kw),
        dVdz=interp1d(z_grid, dV_dz,  fill_value=0.0,           **kw),
        d2c =interp1d(z_grid, d2c_dz2, fill_value=0.0,          **kw),
        d2U =interp1d(z_grid, d2U_dz2, fill_value=0.0,          **kw),
        d2V =interp1d(z_grid, d2V_dz2, fill_value=0.0,          **kw),
    )
    return z_grid, c_grid, U_grid, V_grid, interps


# ---------------------------------------------------------------------------
# Acoustic absorption and vertical amplitude
# ---------------------------------------------------------------------------

def _transport(z_km, lat, lon, time, freq_hz):
    """
    Shared NRLMSISE-00 evaluation for the absorption coefficient.

    Returns (alpha, rho, c) -- absorption (Np/m), density (kg/m^3), sound speed
    (m/s) -- all on `z_km`.  alpha follows Mikesell et al. (2019) eq. A3:

        alpha = [omega^2 / (2 rho c^3)] [4 mu/3 + kappa (gamma-1)/c_p]

    with a Sutherland viscosity and an Eucken (kinetic) conductivity.
    """
    z_km = np.asarray(z_km, dtype=float)
    out = msise_flat(time, z_km, lat, lon, F107, F107, AP)
    rho = np.asarray(out[:, 5], dtype=float) * 1e3          # g/cm^3 -> kg/m^3
    T = np.asarray(out[:, 10], dtype=float)                 # K

    num = np.asarray(out[:, [0, 1, 2, 3, 4, 6, 7]], dtype=float)  # cm^-3
    masses = np.array([4, 16, 28, 32, 40, 1, 14], dtype=float) * 1e-3 / 6.02214076e23
    ntot = num.sum(axis=1)
    ntot[ntot <= 0] = np.nan
    m_mean = (num * masses).sum(axis=1) / ntot              # kg per particle

    k_B = 1.380649e-23
    c = np.sqrt(GAMMA * k_B * T / m_mean)                   # m/s
    mu = 1.458e-6 * T ** 1.5 / (T + 110.4)                  # Pa s (Sutherland)
    c_p = GAMMA * k_B / (m_mean * (GAMMA - 1.0))            # J/(kg K)
    kappa = 0.25 * mu * c_p * (9 * GAMMA - 5)               # W/(m K), Eucken

    omega = 2.0 * math.pi * freq_hz
    alpha = (omega ** 2 / (2.0 * rho * c ** 3)) * \
            (4.0 * mu / 3.0 + kappa * (GAMMA - 1.0) / c_p)  # Np/m
    return alpha, rho, c


def absorption_profile(z_km, lat=EQ_LAT, lon=EQ_LON, time=EQ_TIME,
                       freq_hz=0.002):
    """
    Viscous + thermal absorption alpha(z) (Np/m) and density rho(z) (kg/m^3)
    for path-integrated attenuation.  Mikesell et al. (2019) eq. A3.

    Returns (alpha, rho).
    """
    alpha, rho, _ = _transport(z_km, lat, lon, time, freq_hz)
    return alpha, rho


