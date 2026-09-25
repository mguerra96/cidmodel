"""Acoustic ray tracing and the ray-parameter cube.

The kinematic backbone of the model.  Because NRLMSISE-00 + HWM14 are
evaluated once above the epicentre, the atmosphere is horizontally
homogeneous and a single fan of rays per azimuth serves every fault element
(the solution just translates).  This module:

    ray_odes_spherical              moving-medium ray equations (Lighthill)
    paraxial_odes_spherical         variational system for dy/dtheta0
    trace_ray                       integrate one ray with solve_ivp
    trace_ray_paraxial              a ray together with its paraxial derivative
    RayCube                         ray parameters on a regular (az,rng,z) cube

Spherical geometry only.  The flat-Earth ray equations and the altitude-crossing
ray table they fed (`build_ray_table` -> `RayTable`, with `adaptive_takeoff` and
`geometric_spreading`) were retired on 2026-09-15; `deposit.py` builds the cube
by path deposit instead.

Amplitude bookkeeping combines a vertical factor A_z (rho^-1/2 growth minus
absorption) with A_w (ray-tube spreading), after Mikesell et al. (2019).  Both
are computed along the path by `deposit._fields`, not here.

Units: distances km, time s, slowness s/m, velocity m/s.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .config import R_EARTH_KM, RAY_ATM_MAX_KM, RAY_STEP_M, RAY_T_MAX

R_EARTH_M = R_EARTH_KM * 1e3


# ---------------------------------------------------------------------------
# Ray equations in a moving atmosphere
# ---------------------------------------------------------------------------

def ray_odes_spherical(t, state, az_rad, interps):
    """
    Ray equations in a spherically stratified moving atmosphere.

    State [s (m), z (m), ps, pz]: s = R*theta is surface arc length (directly
    comparable with the flat-Earth x), z is altitude above the curved surface
    (r = R + z), ps/pz the conjugate slownesses.

    Flat-Earth geometry underestimates the altitude a long-range ray must
    climb -- the surface falls R(1-cos(L/R)) below the tangent plane, 78 km at
    L=1000 km, 20% of a 385 km target.  Horizontal distances are unaffected
    (arc vs chord is 0.1% at 1000 km), so only the vertical geometry needs the
    correction.

    Canonical equations of the spherical eikonal Hamiltonian, written in
    (r, theta, p_r, p_theta) and then mapped to this module's state via
    s = R theta, ps = p_theta / R, z = r - R, g = R/r:

        H = c |p| + g ps W - 1 = 0,      |p| = sqrt(pz^2 + (g ps)^2)

        ds/dt  =  c (g^2 ps) / |p| + g W
        dz/dt  =  c pz / |p|
        dps/dt =  0
        dpz/dt = -(dc/dz)|p| + c (g^2 ps^2)/(r |p|) + g ps W / r - g ps dW/dz

    The two 1/r terms are the curvature correction (they bend rays upward
    relative to flat geometry) and arise as -dH/dr through dg/dr = -g/r.
    p_theta is conserved exactly because H has no explicit theta dependence, so
    solutions remain azimuth translation-invariant and the one-fan-per-azimuth
    optimisation holds.

    CORRECTED 2026-08-03: the earlier form substituted |p| = 1/c, true only
    without wind, which made rays over-respond to it.  Verified here by
    (a) |H| <= 7.6e-3 along every ray at 8 azimuths, at the atmosphere
    interpolation floor; (b) dp_theta identically 0; (c) travel time within
    +0.06..+0.14 s of infraGA-sph at az 90 and az 270 over 100-450 km on an
    identical NRLMSISE-00/HWM14 profile.
    """
    s, z, ps, pz = state
    z_km = z / 1e3
    c    = float(interps['c'](z_km))
    dcdz = float(interps['dcdz'](z_km))
    U    = float(interps['U'](z_km))
    V    = float(interps['V'](z_km))
    dUdz = float(interps['dUdz'](z_km))
    dVdz = float(interps['dVdz'](z_km))

    W_along = U * math.sin(az_rad) + V * math.cos(az_rad)
    dWdz    = dUdz * math.sin(az_rad) + dVdz * math.cos(az_rad)

    r = R_EARTH_M + z
    g = R_EARTH_M / r

    gps = g * ps
    p_mag = math.hypot(pz, gps)

    return [c * g * gps / p_mag + g * W_along,
            c * pz / p_mag,
            0.0,
            (-dcdz * p_mag
             + c * gps * gps / (r * p_mag)
             + gps * W_along / r
             - gps * dWdz)]


def paraxial_odes_spherical(t, state8, az_rad, interps):
    """
    Ray + paraxial system in spherical geometry, state
    [s, z, ps, pz, Ps, Pz, Pps, Ppz].

    The trailing four are P = dy/dtheta0, the derivative of the ray state with
    respect to TAKE-OFF ANGLE (radians), obeying the variational equation

        dP/dt = (dF/dy) P

    integrated alongside the ray.  This replaces differencing adjacent rays:
    each ray carries its own exact ds/dtheta0, so the ray-tube Jacobian needs
    no neighbours, has no finite-difference noise, and stays valid where the
    fan is sparse or non-uniform (near the turning-point cutoff, where the old
    np.gradient estimate was worst).  infraGA does the same thing -- its
    solution vector carries d(r,theta,phi)/d(launch angles) as state and
    assembles `jacobian()` from them.

    dF/dy is written out analytically and FUSED with F so the six atmosphere
    lookups are done once per call and shared.  That matters: a plain RHS call
    is ~100% interpolator overhead, so evaluating dF/dy by differencing F
    (8 extra RHS calls) costs 71x, while this costs 2.3x.  Verified against
    that numeric-Jacobian version to 2e-3, and against two-ray finite
    differencing of ds/dtheta0 to 3e-3.
    """
    s, z, ps, pz = state8[0], state8[1], state8[2], state8[3]
    z_km = z / 1e3

    c    = float(interps['c'](z_km))
    dcdz = float(interps['dcdz'](z_km))
    U    = float(interps['U'](z_km))
    V    = float(interps['V'](z_km))
    dUdz = float(interps['dUdz'](z_km))
    dVdz = float(interps['dVdz'](z_km))
    sa, ca = math.sin(az_rad), math.cos(az_rad)
    d2c = float(interps['d2c'](z_km))
    d2W = float(interps['d2U'](z_km)) * sa + float(interps['d2V'](z_km)) * ca

    W_along = U * sa + V * ca
    dWdz    = dUdz * sa + dVdz * ca

    r = R_EARTH_M + z
    g = R_EARTH_M / r
    gps = g * ps
    pm = math.hypot(pz, gps)
    dgdz = -g / r

    F0 = c * g * gps / pm + g * W_along
    F1 = c * pz / pm
    F3 = (-dcdz * pm + c * gps * gps / (r * pm)
          + gps * W_along / r - gps * dWdz)

    dpm_dz  = (gps * ps * dgdz) / pm
    dpm_dps = (gps * g) / pm
    dpm_dpz = pz / pm

    J01 = ((dcdz * g * gps + c * 2 * g * ps * dgdz) / pm
           - c * g * gps * dpm_dz / pm ** 2 + dgdz * W_along + g * dWdz)
    J02 = c * g * g / pm - c * g * gps * dpm_dps / pm ** 2
    J03 = -c * g * gps * dpm_dpz / pm ** 2

    J11 = dcdz * pz / pm - c * pz * dpm_dz / pm ** 2
    J12 = -c * pz * dpm_dps / pm ** 2
    J13 = c / pm - c * pz * dpm_dpz / pm ** 2

    A = c * gps * gps / (r * pm)
    J31 = (-d2c * pm - dcdz * dpm_dz
           + (dcdz * gps * gps + c * 2 * gps * ps * dgdz) / (r * pm)
           - A * (1.0 / r + dpm_dz / pm)
           + (dgdz * ps * W_along + gps * dWdz) / r - gps * W_along / r ** 2
           - (dgdz * ps * dWdz + gps * d2W))
    J32 = (-dcdz * dpm_dps + c * 2 * gps * g / (r * pm)
           - A * dpm_dps / pm + g * W_along / r - g * dWdz)
    J33 = -dcdz * dpm_dpz - A * dpm_dpz / pm

    P1, P2, P3 = state8[5], state8[6], state8[7]
    return [F0, F1, 0.0, F3,
            J01 * P1 + J02 * P2 + J03 * P3,
            J11 * P1 + J12 * P2 + J13 * P3,
            0.0,
            J31 * P1 + J32 * P2 + J33 * P3]


def make_events(alt_max_m):
    """Terminal solve_ivp events: ray hits ground (z=0) or the ceiling."""
    def hit_ground(t, state, az_rad, interps):  return state[1]
    def hit_ceiling(t, state, az_rad, interps): return state[1] - alt_max_m
    hit_ground.terminal  = True;  hit_ground.direction  = -1
    hit_ceiling.terminal = True;  hit_ceiling.direction = +1
    return [hit_ground, hit_ceiling]


def trace_ray(zenith_deg, az_deg, interps,
              x0=0.0, z0=0.0, t_max=RAY_T_MAX, step_m=RAY_STEP_M,
              alt_max_km=None):
    """
    Trace a single acoustic ray.

    zenith_deg : launch zenith angle (0 = straight up, 90 = horizontal).
    az_deg     : propagation azimuth (deg CW from N).
    interps    : atmospheric interpolators from atmosphere.build_atmosphere().
    alt_max_km : ceiling at which integration terminates.  Defaults to the
        ray-tracing atmosphere ceiling (RAY_ATM_MAX_KM).  MUST match the
        ceiling the atmosphere was built to -- if the atmosphere reaches higher
        but this is left low, every ray silently terminates early and targets
        above it look unreachable.

    Returns a scipy ODE solution with dense_output=True.
    """
    if alt_max_km is None:
        alt_max_km = RAY_ATM_MAX_KM
    az_rad = math.radians(az_deg)
    launch_deg = 90.0 - zenith_deg          # zenith -> elevation
    c0 = float(interps['c'](z0 / 1e3))
    U0 = float(interps['U'](z0 / 1e3))
    V0 = float(interps['V'](z0 / 1e3))
    W0 = U0 * math.sin(az_rad) + V0 * math.cos(az_rad)

    ang = math.radians(launch_deg)
    # Launch ON the eikonal manifold H = c|p| + px W - 1 = 0.  For a unit
    # direction (cos ang, sin ang) the wind projects onto the ray as
    # cos(ang) W, NOT the bare W: only a horizontally-travelling ray is
    # advected at the full wind speed.  The previous c_eff = c0 + W0 left
    # H(0) up to 4e-3 off the manifold, with a sign set by the wind direction,
    # so east- and west-going fans started with opposite systematic errors.
    c_eff0 = c0 + math.cos(ang) * W0

    px = math.cos(ang) / c_eff0             # horizontal slowness
    pz = math.sin(ang) / c_eff0             # vertical slowness

    return solve_ivp(
        ray_odes_spherical,
        [0, t_max], [x0, z0, px, pz],
        args=(az_rad, interps),
        events=make_events(alt_max_km * 1e3),
        max_step=step_m, rtol=1e-6, atol=1e-8,
        dense_output=True,
    )


def _launch_state(zenith_deg, az_rad, interps, x0=0.0, z0=0.0):
    """Initial [x, z, px, pz] on the eikonal manifold (see `trace_ray`)."""
    ang = math.radians(90.0 - zenith_deg)
    c0 = float(interps['c'](z0 / 1e3))
    U0 = float(interps['U'](z0 / 1e3))
    V0 = float(interps['V'](z0 / 1e3))
    W0 = U0 * math.sin(az_rad) + V0 * math.cos(az_rad)
    c_eff0 = c0 + math.cos(ang) * W0
    return [x0, z0, math.cos(ang) / c_eff0, math.sin(ang) / c_eff0]


def trace_ray_paraxial(zenith_deg, az_deg, interps,
                       x0=0.0, z0=0.0, t_max=RAY_T_MAX, step_m=RAY_STEP_M,
                       alt_max_km=None, dtheta_deg=1e-5,
                       stop_at_apogee=False):
    """
    Trace a ray together with its paraxial derivative dy/dtheta0.

    Returns a solution whose state is 8 long: the usual [x, z, px, pz] followed
    by [dx/dtheta0, dz/dtheta0, dpx/dtheta0, dpz/dtheta0] in radians^-1.

    The launch derivative is taken by differencing `_launch_state` about
    zenith_deg -- the initial condition depends on the take-off angle through
    both the direction cosines and the wind-projected c_eff, so it is not
    simply (-sin, cos)/c.

    `deposit._fields` turns the result into A_w.  See
    `paraxial_odes_spherical` for why this replaces neighbour differencing.

    `stop_at_apogee` adds a third terminal event at the turning point (pz
    changing sign), so the trace returns the ASCENDING limb only.  The deposit
    uses it for pass 1, where every ray is taken to apogee-or-ceiling and the
    fold is then located as argmax of the apogee ranges; only rays past the
    fold are re-traced to the ground.  See `deposit.deposit_azimuth`.

    Spherical only: the flat-Earth variational system is not written out.
    """
    if alt_max_km is None:
        alt_max_km = RAY_ATM_MAX_KM
    az_rad = math.radians(az_deg)

    y0 = _launch_state(zenith_deg, az_rad, interps, x0, z0)
    lo = _launch_state(zenith_deg - dtheta_deg, az_rad, interps, x0, z0)
    hi = _launch_state(zenith_deg + dtheta_deg, az_rad, interps, x0, z0)
    # per RADIAN of take-off, and note zenith increases as elevation decreases
    scale = 2.0 * math.radians(dtheta_deg)
    P0 = [(h - l) / scale for h, l in zip(hi, lo)]

    def hit_ground(t, s, *a):   return s[1]
    def hit_ceiling(t, s, *a):  return s[1] - alt_max_km * 1e3
    hit_ground.terminal = True;  hit_ground.direction = -1
    hit_ceiling.terminal = True; hit_ceiling.direction = +1

    events = [hit_ground, hit_ceiling]
    if stop_at_apogee:
        # Turning point: vertical slowness changes sign ascending -> descending.
        # Lets a caller take the ascending limb alone without knowing in advance
        # where the ray turns, which is what the deposit's pass 1 needs.
        def hit_apogee(t, s, *a):   return s[3]
        hit_apogee.terminal = True; hit_apogee.direction = -1
        events.append(hit_apogee)

    return solve_ivp(
        paraxial_odes_spherical, [0, t_max], y0 + P0,
        args=(az_rad, interps),
        events=events,
        max_step=step_m, rtol=1e-6, atol=1e-8,
        dense_output=True,
    )



# ---------------------------------------------------------------------------
# Resampled ray cube
# ---------------------------------------------------------------------------

@dataclass
class RayCube:
    """
    Ray parameters resampled onto a regular (azimuth, range, altitude) cube.

    Scattered-ray interpolation is a scalar Python call (~21 us).  A
    400-element source over a 105k-node grid needs ~42 million per timestep --
    about 18 minutes, which makes time integration impossible.  Resampling once
    onto a regular cube lets every node be evaluated with vectorised numpy
    lookups.  Written by `interpolate_deposit`; read back with `load`.

    Fields are NaN where the direct ray fan does not reach.
    """
    az: np.ndarray          # (n_az,) deg, uniform, wraps at 360
    rng: np.ndarray         # (n_rng,) km, uniform
    z: np.ndarray           # (n_z,) km, matches the model grid
    t_w: np.ndarray         # (n_az, n_rng, n_z) travel time (s)
    kx: np.ndarray          # (n_az, n_rng, n_z) k_hat east
    ky: np.ndarray          # k_hat north
    kz: np.ndarray          # k_hat up
    amp: np.ndarray = None  # (n_az, n_rng, n_z) A_z * A_w, combined; 1.0
                            # everywhere if the deposit carried no amplitude

    def save(self, path):
        """Cache the resampled cube to a .npz file."""
        np.savez_compressed(path, az=self.az, rng=self.rng, z=self.z,
                            t_w=self.t_w, kx=self.kx, ky=self.ky, kz=self.kz,
                            amp=self.amp)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(az=d['az'], rng=d['rng'], z=d['z'], t_w=d['t_w'],
                   kx=d['kx'], ky=d['ky'], kz=d['kz'], amp=d['amp'])


    def _lookup_nearest(self, az_deg, rng_km):
        """Nearest-neighbour lookup: one gather per field, minimal memory."""
        d_az = self.az[1] - self.az[0]
        ia = (np.rint((np.asarray(az_deg, dtype=float) % 360.0) / d_az)
              .astype(int)) % len(self.az)
        d_r = self.rng[1] - self.rng[0]
        ir = np.rint(np.asarray(rng_km, dtype=float) / d_r).astype(int)
        valid = ir < len(self.rng)
        ir = np.clip(ir, 0, len(self.rng) - 1)

        t = self.t_w[ia, ir, :]
        kx = self.kx[ia, ir, :]
        ky = self.ky[ia, ir, :]
        kz = self.kz[ia, ir, :]
        a = self.amp[ia, ir, :] if self.amp is not None else np.ones_like(t)
        if not valid.all():
            bad = ~valid
            for arr in (t, kx, ky, kz, a):
                arr[bad] = np.nan
        return t, kx, ky, kz, a

    def lookup(self, az_deg, rng_km, bilinear=False):
        """
        Lookup in (az, range) for whole altitude columns.

        `az_deg` and `rng_km` are arrays of the same shape; returns
        (t_w, kx, ky, kz, amp) each with shape az_deg.shape + (n_z,).  Entries
        outside the ray fan are NaN.

        `bilinear=False` (default) is nearest-neighbour: one gather, cheap,
        ~5x faster and far lighter on memory -- the working default for
        iteration and any full time series.  Its cost is quantising arrival
        time into cube cells; since the wave is a narrow N-wave in (t - t_w),
        those cells show up as radial/annular striping in dNe at late times.

        `bilinear=True` interpolates over the four surrounding cells, removing
        that quantisation at ~5x the cost and memory.  Reserve it for final
        figures.  NaN handling: the fan edge has NaN corners; weights are
        renormalised over the finite corners so an interior point next to the
        edge is not dragged toward zero, but t_w additionally requires all four
        corners finite (interpolating travel time across the boundary would
        smear the wavefront into unreachable ground).
        """
        if not bilinear:
            return self._lookup_nearest(az_deg, rng_km)

        az_q = np.asarray(az_deg, dtype=float) % 360.0
        rng_q = np.asarray(rng_km, dtype=float)

        d_az = self.az[1] - self.az[0]
        fa = az_q / d_az
        ia0 = np.floor(fa).astype(int)
        wa = fa - ia0
        n_az = len(self.az)
        ia0 %= n_az                          # azimuth is periodic (357 -> 0)
        ia1 = (ia0 + 1) % n_az

        d_r = self.rng[1] - self.rng[0]
        fr = rng_q / d_r
        ir0 = np.floor(fr).astype(int)
        wr = fr - ir0
        n_r = len(self.rng)
        valid = (ir0 >= 0) & (ir0 + 1 < n_r)
        ir0 = np.clip(ir0, 0, n_r - 2)
        ir1 = ir0 + 1

        w00 = ((1 - wa) * (1 - wr))[..., None]
        w01 = ((1 - wa) * wr)[..., None]
        w10 = (wa * (1 - wr))[..., None]
        w11 = (wa * wr)[..., None]

        # Finite-corner masks depend only on t_w (all fields share the same fan
        # reach), so compute once rather than per field.  This, plus in-place
        # accumulation below, keeps `lookup` from allocating ~15 grid-sized
        # temporaries per call (an earlier version OOM'd for exactly that).
        tw = self.t_w
        f00 = np.isfinite(tw[ia0, ir0, :])
        f01 = np.isfinite(tw[ia0, ir1, :])
        f10 = np.isfinite(tw[ia1, ir0, :])
        f11 = np.isfinite(tw[ia1, ir1, :])
        wm00 = np.where(f00, w00, 0.0)
        wm01 = np.where(f01, w01, 0.0)
        wm10 = np.where(f10, w10, 0.0)
        wm11 = np.where(f11, w11, 0.0)
        wsum = wm00 + wm01 + wm10 + wm11
        all_finite = f00 & f01 & f10 & f11
        any_finite = wsum > 0

        def interp(arr, require_all):
            # acc = sum_corner value*weight, each term guarded: a masked corner
            # has arr=NaN and 0*NaN=NaN, so the zero weight alone is not enough.
            acc = np.where(f00, arr[ia0, ir0, :] * wm00, 0.0)
            acc += np.where(f01, arr[ia0, ir1, :] * wm01, 0.0)
            acc += np.where(f10, arr[ia1, ir0, :] * wm10, 0.0)
            acc += np.where(f11, arr[ia1, ir1, :] * wm11, 0.0)
            with np.errstate(invalid='ignore', divide='ignore'):
                acc /= wsum
            good = all_finite if require_all else any_finite
            return np.where(good, acc, np.nan)

        t = interp(self.t_w, True)
        kx = interp(self.kx, False)
        ky = interp(self.ky, False)
        kz = interp(self.kz, False)
        a = (interp(self.amp, False) if self.amp is not None
             else np.ones_like(t))

        # Component-wise interpolation does not preserve unit length
        norm = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
        with np.errstate(invalid='ignore', divide='ignore'):
            kx, ky, kz = kx / norm, ky / norm, kz / norm

        if not valid.all():
            bad = ~valid
            t[bad] = np.nan
            kx[bad] = np.nan
            ky[bad] = np.nan
            kz[bad] = np.nan
            a[bad] = np.nan
        return t, kx, ky, kz, a
