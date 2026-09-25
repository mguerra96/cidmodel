"""
Ray cube by PATH DEPOSIT, replacing the altitude-crossing table.

The crossing scheme asked each ray "where do you cross altitude z?" and stored
that one sample per grid altitude.  Near apogee a ray is nearly horizontal, so
it spends hundreds of km of range within a few km of its turning height and
still contributes only two crossings there -- which is why the cube had a blank
band above the F region that no amount of interpolation could honestly fill.

Here each ray is instead walked in uniform arc-length steps and every sample is
deposited into whichever (range, altitude) cell it lands in.  A cell's value is
the mean over the samples inside it; because the step is uniform, that mean is
automatically PATH-LENGTH weighted -- a ray that grazes through a cell for
40 km counts forty times a ray that clips its corner for 1 km.

Three rules shape what gets deposited:

  apogee-range fold
      Sweeping take-off angle downward, the range at which a ray turns over
      rises to a maximum and then falls.  Past that turning point a descending
      branch is only re-covering ground a shallower ray already reached, and
      those rays have arced high above the F region and been absorbed to
      ~1e-120 on the way.  Recording them put 130 decades of amplitude between
      neighbouring cells -- the striping that made the far field
      uninterpolable.  Past the fold, the descending half is dropped and the
      ascending half kept.  (Marco's rule, 2026-08-03.)

      Applied by NOT TRACING rather than by masking: pass 1 stops every ray at
      apogee or the ceiling, the fold is argmax of the apogee ranges, and only
      rays past it are re-traced to the ground.  Nothing integrates above the
      model top.  (2026-08-04, replacing a scheme that traced the whole fan to
      1500 km purely to give every ray a z = 0 landing range.)

  branch split
      Ascending and descending samples are accumulated separately, so the
      choice of what to do where both reach one cell stays open after the
      trace.  With the fold rule in force this affects ~0.01% of cells with
      travel times within ~2 s, so `collapse` simply averages them.

  alpha shape
      Interpolation fills the CONVEX hull of the samples, which spans the
      near-field dome -- a region no ray enters, where the interpolant joins
      one arc back to itself.  The alpha shape is the non-convex hull, and the
      dome is exactly where the sample region is concave, so it comes out by
      construction.  See `alpha_mask`; it superseded both a distance mask (too
      blunt: the dome's interior is close to the samples ringing it) and a
      traced lower envelope (correct but a 200-ray fan per azimuth).

`deposit_azimuth` returns raw accumulators and nothing else; `collapse` and
`interpolate` are separate and cheap, so a build can be re-interpolated at a
different threshold without re-tracing.  See pipeline/02_deposit.py.
"""
import math

import numpy as np
from scipy.interpolate import griddata

from .atmosphere import absorption_profile
from .raytracing import (trace_ray_paraxial, R_EARTH_M)
from .config import RAY_CEILING_KM, RAY_T_MAX, TAKEOFF_RANGE

# Arc-length step along each ray.  Smaller than the 2.5 km cell diagonal so no
# cell a ray passes through is skipped.
DS_KM = 1.0
# Rays per azimuth, uniform in take-off zenith over TAKEOFF_RANGE: 200 over
# 0.25-34 deg is 0.169 deg spacing.
N_TAKEOFF = 200
# Alpha-shape radius (km) for `alpha_mask`.  Must exceed the gap between
# adjacent ray arcs and stay well below the near-field dome's width.  Measured
# at az 97: the dome is carved identically anywhere from 40 to 400 km (its span
# is hundreds of km, far wider than any of them), so the choice is decided by
# the far field, where 40 km leaves the upper corner past 1500 km ragged.  A
# SAMPLING parameter -- it tracks ray spacing, so it wants revisiting if
# N_TAKEOFF changes materially.
ALPHA_KM = 100.0


def _fields(sol, az_rad, theta0_deg, interps, zf, alpha, rho, rho_ground,
            ds_km=DS_KM, band=None, fan_axis=None):
    """Resample one ray at uniform arc length; return the deposited quantities.

    Returns (t, x_km, z_km, A_z, A_w, px, pz, descending) or None if the ray
    is degenerate.

    `theta0_deg` is the ray's take-off zenith, needed for the solid-angle
    weight in A_w (see below).

    `fan_axis` = (z_km, x_km) of the theta0 = 0 ray of this azimuth, used to
    measure the azimuthal lever arm from the fan's own axis rather than from
    the source (see the A_w block).  None reverts to the source-centred form.

    `band` = (z_min, z_max, rng_max) km restricts the EXPENSIVE part -- A_w and
    the returned samples -- to the cube's extent.  Part of every ray lies
    outside it (beyond rng_max, or below the cube's floor); computing A_w
    there is pure waste.  tau is still
    accumulated along the WHOLE path, since absorption at a point depends on
    everything the ray traversed to get there.
    """
    x_m, z_m = sol.y[0], sol.y[1]
    seg = np.hypot(np.diff(x_m), np.diff(z_m))
    L = np.concatenate([[0.0], np.cumsum(seg)])
    if L[-1] <= 0:
        return None
    n_d = max(int(L[-1] / (ds_km * 1e3)) + 1, 2)
    t_d = np.interp(np.linspace(0.0, L[-1], n_d), L, sol.t)
    st = sol.sol(t_d)                                   # (8, n_d)
    xs, zs = st[0] / 1e3, st[1] / 1e3

    # tau on the dense arc-length grid (not on the ODE's own steps).  Must run
    # over the full path -- it is a path integral, not a local quantity.
    ds = np.concatenate([[0.0], np.hypot(np.diff(st[0]), np.diff(st[1]))])
    tau = np.cumsum(np.interp(zs, zf, alpha) * ds)
    A_z = np.sqrt(rho_ground / np.interp(zs, zf, rho)) * np.exp(-tau)

    if band is not None:
        zlo, zhi, rmax = band
        keep = (zs >= zlo) & (zs <= zhi) & (xs <= rmax)
        if not keep.any():
            return None
        t_d, st, xs, zs, A_z = t_d[keep], st[:, keep], xs[keep], zs[keep], A_z[keep]

    # A_w from the ray tube taken PERPENDICULAR to the ray.  The constant-
    # altitude form used by the crossing table divides by dz/dt, which vanishes
    # at apogee -- exactly the band the deposit exists to fill.  |P x t_hat| is
    # finite there and agrees with it wherever the ray is steep.
    #
    # The tube's arrival area is the product of the in-plane width |P x t_hat|
    # and the azimuthal width x.  Energy conservation is per unit SOLID ANGLE
    # at the source, though, and the tube launched with sin(theta0) dtheta dphi
    # of it -- so the area is divided by sin(theta0) to give area per steradian.
    # Without that weight near-vertical tubes are credited with far more flux
    # than they carry, and A_w picks up a spurious tilt across the fan: in a
    # homogeneous atmosphere, where spreading must be an isotropic 1/s, the
    # unweighted form varies strongly with take-off angle while the weighted
    # one is isotropic.
    #
    # Note the perpendicular width carries no theta0 dependence of its own --
    # |P x t_hat| = s exactly in the straight-ray limit -- so the 2026-09-15
    # switch away from the constant-altitude form did not absorb this factor.
    #
    # The azimuthal leg is measured from the FAN'S OWN AXIS, not from the
    # source.  x ~ s sin(theta0) -- so that x/sin(theta0) tends to a finite
    # limit and the area stays regular on axis -- holds only if the fan is
    # symmetric about the vertical through the source.  It is not: the wind
    # advects the whole ray family sideways, by 2.6 km at 200 km altitude,
    # 10.3 km at 300 and 23.5 km at 500 (az 90, where the along-azimuth wind is
    # -63 m/s).  Measured from the source, x therefore tends to that non-zero
    # offset while sin(theta0) -> 0, and x/sin(theta0) runs away: it swings
    # 520 -> 1792 over 8 -> 0.25 deg and changes sign.  Measured from the
    # advected axis x_c(z), the cancellation is restored -- (x-x_c)/sin(theta0)
    # is flat to 1.3% from 8 deg down to 0.05 deg.
    #
    # The axis is traced once per azimuth as the theta0 = 0 ray, which carries
    # no amplitude of its own (a tube of zero angular width) but is a perfectly
    # good geometric reference.  It escapes, so it exists only on the ASCENDING
    # limb; descending samples keep the source-centred form.  That costs
    # nothing here: at z = 300 km the ascending branch occupies 10-510 km and
    # the descending 730-1330 km, so no cell holds both and there is no seam,
    # and the correction out there is under 1% anyway.
    #
    # |sin(theta0)| and |lever|, not the signed values: the wind carries
    # near-vertical rays ACROSS the axis (xs < 0 for a ray launched east), and
    # a signed area then comes out negative and is nan'd away -- silently
    # deleting every positive-zenith ray below ~1.5 deg.
    #
    # Effect: E/W amplitude asymmetry at 10 km from the epicentre, which must
    # be ~1 because both rays climbed the same column of air, goes 2.91 -> 1.01
    # while the genuine far-field wind asymmetry is untouched (1.15 at 800 km,
    # both ways).  Near-axis roughness drops 14-48x across azimuths.
    # This replaced the old `max(xs, 10.0)` clamp's job of capping the
    # divergence, which masked two thirds of it (11.5x -> 3.3x).
    #
    # ds/dt and dz/dt are inlined from ray_odes_spherical and evaluated on the
    # whole ray at once.  Calling it per sample cost ~400k interpreted calls
    # per azimuth (six scalar interpolator lookups each) and dominated the
    # build -- 45 s of a 63 s azimuth, against 15 s for the ODE solving itself.
    ps, pz_ = st[2], st[3]
    cv = np.asarray(interps['c'](zs), float)
    Uv = np.asarray(interps['U'](zs), float)
    Vv = np.asarray(interps['V'](zs), float)
    W_along = Uv * math.sin(az_rad) + Vv * math.cos(az_rad)
    r = R_EARTH_M + st[1]
    g = R_EARTH_M / r
    gps = g * ps
    p_mag = np.hypot(pz_, gps)
    vx = cv * g * gps / p_mag + g * W_along
    vz = cv * pz_ / p_mag
    vmag = np.hypot(vx, vz)
    perp = np.abs(st[4] * vz - st[5] * vx) / np.where(vmag > 0, vmag, np.nan)
    sin_th0 = abs(math.sin(math.radians(theta0_deg)))
    descending = st[3] < 0
    if fan_axis is not None:
        z_ax, x_ax = fan_axis
        lever = np.where(descending, np.abs(xs),
                         np.abs(xs - np.interp(zs, z_ax, x_ax)))
    else:
        lever = np.abs(xs)
    with np.errstate(divide='ignore', invalid='ignore'):
        area = lever * perp / 1e3 / sin_th0             # km * km/rad / sr
        A_w = np.where(area > 0, 1.0 / np.sqrt(area), np.nan)

    return t_d, xs, zs, A_z, A_w, st[2], st[3], descending


def uniform_takeoff(n=N_TAKEOFF, lo=None, hi=None):
    """Take-off zeniths spread uniformly over the full launch range."""
    if lo is None:
        lo = TAKEOFF_RANGE[0]
    if hi is None:
        hi = TAKEOFF_RANGE[1]
    return np.linspace(float(lo), float(hi), int(n))


def _fan_axis(az_deg, interps, ds_km=DS_KM, ray_ceiling_km=RAY_CEILING_KM):
    """(z_km, x_km) of the theta0 = 0 ray's ascending limb, sorted by altitude.

    The azimuthal lever arm in `_fields` is measured from this, not from the
    source: the wind advects the whole fan sideways, so the ray family's axis
    is not the vertical through the epicentre.  See the A_w block there.

    Returns None if the ray is degenerate, which reverts `_fields` to the
    source-centred form.
    """
    sol = trace_ray_paraxial(0.0, az_deg, interps, t_max=RAY_T_MAX * 3,
                             alt_max_km=ray_ceiling_km)
    x_m, z_m = sol.y[0], sol.y[1]
    seg = np.hypot(np.diff(x_m), np.diff(z_m))
    L = np.concatenate([[0.0], np.cumsum(seg)])
    if L[-1] <= 0:
        return None
    n_d = max(int(L[-1] / (ds_km * 1e3)) + 1, 2)
    st = sol.sol(np.interp(np.linspace(0.0, L[-1], n_d), L, sol.t))
    xs, zs = st[0] / 1e3, st[1] / 1e3
    up = np.arange(len(zs)) <= int(np.argmax(zs))
    if up.sum() < 2:
        return None
    o = np.argsort(zs[up])
    return zs[up][o], xs[up][o]


def deposit_azimuth(az_deg, interps, rng_edges, z_edges, freq_hz=0.002,
                    n_takeoff=N_TAKEOFF, ds_km=DS_KM,
                    ray_ceiling_km=RAY_CEILING_KM, apply_fold=True,
                    takeoff=None):
    """Trace one azimuth's fan and deposit it into a (range, altitude) grid.

    Returns a dict of float64 accumulators, each (n_r, n_z), suffixed `_up` /
    `_dn` for the ascending and descending branches:

        n      samples in the cell  (= path length / ds_km)
        rays   DISTINCT rays contributing -- independent information, whereas
               `n` counts a single grazing ray many times
        t, az_, aw, px, pz    sums, to be divided by `n`

    plus `fold` (the take-off angle where the rule fired, NaN if it never did)
    and `n_dropped`.

    Two-phase tracing
    -----------------
    Nothing is integrated above the model top.  Pass 1 traces every ray with a
    terminal event at the ceiling AND at apogee, so each ray stops at whichever
    comes first: escaping rays end at `ray_ceiling_km`, turning rays end at
    their turning point.  The fold is then the take-off angle of the LARGEST
    APOGEE RANGE, and pass 2 re-traces only the rays past it, down to z = 0.

    This replaced a scheme that traced the whole fan to FOLD_CEILING_KM (1500
    km, far above the model top) purely so every ray would have a z = 0 landing
    range for `landing_fold`.  Those extra kilometres deposited nothing -- the
    samples are cut back to the cube's band regardless -- and the escaping rays
    that made the tall ceiling necessary are exactly the ones that never land,
    so they never needed classifying at all.

    Apogee range peaks at the same take-off as landing range: measured over 24
    azimuths at 300 rays, argmax agrees exactly on 20 and within one ray
    spacing (0.113 deg) on 23, with no systematic bias.  It must be taken as an
    ARGMAX over the whole fan, not as the first ray-to-ray decrease while
    sweeping: at 300-ray spacing the apogee curve is not monotone, and a
    first-decrease test trips 75-103 steps early, at 33.4-33.9 deg instead of
    the true 22.2-25.4 deg.
    """
    az_rad = math.radians(az_deg)
    n_r, n_z = len(rng_edges) - 1, len(z_edges) - 1
    r0, dr = rng_edges[0], rng_edges[1] - rng_edges[0]
    z0, dz = z_edges[0], z_edges[1] - z_edges[0]

    zf = np.arange(0.0, ray_ceiling_km + 1.0, 1.0)
    alpha, rho = absorption_profile(zf, freq_hz=freq_hz)
    rho_ground = float(rho[0])

    if takeoff is None:
        takeoff = uniform_takeoff(n_takeoff)
    takeoff = np.unique(np.asarray(takeoff, float))

    # The fan's own axis, for the azimuthal lever arm in `_fields`: one extra
    # ray at theta0 = 0, traced for GEOMETRY only (its tube has zero angular
    # width, so it carries no amplitude).  It escapes, so it gives x_c(z) on
    # the ascending limb, which is where the correction is needed.
    fan_axis = _fan_axis(az_deg, interps, ds_km, ray_ceiling_km)

    # ---- pass 1: every ray to apogee-or-ceiling, nothing above the model top
    traced = []
    x_apogee = np.full(len(takeoff), np.nan)
    escaped = np.zeros(len(takeoff), bool)
    for i, za in enumerate(takeoff):
        sol = trace_ray_paraxial(float(za), az_deg, interps,
                                 t_max=RAY_T_MAX * 3,
                                 alt_max_km=ray_ceiling_km,
                                 stop_at_apogee=True)
        traced.append(sol)
        zk = sol.y[1] / 1e3
        escaped[i] = zk[-1] >= ray_ceiling_km - 1.0
        if not escaped[i]:
            x_apogee[i] = sol.y[0][-1] / 1e3       # the apogee event's own x

    # ---- fold = take-off of the largest apogee range, over TURNING rays only
    fold = None
    if apply_fold and np.isfinite(x_apogee).any():
        fold = float(takeoff[int(np.nanargmax(x_apogee))])

    # ---- pass 2: only rays past the fold continue to the ground
    for i, za in enumerate(takeoff):
        if escaped[i]:
            continue
        if fold is not None and za < fold:
            continue                               # apogee-only, already done
        traced[i] = trace_ray_paraxial(float(za), az_deg, interps,
                                       t_max=RAY_T_MAX * 3,
                                       alt_max_km=ray_ceiling_km)

    keys = ('n', 'rays', 't', 'az_', 'aw', 'px', 'pz')
    acc = {f'{k}_{b}': np.zeros((n_r, n_z)) for k in keys for b in ('up', 'dn')}
    n_dropped = 0

    # Rays stopped at apogee in pass 1 carry no descending samples at all, so
    # the fold needs no masking here -- it was applied by not tracing them.
    n_dropped = int(sum(1 for i, za in enumerate(takeoff)
                        if not escaped[i] and fold is not None and za < fold))

    for za, sol in zip(takeoff, traced):
        f = _fields(sol, az_rad, float(za), interps, zf, alpha, rho,
                    rho_ground, ds_km,
                    band=(z_edges[0], z_edges[-1], rng_edges[-1]),
                    fan_axis=fan_axis)
        if f is None:
            continue
        t_d, xs, zs, A_z, A_w, px, pz, descending = f

        ir = np.floor((xs - r0) / dr).astype(int)
        iz = np.floor((zs - z0) / dz).astype(int)
        ok = ((ir >= 0) & (ir < n_r) & (iz >= 0) & (iz < n_z)
              & np.isfinite(A_w) & np.isfinite(A_z))

        for tag, sel in (('up', ok & ~descending), ('dn', ok & descending)):
            if not sel.any():
                continue
            flat = ir[sel] * n_z + iz[sel]
            np.add.at(acc[f'n_{tag}'].reshape(-1), flat, 1.0)
            np.add.at(acc[f't_{tag}'].reshape(-1), flat, t_d[sel])
            np.add.at(acc[f'az__{tag}'].reshape(-1), flat, A_z[sel])
            np.add.at(acc[f'aw_{tag}'].reshape(-1), flat, A_w[sel])
            np.add.at(acc[f'px_{tag}'].reshape(-1), flat, px[sel])
            np.add.at(acc[f'pz_{tag}'].reshape(-1), flat, pz[sel])
            touched = np.zeros(n_r * n_z, bool)
            touched[flat] = True
            acc[f'rays_{tag}'].reshape(-1)[touched] += 1.0

    acc['fold'] = np.array(fold if fold is not None else np.nan)
    acc['n_dropped'] = np.array(n_dropped)
    return acc


def collapse(acc):
    """Path-length-weighted means over both branches.

    Where ascending and descending samples share a cell the two are AVERAGED,
    weighted by how much path each spent there.  With the fold rule in force
    that is ~0.01% of cells with travel times within ~2 s of each other, so the
    averaging cannot manufacture a meaningful phantom arrival; without it, it
    would (branches hundreds of seconds apart -- see the module docstring).

    Slowness is averaged as a VECTOR and renormalised, not as a direction:
    averaging unit vectors biases toward whichever branch has more samples.
    """
    n = acc['n_up'] + acc['n_dn']
    have = n > 0
    d = np.where(have, n, 1.0)
    out = {'n': n, 'rays': acc['rays_up'] + acc['rays_dn'],
           'have': have}
    for k in ('t', 'aw', 'px', 'pz'):
        out[k] = np.where(have, (acc[f'{k}_up'] + acc[f'{k}_dn']) / d, np.nan)
    out['az_'] = np.where(have, (acc['az__up'] + acc['az__dn']) / d, np.nan)
    out['amp'] = out['az_'] * out['aw']
    return out


def alpha_mask(have, rng, z, alpha_km=ALPHA_KM):
    """Cells inside the ALPHA SHAPE of the deposited samples.

    `griddata` fills the CONVEX hull, and the near-field dome is exactly where
    the real sample region is CONCAVE -- so no distance threshold or traced
    envelope can separate them properly: the dome interior sits close to the
    samples ringing it, and a traced underside costs a full 200-ray fan per
    azimuth.  The alpha shape is the non-convex hull, and it excludes the dome
    for free, from the samples already in the file.

    Construction is the standard one: Delaunay-triangulate the sample points,
    then drop every triangle whose CIRCUMRADIUS exceeds `alpha_km`.  A triangle
    spanning the dome has a huge circumcircle (nothing inside it), so it goes;
    triangles between neighbouring ray arcs are small and stay.  What remains
    is the region genuinely wrapped by the rays.

    `alpha_km` is a SAMPLING parameter, like MAX_DIST_KM before it: it must
    exceed the spacing between adjacent ray arcs (or the shape fragments into
    disconnected slivers) and stay below the width of the dome (or the dome is
    bridged).  It replaces both MAX_DIST_KM and the traced envelope -- one
    criterion instead of two, and no ray tracing at all.

    Returns a boolean array shaped like `have`.
    """
    from scipy.spatial import Delaunay

    pts_r, pts_z = np.nonzero(have)
    if len(pts_r) < 4:
        return np.zeros(have.shape, bool)
    P = np.column_stack([rng[pts_r], z[pts_z]])

    tri = Delaunay(P)
    s = tri.simplices
    a = P[s[:, 0]], P[s[:, 1]], P[s[:, 2]]
    # circumradius = abc / 4A
    la = np.linalg.norm(a[1] - a[2], axis=1)
    lb = np.linalg.norm(a[0] - a[2], axis=1)
    lc = np.linalg.norm(a[0] - a[1], axis=1)
    area = 0.5 * np.abs(np.cross(a[1] - a[0], a[2] - a[0]))
    with np.errstate(divide='ignore', invalid='ignore'):
        circum = la * lb * lc / (4.0 * area)
    keep = np.isfinite(circum) & (circum <= alpha_km)

    # Locate every grid cell in the surviving triangles.
    R, Z = np.meshgrid(rng, z, indexing='ij')
    tgt = np.column_stack([R.ravel(), Z.ravel()])
    which = tri.find_simplex(tgt)
    inside = (which >= 0) & keep[np.clip(which, 0, None)]
    return inside.reshape(have.shape)


def interpolate_plane(fa, ha, fb, hb, rng, z, log=False, alpha_km=ALPHA_KM):
    """Interpolate az and az+180 together, as one vertical plane.

    Azimuth a and a+180 are the two halves of a single plane through the
    epicentre.  Interpolated separately, each has the epicentre column as the
    EDGE of its grid, so the near-vertical rays there -- the densest, most
    strongly constrained part of the fan, and the part directly over the source
    -- sit on the convex-hull boundary where `griddata` has support on one side
    only.  Joined on a signed range axis the epicentre becomes interior: the
    escaping rays of one half constrain the fill of the other, which is correct
    because they are the same rays crossing the same plane.

    The join is measured, not assumed.  At the innermost range bin the two
    halves agree to a median 0.001 in log10 amplitude (~0.2%) and 0.5 s in
    travel time across az 0/45/90/97, so nothing physical is being smoothed
    away.  (The along-azimuth wind projection U sin(az) + V cos(az) does flip
    sign between the halves, but at 5 km range the rays are still near-vertical
    and it has barely acted.)

    Pairing matters with `alpha_km` and not without it.  Against the convex
    hull it is a no-op (0 cells gained): the deposit is dense at small range,
    so the hull boundary at range 0 is already well supported.  With the alpha
    shape one half of each pair gains 292-658 cells and the other 7-15, values
    identical wherever both are defined (median |dlog10| 0.00000) -- so pairing
    changes only WHICH cells exist, never what is in them.

    The gains are in the FAR field (out to 945-1945 km), not at the epicentre,
    so this is not the grid-edge effect one might expect.  It is a sampling
    interaction: the two halves have different far-field arc spacing, and
    triangulating them together lets the half whose arcs have spread past
    `alpha_km` borrow connectivity from the denser one.  Which half benefits
    therefore depends on the azimuth, and the effect would shrink if the fan
    were denser.  Each pairing also loses 8-14 cells near the seam.

    Half `b` is mirrored to negative range and the pair interpolated as one
    image; the two halves are then split back out.  Returns (out_a, out_b).
    """
    n_r = len(rng)
    # signed range: -rng[::-1] for half b, +rng for half a
    rr = np.concatenate([-rng[::-1], rng])
    field = np.concatenate([fb[::-1], fa], axis=0)
    have = np.concatenate([hb[::-1], ha], axis=0)

    out = interpolate(field, have, rr, z, log, alpha_km=alpha_km)
    return out[n_r:], out[:n_r][::-1]


def interpolate(field, have, rng, z, log=False, alpha_km=ALPHA_KM):
    """Linear interpolation over the deposited samples, cut to the alpha shape.

    `griddata` fills the CONVEX hull of the samples, which spans the near-field
    dome -- a region no ray enters, where the interpolant joins one arc back to
    itself and is pure invention.  The alpha shape is the non-convex hull, and
    the dome is exactly where the sample region is concave, so it is excluded
    by construction.  See `alpha_mask`.

    This replaced two earlier rules.  A distance mask (blank past 100 km from a
    real sample) cannot see the dome at all -- its interior is close to the
    samples ringing it, so only a notch of the dome's top came off.  A traced
    lower envelope was correct but cost a 200-ray fan per azimuth; the alpha
    shape gets the same region from the samples already in the file, in ~2.7 s.

    Amplitude should be interpolated with log=True: it spans decades, and
    linear interpolation of a near-exponential field undershoots badly.

    `alpha_km=None` keeps the whole convex hull.
    """
    m = np.isfinite(field) & have
    if log:
        m = m & (field > 0)
    if m.sum() < 3:
        return np.full(field.shape, np.nan)

    R, Z = np.meshgrid(rng, z, indexing='ij')
    # Normalisation must depend on the SPAN OF ONE HALF, not on the array's
    # extent: `interpolate_plane` passes a signed axis of twice the length, and
    # keying off rng[-1]-rng[0] would halve the range/altitude aspect ratio for
    # paired planes only, silently retriangulating them differently from the
    # unpaired case.  `sr` is therefore taken from the positive half-span.
    sr = float(max(abs(rng[0]), abs(rng[-1])))
    sz = float(z[-1] - z[0])
    pts = np.column_stack([R[m] / sr, Z[m] / sz])
    val = np.log10(field[m]) if log else field[m]
    tgt = np.column_stack([R.ravel() / sr, Z.ravel() / sz])
    out = griddata(pts, val, tgt, method='linear').reshape(field.shape)
    if log:
        out = 10.0 ** out

    if alpha_km is not None:
        out = np.where(alpha_mask(m, rng, z, alpha_km), out, np.nan)
    return out
