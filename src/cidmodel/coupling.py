"""Neutral velocity field: source superposition through the ray cube.

The step that turns the source list + ray cube into the neutral particle
velocity v_n(r, t) at one instant, by superposing every fault element
(Mikesell et al. 2019, eq. 7):

    v_n = sum_i  weight_i * n_wave(t - t_rupture_i - t_w_i) * A_i * k_hat_i

Implemented as a fused @njit(parallel) kernel, `neutral_velocity_field_numba`
(it was validated against a pure-numpy implementation of the same formula).
The kernel exists because 88% of a frame is the per-source arithmetic
after the cube lookup, each numpy op allocating a ~45 MB temporary, single-
threaded.  The kernel fuses that arithmetic and the 3-component accumulation
into one parallel pass with no temporaries; the cube lookup itself stays numpy.

Pulse width -- Mikesell linear broadening
-----------------------------------------
    sig_eff = B_LIN * t_w          (linear in travel time, through the ORIGIN)

Mikesell et al. (2019) use sig = b*t.  This is the ONLY law here: the sqrt
"old-age" form used until 2026-09-18 was removed after it failed on its own
terms (the exact formula it used is preserved at the bottom of this docstring
so the pre-2026-09-18 cubes remain reconstructible).

Why linear, measured on the 249 hand picks:

  * Fitting T = A*t + C to binned medians of picked period against time after
    rupture gives C = +11 +/- 92 s -- 0.1 sigma from zero -- so the origin
    constraint costs nothing.  Anchoring instead at the 150 km nonlinear onset
    (t0 = 454 s) is rejected outright (R^2 < 0).
  * The sqrt form cannot fit those medians with a PHYSICAL pulse width: it
    needs sigma < 0 (-57 s at t_w0 = 454, -6.6 s at 631) to straighten itself
    into a line, which is the data saying it wants a line.  RMSE on the binned
    medians: linear 28.3 s, sqrt with sigma >= 0 enforced 40.0 s, the old p80
    constants as coded 102.8 s.
  * A quadratic term is rejected by AICc (worse by 5.6; it bends the line by
    -15 s against a 28 s residual), and the near-field refit (d < 600 km)
    agrees with the full-range slope to 0.16 sigma -- no curvature.

Both candidate mechanisms are linear in t to first order, so the FORM is far
better justified than the coefficient.  Acoustic-gravity dispersion near the
cutoff (f_a = gamma*g/2c = 1.17 mHz at 300 km, cutoff period 856 s, with the
observed periods sitting just below it) predicts an amplitude-INDEPENDENT rate
of 0.04-0.55 s/s depending on band; nonlinear steepening predicts an
amplitude-SET rate of 0.07-0.49 s/s for particle velocities of 50-350 m/s.
Both bracket the measurement, so the rate alone does not discriminate them.
They are separable by whether the broadening correlates with per-arc amplitude
at matched range -- untested.

B_LIN = 0.0859 is fitted CONSISTENTLY: binned medians of picked period against
the CUBE's own t_w, which is the variable sig_eff is evaluated at.  An earlier
fit against the observed arrival time (t_arr - t_rup) gave 0.1562 and produced
visibly over-wide far-field waveforms -- the cube runs ~1.86x slower than the
picked arrivals, so fitting on one abscissa and applying on the other inflated
sigma by exactly that factor.  0.1562/1.86 = 0.084, and the consistent refit
independently gives 0.0859.

KNOWN LIMITATION: the picks are dominated by the Rayleigh-forced TID, whose
period is set by the surface wave and should not broaden with range at all;
27% of them exceed the local acoustic cutoff period at 150 km.  The residual
model period is ~1.38x the observed at every range, which no single b fixes
(lowering b would break the amplitude, since the two couple through sig^-1.5).
That residual most likely traces to the same 1.86x travel-time gap.

Removed sqrt law, for reconstructing pre-2026-09-18 cubes:

    sig_eff = sigma + b*sqrt(max(t_w - t_w0, 0))
    with the "p80" constants sigma = 16.3, b = 2.94, t_w0 = 631.0

v_n is returned in arbitrary amplitude units; the free source scale A_0 is
fitted against the observations afterwards.  The whole chain from dNe to
synthetic sTEC is LINEAR in that scale (verified to 3.5e-08 relative), so it is
applied once at analysis time and never enters a cube.
"""
import math

import numpy as np
from numba import njit, prange

SQRT2 = math.sqrt(2.0)
PI_QUARTER = math.pi ** 0.25

# The one broadening coefficient: sig_eff = B_LIN * t_w (Mikesell form).
# Fitted 2026-09-18 against the CUBE's t_w -- the same variable sig_eff is
# evaluated at -- on binned medians of picked period, d <= 1200 km (the bins
# beyond that reverse as low-SNR arcs lose their long-period tail).
# period = 4*sig for the Gaussian-derivative pulse, so B_LIN = slope/4.
B_LIN = 0.0859


# NOTE: fastmath is OFF on purpose.  fastmath lets LLVM assume no NaN/Inf, which
# (a) can optimise the math.isfinite() masking guards away -> NaN leaks from
# fan-edge cells into v_n and then explodes through the divergence, and (b) buys
# little here since the kernel is memory-bandwidth bound, not compute bound.
# cache is also OFF: a stale on-disk cache can silently run old kernel logic.
@njit(parallel=True, cache=False)
def _accumulate_source(t_w, kx, ky, kz, a_prop, t_eval, t_rupture,
                       b_lin, weight, vnx, vny, vnz):
    """Fused per-source contribution, accumulated in place into vnx/vny/vnz.

    All arrays are FLAT (n_cells,).  A cell with non-finite t_w or k is skipped
    (contributes exactly 0).

    `b_lin` is passed in rather than read from the module global so the kernel
    stays a pure function of its arguments (numba would freeze a global at
    compile time, and a later edit to B_LIN would silently not take effect).
    """
    n = t_w.shape[0]
    for i in prange(n):
        tw = t_w[i]
        kxi = kx[i]
        kyi = ky[i]
        kzi = kz[i]
        # A masked cell must contribute EXACTLY 0.
        # Guard all three k components so a NaN in ky/kz cannot leak in.
        if not (math.isfinite(tw) and math.isfinite(kxi)
                and math.isfinite(kyi) and math.isfinite(kzi)):
            continue
        # Mikesell sig = b*t_w, through the origin (no t_w0, no floor).
        sig = b_lin * tw
        # compression-first N-wave (negated Gaussian derivative)
        tau = t_eval - (t_rupture + tw)
        amp = -(SQRT2 / (sig ** 1.5 * PI_QUARTER)) * tau * \
            math.exp(-tau * tau / (2.0 * sig * sig))
        ap = a_prop[i]
        if math.isfinite(ap):
            amp *= ap
        else:
            amp = 0.0
        amp *= weight
        vnx[i] += amp * kxi
        vny[i] += amp * kyi
        vnz[i] += amp * kzi


def neutral_velocity_field_numba(grid, cube, sources, t_eval,
                                 bilinear=True, verbose=False):
    """Neutral particle velocity v_n(r, t) on the grid at one instant.

    Returns vn : (nx, ny, nz, 3) ENU, arbitrary amplitude units -- the source
    scale A_0 is fitted downstream, at analysis time.  Fuses the post-lookup
    arithmetic + accumulation into one jitted pass per source.  Pulse width is
    sig_eff = B_LIN * t_w, with no per-call override.
    """
    nx, ny, nz = grid.shape
    n_cells = nx * ny * nz
    vnx = np.zeros(n_cells, dtype=np.float64)
    vny = np.zeros(n_cells, dtype=np.float64)
    vnz = np.zeros(n_cells, dtype=np.float64)

    XX, YY = np.meshgrid(grid.x, grid.y, indexing='ij')

    for si, src in enumerate(sources):
        dx = XX - src['x']
        dy = YY - src['y']
        rng = np.hypot(dx, dy)
        az = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0

        t_w, kx, ky, kz, a_prop = cube.lookup(az, rng, bilinear=bilinear)
        _accumulate_source(
            np.ascontiguousarray(t_w).ravel(),
            np.ascontiguousarray(kx).ravel(),
            np.ascontiguousarray(ky).ravel(),
            np.ascontiguousarray(kz).ravel(),
            np.ascontiguousarray(a_prop).ravel(),
            float(t_eval), float(src['t_rupture_s']),
            float(B_LIN), float(src['weight']),
            vnx, vny, vnz)
        if verbose and (si + 1) % 100 == 0:
            print(f"    {si+1}/{len(sources)} elements")

    vn = np.empty((nx, ny, nz, 3), dtype=np.float64)
    vn[..., 0] = vnx.reshape(nx, ny, nz)
    vn[..., 1] = vny.reshape(nx, ny, nz)
    vn[..., 2] = vnz.reshape(nx, ny, nz)
    return vn
