"""Electron-density perturbation from the linearised continuity equation.

The coupling step's v_i field drives dNe through continuity (Mikesell et al.
2019, eq. 5), and dNe is what a line of sight integrates into sTEC:

    divergence        full 3-D div of a vector field on the grid

The time integration of d(dNe)/dt = -div[Ne0 v_i] is done by 04_cube.py,
which writes dNe(r, t) straight to a memmap frame by frame.

Sign convention: continuity carries its physical minus (rate = -div[Ne0 v_i]),
paired with the compression-first N-wave source in coupling/sources.  The two
sign choices are consistent, so the calibrated observable is unchanged while
each piece stays textbook-correct (TEC increase on arrival; Heki & Ping 2005).

The LOS integration -- per-arc IPP-height reference, time-varying satellite
geometry -- lives in stec_los.
"""
import numpy as np


def divergence(field, grid):
    """
    Full 3-D divergence of a vector field on the model grid.

    Centred differences interior, one-sided at boundaries (np.gradient).
    Mikesell et al. (2019) appendix B use spherical coordinates; on a 1400 km
    domain the flat-Earth Cartesian form differs by O(L/R_earth) ~ 10%, small
    against the other approximations (nearest-neighbour ray lookup, 20 km
    horizontal spacing).  IonoSeis stresses the FULL 3-D divergence matters:
    the radial term alone is valid only in the far field where the wave is
    locally plane, and lateral Ne variation changes the LOS integral.

    field : (nx, ny, nz, 3) ENU components.
    grid  : Grid (axes in km; converted to metres here).
    Returns (nx, ny, nz), divergence in [field]/m.
    """
    dx = np.gradient(field[..., 0], grid.x * 1e3, axis=0)
    dy = np.gradient(field[..., 1], grid.y * 1e3, axis=1)
    dz = np.gradient(field[..., 2], grid.z * 1e3, axis=2)
    return dx + dy + dz




