"""Coordinate transforms and the model grid.

Flat-Earth projection centred on the epicentre (x east, y north, z altitude),
the `Grid` container, and the domain-of-validity mask.
"""
from dataclasses import dataclass

import numpy as np

from .config import (EQ_LAT, EQ_LON, R_EARTH_KM, ALT_MIN_KM, ALT_MAX_KM,
                    ALT_STEP_KM, HORIZ_HALF_KM, HORIZ_STEP_KM)


def geo_to_km(lat, lon, lat0=EQ_LAT, lon0=EQ_LON):
    """Flat-Earth projection centred on the epicentre. Returns (x_km, y_km)."""
    x = (np.asarray(lon) - lon0) * np.pi / 180.0 * R_EARTH_KM * np.cos(np.radians(lat0))
    y = (np.asarray(lat) - lat0) * np.pi / 180.0 * R_EARTH_KM
    return x, y


def km_to_geo(x_km, y_km, lat0=EQ_LAT, lon0=EQ_LON):
    """Inverse of geo_to_km. Returns (lat, lon) in degrees."""
    lat = lat0 + np.asarray(y_km) / (np.pi / 180.0 * R_EARTH_KM)
    lon = lon0 + np.asarray(x_km) / (np.pi / 180.0 * R_EARTH_KM * np.cos(np.radians(lat0)))
    return lat, lon


@dataclass
class Grid:
    """
    Regular 3-D grid in local flat-Earth coordinates centred on the epicentre.

    Axes: x east (km), y north (km), z altitude (km).  Field arrays are indexed
    [ix, iy, iz] throughout.  Build the standard model grid with `model_grid()`.
    """
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    @property
    def shape(self):
        return (len(self.x), len(self.y), len(self.z))

    @property
    def n_nodes(self):
        return len(self.x) * len(self.y) * len(self.z)

    def meshgrid(self):
        """Return XX, YY, ZZ each of shape (nx, ny, nz)."""
        return np.meshgrid(self.x, self.y, self.z, indexing='ij')

    def horizontal_polar(self):
        """
        Range (km) and azimuth (deg CW from N) of each column from the origin.
        Returns two (nx, ny) arrays.
        """
        XX, YY = np.meshgrid(self.x, self.y, indexing='ij')
        rng = np.hypot(XX, YY)
        az = (np.degrees(np.arctan2(XX, YY)) + 360.0) % 360.0
        return rng, az

    def describe(self):
        nx, ny, nz = self.shape
        return (f"Grid {nx} x {ny} x {nz} = {self.n_nodes:,} nodes\n"
                f"  x: {self.x.min():+.0f} .. {self.x.max():+.0f} km "
                f"(step {self.x[1]-self.x[0]:.0f})\n"
                f"  y: {self.y.min():+.0f} .. {self.y.max():+.0f} km "
                f"(step {self.y[1]-self.y[0]:.0f})\n"
                f"  z: {self.z.min():.0f} .. {self.z.max():.0f} km "
                f"(step {self.z[1]-self.z[0]:.0f})")


def model_grid(half_km=HORIZ_HALF_KM, step_km=HORIZ_STEP_KM,
               alt_min=ALT_MIN_KM, alt_max=ALT_MAX_KM, alt_step=ALT_STEP_KM):
    """Build the standard model grid (141 x 141 x 281 nodes)."""
    x = np.arange(-half_km, half_km + step_km, step_km, dtype=float)
    y = np.arange(-half_km, half_km + step_km, step_km, dtype=float)
    z = np.arange(alt_min, alt_max + alt_step, alt_step, dtype=float)
    return Grid(x=x, y=y, z=z)


