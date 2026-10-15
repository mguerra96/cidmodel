"""Coordinate transforms and the model grid.

Flat-Earth projection centred on the epicentre (x east, y north, z altitude),
and the `Grid` container.
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

    def meshgrid(self):
        """Return XX, YY, ZZ each of shape (nx, ny, nz)."""
        return np.meshgrid(self.x, self.y, self.z, indexing='ij')


def model_grid(half_km=HORIZ_HALF_KM, step_km=HORIZ_STEP_KM,
               alt_min=ALT_MIN_KM, alt_max=ALT_MAX_KM, alt_step=ALT_STEP_KM):
    """Build the standard model grid (141 x 141 x 281 nodes)."""
    x = np.arange(-half_km, half_km + step_km, step_km, dtype=float)
    y = np.arange(-half_km, half_km + step_km, step_km, dtype=float)
    z = np.arange(alt_min, alt_max + alt_step, alt_step, dtype=float)
    return Grid(x=x, y=y, z=z)


