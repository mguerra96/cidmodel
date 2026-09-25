"""Model configuration: event parameters, grid dimensions, ray-fan sampling,
and domain-of-validity limits.

Single fixed resolution (the former "hi-res" grid is now THE grid): 20 km
horizontal / 2.5 km vertical, +/-1400 km, 100-800 km altitude.  There is no
low-res variant.
"""
import datetime
import os

import numpy as np

# --- Event -----------------------------------------------------------------
EQ_LAT  = 22.001
EQ_LON  = 95.925
EQ_TIME = datetime.datetime(2025, 3, 28, 6, 20, 56)

R_EARTH_KM = 6371.0

# --- Neutral atmosphere (NRLMSISE-00 + HWM14) ------------------------------
# Space-weather indices for the event epoch and gas constants for the sound
# speed c = sqrt(gamma R T / M).
GAMMA  = 1.4          # ratio of specific heats
R_GAS  = 8.314        # universal gas constant (J/mol/K)
F107   = 150.0        # solar 10.7 cm flux index
AP     = 4.0          # geomagnetic activity index
# Mean molecular masses (kg/mol): He, O, N2, O2, Ar, -, H, N
MOL_MASS = np.array([4, 16, 28, 32, 40, 0, 1, 14]) * 1e-3

# HWM14 horizontal wind model.  Unlike NRLMSISE-00 (`nrlmsise00`), IRI-2020
# (`iri20py`) and IGRF (`ppigrf`), which are all pip-installable, HWM14 has
# no usable distribution: the PyPI sdist builds through
# `numpy.distutils` (removed in numpy >= 1.26), calls pip/conda from setup.py,
# and ships without its .dat/.bin data files.  It therefore stays in a separate
# conda env (Python 3.11) and is driven by subprocess.
#
# That would make the pipeline unrunnable off this machine, except that HWM14
# is needed exactly ONCE per event -- a single 1-D profile U(z), V(z) at the
# epicentre, ~6 KB.  `atmosphere.build_atmosphere` caches it under data/inputs/
# and only shells out when the cache is missing or stale, so everything else
# runs anywhere with pip alone.  Regenerating for a NEW event needs conda; see
# `atmosphere.query_hwm14_profile`.
#
# The interpreter of that env is given by the HWM14_PY environment variable
# (e.g. .../envs/hwm14/python); nothing machine-specific is baked in here.
HWM14_PY  = os.environ.get('HWM14_PY')
# Cached wind profile for this event (written by
# `atmosphere.cached_wind_profile` on a cache miss).
WIND_CACHE = 'wind_profile_mandalay.npz'

# --- Ray-tracing atmosphere -------------------------------------------------
RAY_ATM_STEP_KM = 2.0      # atmosphere profile spacing (km)
RAY_STEP_M      = 100      # max solve_ivp integration step (m)
RAY_T_MAX       = 3600     # max integration time (s), ample for <=34 deg rays

# --- Model grid ------------------------------------------------------------
# 141 x 141 x 281 nodes.
ALT_MIN_KM    = 100.0
ALT_MAX_KM    = 800.0
ALT_STEP_KM   = 2.5
HORIZ_HALF_KM = 1400.0
HORIZ_STEP_KM = 20.0

# Ray-tracing ceiling must sit above the grid top or rays terminate early.
RAY_CEILING_KM = ALT_MAX_KM + 20.0

# Ceiling of the tracer's own vertical profile.  This was 350 km on the
# reasoning that the sound-speed gradient goes linear there, so upgoing rays
# stop refracting and integrating higher adds cost without bending.  That is
# true of the RAY PATH, but the profile is also what the absorption integral
# alpha(z) is evaluated on, and the tracer runs rays to RAY_CEILING_KM (820 km)
# regardless -- so every ray reaching above 350 km accumulated tau against an
# alpha extrapolated up to 470 km beyond its validity.  Measured at az 90, the
# descending rays at takeoff 27-31 deg carried tau ~1000-3400 of which >99.9%
# came from above 350 km: numerology, not absorption.  Tying the profile to the
# grid top makes alpha computed rather than extrapolated everywhere a ray goes.
RAY_ATM_MAX_KM  = ALT_MAX_KM

# --- Ray-fan sampling ------------------------------------------------------
# Innermost take-off zenith 0.25 deg: near-vertical rays reach the source axis
# so the fan has no on-axis un-illuminated cylinder (the old x=0 hole).
TAKEOFF_RANGE = (0.25, 34.0)     # zenith angle limits (deg)
# The ceiling must clear the turning-point cutoff, which depends on azimuth
# through the HWM14 winds.  This was briefly raised to 40 deg on cutoffs of
# 35-37 deg measured at az 90-165 -- but those came from the pre-2026-08-03
# ray equations, which over-responded to wind.  With the
# corrected equations the cutoff runs 21.6-26.1 deg over all azimuths, against
# a flat-Earth Snell bound of 21.2-24.3 deg, so 34 deg carries ample margin and
# nothing was ever clipped.  Reverted.
