"""cidmodel -- forward model of coseismic ionospheric disturbances (CIDs).

An acoustic pulse radiated by an extended (finite-fault) or point source is
ray-traced through a windy, absorbing atmosphere, coupled into the ionosphere
along the geomagnetic field, and integrated along GNSS lines of sight into
synthetic slant TEC.  Written for the 28 March 2025 Mw 7.7 Myanmar earthquake.

Modules, in pipeline order:

    config        event, grid and physical constants
    paths         data-file locations (data/inputs, data/cubes, figures)
    geometry      model grid and local flat-Earth <-> geographic conversion
    atmosphere    NRLMSISE-00 sound speed, HWM14 winds, acoustic absorption
    ionosphere    IRI-2020 background Ne, IGRF field, ion velocity
    raytracing    ray equations, paraxial tracing, the RayCube container
    deposit       ray path deposit onto the (range, altitude) grid
    interpolation deposit -> ray cube
    sources       USGS finite-fault and binned point sources
    coupling      neutral velocity field (source superposition)
    continuity    dNe from the continuity equation
    stec_los      synthetic slant TEC along GNSS lines of sight
"""
__version__ = '1.0.0'
