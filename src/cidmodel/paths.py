"""Central data-file locations.

All scripts resolve inputs/outputs through here so filenames live in ONE place.
Paths are absolute, derived from this file's location, so scripts work no matter
the current working directory.

Layout::

    <repo>/data/inputs/   raw, never regenerated (fault model, background state,
                          GNSS arc CSVs, ShakeMaps/pgv shapefile)
    <repo>/data/cubes/    derived, rebuildable (ray table/cube, dNe cubes,
                          per-arc synthetic sTEC)
    <repo>/figures/       output PNGs -- publication artifacts only
    <repo>/scratch/       throwaway diagnostics (figures/ and scripts/),
                          gitignored; see `scratch_fig`
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))       # <repo>/src/cidmodel
ROOT = os.path.dirname(os.path.dirname(_HERE))           # <repo>

INPUTS  = os.path.join(ROOT, 'data', 'inputs')
CUBES   = os.path.join(ROOT, 'data', 'cubes')
# Figure output; CIDMODEL_FIGURES redirects it (e.g. to regenerate the paper
# figures somewhere else and compare).
FIGURES = os.environ.get('CIDMODEL_FIGURES') or os.path.join(ROOT, 'figures')
SCRATCH = os.path.join(ROOT, 'scratch')


def inp(name):
    """Absolute path to a raw input file."""
    return os.path.join(INPUTS, name)


def cube(name):
    """Absolute path to a derived-cube file."""
    return os.path.join(CUBES, name)


def fig(name):
    """Absolute path to an output figure.

    `figures/` is for PUBLICATION artifacts -- output of a `plots/*.py` script
    that states its generation and stamps it into the filename.  Diagnostic
    plots made to understand what is happening go to `scratch_fig` instead, so
    exploratory work never overwrites a figure the paper cites.
    """
    os.makedirs(FIGURES, exist_ok=True)
    return os.path.join(FIGURES, name)


def scratch_fig(name, date=None):
    """Absolute path to an EXPLORATORY figure, date-prefixed.  Not an artifact.

    Diagnostics answer a question on a particular day, so the name carries one:
    `scratch_fig('aw_range_slope.png')` -> `scratch/figures/20260917_aw_range_slope.png`.
    That keeps them sortable and makes a stale one obvious, which matters
    because nothing else in `scratch/` is maintained.  Pass `date` (YYYYMMDD) to
    label a plot as of a day other than today.  Already-prefixed names are left
    alone, so re-running a script does not stack prefixes.

    The directory is created on demand and is gitignored: everything under
    `scratch/` is disposable by construction.  A diagnostic worth keeping gets
    REWRITTEN as a `plots/*.py` script with a required `--gen` -- never copied,
    since a one-off's hardcoded paths are exactly what PATHS-1/PATHS-2 removed.
    """
    import datetime
    stamp = date or datetime.date.today().strftime('%Y%m%d')
    base = os.path.basename(name)
    if not (len(base) > 8 and base[:8].isdigit() and base[8] == '_'):
        name = os.path.join(os.path.dirname(name), '%s_%s' % (stamp, base))
    dest = os.path.join(SCRATCH, 'figures', name)
    d = os.path.dirname(dest)
    if not os.path.isdir(d):
        os.makedirs(d)
    return dest


# --- Canonical input files -------------------------------------------------
FSP        = inp('myanmar_m7.7_usgs_finitefault.fsp')  # USGS finite fault
NE0        = inp('ne0.npy')                # IRI-2020 background Ne (m^-3)
B_FIELD    = inp('B_field.npy')            # IGRF unit field vectors
SHAKEMAPS  = inp('ShakeMaps')              # USGS ShakeMap shapefiles (dir)
# The observations: the data starting point of the pipeline.  Both tables are
# built from the raw GNSS arcs by the preprocessing step, which is not part of
# this repository.  Both hold ONLY the arcs
# carrying a usable hand pick, so they always join on ArcID.
TEC_OBS    = inp('tec_observations.parquet')   # per-sample sTEC + LOS geometry
ARRIVAL_PICKS = inp('arrival_picks.parquet')    # one row per picked arrival

# --- Derived cubes -----------------------------------------------------------
# Each kind has ONE canonical file in data/cubes/ (ray_cube.npz, dne_binned.npz,
# dne_point.npz -- the ones deposited on Zenodo), and every script defaults to
# it.  A tag ('_v3') selects a side-by-side generation instead.  There are no
# RAY_CUBE / DNE_* path constants: callers resolve through `require_cube`,
# which reports what the file actually contains (`describe_cube`) and lists
# what is on disk when the requested cube is missing.  Constants of that kind
# rotted once already -- until 2026-09-17 they pointed at a pre-fix generation
# long after the rebuild, and nothing in the output said so.
#
# dNe cubes are a pair: a float32 memmap (.npy, the heavy frames) and a small
# .npz sidecar carrying A0/x/y/z/times, the memmap's filename (`dne_memmap`)
# and, since 2026-09-17, the ray cube it was integrated against (`ray_cube`).
# The sidecar is what the sTEC scan loads; it points at the memmap.

# Cube-kind -> (filename prefix, extension) used to resolve a bare tag such as
# '_v3' and to list the candidates when resolution fails.
_CUBE_KINDS = {
    'ray':        ('ray_cube',    '.npz'),
    'deposit':    ('deposit_raw', '.npz'),
    'dne_binned': ('dne_binned',  '.npz'),
    'dne_point':  ('dne_point',   '.npz'),
}


def list_cubes(kind):
    """Filenames of every `kind` cube present in data/cubes/, newest first."""
    prefix, ext = _CUBE_KINDS[kind]
    if not os.path.isdir(CUBES):
        return []
    found = [f for f in os.listdir(CUBES)
             if f.startswith(prefix) and f.endswith(ext)]
    return sorted(found, key=lambda f: os.path.getmtime(cube(f)), reverse=True)


def describe_cube(name):
    """One-line provenance for a cube file, read from the file itself.

    dNe sidecars record the ray cube they were built from, so this reports
    what a cube CONTAINS rather than what its name claims -- the distinction
    that made the 2026-09-17 mislabelling invisible.  Returns '' when the file
    says nothing, rather than guessing from the filename.
    """
    import numpy as np
    try:
        with np.load(cube(name), allow_pickle=False, mmap_mode='r') as d:
            bits = []
            if 'ray_cube' in d:
                bits.append(f'from {d["ray_cube"]}')
            if 'freq_hz' in d:
                bits.append(f'{float(d["freq_hz"])*1e3:.0f} mHz')
            return ', '.join(bits)
    except Exception:
        return ''


def gen_suffix(gen):
    """Filename suffix for a generation: '' stays empty, '_v3'/'v3' -> '_v3'.

    Keeps figure names clean for the final (unsuffixed) model while a
    side-by-side generation still stamps itself into the filename.
    """
    return '' if not gen else (gen if gen.startswith('_') else '_' + gen)


def synth_columns(gen='', obs=None):
    """(fault, point) synthetic-sTEC column names for a model generation.

    `05_synth_stec.py --suffix X` writes `synth_stecX` /
    `synth_stec_pointX`, so a generation is a column suffix.  The FINAL model
    (2 mHz, A0 5.7) carries no suffix, so the default empty `gen` names the
    published columns; pass a suffix only while a second generation is being
    evaluated side by side, as `_v3`/`_v4` were in September 2026.

    Pass `obs` to check the columns exist, and get the available generations
    listed instead of a KeyError deep in a plot.
    """
    suffix = gen_suffix(gen)
    cols = (f'synth_stec{suffix}', f'synth_stec_point{suffix}')
    if obs is not None:
        missing = [c for c in cols if c not in obs.columns]
        if missing:
            have = sorted({c.split('synth_stec')[-1].replace('_point', '')
                           for c in obs.columns if c.startswith('synth_stec')})
            raise SystemExit(
                f'no {missing} column(s) in the observations table.\n'
                f'generations present: {have or "(none)"}\n'
                f'build one with:  python pipeline/05_synth_stec.py '
                f'--tag {suffix} --suffix {suffix}')
    return cols


def require_cube(arg, kind='ray'):
    """Resolve a cube argument to an absolute path.

    `arg` may be None or '' (the canonical cube of that kind, e.g.
    dne_binned.npz), a full path, a bare filename, or a tag ('_v3' / 'v3')
    which is expanded to the `kind`'s canonical name plus the tag.  A cube that
    does not exist raises SystemExit with the list of cubes actually on disk.
    """
    prefix, ext = _CUBE_KINDS[kind]

    def _fail(msg):
        lines = [msg, '', f'{kind} cubes in {CUBES}:']
        names = list_cubes(kind)
        if not names:
            lines.append('  (none)')
        for n in names:
            why = describe_cube(n)
            lines.append(f'  {n}' + (f'   [{why}]' if why else ''))
        raise SystemExit('\n'.join(lines))

    if not arg:                                         # canonical cube
        path = cube(f'{prefix}{ext}')
        if not os.path.exists(path):
            _fail(f'No canonical {kind} cube: {path}')
        return path

    if os.path.sep in arg or (os.path.altsep and os.path.altsep in arg):
        path = os.path.abspath(arg)                     # explicit path
    elif arg.endswith(ext):
        path = cube(arg)                                # bare filename
    else:
        tag = arg if arg.startswith('_') else '_' + arg
        path = cube(f'{prefix}{tag}{ext}')              # tag -> canonical name

    if not os.path.exists(path):
        _fail(f'No such {kind} cube: {arg}  ->  {path}')
    return path
