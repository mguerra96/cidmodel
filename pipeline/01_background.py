"""Step 1 -- the static background on the model grid: Ne0 (IRI-2020) and B_hat (IGRF).

The first step of the pipeline.  Both fields depend only on the event time and
the grid, not on the source or the ray cube, so they are computed once and
read by 04_cube.py:

    data/inputs/ne0.npy       (nx, ny, nz)     background electron density, m^-3
    data/inputs/B_field.npy   (nx, ny, nz, 3)  IGRF unit field vector, ENU

Ne0 is evaluated on a 1 deg (lat, lon) subgrid and interpolated onto the model
columns (`ionosphere.background_ne`); B_hat is evaluated at every node
(`ionosphere.magnetic_field_grid`).

The HWM14 winds are the third background ingredient.  They are not built here:
the ray-tracing step reads them from the tracked cache
data/inputs/wind_profile_mandalay.npz (see `atmosphere.cached_wind_profile`).

Existing files are never overwritten unless --force is given; --out-dir writes
the pair somewhere else, e.g. to compare a rebuild against the files in use.

    python pipeline/01_background.py
    python pipeline/01_background.py --out-dir background_check
"""
import argparse
import os
import sys
import time

import numpy as np

from cidmodel import paths
from cidmodel.geometry import model_grid
from cidmodel.ionosphere import background_ne, magnetic_field_grid, profile_summary


def _atomic_save_npy(path, arr):
    tmp = path + '.tmp.npy'
    np.save(tmp, arr)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out-dir', default=None,
                    help='write ne0.npy / B_field.npy here instead of data/inputs/')
    ap.add_argument('--force', action='store_true',
                    help='overwrite existing output files')
    args = ap.parse_args()

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        ne0_path = os.path.join(args.out_dir, os.path.basename(paths.NE0))
        b_path = os.path.join(args.out_dir, os.path.basename(paths.B_FIELD))
    else:
        ne0_path, b_path = paths.NE0, paths.B_FIELD
    existing = [p for p in (ne0_path, b_path) if os.path.exists(p)]
    if existing and not args.force:
        sys.exit('refusing to overwrite (pass --force):\n  ' + '\n  '.join(existing))

    g = model_grid()
    print(f'grid {g.shape[0]} x {g.shape[1]} x {g.shape[2]}', flush=True)

    t0 = time.time()
    print('Ne0 from IRI-2020 ...', flush=True)
    ne0 = background_ne(g)
    s = profile_summary(ne0, g)
    print(f'  epicentral column: NmF2 {s["NmF2"]:.3e} m^-3, hmF2 {s["hmF2"]:.0f} km, '
          f'vTEC {s["vTEC_TECU"]:.1f} TECU  ({time.time() - t0:.0f} s)', flush=True)
    _atomic_save_npy(ne0_path, ne0)
    print(f'  -> {ne0_path}', flush=True)

    t0 = time.time()
    print('B_hat from IGRF ...', flush=True)
    B = magnetic_field_grid(g)
    _atomic_save_npy(b_path, B)
    print(f'  -> {b_path}  ({time.time() - t0:.0f} s)', flush=True)


if __name__ == '__main__':
    main()
