"""Step 6 -- per-source sTEC basis, for the source-weighting sensitivity tests.

The chain source -> v_n -> dNe -> sTEC is linear in each source's weight, so the
finite-fault synthetic is  sum_k w_k * S_k,  where S_k is the UNSCALED sTEC of
merged source k alone at unit weight.  With every S_k on disk, a re-weighting
(uniform weights, random polarities, ...) is a linear combination plus an A0
refit, not a rebuild.

For each merged source k: build dne_binned_basisKK (04_cube.py --only k),
integrate it on every arc (05_synth_stec.synth_for_source, unscaled), then
DELETE the 3 GB cube.  Resumable: sources already in the output are skipped.
Ends with a self-check that sum_k w_k S_k reproduces the canonical
`synth_stec` column divided by its A0.

    python pipeline/06_source_basis.py
    python pipeline/06_source_basis.py --only 0 3      # a subset (testing)

Output (data/cubes/, derived, gitignored):
    stec_basis.parquet          ArcID, t_s, s00..sNN  (row-aligned with
                                tec_observations.parquet)
    stec_basis_sources.npz      x, y, t_rupture_s, weight of each source
"""
import argparse
import importlib.util
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from cidmodel import paths
from cidmodel.geometry import model_grid
from cidmodel.sources import binned_fault_sources

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = paths.cube('stec_basis.parquet')

# 05_synth_stec.py starts with a digit, so it is loaded by path.
_spec = importlib.util.spec_from_file_location(
    'synth_stec', os.path.join(HERE, '05_synth_stec.py'))
synth = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(synth)


def check(basis, w, obs):
    """sum_k w_k S_k must equal synth_stec / A0 of the canonical fault cube."""
    with np.load(paths.require_cube('', 'dne_binned')) as d:
        a0 = float(d['A0'])
    cols = [f's{k:02d}' for k in range(len(w))]
    total = basis[cols].to_numpy() @ w
    ref = obs['synth_stec'].to_numpy(float) / a0
    m = np.isfinite(ref)
    rel = np.nanmax(np.abs(total[m] - ref[m])) / np.nanmax(np.abs(ref[m]))
    print(f'self-check: max |sum w_k S_k - synth_stec/A0| = {rel:.2e} of the peak')
    assert np.array_equal(np.isfinite(total), m), 'NaN pattern differs'
    assert rel < 1e-3, 'basis does not reproduce the fault synthetic'


def main(only=None):
    src = binned_fault_sources(model_grid(), verbose=False)
    w = np.array([s['weight'] for s in src])
    np.savez(paths.cube('stec_basis_sources.npz'),
             **{k: np.array([s[k] for s in src])
                for k in ('x', 'y', 't_rupture_s', 'weight')})

    obs = pd.read_parquet(paths.TEC_OBS)
    if os.path.exists(OUT):
        basis = pd.read_parquet(OUT)
        assert (basis['ArcID'].equals(obs['ArcID'])
                and basis['t_s'].equals(obs['t_s'])), \
            'observation table changed since the basis was started: delete ' + OUT
    else:
        basis = obs[['ArcID', 't_s']].copy()

    for k in (range(len(src)) if only is None else only):
        col = f's{k:02d}'
        if col in basis:
            continue
        tag = f'_basis{k:02d}'
        print(f'\n=== source {k + 1}/{len(src)} (w={w[k]:.4f}) ===', flush=True)
        subprocess.run([sys.executable, '-u', os.path.join(HERE, '04_cube.py'),
                        '--sources', 'binned', '--only', str(k),
                        '--out-tag', tag], check=True)
        basis[col] = synth.synth_for_source(obs, 'binned', tag).to_numpy()
        tmp = OUT + '.tmp'
        basis.to_parquet(tmp, index=False)
        os.replace(tmp, OUT)
        for ext in ('.npy', '.npz'):    # only after the column is safely written
            os.remove(paths.cube(f'dne_binned{tag}{ext}'))

    if all(f's{k:02d}' in basis for k in range(len(src))):
        check(basis, w, obs)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--only', type=int, nargs='+', default=None,
                    help='build only these source indices')
    main(ap.parse_args().only)
