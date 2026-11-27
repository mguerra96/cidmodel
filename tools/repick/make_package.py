"""Build a BLIND re-pick package for an independent picker (revision 1, R1-M4).

Draws a stratified subset of the picked arcs (off-rupture arcs + a fixed number
per along-rupture epicentral-range bin), anonymises them (R01, R02, ...) and
shuffles them.  The package holds the raw sTEC only: no original picks, no
synthetics, no station, satellite, range or azimuth.  The key (anon -> ArcID)
is written OUTSIDE the package, so the picker never sees it.

With the defaults this rebuilds exactly the 48-arc subset re-picked for the
paper (data/inputs/repick_pica_key.csv).

    python tools/repick/make_package.py --out repick_package --key repick_key.csv
    python repick_package/repick.py          # the picker; writes picks.csv there
    python tools/repick/compare.py --picks repick_package/picks.csv --key repick_key.csv
"""
import argparse
import os
import shutil

import numpy as np
import pandas as pd

from cidmodel import paths

HERE = os.path.dirname(os.path.abspath(__file__))
BINS = [0, 500, 800, 1100, 9999]          # epicentral range bins, km


def main(out, key, seed=20261124, n_off=12, n_per_bin=9):
    p = pd.read_parquet(paths.ARRIVAL_PICKS)
    obs = pd.read_parquet(paths.TEC_OBS, columns=['ArcID', 't_s', 'GFLC'])
    dt = obs.groupby('ArcID').t_s.apply(lambda s: s.diff().median()).rename('dt_s')
    p = p.merge(dt, on='ArcID')
    off = ~p.az_deg.between(100, 220)       # along-rupture sector = azimuth 100-220 deg
    p['stratum'] = np.where(off, 'off', 'along_' + pd.cut(p.dist_km, BINS).astype(str))

    take = [p[off].sample(n_off, random_state=seed)]
    for _, g in p[~off].groupby('stratum'):
        take.append(g.sample(min(n_per_bin, len(g)), random_state=seed))
    sub = pd.concat(take).sample(frac=1, random_state=seed).reset_index(drop=True)
    sub['anon'] = [f'R{i + 1:02d}' for i in range(len(sub))]
    print(f'{len(sub)} arcs, {sub.station.nunique()} receivers')
    print(sub.stratum.value_counts().sort_index().to_string())

    if os.path.exists(out):
        raise SystemExit(f'{out} exists: remove it first (it may hold picks)')
    os.makedirs(out)
    arcs = obs[obs.ArcID.isin(sub.ArcID)].merge(sub[['ArcID', 'anon']], on='ArcID')
    arcs.rename(columns={'GFLC': 'stec_tecu'})[['anon', 't_s', 'stec_tecu']] \
        .to_parquet(os.path.join(out, 'arcs.parquet'), index=False)
    pd.DataFrame({'anon': sub.anon}).to_csv(os.path.join(out, 'order.csv'), index=False)
    shutil.copy(os.path.join(HERE, 'repick.py'), out)
    shutil.copy(os.path.join(HERE, 'PICKER_README.md'), os.path.join(out, 'README.md'))
    sub[['anon', 'ArcID', 'stratum', 'dt_s', 'az_deg', 'dist_km']].to_csv(key, index=False)
    print('package ->', out, '| key ->', key)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', default='repick_package', help='package folder (must not exist)')
    ap.add_argument('--key', default='repick_key.csv',
                    help='anon -> ArcID key; keep it away from the picker')
    ap.add_argument('--seed', type=int, default=20261124)
    ap.add_argument('--n-off', type=int, default=12, help='off-rupture arcs')
    ap.add_argument('--n-per-bin', type=int, default=9, help='along-rupture arcs per range bin')
    a = ap.parse_args()
    main(a.out, a.key, a.seed, a.n_off, a.n_per_bin)
