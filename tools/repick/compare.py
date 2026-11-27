"""Compare a blind re-pick with the original picks (data/inputs/arrival_picks.parquet).

Differences re-pick minus original for onset, crest and trough: median with a
receiver-clustered bootstrap 95% CI, median absolute difference, fraction
within 30 s (the sampling interval of most arcs) and 60 s, and the largest
crest differences.  Defaults = the paper's re-pick (E. Pica, 48 arcs).

    python tools/repick/compare.py
    python tools/repick/compare.py --picks repick_package/picks.csv --key repick_key.csv
"""
import argparse

import numpy as np
import pandas as pd

from cidmodel import paths


def main(picks, key, nboot=4000):
    rp = pd.read_csv(picks)
    k = pd.read_csv(key)
    orig = pd.read_parquet(paths.ARRIVAL_PICKS)
    d = rp.merge(k, on='anon').merge(orig[['ArcID', 'station', 't_arr_s', 't_max_s', 't_min_s']],
                                     on='ArcID', suffixes=('_re', '_orig'))
    flag = d['flag'].fillna('').astype(str).str.len() > 0
    print(f'{len(d)} arcs; flagged unusable by the re-picker: {int(flag.sum())} {sorted(d.ArcID[flag])}')
    u = d[~flag]
    rng = np.random.default_rng(0)
    for c, name in (('t_arr_s', 'onset'), ('t_max_s', 'crest'), ('t_min_s', 'trough')):
        x = (u[f'{c}_re'] - u[f'{c}_orig']).dropna()
        st = u.loc[x.index, 'station'].to_numpy()
        groups = [x.to_numpy()[st == s] for s in np.unique(st)]
        b = [np.median(np.concatenate([groups[j] for j in rng.integers(len(groups), size=len(groups))]))
             for _ in range(nboot)]
        lo, hi = np.percentile(b, [2.5, 97.5])
        print(f'{name:7s} n={len(x):2d}  re-pick - original: median {x.median():+6.1f} s [{lo:+.0f}, {hi:+.0f}], '
              f'MedAE {x.abs().median():5.1f} s, |d|<=30 s {np.mean(x.abs() <= 30):.0%}, '
              f'|d|<=60 s {np.mean(x.abs() <= 60):.0%}, max |d| {x.abs().max():.0f} s')
    x = u.t_max_s_re - u.t_max_s_orig
    print('\nlargest crest differences:')
    print(u.assign(d=x).reindex(x.abs().sort_values(ascending=False).index)
          [['anon', 'ArcID', 't_max_s_orig', 't_max_s_re', 'd']].head(5).round(0).to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--picks', default='data/inputs/repick_pica_picks.csv')
    ap.add_argument('--key', default='data/inputs/repick_pica_key.csv')
    a = ap.parse_args()
    main(a.picks, a.key)
