"""Step 3 -- interpolate the saved ray deposit into the ray cube.

    data/cubes/deposit_raw.npz  ->  data/cubes/ray_cube.npz

Step 2 runs this automatically after tracing; run it on its own to change the
interpolation region without re-tracing (seconds per azimuth).  The method and
its settled defaults are documented in `cidmodel.interpolation`.

    python pipeline/03_interpolate.py                 # the settled defaults
    python pipeline/03_interpolate.py --alpha-km 0    # keep the whole hull
    python pipeline/03_interpolate.py --no-paired
"""
import argparse

from cidmodel import deposit as D
from cidmodel.interpolation import AXISYM_BINS, interpolate_deposit

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # '' = the canonical deposit_raw.npz.
    ap.add_argument('--tag', default='', help='source deposit tag')
    ap.add_argument('--out-tag', default=None)
    ap.add_argument('--alpha-km', type=float, default=D.ALPHA_KM,
                    help=f'alpha-shape radius (km); 0 keeps the whole convex '
                         f'hull (default {D.ALPHA_KM:.0f})')
    ap.add_argument('--no-paired', dest='paired', action='store_false',
                    help='interpolate each azimuth alone instead of pairing '
                         'it with az+180 as one signed-range plane')
    ap.add_argument('--axisym-bins', type=int, default=AXISYM_BINS,
                    help=f'innermost range bins replaced by their azimuthal '
                         f'median; 0 disables (default {AXISYM_BINS})')
    ap.add_argument('--n-jobs', type=int, default=None)
    ap.set_defaults(paired=True)
    a = ap.parse_args()
    interpolate_deposit(src_tag=a.tag, out_tag=a.out_tag, n_jobs=a.n_jobs,
                        alpha_km=(a.alpha_km or None), paired=a.paired,
                        axisym_bins=a.axisym_bins)
