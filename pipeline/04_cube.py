"""Step 4 -- build a dNe cube: integrate continuity over the ray cube with the numba v_n
kernel (nearest-neighbour lookup).

On the 2.5 km grid the arrival-time quantisation is small (Dt_w ~ 0.10 sigma),
so nearest-neighbour lookup matches the bilinear cube per arc while the numba
kernel runs it ~5x faster (numba only accelerates the nearest path; bilinear is
bottlenecked on the single-threaded numpy lookup).

Integrates d(dNe)/dt = -div[Ne0 v_i] by trapezoid in time, reusing the cached
ne0 / B_field / ray_cube.  Streams float32 frames to a disk memmap
(data/cubes/dne_*.npy) with a small .npz sidecar.

  --sources binned   merged fault elements onto grid nodes (~30; default)
  --sources full     all ~124 USGS elements
  --sources point    single source at the epicentre (finite-fault vs point)

The N-wave pulse width follows sig_eff = B_LIN * t_w, a constant of
cidmodel.coupling, not a command-line option.

    python pipeline/04_cube.py --sources binned
    python pipeline/04_cube.py --sources point
"""
import argparse
import os
import time

import numpy as np

from cidmodel import paths
from cidmodel.geometry import model_grid
from cidmodel.raytracing import RayCube
from cidmodel.sources import binned_fault_sources, usgs_fault_sources
from cidmodel.ionosphere import ion_velocity
from cidmodel.continuity import divergence
from cidmodel.coupling import neutral_velocity_field_numba, B_LIN

# NO A0 HERE.  The source scale is a FITTED result, not a build input: it is
# derived by comparing the synthetic sTEC against the observed peak-to-peak, so
# it does not exist until after this cube has been integrated and scanned.
# Writing it here is what let a cube claim A0=5.7 while its own generation
# wanted 5.14.  The whole dNe -> sTEC chain is linear in the scale (verified to
# 3.5e-08 relative), so it is applied once, at analysis time, in
# pipeline/05_synth_stec.py --a0.  The dNe cubes hold unscaled physics.
T_START = 0.0           # s
T_STEP  = 15.0          # s
# 35 min window (the observed CIDs run to ~35 min; a 25 min window cut the far
# arcs off mid-wave).
T_END = 2100.0


def _atomic_save_npz(path, **arrays):
    tmp = path + '.tmp.npz'
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sources', choices=['full', 'binned', 'point'],
                    default='binned',
                    help="'full' = 124 USGS elements; 'binned' = merged onto the "
                         "grid nodes (~30, faster + smoother); 'point' = a single "
                         "source at the epicentre for the finite-fault-vs-point "
                         "comparison")
    # REQUIRED, and deliberately so.  This defaulted to `paths.RAY_CUBE` until
    # 2026-09-17; that constant still named the Gen-1 `_uni` cube long after the
    # sin(theta0) rebuild, so `--out-tag _v4` quietly integrated the PRE-FIX
    # cube and wrote 3 GB of dNe labelled with a generation it did not contain.
    ap.add_argument('--cube', default='',
                    help='ray cube to integrate against: a path, a filename or '
                         'a tag (default: the canonical data/cubes/ray_cube.npz)')
    ap.add_argument('--out-tag', default='',
                    help='suffix for the output dNe memmap + sidecar, e.g. '
                         '"_v4" -> dne_binned_v4.npy/.npz (default: none, i.e. '
                         'the canonical dne_binned / dne_point)')
    # The N-wave pulse width is sig_eff = B_LIN * t_w, a MODULE constant in
    # coupling.py, not a CLI argument.  It was briefly selectable (sqrt vs
    # linear, with sigma/b/tw0/period-scale); that is gone.  A law is a physics
    # choice the whole repo must agree on, and making it a flag is what let a
    # plain build silently produce cubes on a period nothing else assumed.
    args = ap.parse_args()


    # Name both halves from the same tag: the sidecar records the memmap's
    # basename as `dne_memmap`, so they cannot be tagged independently without
    # the sidecar pointing at the wrong array.
    stem = 'dne_point' if args.sources == 'point' else 'dne_binned'
    tag = paths.gen_suffix(args.out_tag)
    dne_mmap = paths.cube(f'{stem}{tag}.npy')
    dne_meta = paths.cube(f'{stem}{tag}.npz')

    t0 = time.time()
    g = model_grid()
    cube_path = paths.require_cube(args.cube, 'ray')
    cube = RayCube.load(cube_path)
    print(f'  ray cube: {cube_path}', flush=True)
    if args.sources == 'binned':
        src = binned_fault_sources(g, verbose=True)
    elif args.sources == 'point':
        src = [dict(x=0.0, y=0.0, t_rupture_s=0.0, weight=1.0)]
        print('  POINT source: single element at epicentre (0,0), t_rup=0',
              flush=True)
    else:
        src = usgs_fault_sources(verbose=False)
    ne0 = np.load(paths.NE0)
    B = np.load(paths.B_FIELD)
    times = np.arange(T_START, T_END + 1e-6, T_STEP)
    print(f'NEAREST+numba build [{args.sources}, {len(src)} src]: '
          f'{len(times)} frames to {T_END:.0f}s (35 min) -> {dne_mmap}',
          flush=True)
    print(f'  N-wave: sig = {B_LIN:.4f} * t_w (Mikesell linear); '
          f'sig at t_w=650/1200 s = {B_LIN*650:.1f}/{B_LIN*1200:.1f} s',
          flush=True)

    out = np.lib.format.open_memmap(dne_mmap, mode='w+', dtype=np.float32,
                                    shape=(len(times),) + g.shape)
    accum = np.zeros(g.shape)
    prev = None
    for i, t in enumerate(times):
        tf = time.time()
        vn = neutral_velocity_field_numba(g, cube, src, float(t),
                                          bilinear=False)
        vi, _ = ion_velocity(vn, B)
        r = -divergence(ne0[..., None] * vi, g)
        if prev is not None:
            accum += 0.5 * (times[i] - times[i - 1]) * (r + prev)
        prev = r
        out[i] = accum.astype(np.float32)
        out.flush()
        print('  frame %3d t=%4.0fs  %.1f s  |dNe|max=%.3e'
              % (i, t, time.time() - tf, np.abs(accum).max()), flush=True)

    # `ray_cube` records the cube this dNe was actually integrated against, so
    # the file answers its own provenance instead of relying on its name.  The
    # 2026-09-17 mislabelling was invisible precisely because nothing here did.
    # `b_lin` pins the pulse width the frames were built with -- a physics
    # choice the filename does not carry.
    #
    # `A0` is written as 1.0: these frames ARE unscaled, and that is the honest
    # value for them.  The source scale cannot be known here -- it is fitted by
    # comparing this cube's own sTEC against the observed peak-to-peak -- so
    # 05_synth_stec.py writes the fitted value back into this field as the last
    # step of the chain that derives it.  The field is therefore always present
    # and always true of the frames beside it: 1.0 until the fit has been run,
    # the fitted scale afterwards.  Readers (the dNe figures) take it as the
    # default scale, so a cube that has not been fitted yet draws
    # as the raw field rather than at someone else's number.
    _atomic_save_npz(dne_meta, x=g.x, y=g.y, z=g.z, times=times,
                     dne_memmap=np.array(os.path.basename(dne_mmap)),
                     ray_cube=np.array(os.path.basename(cube_path)),
                     sources=np.array(args.sources),
                     b_lin=np.array(B_LIN),
                     A0=np.array(1.0))
    print(f'DONE {dne_meta} in {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
