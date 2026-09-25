"""Regenerate every figure of the paper and its Supporting Information.

Each figure is drawn by its own script in plots/, run here with the exact
arguments used for the paper.  The scripts write into figures/ under their own
names; each result is then copied to figures/paper/ under its figure number,
e.g. figures/paper/Figure_7.png.

Needs the observation tables in data/inputs/ and the cubes in data/cubes/ (see
README.md); nothing is recomputed except the neutral-velocity field of Figure 5.

    python make_figures.py              # all figures
    python make_figures.py 5 7 S1       # just these

Set CIDMODEL_FIGURES to write somewhere other than figures/.
"""
import os
import shutil
import subprocess
import sys
import time

from cidmodel import paths

HERE = os.path.dirname(os.path.abspath(__file__))

# The eleven G11 arcs of Figure 7, in order of pierce-point range.
FIG7_ARCS = ['YNYY_G11_1000', 'BNEU_G11_1000', 'CHMA_G11_1000', 'vast_G11_1000',
             'UDON_G11_1000', 'NKAY_G11_1000', 'NKRM_G11_1000', 'NKNY_G11_1000',
             'KMI6_G11_1000', 'CNBR_G11_1000', 'CHAN_G11_1000']

# figure number -> (script, arguments, file the script writes)
FIGURES = {
    '1':  ('plot_geomap.py', [], 'geomap.png'),
    '2':  ('plot_model_figure.py', [], 'model_figure.png'),
    '3':  ('plot_arrival_map.py', [], 'arrival_map.png'),
    '4':  ('plot_travel_time.py', [], 'travel_time.png'),
    '5':  ('plot_ns_coupling_chain.py', [], 'ns_coupling_chain_t720.png'),
    '6':  ('plot_dne_slices_grid.py', [], 'dne_slices_grid_fault_vs_point.png'),
    '7':  ('plot_synth_arcs.py', ['--sat', 'G11', '--page', '--arcs', *FIG7_ARCS],
           'synth_arcs_G11_page.png'),
    '8':  ('plot_peak_time_and_ptp.py', [], 'peak_time_and_ptp_min0p05.png'),
    'S1': ('plot_si_fault_distance.py', [], 'si_fault_distance.png'),
    'S2': ('plot_si_secondary_population.py', [], 'si_secondary_population.png'),
}


def main(wanted):
    unknown = [k for k in wanted if k not in FIGURES]
    if unknown:
        sys.exit(f'unknown figure(s) {unknown}; choose from {list(FIGURES)}')
    out_dir = os.path.join(paths.FIGURES, 'paper')
    os.makedirs(out_dir, exist_ok=True)

    failed = []
    for key in wanted:
        script, args, produced = FIGURES[key]
        t0 = time.time()
        r = subprocess.run([sys.executable, os.path.join(HERE, 'plots', script), *args],
                           capture_output=True, text=True)
        src = paths.fig(produced)
        if r.returncode or not os.path.exists(src):
            failed.append(key)
            print(f'Figure {key:3s} FAILED ({script})\n{r.stderr[-2000:]}')
            continue
        dst = os.path.join(out_dir, f'Figure_{key}.png')
        shutil.copyfile(src, dst)
        shown = os.path.relpath(os.path.realpath(dst), os.path.realpath(HERE))
        if shown.startswith('..'):
            shown = dst
        print(f'Figure {key:3s} {script:34s} -> {shown}  '
              f'({time.time() - t0:.0f} s)', flush=True)
    if failed:
        sys.exit(f'failed: {failed}')


if __name__ == '__main__':
    main(sys.argv[1:] or list(FIGURES))
