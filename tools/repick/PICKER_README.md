# Blind CID re-pick: instructions

This package lets you pick the coseismic ionospheric disturbances (CIDs) of the
Mw 7.7 Myanmar earthquake blindly, to test how reproducible the original hand
picks are.

## Contents
- `arcs.parquet`: raw sTEC of the satellite-receiver arcs, anonymised (R01, R02, ...),
  time in seconds after the origin time (2025-03-28 06:20:52 UTC).
- `order.csv`: presentation order (random).
- `repick.py`: the picker.

The original picks, the synthetics, and the station, satellite, range and
azimuth of each arc are deliberately left out.

## Usage
You need Python with numpy, pandas (+ pyarrow) and matplotlib with an interactive
backend (Tk or Qt):

    python repick.py

The lower panel of each arc shows the detrended sTEC: a degree-2 polynomial is
removed over 0-45 min after the origin time, as in the original picking.
**Pick on that panel.** Each arc takes three clicks, in this order; the picker
moves on to the next arc by itself:

1. **arrival**: the onset of the disturbance;
2. **max**: the CID crest (positive extreme);
3. **min**: the trough (negative extreme).

Keys: `1` `2` `3` choose which pick to set, `d` deletes the active pick, `D`
deletes all picks of the arc, `f` marks the arc unusable (no recognisable CID),
arrows move between arcs, `q` saves and quits.

Picks are saved after every click to `picks.csv`; rerunning resumes where you
stopped. Send `picks.csv` back to whoever holds the key, or compare it yourself
with `tools/repick/compare.py`.

## Criterion
Pick as you normally would, without trying to guess what is expected. If an arc
shows two wave packets, pick the one you consider the main CID and note the arc
number if it is ambiguous. If you see no CID, press `f`.
