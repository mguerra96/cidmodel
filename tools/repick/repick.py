"""Blind CID picker: arrival, crest and trough on each arc.

Same display and detrending as the original picking (degree-2 polynomial over
0-45 min after origin time; picks on the detrended trace), but with no model
and no geometry shown.  Needs numpy, pandas (+pyarrow) and matplotlib.

    python repick.py

Picks save after every click to picks.csv in this folder; rerunning resumes.
"""
import os

import numpy as np
import pandas as pd
import matplotlib

for _bk in ('TkAgg', 'QtAgg', 'Qt5Agg'):
    try:
        matplotlib.use(_bk, force=True)
        break
    except Exception:
        continue
import matplotlib.pyplot as plt

for _k in list(matplotlib.rcParams):          # free the keys the picker uses
    if _k.startswith('keymap.'):
        matplotlib.rcParams[_k] = []

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'picks.csv')
T0, T1 = 0.0, 45.0                            # display window, min after origin
MODES = ('arr', 'max', 'min')
COL = {'arr': 'lime', 'max': 'tab:red', 'min': 'tab:blue'}
LBL = {'arr': 'ARRIVAL (onset)', 'max': 'MAX (crest)', 'min': 'MIN (trough)'}


def detrend(t, y, deg=2):
    return y - np.polyval(np.polyfit(t, y, deg), t)


class Picker:
    def __init__(self):
        arcs = pd.read_parquet(os.path.join(HERE, 'arcs.parquet'))
        self.ids = pd.read_csv(os.path.join(HERE, 'order.csv'))['anon'].tolist()
        self.data = {}
        for a, g in arcs.groupby('anon'):
            g = g.sort_values('t_s')
            t = g.t_s.to_numpy(float) / 60.0
            y = g.stec_tecu.to_numpy(float)
            m = (t >= T0) & (t <= T1) & np.isfinite(y)
            t, y = t[m], y[m]
            self.data[a] = (t, y - np.median(y), detrend(t, y))
        self.picks = {a: dict(arr=np.nan, max=np.nan, min=np.nan, flag='')
                      for a in self.ids}
        if os.path.exists(OUT):
            for r in pd.read_csv(OUT).itertuples():
                self.picks[r.anon] = dict(arr=r.t_arr_s, max=r.t_max_s, min=r.t_min_s,
                                          flag=r.flag if isinstance(r.flag, str) else '')
        self.i = next((j for j, a in enumerate(self.ids)
                       if not np.isfinite(self.picks[a]['arr'])
                       and self.picks[a]['flag'] != 'unusable'), 0)
        self.mode = 'arr'
        self.fig, (self.axr, self.ax) = plt.subplots(
            2, 1, figsize=(12, 8), sharex=True,
            gridspec_kw=dict(height_ratios=[1, 1.7], hspace=0.12))
        self.fig.canvas.mpl_connect('button_press_event', self.click)
        self.fig.canvas.mpl_connect('key_press_event', self.key)
        self.draw()

    def draw(self):
        a = self.ids[self.i]
        t, raw, det = self.data[a]
        p = self.picks[a]
        self.axr.clear()
        self.ax.clear()
        self.axr.plot(t, raw, 'k-', lw=1.4, label='sTEC (median removed)')
        self.axr.plot(t, raw - det - np.median(raw - det) + np.median(raw), '--',
                      color='tab:orange', lw=1.3, label='degree-2 trend removed')
        self.ax.plot(t, det, 'k-', lw=1.8)
        self.ax.axhline(0, color='0.6', lw=0.5)
        for m in MODES:
            if np.isfinite(p[m]):
                for ax in (self.axr, self.ax):
                    ax.axvline(p[m] / 60, color=COL[m], lw=2.4 if m == self.mode else 1.4)
        if np.isfinite(p['max']) and np.isfinite(p['min']):
            self.ax.axvspan(min(p['max'], p['min']) / 60, max(p['max'], p['min']) / 60,
                            color='0.75', alpha=0.25)
        done = sum(np.isfinite(v['arr']) or v['flag'] == 'unusable' for v in self.picks.values())
        self.axr.set_title(
            f'[{self.i + 1}/{len(self.ids)}]  arc {a}' + (f'   [{p["flag"]}]' if p['flag'] else '')
            + f'   |   done {done}/{len(self.ids)}\nNEXT CLICK SETS: {LBL[self.mode]}'
            '   (1 arrival, 2 max, 3 min)\nd delete pick   D delete all   '
            'f unusable   arrows: arc   q save+quit', fontsize=10)
        self.axr.legend(loc='upper right', fontsize=8)
        self.ax.set_xlim(T0, T1)
        self.ax.set_xlabel('minutes after origin time (06:20:52 UTC)')
        self.ax.set_ylabel('detrended sTEC (TECU) - PICK HERE')
        self.axr.set_ylabel('sTEC (TECU)')
        for ax in (self.axr, self.ax):
            ax.grid(True, lw=0.5, alpha=0.5, ls=':')
        self.fig.canvas.draw_idle()

    def click(self, ev):
        if ev.inaxes not in (self.ax, self.axr) or ev.button != 1 or ev.xdata is None:
            return
        p = self.picks[self.ids[self.i]]
        p[self.mode] = float(ev.xdata) * 60.0
        self.save()
        nxt = [m for m in MODES if not np.isfinite(p[m])]
        if nxt:
            self.mode = nxt[0]
        elif self.i < len(self.ids) - 1:
            self.i += 1
            self.mode = 'arr'
        self.draw()

    def key(self, ev):
        p = self.picks[self.ids[self.i]]
        k = ev.key
        if k in ('1', '2', '3'):
            self.mode = MODES[int(k) - 1]
        elif k == 'd':
            p[self.mode] = np.nan
            self.save()
        elif k == 'D':
            p.update(arr=np.nan, max=np.nan, min=np.nan)
            self.save()
        elif k == 'f':
            p['flag'] = '' if p['flag'] == 'unusable' else 'unusable'
            self.save()
            if p['flag'] and self.i < len(self.ids) - 1:
                self.i += 1
                self.mode = 'arr'
        elif k in ('right', 'down'):
            self.i = min(self.i + 1, len(self.ids) - 1)
            self.mode = 'arr'
        elif k in ('left', 'up'):
            self.i = max(self.i - 1, 0)
            self.mode = 'arr'
        elif k in ('q', 'escape'):
            self.save()
            plt.close(self.fig)
            return
        else:
            return
        self.draw()

    def save(self):
        rows = [dict(anon=a, t_arr_s=v['arr'], t_max_s=v['max'], t_min_s=v['min'],
                     flag=v['flag']) for a, v in self.picks.items()]
        tmp = OUT + '.tmp'
        pd.DataFrame(rows).to_csv(tmp, index=False)
        os.replace(tmp, OUT)


if __name__ == '__main__':
    pk = Picker()
    plt.show()
    pk.save()
    print('saved', OUT)
