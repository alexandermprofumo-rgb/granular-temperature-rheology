"""The setup figure: what one run is, and what one scan produces.

WHY THIS EXISTS.  A reader who has not run a discrete-element simulation cannot
tell from the contact model alone what is actually being done.  Panel (a) draws
a real packing from the data file the runs start from, with the three things
that are imposed on it: the shear rate at the periodic boundary, the pressure
along the gradient direction, and the random forces of the bath.  Panel (b)
takes one real scan and shows the measurement made from it, mu against Theta at
fixed I, with the local slope n at Thetaz.

Panel (a) draws a 22-diameter window of the real 4000-grain box, so that
individual grains are visible; every run uses the whole box.

Usage:  python3 make_fig_schematic.py      (run from the directory holding the sweeps)
Writes: fig0_setup.png
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nfit

PACKING = 'sweep_matched2d/data.m2d_seed1'
SCAN = 'sweep_steady2d/log.mu0.3_T*_s*'
RX = r'log\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$'
THZ = 8.4664e-04
WIN = 22.0                      # side of the sub-box drawn, in grain diameters
INK = '#2b2b2b'
GRAIN = '#c9d6e4'
EDGE = '#5b7fa6'
HOT = '#c0504d'


def read_packing(path):
    xs, ys, rs = [], [], []
    with open(path) as f:
        rows = f.read().split('Atoms # sphere')[1].strip().split('\n')
    for r in rows:
        t = r.split()
        if len(t) < 7:
            continue
        rs.append(float(t[2]) / 2.0)
        xs.append(float(t[4]))
        ys.append(float(t[5]))
    return np.array(xs), np.array(ys), np.array(rs)


def panel_box(ax, x, y, r):
    m = (x - r > 0) & (x + r < WIN) & (y - r > 0) & (y + r < WIN)
    x, y, r = x[m], y[m], r[m]
    rng = np.random.default_rng(7)
    kicked = rng.choice(len(x), size=max(6, len(x) // 9), replace=False)
    for i in range(len(x)):
        ax.add_patch(Circle((x[i], y[i]), r[i], facecolor=GRAIN,
                            edgecolor=EDGE, lw=0.6, zorder=2))
    a = rng.uniform(0, 2 * np.pi, size=len(kicked))
    ax.quiver(x[kicked], y[kicked], np.cos(a), np.sin(a), color=HOT,
              angles='xy', scale_units='xy', scale=0.55, width=0.007,
              zorder=4)
    ax.plot([0, WIN, WIN, 0, 0], [0, 0, WIN, WIN, 0], color=INK, lw=1.1,
            zorder=3)
    for s, yy in ((+1, WIN), (-1, 0.0)):  # Lees-Edwards shear at top and bottom
        ax.annotate('', xy=(WIN / 2 + s * 6.5, yy + s * 1.9),
                    xytext=(WIN / 2 - s * 6.5, yy + s * 1.9),
                    arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.6))
    for xx in (-2.2, WIN + 2.2):          # barostat along the gradient only
        for s in (1, -1):
            ax.annotate('', xy=(xx, WIN / 2 + s * 2.0),
                        xytext=(xx, WIN / 2 + s * 6.0),
                        arrowprops=dict(arrowstyle='-|>', color='#4a7d52',
                                        lw=1.5))
    ax.text(WIN / 2, WIN + 3.4, r'shear rate $\dot\gamma$ (imposed)',
            ha='center', va='bottom', color=INK, fontsize=9)
    ax.text(WIN + 5.0, WIN / 2, 'pressure $P$ (imposed)', ha='center',
            va='center', color='#4a7d52', fontsize=9, rotation=270)
    ax.text(0.5, -3.4, 'random forces from the bath, set by $\\Theta$',
            ha='left', va='top', color=HOT, fontsize=9)
    ax.set_xlim(-6.0, WIN + 7.0)
    ax.set_ylim(-7.5, WIN + 6.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(a)  one run', fontsize=9, loc='left')


def panel_scan(ax):
    cells = nfit.load_sweep(SCAN, RX)
    key = [k for k in cells if dict(k).get('mu') == '0.3'][0]
    runs = cells[key]
    Ts, byT = nfit.surviving_setpoints(runs, z_min=3.0)
    th = np.array([np.mean([x['Theta'] for x in byT[t]]) for t in Ts])
    mu = np.array([np.mean([x['mu'] for x in byT[t]]) for t in Ts])
    o = np.argsort(th)
    th, mu = th[o], mu[o]
    ax.plot(th, mu, 'o', color=EDGE, mec=INK, mew=0.5, ms=6, zorder=3,
            label='one run at each temperature setpoint')
    r = nfit.fit_local_sys(runs, THZ, fn=nfit.fit_local_adaptive, z_min=3.0)
    n = r['n']
    mu0 = np.interp(np.log(THZ), np.log(th), np.log(mu))
    tt = np.array([THZ / 2.2, THZ * 2.2])
    ax.plot(tt, np.exp(mu0) * (tt / THZ) ** (-n), '-', color=HOT, lw=1.6,
            zorder=2, label=fr'local slope $n = {n:.3f}$ at $\Theta_0$')
    ax.axvline(THZ, color=INK, lw=0.8, ls=':', zorder=1)
    ax.text(THZ * 1.08, mu[0] * 0.98, r'$\Theta_0$', fontsize=9, color=INK)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'granular temperature $\Theta$')
    ax.set_ylabel(r'effective friction $\mu$')
    ax.legend(frameon=False, fontsize=8, loc='lower left',
              borderaxespad=0.2)
    ax.set_title('(b)  one scan, at one shear rate', fontsize=9, loc='left')


def main():
    plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8})
    fig, ax = plt.subplots(1, 2, figsize=(7.1, 3.4))
    x, y, r = read_packing(PACKING)
    panel_box(ax[0], x, y, r)
    panel_scan(ax[1])
    fig.tight_layout()
    fig.savefig('fig0_setup.png', dpi=400)
    print('wrote fig0_setup.png')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
