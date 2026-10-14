"""Candidate Letter panel: the three shear-rate series in Kim and Kamrin's
coordinates, mu Theta^p against I, at the single p that collapses them best.

Reads collapse_test.csv (collapse_test.py).  Under Eq. (1) every point at one
inertial number has one value of mu Theta^p, so each series should shrink to
a point.  Colour runs from the coldest setpoint to the hottest, so a series
that stays spread also shows which way heating moves it.

Usage:  python3 make_collapse_panel.py     (beside collapse_test.csv)
"""
import csv

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

W = 3.375
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'legend.fontsize': 7, 'figure.dpi': 400})
CANON = 8.4664e-04

rows = list(csv.DictReader(open('collapse_test.csv')))
LAB = {'slow': r'$I=10^{-4}$', 'mid': r'$I=10^{-3.5}$', 'fast': r'$I=10^{-3}$'}
COL = dict(zip(('slow', 'mid', 'fast'), plt.cm.plasma(np.linspace(0.12, 0.72, 3))))
fig, ax = plt.subplots(2, 1, figsize=(W, 3.6), sharex=False)
for a, mg, ttl in ((ax[0], '0.3', r'(b)  frictional grains, $\mu_g=0.3$'),
                   (ax[1], '0.0', r'(c)  frictionless grains')):
    sel = [r for r in rows if r['mu_g'] == mg]
    p = float(sel[0]['p_star'])
    for arm in ('slow', 'mid', 'fast'):
        s_ = [r for r in sel if r['arm'] == arm]
        th = np.array([float(r['Theta']) for r in s_]) / CANON
        y = np.array([float(r['mu_Theta_pstar']) for r in s_])
        # each series scaled by its value at Theta_0, from a local quadratic,
        # so a series that obeys Eq. (1) lies on y = 1 across the window
        u = np.log(th)
        c = np.polyfit(u, np.log(y), 2)
        y0 = np.exp(np.polyval(c, 0.0))
        a.plot(th, y / y0, 'o', ms=3.2, color=COL[arm], label=LAB[arm], zorder=3)
        uu = np.linspace(u.min(), u.max(), 50)
        a.plot(np.exp(uu), np.exp(np.polyval(c, uu)) / y0, '-', color=COL[arm],
               lw=1.0, zorder=2)
    a.axhline(1.0, color='0.6', lw=0.6, zorder=1)
    a.set_xscale('log'); a.set_yscale('log')
    a.set_ylim(0.88, 1.12)
    a.set_yticks([0.9, 1.0, 1.1]); a.set_yticklabels(['0.9', '1', '1.1'])
    a.set_yticks([], minor=True)
    a.set_xticks([0.3, 1, 3, 10]); a.set_xticklabels(['0.3', '1', '3', '10'])
    a.set_xticks([], minor=True)
    a.set_xlabel(r'$\Theta/\Theta_0$')
    a.set_ylabel(r'$\mu\Theta^{p}$, scaled at $\Theta_0$')
    a.set_title(ttl + rf',  best single $p={p:.3f}$', loc='left')
# the upper right of (b) is empty, so the key sits there, off the points
ax[0].legend(frameon=False, loc='upper right', handletextpad=0.2)
fig.tight_layout(pad=0.4)
fig.savefig('figL1_collapse_candidate.png')
print('wrote figL1_collapse_candidate.png')
