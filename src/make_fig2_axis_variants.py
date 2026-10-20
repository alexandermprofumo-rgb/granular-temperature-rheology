"""Three ways to draw the x-axis of the Letter's Fig. 2(a), for comparison.

WHY THIS EXISTS.  The current panel uses a symlog axis, linear below 0.05 and
logarithmic above, so mu_g = 0 can sit on it.  The friction law is a straight
line in mu_g, but on that axis it is drawn curved, and the text calls it
straight.  The three candidates:

  A  logarithmic axis from 0.02 to 1, with the frictionless value on its own
     narrow axis to the left, past a break
  B  linear axis from 0 to 1
  C  B, with an inset on the rising branch

Reads friction_law_points.json, written by make_letter_figures.py.
Usage:  python3 make_fig2_axis_variants.py     (beside that file)
"""
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

W = 3.375
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'legend.fontsize': 7, 'figure.dpi': 400,
                     'lines.linewidth': 1.2})
C = {'2D': '#C44E52', '3D': '#4C72B0'}
J = json.load(open('friction_law_points.json'))
R = {t: {float(k): v for k, v in J[t].items()} for t in ('R2', 'R3')}
BRANCH, PUB = J['branch'], J['published']


def fit(tag):
    D = R['R2' if tag == '2D' else 'R3']
    x = np.array([m for m in sorted(D) if m <= BRANCH[tag]])
    y = np.array([D[m][0] for m in x]); e = np.array([D[m][1] for m in x])
    w = 1 / e ** 2
    A = np.column_stack([np.ones_like(x), x])
    return np.linalg.solve(A.T @ (A * w[:, None]), A.T @ (w * y))


def draw(ax, xmin=None, inset=False):
    for tag in ('2D', '3D'):
        D = R['R2' if tag == '2D' else 'R3']
        m = np.array(sorted(D))
        if xmin is not None:
            m = m[m >= xmin]
        ax.errorbar(m, [D[x][0] for x in m], yerr=[D[x][1] for x in m],
                    marker='o', ms=3.2 if inset else 3.6, lw=0,
                    elinewidth=1.0, capsize=2, color=C[tag],
                    label=None if inset else tag)
        c = fit(tag)
        xs = np.linspace(max(xmin or 0, 0), BRANCH[tag], 50)
        ax.plot(xs, c[0] + c[1] * xs, '-', color=C[tag], lw=1.1)
        ax.axhline(PUB[tag], ls=':', lw=1.0, color=C[tag])


def labels(ax, x):
    ax.text(x, PUB['2D'] + 0.006, r'$p=1/8$ fitted for disks', fontsize=6.5,
            color=C['2D'])
    ax.text(x, PUB['3D'] + 0.006, r'$p=1/6$ fitted for spheres',
            fontsize=6.5, color=C['3D'])


fig = plt.figure(figsize=(3 * W + 0.4, 2.6))
outer = fig.add_gridspec(1, 3, wspace=0.32)

# ---- A: log axis, frictionless value on its own axis past a break
gA = outer[0].subgridspec(1, 2, width_ratios=[1, 7], wspace=0.06)
a0 = fig.add_subplot(gA[0]); a1 = fig.add_subplot(gA[1], sharey=a0)
for tag in ('2D', '3D'):
    D = R['R2' if tag == '2D' else 'R3']
    off = -0.15 if tag == '2D' else 0.15
    a0.errorbar([off], [D[0.0][0]], yerr=[D[0.0][1]], marker='o', ms=3.6,
                lw=0, elinewidth=1.0, capsize=2, color=C[tag])
    a0.axhline(PUB[tag], ls=':', lw=1.0, color=C[tag])
a0.set_xlim(-0.6, 0.6); a0.set_xticks([0]); a0.set_xticklabels(['0'])
a0.set_ylim(0, 0.34); a0.set_ylabel(r'$n$')
a0.spines['right'].set_visible(False)
draw(a1, xmin=0.02)
a1.set_xscale('log'); a1.set_xlim(0.015, 1.4)
a1.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5, 1])
a1.set_xticklabels(['0.02', '0.05', '0.1', '0.2', '0.5', '1'])
a1.set_xticks([], minor=True)
a1.spines['left'].set_visible(False)
a1.tick_params(axis='y', left=False, labelleft=False)
for ax_, x0 in ((a0, 1), (a1, 0)):            # the break marks
    kw = dict(transform=ax_.transAxes, color='k', clip_on=False, lw=0.8)
    for y0 in (0, 1):
        ax_.plot([x0 - 0.04 * (1 if ax_ is a1 else 7) / 1, x0 + 0.04 *
                  (1 if ax_ is a1 else 7) / 1], [y0 - 0.02, y0 + 0.02], **kw)
labels(a1, 0.022)
a1.set_xlabel(r'grain friction  $\mu_g$')
a1.legend(frameon=False, loc='upper left', handletextpad=0.4)
a0.set_title('A  log axis, with a break', loc='left')

# ---- B: linear axis
b = fig.add_subplot(outer[1])
draw(b)
b.set_xlim(-0.03, 1.05); b.set_ylim(0, 0.34)
labels(b, 0.45)
b.set_xlabel(r'grain friction  $\mu_g$'); b.set_ylabel(r'$n$')
b.legend(frameon=False, loc='upper right', handletextpad=0.4)
b.set_title('B  linear axis', loc='left')

# ---- C: linear axis with an inset on the rising branch
c = fig.add_subplot(outer[2])
draw(c)
c.set_xlim(-0.03, 1.05); c.set_ylim(0, 0.34)
c.set_xlabel(r'grain friction  $\mu_g$'); c.set_ylabel(r'$n$')
c.legend(frameon=False, loc='upper left', handletextpad=0.4,
         bbox_to_anchor=(0.0, 1.0))
ins = c.inset_axes([0.47, 0.50, 0.50, 0.45])
draw(ins, inset=True)
ins.set_xlim(-0.01, 0.22); ins.set_ylim(0, 0.32)
ins.set_xticks([0, 0.1, 0.2]); ins.set_yticks([0, 0.1, 0.2, 0.3])
ins.tick_params(labelsize=6)
c.indicate_inset_zoom(ins, edgecolor='0.5')
c.set_title('C  linear, with an inset', loc='left')

fig.savefig('fig2a_axis_variants.png', bbox_inches='tight')
print('wrote fig2a_axis_variants.png')
