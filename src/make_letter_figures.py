"""The Letter's three figures, drawn at the width PRL prints them.

WHY THIS EXISTS.  make_paper_figures.py draws six figures 7.1 in across, sized
for the two-column spread of the companion.  A Letter figure is one column
wide, 3.375 in, and a panel shrunk into that space loses its tick labels first
and its legend second.  These are drawn at the printed width with the panels
stacked, so nothing is scaled in LaTeX and an 8 pt label stays 8 pt on paper.

Nothing is hardcoded that the data can supply.  The two branch limits are the
exception: they are the frictions out to which n(mu_g) is linear, located in
the companion by extending a weighted fit one friction at a time, and they are
inputs to this figure rather than results of it.

  figL1   the system and the headline
          (a) one run; (b) three shear rates at mu_g = 0.3, not parallel;
          (c) the same for frictionless grains, parallel
  figL2   what sets the exponent
          (a) n against grain friction in both dimensions, with the branch fit
          (b) the share of n carried by each contact channel
  figL3   stiffness, and the coupling
          (a) dn/dln(kappa) against grain friction, sign changing at the peak
          (b) dn/dln(gdot) at four stiffnesses, frictional and frictionless

Usage:  python3 make_letter_figures.py     (from the directory holding sweeps)
"""
import glob
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nfit
import levers_matched as lev

W = 3.375                      # PRL single column, inches
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'legend.fontsize': 7, 'figure.dpi': 400,
                     'lines.linewidth': 1.2})

ZMIN, ZMIN3 = 3.0, 4.0
CANON = 8.4664e-04
CORE = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
BRANCH = {'2D': 0.15, '3D': 0.2}          # see the docstring
PUBLISHED = {'2D': 0.125, '3D': 1 / 6}    # Kim and Kamrin, disks and spheres
C = {'2D': '#C44E52', '3D': '#4C72B0', 'mod': '#4C72B0', 'prs': '#55A868',
     'geo': '#8172B2', 'nrm': '#4C72B0', 'tan': '#C44E52', 'fl': '0.45'}
RXM = r'log\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$'
OUT = {}

# Panel 1(a) draws a real packing rather than a cartoon, because the point of
# it is that a reader can see what one run is.  Both comparison Letters open
# with a picture of the system and this one did not.
# the last frame of a steady sheared run, 2D, mu_g = 0.3, default stiffness
PACKING = 'sweep_tracking_matched2d/dump.atoms.mu0.3_T0.001_s1'
WX, WY = 62.0, 11.0             # the window drawn, in grain diameters
INK, GRAIN, EDGE, HOT = '#2b2b2b', '#c9d6e4', '#5b7fa6', '#c0504d'


def read_packing(path, x0=2.0, y0=25.0):
    """Grain positions and radii from the last frame of a LAMMPS atom dump,
    wrapped into the sheared periodic cell and shifted so the drawn window
    starts at (x0, y0)."""
    with open(path) as f:
        frame = f.read().split('ITEM: TIMESTEP')[-1].split('\n')
    (xlo, xhi, xy), (ylo, yhi, _) = [tuple(map(float, frame[i].split()))
                                     for i in (5, 6)]
    rows = np.array([list(map(float, r.split())) for r in frame[9:] if r.strip()])
    x, y, rad = rows[:, 1], rows[:, 2], rows[:, 6]
    # the box is a tilted (Lees-Edwards) cell, so the periodic image in x is
    # offset by the tilt at the grain's height
    # dump bounds enclose the tilted cell, so its own length is shorter by xy
    lx, ly = xhi - xlo - abs(xy), yhi - ylo
    y = ylo + np.mod(y - ylo, ly)
    shift = xy * (y - ylo) / ly
    x = xlo + shift + np.mod(x - xlo - shift, lx)
    return x - x0, y - y0, rad


def panel_setup(ax):
    x, y, r = read_packing(PACKING)
    m = (x - r > 0) & (x + r < WX) & (y - r > 0) & (y + r < WY)
    x, y, r = x[m], y[m], r[m]
    rng = np.random.default_rng(7)
    kicked = rng.choice(len(x), size=max(6, len(x) // 8), replace=False)
    for i in range(len(x)):
        ax.add_patch(Circle((x[i], y[i]), r[i], facecolor=GRAIN,
                            edgecolor=EDGE, lw=0.35, zorder=2))
    a = rng.uniform(0, 2 * np.pi, size=len(kicked))
    ax.quiver(x[kicked], y[kicked], np.cos(a), np.sin(a), color=HOT,
              angles='xy', scale_units='xy', scale=0.5, width=0.005, zorder=4)
    ax.plot([0, WX, WX, 0, 0], [0, 0, WY, WY, 0], color=INK, lw=0.9, zorder=3)
    for s, yy in ((+1, WY), (-1, 0.0)):          # Lees-Edwards shear
        ax.annotate('', xy=(WX / 2 + s * 9.0, yy + s * 1.4),
                    xytext=(WX / 2 - s * 9.0, yy + s * 1.4),
                    arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.2))
    for xx in (-1.8, WX + 1.8):                  # barostat, along the gradient
        for s in (1, -1):
            ax.annotate('', xy=(xx, WY / 2 + s * 0.9),
                        xytext=(xx, WY / 2 + s * 6.5),
                        arrowprops=dict(arrowstyle='-|>', color='#4a7d52',
                                        lw=1.2, mutation_scale=8))
    ax.text(WX / 2, WY + 2.4, r'shear rate $\dot\gamma$', ha='center',
            va='bottom', color=INK, fontsize=7)
    ax.text(WX + 3.6, WY / 2, r'pressure $P$', ha='center', va='center',
            color='#4a7d52', fontsize=7, rotation=270)
    ax.text(0.0, -2.2, r'random forces from the bath, set by $\Theta$',
            ha='left', va='top', color=HOT, fontsize=7)
    ax.set_xlim(-5.5, WX + 5.5)
    ax.set_ylim(-4.0, WY + 3.4)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(a)  part of a sheared packing', loc='left')


def cells(d, rx=RXM):
    out = {}
    for p in glob.glob(f'{d}/log.*'):
        m = re.match(rx, os.path.basename(p))
        if not m:
            continue
        q = nfit.parse_log(p)
        if not q:
            continue
        q['Tgran'] = float(m.group('T'))
        out.setdefault(float(m.group('mu')), []).append(q)
    return out


def jammed(runs, zmin):
    Ts, byT = nfit.surviving_setpoints(runs, z_min=zmin)
    return [t for t in Ts
            if not [x['Z'] for x in byT[t] if 'Z' in x]
            or np.mean([x['Z'] for x in byT[t] if 'Z' in x]) >= zmin]


def nval(runs, keep, zmin, theta0=CANON):
    r = nfit.fit_local_sys(runs, theta0, fn=nfit.fit_local_adaptive,
                           restrict_to=keep, z_min=zmin)
    if not r:
        return None
    return r['n'], float(np.hypot(r['tot'], r.get('sys_deg', 0.0)))


def setpoint_means(runs, keep, zmin):
    """One (Theta, mu) point per surviving setpoint, averaged over seeds."""
    _, byT = nfit.surviving_setpoints(runs, z_min=zmin, restrict_to=keep)
    pts = []
    for t in keep:
        sel = [r for r in byT.get(t, []) if r['Theta'] > 0 and r['mu'] > 0]
        if sel:
            pts.append((float(np.mean([r['Theta'] for r in sel])),
                        float(np.mean([r['mu'] for r in sel]))))
    return np.array(sorted(pts)).T if pts else None


def wline(x, y, e):
    w = 1.0 / np.asarray(e) ** 2
    A = np.column_stack([np.ones_like(x), x])
    cov = np.linalg.inv(A.T @ np.diag(w) @ A)
    c = cov @ A.T @ np.diag(w) @ y
    return c, np.sqrt(np.diag(cov))


def at_theta0(th, mu, theta0):
    """mu at theta0 from a local quadratic in log-log, for normalizing a curve.

    Dividing by the nearest measured point instead would put each series
    through 1 at a slightly different temperature, which is exactly the
    displacement these panels are drawn to show.
    """
    u = np.log(th / theta0)
    c = np.polyfit(u, np.log(mu), min(2, len(u) - 1))
    return float(np.exp(np.polyval(c, 0.0)))


def plain_ticks(ax, which, ticks):
    """Log axes with readable labels: 0.3, 1, 3 rather than 3 x 10^-1."""
    def lab(v):
        return f'{v:g}'
    if 'x' in which:
        ax.set_xticks(ticks); ax.set_xticklabels([lab(t) for t in ticks])
        ax.set_xticks([], minor=True)
    if 'y' in which:
        ax.set_yticks(ticks); ax.set_yticklabels([lab(t) for t in ticks])
        ax.set_yticks([], minor=True)


print('loading the friction scans ...')
D2, D3 = cells('sweep_steady2d'), cells('sweep_steady3d')
K2 = {m: jammed(v, ZMIN) for m, v in D2.items()}
K3 = {m: jammed(v, ZMIN3) for m, v in D3.items()}
R2 = {m: nval(D2[m], K2[m], ZMIN) for m in sorted(D2) if len(K2[m]) >= 4}
R3 = {m: nval(D3[m], K3[m], ZMIN3) for m in sorted(D3) if len(K3[m]) >= 4}
R2 = {k: v for k, v in R2.items() if v}
R3 = {k: v for k, v in R3.items() if v}


# ------------------------------------------- figure 1: the system and the headline
ARMS = (('slow', 1.0e-4, 'sweep_iscan2/log.g3.162e-4_mu*',
         r'log\.g(?P<g>[0-9.eE+-]+)_mu(?P<mu>[0-9.]+)'
         r'_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$'),
        ('mid', 3.162e-4, 'sweep_steady2d/log.mu*', RXM),
        ('fast', 1.0e-3, 'sweep_iscan2/log.g3.162e-3_mu*',
         r'log\.g(?P<g>[0-9.eE+-]+)_mu(?P<mu>[0-9.]+)'
         r'_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$'))


def arm_cells(pattern, rx):
    out = {}
    for key, runs in nfit.load_sweep(pattern, rx).items():
        out.setdefault(float(dict(key)['mu']), []).extend(runs)
    return out


ARMDATA = {tag: (I, arm_cells(p, rx)) for tag, I, p, rx in ARMS}


def panel_arms(a, mg, ttl, leg='lower left'):
    """mu(Theta) for the three shear rates at one friction, over the shared window."""
    pools = [(tag, ARMDATA[tag][0], ARMDATA[tag][1][mg])
             for tag, _, _, _ in ARMS if mg in ARMDATA[tag][1]]
    got = nfit.common_theta0([p[2] for p in pools])
    if got is None:
        print(f'  mu_g = {mg}: no shared window')
        return
    T0, (lo, hi) = got
    Tuse = CANON if lo * 1.3 <= CANON <= hi / 1.3 else T0
    shades = plt.cm.plasma(np.linspace(0.12, 0.72, len(pools)))
    for col, (tag, I, runs) in zip(shades, pools):
        keep = [t for t in jammed(runs, ZMIN)
                if all(lo * 0.999 <= r['Theta'] <= hi * 1.001
                       for r in nfit.surviving_setpoints(
                           runs, z_min=ZMIN, restrict_to=[t])[1].get(t, [])
                       if r['Theta'] > 0)]
        if len(keep) < 4:
            print(f'  mu_g = {mg}, {tag}: {len(keep)} setpoints in the window')
            continue
        pts = setpoint_means(runs, keep, ZMIN)
        if pts is None:
            continue
        th, mu = pts
        got_n = nval(runs, keep, ZMIN, Tuse)
        e10 = np.log10(I)
        ex = f'{int(round(e10))}' if abs(e10 - round(e10)) < 0.05 \
            else f'{e10:.1f}'
        a.plot(th / Tuse, mu / at_theta0(th, mu, Tuse), 'o', ms=3.2, color=col,
               label=rf'$I=10^{{{ex}}}$,  $n={got_n[0]:.3f}$')
        # a smooth local quadratic through the series, not the tangent at
        # Theta_0, which overshoots the points wherever mu(Theta) curves
        u = np.log(th / Tuse)
        c = np.polyfit(u, np.log(mu), min(2, len(u) - 1))
        m0 = np.exp(np.polyval(c, 0.0))
        uu = np.linspace(u.min(), u.max(), 60)
        a.plot(np.exp(uu), np.exp(np.polyval(c, uu)) / m0, '-', color=col, lw=1.0)
        print(f'  mu_g = {mg}, {tag}: n = {got_n[0]:.4f} +/- {got_n[1]:.4f}'
              f'  over {len(keep)} setpoints')
    a.set_xscale('log'); a.set_yscale('log')
    plain_ticks(a, 'x', [0.3, 0.5, 1, 2, 3])
    plain_ticks(a, 'y', [0.9, 1.0, 1.1])
    a.set_xlabel(r'$\Theta/\Theta_0$')
    a.set_ylabel(r'$\mu/\mu(\Theta_0)$')
    a.set_title(ttl, loc='left')
    a.legend(frameon=False, loc=leg)


# panel (a) is drawn at equal aspect, so its allocated height has to be large
# enough for the packing to span the column rather than being centred in it
fig, ax = plt.subplots(3, 1, figsize=(W, 5.6),
                       gridspec_kw=dict(height_ratios=[0.78, 1.5, 1.5]))
panel_setup(ax[0])
panel_arms(ax[1], 0.3, r'(b)  frictional grains, $\mu_g=0.3$')
# the frictionless series lie low on the shared scale, so their legend goes
# in the empty upper right rather than over the points
panel_arms(ax[2], 0.0, '(c)  frictionless grains', leg='upper right')
# one vertical scale for both, so the fan in (b) and its absence in (c) are
# compared by eye rather than through two different axes
lo_ = min(ax[1].get_ylim()[0], ax[2].get_ylim()[0])
hi_ = max(ax[1].get_ylim()[1], ax[2].get_ylim()[1])
for a_ in (ax[1], ax[2]):
    a_.set_ylim(lo_, hi_)
    plain_ticks(a_, 'y', [0.9, 1.0, 1.1, 1.2])
fig.tight_layout(pad=0.4); fig.savefig('figL1_separability.png')
print('wrote figL1_separability.png')


# ------------------------------------------- figure 2: what sets the exponent
def load_csv(name):
    import csv
    for d in ('.', os.path.dirname(os.path.abspath(__file__))):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return [{k: (float(v) if k != 'seed' else v) for k, v in r.items()}
                    for r in csv.DictReader(open(p))]
    raise SystemExit(f'{name} not found beside the sweeps or the code')


def local_value_slope(x, y, x0, deg=2):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < deg + 2:
        return None
    A = np.vander(x - x0, deg + 1, increasing=True)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(c[0]), float(c[1])


def channel_shares():
    """Share of n carried by the geometric, normal-force and tangential terms.

    The same construction as the companion paper's channel figure: each term's
    contribution is its local slope against ln Theta at Theta_0 divided by
    2 mu, and the share is that contribution over their sum.
    """
    ch = load_csv('chan_steady2d.csv')
    out = []
    for m in sorted(set(r['mu_g'] for r in ch)):
        # The frictionless exponent is an order of magnitude below the rest, so
        # its shares are ratios of small numbers and swing past -60% and +160%.
        # They say nothing about which term carries the weakening, so the
        # panel starts at the first friction.
        if m == 0.0 or m not in K2 or len(K2[m]) < 4:
            continue
        keep = {round(t, 12) for t in K2[m]}
        sel = [r for r in ch if r['mu_g'] == m and round(r['Tgran'], 12) in keep]
        if len(sel) < 8:
            continue
        if 'Theta' in sel[0]:
            x = np.log([r['Theta'] for r in sel])
        else:
            th = {}
            for p in (glob.glob(f'sweep_steady2d/log.mu{m:g}_T*')
                      + glob.glob(f'sweep_steady2d/log.mu{m}_T*')):
                mm = re.match(RXM, os.path.basename(p)); q = nfit.parse_log(p)
                if mm and q:
                    th[(float(mm.group('T')), mm.group('s'))] = q['Theta']
            sel = [r for r in sel if (r['Tgran'], str(int(r['seed']))) in th]
            if len(sel) < 8:
                continue
            x = np.log([th[(r['Tgran'], str(int(r['seed'])))] for r in sel])
        x0 = np.log(CANON)
        gm = local_value_slope(x, [r['mu'] for r in sel], x0)
        if not gm or gm[0] <= 0:
            continue
        v = {k: -local_value_slope(x, [r[k] for r in sel], x0)[1] / (2 * gm[0])
             for k in ('a_c', 'a_n', 'a_t')}
        tot = sum(v.values())
        if abs(tot) > 1e-9:
            out.append((m, v['a_c'] / tot, v['a_n'] / tot, v['a_t'] / tot))
    return np.array(out)


fig, ax = plt.subplots(2, 1, figsize=(W, 4.5))
b = ax[0]
for tag, R in (('2D', R2), ('3D', R3)):
    m = np.array(sorted(R))
    b.errorbar(m, [R[x][0] for x in m], yerr=[R[x][1] for x in m],
               marker='o', ms=3.6, lw=0, elinewidth=1.0, capsize=2,
               color=C[tag], label=tag)
    sel = [x for x in m if x <= BRANCH[tag]]
    c, s = wline(np.array(sel), np.array([R[x][0] for x in sel]),
                 np.array([R[x][1] for x in sel]))
    xs = np.linspace(0, BRANCH[tag], 20)
    b.plot(xs, c[0] + c[1] * xs, '-', color=C[tag], lw=1.1)
    b.axhline(PUBLISHED[tag], ls=':', lw=1.0, color=C[tag])
    OUT[f'slope_{tag}'] = (c[1], s[1])
    print(f'  branch slope {tag}: {c[1]:+.3f} +/- {s[1]:.3f}')
b.text(0.0015, PUBLISHED['2D'] + 0.006, r'$p=1/8$ fitted for disks',
       fontsize=6.5, color=C['2D'])
b.text(0.0015, PUBLISHED['3D'] + 0.006, r'$p=1/6$ fitted for spheres',
       fontsize=6.5, color=C['3D'])
b.set_xscale('symlog', linthresh=0.05)
b.set_xlim(-0.004, 1.6); b.set_ylim(0, 0.34)
b.set_xticks([0, 0.05, 0.1, 0.2, 0.5, 1.0])
b.set_xticklabels(['0', '0.05', '0.1', '0.2', '0.5', '1'])
b.set_xticks([], minor=True)
b.set_xlabel(r'grain friction  $\mu_g$')
b.set_ylabel(r'$n$')
b.set_title('(a)  the exponent against grain friction', loc='left')
b.legend(frameon=False, loc='upper left', handletextpad=0.4)

a = ax[1]
sh = channel_shares()
for col, key, lab in ((1, 'geo', 'arrangement of contacts'),
                      (2, 'nrm', 'unequal normal forces'),
                      (3, 'tan', 'tangential grip')):
    a.plot(sh[:, 0], sh[:, col], 'o-', ms=3.4, lw=1.1, color=C[key], label=lab)
print('  channel shares (mu_g, geometric, normal, tangential):')
for row in sh:
    print('   ' + '  '.join(f'{v:+.3f}' for v in row))
a.axhline(0, color='k', lw=0.7)
a.set_xscale('symlog', linthresh=0.05)
a.set_xlim(-0.004, 1.6)
a.set_xticks([0, 0.05, 0.1, 0.2, 0.5, 1.0])
a.set_xticklabels(['0', '0.05', '0.1', '0.2', '0.5', '1'])
a.set_xticks([], minor=True)
a.set_xlabel(r'grain friction  $\mu_g$')
a.set_ylabel(r'share of $n$')
a.set_title('(b)  what carries the weakening', loc='left')
lo_, hi_ = a.get_ylim()
a.set_ylim(lo_, hi_ + 0.30 * (hi_ - lo_))
a.legend(frameon=False, loc='upper center', ncol=1, handletextpad=0.4)
fig.tight_layout(pad=0.4); fig.savefig('figL2_frictionlaw.png')
# the points behind panel (a), so its axis can be redrawn without rerunning
# the fits (make_fig2_axis_variants.py)
import json
json.dump({'R2': {str(k): v for k, v in R2.items()},
           'R3': {str(k): v for k, v in R3.items()},
           'branch': BRANCH, 'published': PUBLISHED},
          open('friction_law_points.json', 'w'), indent=1)
print('wrote figL2_frictionlaw.png')


# ------------------------------------------- figure 3: stiffness, and the coupling
# kappa = E/P.  The modulus lever holds P = 10 and the pressure lever holds
# E = 1e5, so each cell's kappa comes from the name of its own directory.
P_MOD, E_PRS = 10.0, 1.0e5
LEVERS = {'mod': ('sweep_lev_E', lev.LEVERS[0][2], ZMIN, False, +1,
                  lambda v: v / P_MOD, 'modulus scan'),
          'prs': ('sweep_lev_P', lev.LEVERS[1][2], ZMIN, True, -1,
                  lambda v: E_PRS / v, 'pressure scan')}


def lever_cells(key):
    d, rx, zmin, scale, sign, kof, _ = LEVERS[key]
    return lev.cells(d, rx), zmin, scale, sign, kof


def lever_slope(key, mg):
    """dn/dln(kappa) at one friction, on the window every cell shares."""
    C_, zmin, scale, sign, kof = lever_cells(key)
    vs = sorted(v for (m, v) in C_ if abs(m - mg) < 1e-9)
    keep = {v: lev.gated(C_[(mg, v)], zmin) for v in vs}
    keep = {v: k for v, k in keep.items() if len(k) >= 4}
    if len(keep) < 3:
        return None
    common = sorted(set.intersection(*(set(k) for k in keep.values())))
    if len(common) < 4:
        return None
    x, y, e = [], [], []
    for v in sorted(keep):
        t0 = CANON * (v / 10.0) if scale else CANON
        got = lev.n_of(C_[(mg, v)], common, zmin, t0)
        if got:
            x.append(np.log(v)); y.append(got[0]); e.append(got[1])
    if len(x) < 3:
        return None
    s, se = lev.wslope(np.array(x), np.array(y), np.array(e))
    return sign * s, se


fig, ax = plt.subplots(2, 1, figsize=(W, 4.5))
a = ax[0]
a.axhline(0, color='k', lw=0.8)
for key, mk in (('mod', 'o'), ('prs', 's')):
    C_, *_ = lever_cells(key)
    xs, ys, es = [], [], []
    for mg in sorted({m for (m, _) in C_}):
        got = lever_slope(key, mg)
        if got:
            xs.append(mg); ys.append(got[0]); es.append(got[1])
    # the two scans reach the same frictions, so they are dodged by 3% to keep
    # both error bars readable where they sit on top of each other
    dodge = 1.03 if key == 'mod' else 1 / 1.03
    a.errorbar(np.array(xs) * dodge, ys, yerr=es, marker=mk, ms=3.6, lw=0,
               elinewidth=1.0, capsize=2, color=C[key], label=LEVERS[key][6])
    print(f'  {key}: ' + '  '.join(f'{m:g}:{v:+.4f}' for m, v in zip(xs, ys)))
peak = max(R2, key=lambda m: R2[m][0])
a.axvline(peak, ls='--', lw=0.9, color='0.45')
a.text(peak * 1.1, -0.031, 'peak of\nFig. 2(a)', fontsize=6.5, color='0.35',
       va='bottom')
a.set_xscale('log')
plain_ticks(a, 'x', [0.05, 0.1, 0.2, 0.5, 1.0])
a.set_xlabel(r'grain friction  $\mu_g$')
a.set_ylabel(r'$dn/d\ln\kappa$')
a.set_title('(a)  stiffness moves the exponent both ways', loc='left')
a.legend(frameon=False, loc='upper left')

import confound_grid as cg
b = ax[1]
b.axhline(0, color='k', lw=0.8)
for mg, key, mk, lab in ((0.3, 'tan', 'o', r'$\mu_g=0.3$'),
                         (0.0, 'fl', 's', 'frictionless')):
    sl = cg.arm_slopes(mg)
    if not sl:
        print(f'  grid at mu_g = {mg}: too few cases')
        continue
    k = np.array([v[0] for v in sl.values()])
    s = np.array([v[1] for v in sl.values()])
    e = np.array([v[2] for v in sl.values()])
    o = np.argsort(k)
    b.errorbar(k[o], s[o], yerr=e[o], marker=mk, ms=3.8, lw=1.0,
               elinewidth=1.0, capsize=2, color=C[key], label=lab)
    print(f'  grid mu_g = {mg}: ' + '  '.join(
        f'kappa={kk:.0e} {ss:+.4f}+/-{ee:.4f}' for kk, ss, ee in zip(k[o], s[o], e[o])))
b.set_xscale('log')
b.set_xlabel(r'stiffness number  $\kappa=E/P$')
b.set_ylabel(r'$dn/d\ln\dot\gamma$')
b.set_title('(b)  shear-rate dependence against stiffness', loc='left')
# headroom above the frictionless points, so the key sits top right, clear of
# the data and next to the series it names first
lo_, hi_ = b.get_ylim()
b.set_ylim(lo_, hi_ + 0.30 * (hi_ - lo_))
b.legend(frameon=False, loc='upper right')
fig.tight_layout(pad=0.4); fig.savefig('figL3_stiffness.png')
print('wrote figL3_stiffness.png')


# =================================================== restructured figure set
# The restructured Letter tells the shear-rate result in one place and the
# size of the exponent in another, so its figures follow that grouping.
#   figL2_grid        the shear-rate dependence of n against stiffness
#   figL3_grainprops  (a) n against grain friction, (b) its change with
#                     stiffness, (c) the share of n carried by each term
# All three panels of figL3 share one linear axis in grain friction, so the
# friction law is drawn straight, as the text describes it.

fig, b = plt.subplots(figsize=(W, 2.35))
b.axhline(0, color='k', lw=0.8)
for mg, key, mk, lab in ((0.3, 'tan', 'o', r'$\mu_g=0.3$'),
                         (0.0, 'fl', 's', 'frictionless')):
    sl = cg.arm_slopes(mg)
    k = np.array([v[0] for v in sl.values()])
    s = np.array([v[1] for v in sl.values()])
    e = np.array([v[2] for v in sl.values()])
    o = np.argsort(k)
    b.errorbar(k[o], s[o], yerr=e[o], marker=mk, ms=3.8, lw=1.0,
               elinewidth=1.0, capsize=2, color=C[key], label=lab)
b.set_xscale('log')
b.set_xlabel(r'stiffness number  $\kappa=E/P$')
b.set_ylabel(r'$dn/d\ln\dot\gamma$')
lo_, hi_ = b.get_ylim()
b.set_ylim(lo_, hi_ + 0.30 * (hi_ - lo_))
b.legend(frameon=False, loc='upper right')
fig.tight_layout(pad=0.4); fig.savefig('figL2_grid.png')
print('wrote figL2_grid.png')

fig, ax = plt.subplots(3, 1, figsize=(W, 6.4), sharex=True)
a = ax[0]
for tag, R in (('2D', R2), ('3D', R3)):
    m = np.array(sorted(R))
    a.errorbar(m, [R[x][0] for x in m], yerr=[R[x][1] for x in m],
               marker='o', ms=3.6, lw=0, elinewidth=1.0, capsize=2,
               color=C[tag], label=tag)
    sel = [x for x in m if x <= BRANCH[tag]]
    c_, _ = wline(np.array(sel), np.array([R[x][0] for x in sel]),
                  np.array([R[x][1] for x in sel]))
    xs = np.linspace(0, BRANCH[tag], 20)
    a.plot(xs, c_[0] + c_[1] * xs, '-', color=C[tag], lw=1.1)
    a.axhline(PUBLISHED[tag], ls=':', lw=1.0, color=C[tag])
# the labels sit at the right, above the high-friction points and clear of
# the peak
a.text(1.03, PUBLISHED['3D'] + 0.006, r'spheres, $p=1/6$',
       fontsize=6.5, color=C['3D'], ha='right')
a.text(1.03, PUBLISHED['2D'] + 0.006, r'disks, $p=1/8$',
       fontsize=6.5, color=C['2D'], ha='right')
a.set_ylim(0, 0.34); a.set_ylabel(r'$n$')
a.set_title('(a)  the exponent', loc='left')
a.legend(frameon=False, loc='upper right', handletextpad=0.4)

a = ax[1]
a.axhline(0, color='k', lw=0.8)
for key, mk in (('mod', 'o'), ('prs', 's')):
    C_, *_ = lever_cells(key)
    xs, ys, es = [], [], []
    for mg in sorted({m for (m, _) in C_}):
        got = lever_slope(key, mg)
        if got:
            xs.append(mg); ys.append(got[0]); es.append(got[1])
    dodge = 0.008 if key == 'mod' else -0.008
    a.errorbar(np.array(xs) + dodge, ys, yerr=es, marker=mk, ms=3.6, lw=0,
               elinewidth=1.0, capsize=2, color=C[key], label=LEVERS[key][6])
peak = max(R2, key=lambda m: R2[m][0])
a.axvline(peak, ls='--', lw=0.9, color='0.45')
a.text(peak + 0.02, -0.031, 'peak of $n$\nin (a)', fontsize=6.5, color='0.35',
       va='bottom')
a.set_ylabel(r'$dn/d\ln\kappa$')
a.set_title('(b)  its change with stiffness', loc='left')
a.legend(frameon=False, loc='upper right')

a = ax[2]
sh = channel_shares()
for col, key, lab in ((1, 'geo', 'arrangement of contacts'),
                      (2, 'nrm', 'unequal normal forces'),
                      (3, 'tan', 'tangential grip')):
    a.plot(sh[:, 0], sh[:, col], 'o-', ms=3.4, lw=1.1, color=C[key], label=lab)
a.axhline(0, color='k', lw=0.7)
a.set_ylabel(r'share of $n$')
a.set_title('(c)  the share of $n$ carried by each term', loc='left')
lo_, hi_ = a.get_ylim()
a.set_ylim(lo_, hi_ + 0.30 * (hi_ - lo_))
a.legend(frameon=False, loc='upper center', handletextpad=0.4)
a.set_xlim(-0.03, 1.05)
a.set_xlabel(r'grain friction  $\mu_g$')
fig.tight_layout(pad=0.4); fig.savefig('figL3_grainprops.png')
print('wrote figL3_grainprops.png')
