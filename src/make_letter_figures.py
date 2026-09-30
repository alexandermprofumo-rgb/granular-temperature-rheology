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

  figL1   (a) mu(Theta) at fixed I, normalized, slopes fanning with friction
          (b) n against grain friction in both dimensions, with the branch fit
  figL2   (a) dn/dln(kappa) against grain friction, sign changing at the peak
          (b) n against kappa at mu_g = 0.3, modulus and pressure scans
  figL3   (a) the three shear-rate series at mu_g = 0.3, not parallel
          (b) the same at mu_g = 0, the internal null

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
C = {'2D': '#C44E52', '3D': '#4C72B0', 'mod': '#4C72B0', 'prs': '#55A868'}
RXM = r'log\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$'
OUT = {}

# Panel 1(a) draws a real packing rather than a cartoon, because the point of
# it is that a reader can see what one run is.  Both comparison Letters open
# with a picture of the system and this one did not.
PACKING = 'sweep_matched2d/data.m2d_seed1'
WX, WY = 42.0, 12.0             # the window drawn, in grain diameters
INK, GRAIN, EDGE, HOT = '#2b2b2b', '#c9d6e4', '#5b7fa6', '#c0504d'


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
        ax.annotate('', xy=(WX / 2 + s * 7.0, yy + s * 1.5),
                    xytext=(WX / 2 - s * 7.0, yy + s * 1.5),
                    arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.2))
    for xx in (-2.2, WX + 2.2):                  # barostat, along the gradient
        for s in (1, -1):
            ax.annotate('', xy=(xx, WY / 2 + s * 1.0),
                        xytext=(xx, WY / 2 + s * 7.5),
                        arrowprops=dict(arrowstyle='-|>', color='#4a7d52',
                                        lw=1.2, mutation_scale=8))
    ax.text(WX / 2, WY + 2.6, r'shear rate $\dot\gamma$', ha='center',
            va='bottom', color=INK, fontsize=7)
    ax.text(WX + 4.4, WY / 2, r'pressure $P$', ha='center', va='center',
            color='#4a7d52', fontsize=7, rotation=270)
    ax.text(0.0, -2.4, r'random forces from the bath, set by $\Theta$',
            ha='left', va='top', color=HOT, fontsize=7)
    ax.set_xlim(-7.0, WX + 7.0)
    ax.set_ylim(-4.2, WY + 3.6)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(a)  one run', loc='left')


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

# ---------------------------------------------------------------- figure 1
fig, ax = plt.subplots(3, 1, figsize=(W, 5.15),
                       gridspec_kw=dict(height_ratios=[0.62, 1.5, 1.65]))
panel_setup(ax[0])

a = ax[1]
SHOW = [0.0, 0.1, 0.3, 1.0]
shades = plt.cm.viridis(np.linspace(0.08, 0.82, len(SHOW)))
for col, mg in zip(shades, SHOW):
    if mg not in D2 or len(K2[mg]) < 4:
        continue
    pts = setpoint_means(D2[mg], K2[mg], ZMIN)
    if pts is None:
        continue
    th, mu = pts
    a.plot(th / CANON, mu / at_theta0(th, mu, CANON), 'o', ms=3.2, color=col,
           label=rf'$\mu_g={mg:g}$')
    n, _ = R2[mg]
    # the tangent at Theta_0, over the decade the frictions share, so the fan
    # is read where every series has data rather than where one of them runs on
    xs = np.array([max(0.3, th.min() / CANON), min(3.0, th.max() / CANON)])
    a.plot(xs, xs ** (-n), '-', color=col, lw=1.0)
a.set_xscale('log'); a.set_yscale('log')
a.set_xlim(0.07, 40)
plain_ticks(a, 'x', [0.1, 1, 10])
plain_ticks(a, 'y', [0.8, 0.9, 1.0, 1.1])
a.set_xlabel(r'$\Theta/\Theta_0$')
a.set_ylabel(r'$\mu/\mu(\Theta_0)$')
a.set_title(r'(b)  friction against temperature, $\In$ fixed'.replace(
    r'$\In$', '$I$'), loc='left')
a.legend(frameon=False, ncol=2, handletextpad=0.4, columnspacing=1.0,
         loc='upper right')

b = ax[2]
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
# name the value rather than its status: "published" left the reader to guess
# which published number, and whose
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
b.set_title('(c)  the exponent is a property of the grains', loc='left')
b.legend(frameon=False, loc='upper left', handletextpad=0.4)
fig.tight_layout(pad=0.4); fig.savefig('figL1_frictionlaw.png')
print('wrote figL1_frictionlaw.png')

# ---------------------------------------------------------------- figure 2
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
    x, y, e, ks = [], [], [], []
    for v in sorted(keep):
        t0 = CANON * (v / 10.0) if scale else CANON
        got = lev.n_of(C_[(mg, v)], common, zmin, t0)
        if got:
            x.append(np.log(v)); y.append(got[0]); e.append(got[1])
            ks.append(kof(v))
    if len(x) < 3:
        return None
    s, se = lev.wslope(np.array(x), np.array(y), np.array(e))
    return sign * s, se, np.array(ks), np.array(y), np.array(e)


fig, ax = plt.subplots(2, 1, figsize=(W, 4.5))
a = ax[0]
a.axhline(0, color='k', lw=0.8)
for key, mk in (('mod', 'o'), ('prs', 's')):
    C_, *_ = lever_cells(key)
    mus = sorted({m for (m, _) in C_})
    xs, ys, es = [], [], []
    for mg in mus:
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
a.text(peak * 1.1, -0.031, 'peak of\nFig. 1(b)', fontsize=6.5, color='0.35',
       va='bottom')
a.set_xscale('log')
plain_ticks(a, 'x', [0.05, 0.1, 0.2, 0.5, 1.0])
a.set_xlabel(r'grain friction  $\mu_g$')
a.set_ylabel(r'$dn/d\ln\kappa$')
a.set_title('(a)  stiffness moves the exponent both ways', loc='left')
a.legend(frameon=False, loc='upper left')

b = ax[1]
for key, mk in (('mod', 'o'), ('prs', 's')):
    got = lever_slope(key, 0.3)
    if not got:
        continue
    s, se, ks, y, e = got
    b.errorbar(ks, y, yerr=e, marker=mk, ms=3.6, lw=0, elinewidth=1.0,
               capsize=2, color=C[key],
               label=rf'{LEVERS[key][6]},  {s:+.3f}$\pm${se:.3f}')
    xs = np.array([ks.min(), ks.max()])
    c, _ = wline(np.log(ks), y, e)
    b.plot(xs, c[0] + c[1] * np.log(xs), '-', color=C[key], lw=1.0)
b.set_xscale('log')
b.set_xlabel(r'stiffness number  $\kappa=E/P$')
b.set_ylabel(r'$n$')
b.set_title(r'(b)  $\mu_g=0.3$, moved by modulus and by pressure', loc='left')
b.legend(frameon=False, loc='upper left')
fig.tight_layout(pad=0.4); fig.savefig('figL2_stiffness.png')
print('wrote figL2_stiffness.png')

# ---------------------------------------------------------------- figure 3
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
fig, ax = plt.subplots(2, 1, figsize=(W, 4.5))
for a, mg, ttl in ((ax[0], 0.3, '(a)  frictional grains, not parallel'),
                   (ax[1], 0.0, '(b)  frictionless, the internal null')):
    pools = [(tag, ARMDATA[tag][0], ARMDATA[tag][1][mg])
             for tag, _, _, _ in ARMS if mg in ARMDATA[tag][1]]
    got = nfit.common_theta0([p[2] for p in pools])
    if got is None:
        print(f'  mu_g = {mg}: no shared window')
        continue
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
        j = int(np.argmin(np.abs(np.log(th / Tuse))))
        got_n = nval(runs, keep, ZMIN, Tuse)
        e10 = np.log10(I)
        ex = f'{int(round(e10))}' if abs(e10 - round(e10)) < 0.05 \
            else f'{e10:.1f}'
        a.plot(th / Tuse, mu / at_theta0(th, mu, Tuse), 'o', ms=3.2, color=col,
               label=rf'$I=10^{{{ex}}}$,  $n={got_n[0]:.3f}$')
        xs = np.array([th.min(), th.max()]) / Tuse
        a.plot(xs, xs ** (-got_n[0]), '-', color=col, lw=1.0)
        print(f'  mu_g = {mg}, {tag}: n = {got_n[0]:.4f} +/- {got_n[1]:.4f}'
              f'  over {len(keep)} setpoints')
    a.set_xscale('log'); a.set_yscale('log')
    plain_ticks(a, 'x', [0.3, 0.5, 1, 2, 3])
    plain_ticks(a, 'y', [0.9, 1.0, 1.1])
    a.set_xlabel(r'$\Theta/\Theta_0$')
    a.set_ylabel(r'$\mu/\mu(\Theta_0)$')
    a.set_title(ttl, loc='left')
    a.legend(frameon=False, loc='lower left')
fig.tight_layout(pad=0.4); fig.savefig('figL3_separability.png')
print('wrote figL3_separability.png')
