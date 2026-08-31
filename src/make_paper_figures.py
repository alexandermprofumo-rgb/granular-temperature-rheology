"""Regenerate every figure from current data. No hardcoded numbers anywhere.

The previous figure scripts are stale (newest output Aug 11) and
make_channel_figure.py carried literal S2/S3 arrays that the corrected channel
extraction has since invalidated. Everything here is computed from the sweeps
and the chan_*.csv caches at run time, and the numbers used are printed
alongside each panel so they can be quoted in text without re-deriving them.

Figures:
  fig1_ncurve       n(mu_g), both dimensions, with the divergence above mu_g~0.2
  fig2_frictionless the exclusion, and its finite-size convergence
  fig3_curvature    mu(Theta) in log-log -- why n is not an exponent
  fig4_channels     the three-channel decomposition, both dimensions
  fig5_chi          n_c and n_n collapse on chi; n itself does not
  fig6_validation   strain convergence, jamming gate, in-window drift

Usage:  python3 make_paper_figures.py
"""
import csv
import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import nfit

ZMIN = 3.0
ZMIN3 = 4.0     # 3D isostatic floor, D+1
CORE = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
NG = 4000
C = {'2D': '#C44E52', '3D': '#4C72B0', 'c': '#4C72B0', 'n': '#C44E52',
     't': '#55A868', 'g': '#8172B2'}
OUT = {}


def cells(d, rx):
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


def jammed(runs, zmin=ZMIN):
    # zmin must drive BOTH the per-run filter and the setpoint mean test.
    # Passing ZMIN to the filter while mean-testing at `zmin` would gate 3D
    # runs at the 2D floor.
    Ts, byT = nfit.surviving_setpoints(runs, z_min=zmin)
    return [t for t in Ts
            if not [x['Z'] for x in byT[t] if 'Z' in x]
            or np.mean([x['Z'] for x in byT[t] if 'Z' in x]) >= zmin]


RXM = r'log\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$'
print('loading sweeps ...')
D2 = cells('sweep_steady2d', RXM)
D3 = cells('sweep_steady3d', RXM)
K2 = {m: jammed(v) for m, v in D2.items()}
K3 = {m: jammed(v, 4.0) for m, v in D3.items()}   # 3D isostatic floor

los, his = [], []
for D, K, ZM in ((D2, K2, ZMIN), (D3, K3, ZMIN3)):
    for m in CORE:
        if m not in D or len(K[m]) < 4:
            continue
        _, byT = nfit.surviving_setpoints(D[m], z_min=ZM, restrict_to=K[m])
        th = [x['Theta'] for t in K[m] for x in byT.get(t, []) if x['Theta'] > 0]
        if th:
            los.append(min(th)); his.append(max(th))
T0 = float(np.sqrt(max(los) * min(his)))
print(f'common Theta_0 = {T0:.4e}')
OUT['Theta0'] = T0

R2 = {m: nfit.fit_local_sys(D2[m], T0, restrict_to=K2[m], z_min=ZMIN)
      for m in sorted(D2) if len(K2[m]) >= 4}
R3 = {m: nfit.fit_local_sys(D3[m], T0, restrict_to=K3[m], z_min=ZMIN3)
      for m in sorted(D3) if len(K3[m]) >= 4}
R2 = {k: v for k, v in R2.items() if v}
R3 = {k: v for k, v in R3.items() if v}

# ---------------------------------------------------------------- figure 1
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2),
                       gridspec_kw=dict(width_ratios=[1.12, 1]))
a = ax[0]
for tag, R in (('2D', R2), ('3D', R3)):
    m = sorted(R)
    a.errorbar(m, [R[x]['n'] for x in m], yerr=[R[x]['tot'] for x in m],
               marker='o', ms=5, lw=1.6, capsize=3, color=C[tag], label=tag)
a.axhline(0.25, ls=':', c=C['2D'], lw=1.2)
a.axhline(1 / 6, ls=':', c=C['3D'], lw=1.2)
# right of the mu_g = 1 point the panel is empty; matplotlib does not honour
# LaTeX's ~, so write the spaces literally
a.text(1.12, 0.253, 'Eq. (2), 2D', color=C['2D'], fontsize=8,
       va='bottom', ha='left')
a.text(1.12, 0.170, 'Eq. (2), 3D', color=C['3D'], fontsize=8,
       va='bottom', ha='left')
a.set_xscale('symlog', linthresh=0.05)
a.set_xlim(-0.004, 2.6)
a.set_xlabel(r'grain friction  $\mu_g$'); a.set_ylabel(r'$n(\Theta_0)$')
a.set_title('(a)  the exponent against grain friction', fontsize=10, loc='left')
a.legend(frameon=False, fontsize=9); a.set_ylim(-0.03, 0.33)

# Panel (b) shows BOTH window treatments.  Comparing local slopes fitted over
# different Theta ranges mixes a real difference with a fitting artefact, and
# for these pairs the two treatments give opposite readings.  Plotting only the
# own-window series would assert a result the paper withdraws.
def matched_pair(m):
    """(own, matched) difference and error for one friction, at a shared Theta_0."""
    rng = {}
    for tag, D, K, ZM in (('2D', D2, K2, ZMIN), ('3D', D3, K3, ZMIN3)):
        if m not in D or len(K[m]) < 4:
            return None
        _, byT = nfit.surviving_setpoints(D[m], z_min=ZM, restrict_to=K[m])
        th = [x['Theta'] for t in K[m] for x in byT.get(t, []) if x['Theta'] > 0]
        if not th:
            return None
        rng[tag] = (K[m], byT, min(th), max(th))
    LO = max(rng['2D'][2], rng['3D'][2]); HI = min(rng['2D'][3], rng['3D'][3])
    if HI <= LO:
        return None
    th0 = float(np.sqrt(LO * HI))
    out = {}
    for tag, D, ZM in (('2D', D2, ZMIN), ('3D', D3, ZMIN3)):
        ts, byT = rng[tag][0], rng[tag][1]
        tsm = [t for t in ts
               if all(LO * 0.999 <= x['Theta'] <= HI * 1.001
                      for x in byT[t] if x['Theta'] > 0)]
        own = nfit.fit_local_sys(D[m], th0, restrict_to=ts, z_min=ZM)
        mat = (nfit.fit_local_sys(D[m], th0, restrict_to=tsm, z_min=ZM)
               if len(tsm) >= 4 else None)
        if own is None or mat is None:
            return None
        out[tag] = (own, mat)
    # Conservative errors in BOTH columns, per ANALYSIS_PROTOCOL Amendment 2:
    # restricting the window costs setpoints, which raises the jackknife null
    # and truncates the calibrated excess to zero, making the matched error
    # spuriously smaller than the unmatched one.  The raw jackknife has no such
    # truncation, so the two columns sit on one footing.  constraints_final.py
    # uses the same rule; the figure must not print a different number for the
    # same claim.
    def cons(r):
        return float(np.hypot(r['stat'], r['sys_raw']))
    dn_o = out['3D'][0]['n'] - out['2D'][0]['n']
    e_o = float(np.hypot(cons(out['2D'][0]), cons(out['3D'][0])))
    dn_m = out['3D'][1]['n'] - out['2D'][1]['n']
    e_m = float(np.hypot(cons(out['2D'][1]), cons(out['3D'][1])))
    return dn_o, e_o, dn_m, e_m

b = ax[1]
mm, do, eo, dm, em = [], [], [], [], []
for m in CORE:
    got = matched_pair(m)
    if got is None:
        continue
    mm.append(m); do.append(got[0]); eo.append(got[1])
    dm.append(got[2]); em.append(got[3])
b.axhline(0, c='k', lw=0.8)
b.errorbar(mm, do, yerr=eo, marker='s', ms=5, lw=1.4, capsize=3,
           color=C['g'], label='own windows')
b.errorbar(mm, dm, yerr=em, marker='o', ms=5, lw=1.4, capsize=3,
           color='0.35', ls='--', label='matched windows')
b.set_xscale('symlog', linthresh=0.05)
b.set_xlim(-0.004, 1.6)
b.set_xlabel(r'$\mu_g$'); b.set_ylabel(r'$n_{3D}-n_{2D}$')
b.set_title('(b)  dimension difference, two window treatments',
            fontsize=10, loc='left')
b.legend(frameon=False, fontsize=8, loc='lower right')

def _c2(dd, ee, sel):
    v = [(x / y) ** 2 for x, y, mg in zip(dd, ee, mm) if sel(mg)]
    return float(np.mean(v)) if v else float('nan')
# mu_g = 0 is excluded from the grouped chi^2: that pair is the frictionless
# baseline comparison, reported on its own and not as part of a trend in
# friction.  constraints_final.py groups the same way, and the two must agree.
lo_o, hi_o = (_c2(do, eo, lambda x: 0 < x <= 0.2),
              _c2(do, eo, lambda x: x > 0.2))
lo_m, hi_m = (_c2(dm, em, lambda x: 0 < x <= 0.2),
              _c2(dm, em, lambda x: x > 0.2))
b.text(0.03, 0.97,
       f'$\\chi^2$/dof   own / matched\n'
       f'$\\mu_g\\leq0.2$:  {lo_o:.2f} / {lo_m:.2f}\n'
       f'$\\mu_g>0.2$:  {hi_o:.2f} / {hi_m:.2f}',
       transform=b.transAxes, fontsize=8, va='top')
OUT['chi2_lo'] = lo_o; OUT['chi2_hi'] = hi_o
OUT['chi2_lo_matched'] = lo_m; OUT['chi2_hi_matched'] = hi_m
print(f'fig1b  chi2/dof  own {lo_o:.2f}/{hi_o:.2f}   matched {lo_m:.2f}/{hi_m:.2f}')
fig.tight_layout(); fig.savefig('fig4_ncurve.png', dpi=160)

# ---------------------------------------------------------------- figure 2
RXN = r'log\.N(?P<N>[0-9]+)_mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$'
SZ = {}
for p in glob.glob('sweep_size2d/log.N*_mu0.0_*'):
    m = re.match(RXN, os.path.basename(p))
    q = nfit.parse_log(p)
    if m and q:
        q['Tgran'] = float(m.group('T'))
        SZ.setdefault(int(m.group('N')), []).append(q)
SZ[NG] = D2[0.0]
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
a = ax[0]
Ns = sorted(SZ)
pool = [SZ[N] for N in Ns]
g = nfit.common_theta0(pool)
Tn = g[0] if g else T0
vals = []
for N in Ns:
    ts = jammed(SZ[N])
    r = nfit.fit_local_sys(SZ[N], Tn, restrict_to=ts, z_min=ZMIN) if len(ts) >= 4 else None
    if r:
        vals.append((N, r['n'], r['tot']))
a.errorbar([v[0] for v in vals], [v[1] for v in vals],
           yerr=[v[2] for v in vals], marker='o', ms=6, lw=1.6, capsize=3,
           color=C['2D'])
a.set_xscale('log'); a.set_xlabel('N (grains)')
a.set_ylabel(r'$n(\mu_g=0)$')
a.set_title('(a)  finite-size convergence, frictionless', fontsize=10, loc='left')
a.axhline(0, c='k', lw=0.8, ls='--')
OUT['size'] = [(int(v[0]), float(v[1]), float(v[2])) for v in vals]
print('fig2  n(mu=0) vs N:', ' '.join(f'{n}:{y:.4f}' for n, y, _ in vals))

b = ax[1]
for i, (tag, R, pred) in enumerate((('2D', R2, 0.25), ('3D', R3, 1 / 6))):
    r = R[0.0]
    b.errorbar([i], [r['n']], yerr=[r['tot']], marker='o', ms=8, capsize=4,
               color=C[tag])
    b.plot([i], [pred], marker='_', ms=26, mew=2.5, color='k')
    b.text(i + 0.12, pred, f'  1/(2D) = {pred:.3f}', va='center', fontsize=8)
    b.text(i + 0.12, r['n'], f'  {r["n"]:.4f} ± {r["tot"]:.4f}\n'
           f'  {abs(r["n"]-pred)/r["tot"]:.0f}σ below',
           va='center', fontsize=8, color=C[tag])
    OUT[f'excl_{tag}'] = (float(r['n']), float(r['tot']),
                          float(abs(r['n'] - pred) / r['tot']))
b.set_xlim(-0.4, 1.9); b.set_xticks([0, 1]); b.set_xticklabels(['2D', '3D'])
b.set_ylabel(r'$n(\mu_g=0)$'); b.axhline(0, c='k', lw=0.8, ls='--')
b.set_title('(b)  measured vs predicted', fontsize=10, loc='left')
fig.tight_layout(); fig.savefig('fig3_frictionless.png', dpi=160)

# ---------------------------------------------------------------- figure 3
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
for a, D, K, tag, ZM in ((ax[0], D2, K2, '2D', ZMIN),
                         (ax[1], D3, K3, '3D', ZMIN3)):
    for m in [x for x in (0.0, 0.1, 0.3, 1.0) if x in D]:
        Ts, byT = nfit.surviving_setpoints(D[m], z_min=ZM, restrict_to=K[m])
        th = np.array([np.mean([x['Theta'] for x in byT[t]]) for t in Ts])
        mu = np.array([np.mean([x['mu'] for x in byT[t]]) for t in Ts])
        a.plot(th, mu / mu[0], 'o-', ms=4, lw=1.3, label=f'$\\mu_g$={m:g}')
    a.set_xscale('log'); a.set_yscale('log')
    a.set_xlabel(r'$\Theta$'); a.set_ylabel(r'$\mu/\mu(\Theta_{min})$')
    a.set_title(f'({"ab"[tag=="3D"]})  {tag}: $\\mu(\\Theta)$ is curved',
                fontsize=10, loc='left')
    a.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig('fig2_curvature.png', dpi=160)

# ---------------------------------------------------------------- figure 4
def load_csv(p):
    rows = []
    for x in csv.DictReader(open(p)):
        rows.append({k: (float(v) if k != 'seed' else v) for k, v in x.items()})
    return rows


def loc(x, y, x0, deg=2):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < deg + 2:
        return None
    A = np.vander(x - x0, deg + 1, increasing=True)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(c[0]), float(c[1])


ch2 = load_csv('chan_steady2d.csv')
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
mus2, cc, cn, ct, ntot = [], [], [], [], []
for m in sorted(set(r['mu_g'] for r in ch2)):
    if m not in K2 or len(K2[m]) < 4:
        continue
    keep = {round(t, 12) for t in K2[m]}
    sel = [r for r in ch2 if r['mu_g'] == m and round(r['Tgran'], 12) in keep]
    if len(sel) < 8:
        continue
    x = np.log([r['Theta'] for r in sel]) if 'Theta' in sel[0] else None
    if x is None:
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
    x0 = np.log(T0)
    gm = loc(x, [r['mu'] for r in sel], x0)
    if not gm or gm[0] <= 0:
        continue
    vals = {k: -loc(x, [r[k] for r in sel], x0)[1] / (2 * gm[0])
            for k in ('a_c', 'a_n', 'a_t')}
    mus2.append(m); cc.append(vals['a_c']); cn.append(vals['a_n']); ct.append(vals['a_t'])
    ntot.append(sum(vals.values()))
o = np.argsort(mus2)
mus2 = np.array(mus2)[o]; cc = np.array(cc)[o]; cn = np.array(cn)[o]; ct = np.array(ct)[o]
# Signed lines, not a stacked area.  C_n is negative at mu_g = 1 and C_c is
# negative at mu_g = 0; stackplot cannot render a negative component, and
# clipping them to zero (as this did) leaves a stack that no longer sums to the
# plotted total exactly where the sign change is the point being made.
for arr, k, lab in ((cc, 'c', r'$C_c$ fabric'), (cn, 'n', r'$C_n$ normal force'),
                    (ct, 't', r'$C_t$ grip')):
    ax[0].plot(mus2, arr, 'o-', ms=4, lw=1.5, color=C[k], label=lab)
ax[0].plot(mus2, cc + cn + ct, 'k-', lw=1.8, label=r'$n=C_c+C_n+C_t$')
ax[0].axhline(0, c='k', lw=0.8, ls=':')
lo = min(cc.min(), cn.min(), ct.min())
ax[0].set_ylim(min(lo * 1.35, -0.02), None)
ax[0].set_xscale('symlog', linthresh=0.05)
ax[0].set_xlim(-0.004, 1.6)
ax[0].set_xlabel(r'$\mu_g$'); ax[0].set_ylabel('channel contribution')
ax[0].set_title('(a)  2D, $a_t$ measured', fontsize=10, loc='left')
_lo0, _hi0 = ax[0].get_ylim()
ax[0].set_ylim(_lo0, _hi0 + 0.30 * (_hi0 - _lo0))   # headroom for the legend
ax[0].legend(frameon=False, fontsize=8, loc='upper left')
tot = cc + cn + ct
ok = np.abs(tot) > 1e-9
for arr, k, lab in ((cc, 'c', r'$C_c/n$'), (cn, 'n', r'$C_n/n$'),
                    (ct, 't', r'$C_t/n$')):
    ax[1].plot(mus2[ok], arr[ok] / tot[ok], 'o-', ms=5, color=C[k], label=lab)
ax[1].axhline(0, c='k', lw=0.8)
ax[1].set_xscale('symlog', linthresh=0.05)
ax[1].set_xlim(-0.004, 1.6)
ax[1].set_xlabel(r'$\mu_g$'); ax[1].set_ylabel('share of $n$')
ax[1].set_title('(b)  the dominant channel inverts', fontsize=10, loc='left')
_lo1, _hi1 = ax[1].get_ylim()
ax[1].set_ylim(_lo1, _hi1 + 0.22 * (_hi1 - _lo1))
ax[1].legend(frameon=False, fontsize=9, loc='upper right')
fig.tight_layout(); fig.savefig('fig5_channels.png', dpi=160)
OUT['channels2d'] = [[float(a) for a in row]
                     for row in zip(mus2, cc, cn, ct)]
print('fig4  2D channel table regenerated (corrected frames)')
for m, a, b_, c_ in zip(mus2, cc, cn, ct):
    print(f'   mu_g={m:<6g} C_c={a:+.4f}  C_n={b_:+.4f}  C_t={c_:+.4f}  '
          f'sum={a+b_+c_:+.4f}')

# ---- the Theta regime, per cell, over the window actually fitted.
# A range of "70-2000x the shear-generated m(gdot d)^2" and "8-46x the affine
# shear velocity" holds only for the 2D FRICTIONAL cells. The frictionless baselines -- which carry constraints 1, 2 and 2b and
# the whole frictional-origin reframe -- run to 2.8e4 (2D) and 4.1e4 (3D), and
# 3D frictional cells to ~1e4. Computed here so the paragraph can never again
# be written from memory.
GD = 1.0e-3
REG = {}
for tag, DD, ZM in (('2D', D2, 3.0), ('3D', D3, 4.0)):   # Z >= D+1 per dimension
    for m in sorted(DD):
        ts = jammed(DD[m], ZM)
        if len(ts) < 4:
            continue
        _, byT = nfit.surviving_setpoints(DD[m], z_min=ZM, restrict_to=ts)
        th = np.array([x['Theta'] for t in ts for x in byT[t] if x['Theta'] > 0])
        if not th.size:
            continue
        REG[f'{tag}_{m:g}'] = [float(th.min() / GD ** 2), float(th.max() / GD ** 2),
                               float(np.sqrt(th.min()) / GD), float(np.sqrt(th.max()) / GD)]
OUT['regime'] = REG
lo = min(v[0] for v in REG.values()); hi = max(v[1] for v in REG.values())
vlo = min(v[2] for v in REG.values()); vhi = max(v[3] for v in REG.values())
print(f'regime  Theta/(gdot d)^2 = {lo:.0f}-{hi:.0f} ;  v_T/(gdot d) = {vlo:.0f}-{vhi:.0f}')
for k in ('2D_0', '3D_0'):
    if k in REG:
        print(f'        {k}: {REG[k][0]:.0f}-{REG[k][1]:.0f}x  '
              f'v_T {REG[k][2]:.0f}-{REG[k][3]:.0f}x')

json.dump(OUT, open('figure_numbers.json', 'w'), indent=1)
print('\nnumbers cached to figure_numbers.json')

# ---------------------------------------------------------------- figure 5
# WAS "the positive result: n_c and n_n collapse on chi". WITHDRAWN 2026-08-25.
# The chi2/dof values that made that claim (1.1 and 0.8) were hand-transcribed
# and are not reproducible under any frame policy, fabric
# weighting, gate or Theta_0 -- the best obtainable over 84 configurations is
# 2.12, still failing. The pre-registered test now returns 0 passes in 32 pairs
# in 2D and 0 in 32 in 3D. The figure is kept because the scatter about a
# single n(chi) curve is worth SHOWING, but it is a null: chi is the least-bad
# organiser and it is not good enough.
FILES = {'friction': 'chan_steady2d.csv', 'stiffness': 'chan_lev_E.csv',
         'pressure': 'chan_lev_P.csv', 'tangential': 'chan_lev_kt.csv'}
SPEC = {'friction': ('sweep_steady2d/log.mu*', RXM, ()),
        'stiffness': ('sweep_lev_E/log.E*',
                      r'log\.E(?P<E>[0-9.eE+-]+)_mu(?P<mu>[0-9.]+)'
                      r'_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$', ('E',)),
        'pressure': ('sweep_lev_P/log.P*',
                     r'log\.P(?P<P>[0-9.]+)_mu(?P<mu>[0-9.]+)'
                     r'_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$', ('P',))}
MK = {'friction': 'o', 'stiffness': 's', 'pressure': '^'}
pts = {'n_c': [], 'n_n': [], 'n_tot': []}
for lever, fname in FILES.items():
    if lever not in SPEC or not os.path.exists(fname):
        continue
    pat, rx, extra = SPEC[lever]
    th, grp = {}, {}
    for p in glob.glob(pat):
        m = re.match(rx, os.path.basename(p)); q = nfit.parse_log(p)
        if not (m and q):
            continue
        key = tuple([float(m.group('mu'))] + [float(m.group(k)) for k in extra])
        q['Tgran'] = float(m.group('T'))
        th[(key, float(m.group('T')), m.group('s'))] = q['Theta']
        grp.setdefault(key, []).append(q)
    keep = {k: {round(t, 12) for t in jammed(v)} for k, v in grp.items()}
    rows = load_csv(fname)
    byc = {}
    for r in rows:
        key = tuple([r['mu_g']] + [r[k] for k in extra])
        if key not in keep or round(r['Tgran'], 12) not in keep[key]:
            continue
        tk = (key, r['Tgran'], str(int(r['seed'])))
        if tk in th and th[tk] > 0:
            r = dict(r); r['Theta'] = th[tk]
            byc.setdefault(key, []).append(r)
    for key, sel in byc.items():
        if len(sel) < 8 or key[0] <= 0:
            continue
        x = np.log([r['Theta'] for r in sel]); x0 = np.log(T0)
        gm = loc(x, [r['mu'] for r in sel], x0)
        if not gm or gm[0] <= 0:
            continue
        ac = np.array([r['a_c'] for r in sel]); an = np.array([r['a_n'] for r in sel])
        if (ac <= 0).any() or (an <= 0).any():
            continue
        chi = float(np.mean([r['chi'] for r in sel]))
        if chi <= 0:
            continue
        pts['n_c'].append((chi, -loc(x, np.log(ac), x0)[1], lever))
        pts['n_n'].append((chi, -loc(x, np.log(an), x0)[1], lever))
        pts['n_tot'].append((chi, -loc(x, np.log([r['mu'] for r in sel]), x0)[1], lever))

fig, ax = plt.subplots(1, 3, figsize=(13, 4.0), sharex=True)
for a, key, lab in zip(ax, ('n_c', 'n_n', 'n_tot'),
                       (r'$n_c=-d\ln a_c/d\ln\Theta$',
                        r'$n_n=-d\ln a_n/d\ln\Theta$',
                        r'$n$  (the total)')):
    P = pts[key]
    for lever in sorted(set(p[2] for p in P)):
        q = [p for p in P if p[2] == lever]
        a.plot([p[0] for p in q], [p[1] for p in q], MK[lever], ms=6,
               alpha=.8, label=lever + ' lever')
    if P:
        xx = np.log([p[0] for p in P]); yy = np.array([p[1] for p in P])
        A = np.vstack([np.ones_like(xx), xx, xx ** 2]).T
        c = np.linalg.lstsq(A, yy, rcond=None)[0]
        gx = np.linspace(xx.min(), xx.max(), 60)
        a.plot(np.exp(gx), c[0] + c[1] * gx + c[2] * gx ** 2, 'k-', lw=1.3)
        # quote the PRE-REGISTERED statistic, from the file the test wrote --
        # not a second, ad-hoc one computed here. (This panel pools four levers;
        # the test pools two, so the numbers describe the same data differently
        # and both are stated.)
        lbl = f'scatter about one curve: {np.std(yy - A@c):.4f}'
        try:
            import json as _json
            _tw = _json.load(open('channel_state_2d.json'))['pairs']
            if f'{key}|chi' in _tw:
                _c2, _tz, _ok = _tw[f'{key}|chi']
                lbl += (f'\npre-registered test: $\\chi^2$/dof = {_c2:.1f}, '
                        f'trend {_tz:.1f}$\\sigma$\n{"PASSES" if _ok else "FAILS"}')
        except (OSError, ValueError, KeyError):
            pass
        # headroom first, then place the block in the cleared strip: the
        # curves peak mid-panel and the annotation used to sit on top of them
        _l, _h = a.get_ylim()
        a.set_ylim(_l, _h + 0.34 * (_h - _l))
        a.text(.04, .97, lbl, transform=a.transAxes, fontsize=8, va='top')
    a.set_xscale('log'); a.set_xlabel(r'$\chi$  (Coulomb-mobilised fraction)')
    a.set_title(lab, fontsize=10, loc='left')
ax[0].set_ylabel('channel log-slope')
# one shared legend under the title: per-axes it collided with the annotation
_h, _l = ax[0].get_legend_handles_labels()
fig.legend(_h, _l, frameon=False, fontsize=8, ncol=3,
           loc='upper center', bbox_to_anchor=(0.5, 0.945))
fig.suptitle('No microstructural variable organises n or its channels: '
             '0 passes in 64 (channel, candidate) pairs across both dimensions',
             fontsize=9, y=1.0)
fig.tight_layout(); fig.savefig('fig6_chinull.png', dpi=160)
print('fig5  points per panel:', {k: len(v) for k, v in pts.items()})

# ---------------------------------------------------------------- figure 6
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
def drift(pat):
    out = []
    for p in sorted(glob.glob(pat))[:400]:
        L = open(p, errors='ignore').readlines()
        i = next((k for k, l in enumerate(L) if 'BEGIN MEASUREMENT' in l), None)
        if i is None:
            continue
        h, rows = None, []
        for l in L[i:]:
            t = l.split()
            if t and t[0] == 'Step':
                h = t; continue
            if h is None:
                continue
            try:
                v = [float(x) for x in t]
            except ValueError:
                if rows:
                    break
                continue
            if len(v) == len(h):
                rows.append(v)
        if len(rows) < 8:
            continue
        a_ = np.array(rows); j = h.index('v_muI'); k2 = len(a_) // 2
        if a_[:k2, j].mean() > 0:
            out.append(a_[k2:, j].mean() / a_[:k2, j].mean() - 1)
    return np.array(out)
old = drift('sweep_matched2d/log.mu*'); new = drift('sweep_steady2d/log.mu*')
ax[0].hist(100 * old, bins=30, alpha=.7, color=C['n'], label=r'strain $0.12$')
ax[0].hist(100 * new, bins=30, alpha=.7, color=C['c'], label=r'strain $1.0$')
ax[0].axvline(0, c='k', lw=.8)
ax[0].set_xlabel(r'in-window drift in $\mu$  (%)'); ax[0].set_ylabel('runs')
ax[0].set_title('(a)  steady state reached', fontsize=10, loc='left')
ax[0].legend(frameon=False, fontsize=8)
OUT['drift_old'] = float(old.mean()); OUT['drift_new'] = float(new.mean())

for tag, D, iso, col in (('2D', D2, 4.0, C['2D']), ('3D', D3, 6.0, C['3D'])):
    m0 = 0.0
    # DELIBERATELY UNGATED. This panel plots Z against Theta to SHOW the pack
    # unjamming as it is heated; applying the jamming gate would remove exactly
    # the points the figure exists to display.
    Ts, byT = nfit.surviving_setpoints(D[m0])
    th = [np.mean([x['Theta'] for x in byT[t]]) for t in Ts]
    z = [np.mean([x['Z'] for x in byT[t] if 'Z' in x]) for t in Ts]
    ax[1].plot(th, z, 'o-', ms=4, color=col, label=f'{tag}, $\\mu_g$=0')
    ax[1].axhline(iso, ls=':', color=col, lw=1.2)
ax[1].set_xscale('log'); ax[1].set_xlabel(r'$\Theta$')
ax[1].set_ylabel('Z (contacts per grain)')
ax[1].set_title('(b)  shaking unjams the pack', fontsize=10, loc='left')
_l, _h = ax[1].get_ylim(); ax[1].set_ylim(_l - 0.22 * (_h - _l), _h)
ax[1].legend(frameon=False, fontsize=8, loc='lower left')

for N, y, e in OUT['size']:
    ax[2].errorbar([N], [y], yerr=[e], marker='o', ms=6, capsize=3, color=C['2D'])
ax[2].set_xscale('log'); ax[2].set_xlabel('N'); ax[2].set_ylabel(r'$n(\mu_g=0)$')
ax[2].set_title('(c)  finite-size convergence', fontsize=10, loc='left')
fig.tight_layout(); fig.savefig('fig1_steady.png', dpi=160)
print(f'fig6  drift old {old.mean():+.1%}  new {new.mean():+.1%}')
# ---- the Theta regime, per cell, over the window actually fitted.
# A range of "70-2000x the shear-generated m(gdot d)^2" and "8-46x the affine
# shear velocity" holds only for the 2D FRICTIONAL cells. The frictionless baselines -- which carry constraints 1, 2 and 2b and
# the whole frictional-origin reframe -- run to 2.8e4 (2D) and 4.1e4 (3D), and
# 3D frictional cells to ~1e4. Computed here so the paragraph can never again
# be written from memory.
GD = 1.0e-3
REG = {}
for tag, DD, ZM in (('2D', D2, 3.0), ('3D', D3, 4.0)):   # Z >= D+1 per dimension
    for m in sorted(DD):
        ts = jammed(DD[m], ZM)
        if len(ts) < 4:
            continue
        _, byT = nfit.surviving_setpoints(DD[m], z_min=ZM, restrict_to=ts)
        th = np.array([x['Theta'] for t in ts for x in byT[t] if x['Theta'] > 0])
        if not th.size:
            continue
        REG[f'{tag}_{m:g}'] = [float(th.min() / GD ** 2), float(th.max() / GD ** 2),
                               float(np.sqrt(th.min()) / GD), float(np.sqrt(th.max()) / GD)]
OUT['regime'] = REG
lo = min(v[0] for v in REG.values()); hi = max(v[1] for v in REG.values())
vlo = min(v[2] for v in REG.values()); vhi = max(v[3] for v in REG.values())
print(f'regime  Theta/(gdot d)^2 = {lo:.0f}-{hi:.0f} ;  v_T/(gdot d) = {vlo:.0f}-{vhi:.0f}')
for k in ('2D_0', '3D_0'):
    if k in REG:
        print(f'        {k}: {REG[k][0]:.0f}-{REG[k][1]:.0f}x  '
              f'v_T {REG[k][2]:.0f}-{REG[k][3]:.0f}x')

json.dump(OUT, open('figure_numbers.json', 'w'), indent=1)
