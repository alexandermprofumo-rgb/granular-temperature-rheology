"""How much shear strain do the production runs actually accumulate, and has
mu plateaued inside the measurement window?

in.granular_2d / in.granular_3d: dt = 0.001, gdot = 1e-3, nequil = 40000,
nmeas = 80000  ->  total strain gamma = gdot*dt*(nequil+nmeas) = 0.12,
of which the measurement window is 0.08.
"""
import glob
import os
import re
import numpy as np
import nfit

DT, GDOT = 0.001, 1.0e-3


def trace(path):
    L = open(path).readlines()
    i = next((k for k, l in enumerate(L) if 'BEGIN MEASUREMENT WINDOW' in l), None)
    if i is None:
        return None
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
        return None
    a = np.array(rows)
    c = {k: a[:, j] for j, k in enumerate(h)}
    return c


def drift(pattern, label):
    ds, mus0, mus1, strains = [], [], [], []
    for p in sorted(glob.glob(pattern))[:400]:
        c = trace(p)
        if c is None or 'v_muI' not in c:
            continue
        m = c['v_muI']; st = c['Step']
        h = len(m) // 2
        a, b = m[:h].mean(), m[h:].mean()
        if a <= 0:
            continue
        ds.append(b / a - 1.0)
        mus0.append(a); mus1.append(b)
        strains.append((st[-1] - st[0]) * DT * GDOT)
    if not ds:
        print(f'{label}: nothing parsed'); return
    ds = np.array(ds)
    print(f'{label:<34} n={len(ds):>4}   mu(late)/mu(early) - 1 : '
          f'mean {ds.mean():+.3%}, median {np.median(ds):+.3%}, '
          f'{100*(ds>0).mean():.0f}% positive')
    print(f'{"":<34} measurement-window strain = {np.mean(strains):.4f}')


print('SHEAR STRAIN PER RUN')
print(f'   production 2D/3D: equilibration {GDOT*DT*40000:.3f}, '
      f'measurement {GDOT*DT*80000:.3f}, total {GDOT*DT*120000:.3f}')
print('   (dense granular flow reaches a critical state at strain ~1-10)\n')
print('IS mu STILL DRIFTING INSIDE THE MEASUREMENT WINDOW?')
print('   (first half vs second half of the thermo rows after the marker)\n')
drift('sweep_matched2d/log.mu*', 'sweep_matched2d (2D, P=10)')
drift('sweep_run_3d/log.mu*', 'sweep_run_3d (3D, P=10)')
drift('sweep_pichi/log.P*', 'sweep_pichi (2D, P=5/50)')
drift('sweep_steadystate/log.mu*', 'sweep_steadystate (strain 1.5)')
drift('sweep_iscan/log.g3.162e-3*', 'sweep_iscan fast arm (I=1e-3)')
drift('sweep_iscan/log.g3.162e-4*', 'sweep_iscan slow arm (I=1e-4)')

print()
print('=' * 78)
print('n AT STRAIN 0.12 (production) vs STRAIN 1.5 (sweep_steadystate)')
print('=' * 78)
ss = {}
for k, v in nfit.load_sweep(
        'sweep_steadystate/log.mu*',
        r'log\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$').items():
    ss.setdefault(float(dict(k)['mu']), []).extend(v)
m2 = {}
for k, v in nfit.load_sweep(
        'sweep_matched2d/log.mu*',
        r'log\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$').items():
    m2.setdefault(float(dict(k)['mu']), []).extend(v)
print(f'   frictions in sweep_steadystate: {sorted(ss)}')
for mg in sorted(ss):
    if mg not in m2:
        continue
    got = nfit.common_theta0([ss[mg], m2[mg]])
    if got is None:
        print(f'   mu_g={mg}: no common Theta range'); continue
    T0, (lo, hi) = got
    a = nfit.fit_local_sys(m2[mg], T0)
    b = nfit.fit_local_sys(ss[mg], T0)
    if not (a and b):
        print(f'   mu_g={mg}: unfittable'); continue
    d = b['n'] - a['n']
    e = float(np.hypot(a['tot'], b['tot']))
    es = float(np.hypot(a['stat'], b['stat']))
    print(f'   mu_g={mg:<5g} production {a["n"]:+.4f}+/-{a["tot"]:.4f}   '
          f'strain-1.5 {b["n"]:+.4f}+/-{b["tot"]:.4f}   diff {d:+.4f} '
          f'({abs(d)/e:.1f}s tot, {abs(d)/es:.1f}s stat)  '
          f'[Theta_0={T0:.2e}, range {hi/lo:.1f}x]')
