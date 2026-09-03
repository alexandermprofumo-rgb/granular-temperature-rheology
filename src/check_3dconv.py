"""Reproduce section 6a (the 3D convergence correction) from sweep_3dconv.

run_3d_converge.sh runs nmeas = 400000 (5x production) at mu_g = 0, 0.3, 1.0,
six Theta setpoints, two seeds, and n is fitted from successive quarters of the
measurement window.
"""
import glob
import os
import re
import numpy as np
import nfit

RX = re.compile(r'log\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$')


def parse_quarter(path, q, nq=4):
    """parse_log, but restricted to quarter q (0-based) of the window."""
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
    if len(rows) < 4 * nq:
        return None
    a = np.array(rows)
    lo = int(len(a) * q / nq); hi = int(len(a) * (q + 1) / nq)
    a = a[lo:hi]
    c = {k: a[:, j] for j, k in enumerate(h)}
    out = dict(mu=float(c['v_muI'].mean()), Theta=float(c['v_Theta'].mean()),
               I=float(c['v_Iiner'].mean()))
    if 'v_P' in c:
        out['Pm'] = float(c['v_P'].mean()); out['Ps'] = float(c['v_P'].std())
    return out


def cells(q):
    out = {}
    for p in glob.glob('sweep_3dconv/log.mu*'):
        m = RX.match(os.path.basename(p))
        if not m:
            continue
        r = parse_quarter(p, q) if q is not None else nfit.parse_log(p)
        if not r:
            continue
        r['Tgran'] = float(m.group('T'))
        out.setdefault(float(m.group('mu')), []).append(r)
    return out


prod = {}
raw = nfit.load_sweep('sweep_run_3d/log.mu*',
                      r'log\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$')
for k, v in raw.items():
    prod.setdefault(float(dict(k)['mu']), []).extend(v)

MUS = [0.0, 0.3, 1.0]
print('sweep_3dconv: n from successive quarters of a 5x-length window')
print('(Theta_0 per friction = geometric centre common to the four quarters '
      'AND the production cell)\n')
print(f'{"mu_g":>6}{"Q1":>18}{"Q2":>18}{"Q3":>18}{"Q4":>18}{"production":>18}')
QS = [cells(q) for q in range(4)]
full = cells(None)
for mg in MUS:
    legs = [Q.get(mg) for Q in QS] + [prod.get(mg)]
    if any(l is None for l in legs):
        print(f'{mg:>6g}   missing'); continue
    got = nfit.common_theta0(legs)
    if got is None:
        print(f'{mg:>6g}   no common Theta range'); continue
    T0, _ = got
    cellvals = []
    for l in legs:
        f = nfit.fit_local_sys(l, T0)
        cellvals.append(f)
    s = f'{mg:>6g}'
    for f in cellvals:
        s += (f'{f["n"]:>+9.4f}+/-{f["tot"]:<8.4f}' if f else f'{"VOID":>18}')
    print(s)
    if cellvals[3] and cellvals[4]:
        d = cellvals[3]['n'] - cellvals[4]['n']
        e = float(np.hypot(cellvals[3]['tot'], cellvals[4]['tot']))
        print(f'{"":>6}   Q4 - production = {d:+.4f} +/- {e:.4f}  ({abs(d)/e:.1f} sigma)'
              f'   [Theta_0 = {T0:.3e}]')
