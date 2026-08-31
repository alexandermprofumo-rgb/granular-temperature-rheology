"""Finite-size and restitution scans, run exactly as pre-registered.

ANALYSIS_PROTOCOL.md was written 2026-08-19 09:13, before either sweep existed:

  section 5.1  finite size    statistic: weighted slope of n against ln N, at a
                              common Theta_0, per friction. PASS if |slope| <
                              2 sigma at all three frictions.
                              Prediction on record: PASS.
  section 5.2  restitution    statistic: weighted slope of n against e, per
                              friction. PASS if |slope| < 2 sigma.
                              Prediction on record: none.

Both jamming-gated (Z >= 3), calibrated errors, slope covariance propagated
from the per-cell errors and never rescaled by residuals.

Also reports Theta_rot/Theta, which these runs carry for the first time -- the
input to the section 5.4 test on rotational temperature.

Usage:  python3 analyze_size_restitution.py
"""
import glob
import os
import re

import numpy as np

import nfit

ZMIN = 3.0


def parse_extra(path):
    """parse_log plus the rotational temperature column, if present."""
    q = nfit.parse_log(path)
    if q is None:
        return None
    L = open(path).readlines()
    i = next((k for k, l in enumerate(L) if 'BEGIN MEASUREMENT WINDOW' in l), None)
    h, rows = None, []
    for l in L[i:]:
        t = l.split()
        if t and t[0] == 'Step':
            h = t
            continue
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
    if rows and h and 'v_Trot' in h:
        a = np.array(rows)[len(rows) // 3:]
        q['Trot'] = float(a[:, h.index('v_Trot')].mean())
    return q


def cells(sweep, rx, keyname):
    out = {}
    for p in glob.glob(f'{sweep}/log.*'):
        m = re.match(rx, os.path.basename(p))
        if not m:
            continue
        q = parse_extra(p)
        if not q:
            continue
        q['Tgran'] = float(m.group('T'))
        out.setdefault((float(m.group(keyname)), float(m.group('mu'))), []).append(q)
    return out


def jammed(runs):
    Ts, byT = nfit.surviving_setpoints(runs, z_min=ZMIN)
    return [t for t in Ts
            if not [x['Z'] for x in byT[t] if 'Z' in x]
            or np.mean([x['Z'] for x in byT[t] if 'Z' in x]) >= ZMIN]


def wslope(x, y, e):
    x = np.asarray(x, float); y = np.asarray(y, float); e = np.asarray(e, float)
    A = np.vstack([np.ones_like(x), x]).T
    W = np.diag(1 / e ** 2)
    cov = np.linalg.inv(A.T @ W @ A)
    c = cov @ A.T @ W @ y
    chi2 = float((((y - A @ c) / e) ** 2).sum()) / max(len(x) - 2, 1)
    return float(c[1]), float(np.sqrt(cov[1, 1])), chi2


def report(title, C, mus, xlabel, xform, extra_cells=None):
    print('=' * 74)
    print(title)
    print('=' * 74)
    verdicts = []
    for mg in mus:
        arr = sorted((k[0], v) for k, v in C.items() if abs(k[1] - mg) < 1e-9)
        if extra_cells and mg in extra_cells:
            arr = sorted(arr + [extra_cells[mg]])
        if len(arr) < 3:
            print(f'  mu_g = {mg}: only {len(arr)} points'); continue
        pool = [v for _, v in arr]
        got = nfit.common_theta0(pool)
        if got is None:
            print(f'  mu_g = {mg}: no common Theta range'); continue
        T0, _ = got
        xs, ys, es, rot = [], [], [], []
        for xv, v in arr:
            ts = jammed(v)
            r = nfit.fit_local_sys(v, T0, restrict_to=ts, z_min=ZMIN) if len(ts) >= 4 else None
            if not r:
                print(f'     {xlabel}={xv:<8g} VOID ({len(ts)} jammed setpoints)')
                continue
            xs.append(xform(xv)); ys.append(r['n']); es.append(r['tot'])
            _, byT = nfit.surviving_setpoints(v, z_min=ZMIN, restrict_to=ts)
            tr = [x['Trot'] / x['Theta'] for t in ts for x in byT.get(t, [])
                  if 'Trot' in x and x['Theta'] > 0]
            rot.append(np.mean(tr) if tr else np.nan)
            print(f'     {xlabel}={xv:<8g} n = {r["n"]:+.4f} +/- {r["tot"]:.4f}'
                  f'   [{r["k"]} setpoints]'
                  + (f'   Theta_rot/Theta = {rot[-1]:.2f}' if np.isfinite(rot[-1]) else ''))
        if len(xs) < 3:
            print(f'  mu_g = {mg}: too few fittable\n'); continue
        s, se, c2 = wslope(xs, ys, es)
        ok = abs(s) / se < 2
        verdicts.append(ok)
        print(f'  mu_g = {mg}:  slope = {s:+.4f} +/- {se:.4f}  '
              f'({abs(s)/se:.1f} sigma)  chi2/dof {c2:.2f}   '
              f'-> {"PASS" if ok else "FAIL"}   [Theta_0 = {T0:.2e}]\n')
    if verdicts:
        print(f'  VERDICT: {"PASS" if all(verdicts) else "FAIL"} '
              f'({sum(verdicts)}/{len(verdicts)} frictions within 2 sigma)\n')


def main():
    RXN = r'log\.N(?P<N>[0-9]+)_mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$'
    RXE = r'log\.e(?P<e>[0-9.]+)_mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$'
    S = cells('sweep_size2d', RXN, 'N')
    R = cells('sweep_restit', RXE, 'e')

    # the existing N = 4000 production cells, same deck and protocol
    base = {}
    for p in glob.glob('sweep_steady2d/log.mu*'):
        m = re.match(r'log\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$',
                     os.path.basename(p))
        if not m:
            continue
        q = parse_extra(p)
        if not q:
            continue
        q['Tgran'] = float(m.group('T'))
        base.setdefault(float(m.group('mu')), []).append(q)
    extra = {mg: (4000.0, base[mg]) for mg in (0.0, 0.15, 1.0) if mg in base}

    report('SECTION 5.1  FINITE SIZE   n vs ln N   (prediction on record: PASS)',
           S, (0.0, 0.15, 1.0), 'N', np.log, extra)
    report('SECTION 5.2  RESTITUTION   n vs e      (no prediction on record)',
           R, (0.0, 0.15, 1.0), 'e', lambda v: v)


if __name__ == '__main__':
    main()
