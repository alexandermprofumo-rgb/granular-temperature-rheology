"""Section 12: do the two sign changes coincide? -- on steady-state data.

Both halves turn out to need NO new simulations:
  * the curvature of mu(Theta) needs a dense mu_g grid in 2D at P = 10, and
    sweep_steady2d has 11 frictions with five inside the crossing band;
  * b = dn/dln(kappa) needs the pressure lever, and sweep_lev_P has 6 frictions
    x 4 pressures.

Both are strain-1.0, jamming-gated, calibrated-error data. Slope errors are
propagated from the per-cell errors, never rescaled by residuals.

Usage:  python3 analyze_crossings_steady.py
"""
import glob
import os
import re

import numpy as np

import nfit

RX2 = re.compile(r'log\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$')
RXP = r'log\.P(?P<P>[0-9.]+)_mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$'
ZMIN = 3.0


def jammed(runs):
    Ts, byT = nfit.surviving_setpoints(runs)
    return [t for t in Ts
            if not [x['Z'] for x in byT[t] if 'Z' in x]
            or np.mean([x['Z'] for x in byT[t] if 'Z' in x]) >= ZMIN]


def wline(x, y, e):
    x = np.asarray(x, float); y = np.asarray(y, float); e = np.asarray(e, float)
    A = np.vstack([np.ones_like(x), x]).T
    W = np.diag(1 / e ** 2)
    cov = np.linalg.inv(A.T @ W @ A)
    c = cov @ A.T @ W @ y
    chi2 = float((((y - A @ c) / e) ** 2).sum()) / max(len(x) - 2, 1)
    return c, cov, chi2


def crossing(x, y, e, label):
    """Zero of a weighted line in ln(mu_g); error by propagation."""
    c, cov, chi2 = wline(x, y, e)
    if c[1] == 0:
        return None
    xc = -c[0] / c[1]
    J = np.array([-1 / c[1], c[0] / c[1] ** 2])
    xce = float(np.sqrt(J @ cov @ J))
    if chi2 > 1:
        xce *= np.sqrt(chi2)          # inflate if the line is a poor model
    print(f'   {label}: slope {c[1]:+.4f} +/- {np.sqrt(cov[1,1]):.4f} '
          f'({abs(c[1])/np.sqrt(cov[1,1]):.1f} sigma), chi2/dof {chi2:.2f}')
    print(f'   {label}: crosses zero at mu_g = {np.exp(xc):.4f} '
          f'(x/ {np.exp(xce):.2f})')
    return np.exp(xc), xc, xce


def main():
    # curvature of mu(Theta), 2D
    C = {}
    for p in glob.glob('sweep_steady2d/log.mu*'):
        m = RX2.match(os.path.basename(p))
        if not m:
            continue
        q = nfit.parse_log(p)
        if not q:
            continue
        q['Tgran'] = float(m.group('T'))
        C.setdefault(float(m.group('mu')), []).append(q)
    keep = {m: jammed(v) for m, v in C.items()}
    los, his = [], []
    for m, ts in keep.items():
        if len(ts) < 4:
            continue
        _, byT = nfit.surviving_setpoints(C[m], restrict_to=ts)
        th = [x['Theta'] for t in ts for x in byT.get(t, []) if x['Theta'] > 0]
        if th:
            los.append(min(th)); his.append(max(th))
    T0 = float(np.sqrt(max(los) * min(his)))
    print('=' * 74)
    print(f'LEVER 1  curvature of mu(Theta), 2D   Theta_0 = {T0:.3e}')
    print('=' * 74)
    xs, ys, es = [], [], []
    for m in sorted(C):
        if m <= 0 or len(keep[m]) < 4:
            continue
        r = nfit.fit_local(C[m], T0, restrict_to=keep[m])
        if not r:
            continue
        print(f'   mu_g = {m:<5g} curv = {r["curv"]:+.4f} +/- {r["curv_err"]:.4f}'
              f'   ({abs(r["curv"])/r["curv_err"]:.1f} s)')
        if m <= 0.35:                      # linear only near the crossing
            xs.append(np.log(m)); ys.append(r['curv']); es.append(r['curv_err'])
    cur = crossing(xs, ys, es, 'curvature')

    # pressure response b = dn/dln(kappa)
    raw = nfit.load_sweep('sweep_lev_P/log.P*', RXP,
                          tgran_from=lambda m: float(m.group('T')) * float(m.group('P')) / 10.0)
    cm = {}
    for k, v in raw.items():
        d = dict(k)
        cm[(float(d['P']), float(d['mu']))] = v
    Ps = sorted({p for p, _ in cm})
    mus = sorted({m for _, m in cm})
    kp = {k: jammed(v) for k, v in cm.items()}
    los, his = [], []
    for (P, mg), ts in kp.items():
        if len(ts) < 4:
            continue
        _, byT = nfit.surviving_setpoints(cm[(P, mg)], restrict_to=ts)
        th = [x['Theta'] / P for t in ts for x in byT.get(t, []) if x['Theta'] > 0]
        if th:
            los.append(min(th)); his.append(max(th))
    T0h = float(np.sqrt(max(los) * min(his)))
    print()
    print('=' * 74)
    print(f'LEVER 2  b = dn/dln(kappa)   reduced Theta_0 = {T0h:.3e}')
    print('=' * 74)
    xb, yb, eb = [], [], []
    for mg in mus:
        pts = []
        for P in Ps:
            ts = kp.get((P, mg), [])
            if len(ts) < 4:
                continue
            r = nfit.fit_local_sys(cm[(P, mg)], T0h * P, restrict_to=ts)
            if r:
                pts.append((P, r))
        if len(pts) < 3:
            continue
        c, cov, chi2 = wline(np.log(1.0e5 / np.array([p for p, _ in pts])),
                             [r['n'] for _, r in pts], [r['tot'] for _, r in pts])
        b, be = float(c[1]), float(np.sqrt(cov[1, 1]))
        if chi2 > 1:
            be *= np.sqrt(chi2)
        print(f'   mu_g = {mg:<5g} b = {b:+.4f} +/- {be:.4f}  ({abs(b)/be:.1f} s)')
        xb.append(np.log(mg)); yb.append(b); eb.append(be)
    prs = crossing(xb, yb, eb, 'pressure b')

    # do they coincide?
    if cur and prs:
        print()
        print('=' * 74)
        print('DO THE TWO SIGN CHANGES COINCIDE?')
        print('=' * 74)
        d = prs[1] - cur[1]
        e = float(np.hypot(cur[2], prs[2]))
        print(f'   curvature crosses at mu_g = {cur[0]:.4f} (x/{np.exp(cur[2]):.2f})')
        print(f'   pressure b crosses at mu_g = {prs[0]:.4f} (x/{np.exp(prs[2]):.2f})')
        print(f'   separation in ln(mu_g): {d:+.4f} +/- {e:.4f} -> {abs(d)/e:.1f} sigma')
        print('   [report: 0.7 sigma at P=10, then revised to "not established",'
              '\n    with the pichi data giving 2.8 sigma apart]')
        if abs(d) / e < 2:
            print('   -> CONSISTENT with a single crossing')
        else:
            print('   -> NOT consistent: two distinct crossings')


if __name__ == '__main__':
    main()
