"""Is n a simple function of grain friction on the rising branch?

WHY THIS EXISTS. Every other analysis in this project tests a null hypothesis:
is n equal to 1/(2D), are the baselines equal, is the peak where it looks, is
the curvature one-signed. A framework built to stop motivated fitting also
cannot discover a functional form, so nobody ever asked what function n(mu_g)
is. This asks.

It matters because Kim and Kamrin (PRL 125, 088002) propose mu*Theta^p = f(I)
with p = 1/6 in 3D and 1/8 in 2D, and report that p does not depend on surface
friction. They tested two frictions, mu_p = 0.1 and 0.4. If n is a strong
function of mu_g, their two-point test could not have seen it.

WHAT IS TESTED, in order:

  1. The fit range is located by the data, not chosen. The linear fit is
     extended one friction at a time and chi2/dof is reported, so the reader
     sees where linearity fails rather than being told where it holds.

  2. The coefficients are scanned across Theta_0. This is not optional. n is a
     local slope at a stated reference temperature, so a law in n inherits that
     dependence, and a law quoted at one Theta_0 means little. The scan is
     confined to the window common to every cell on the rising branch:
     evaluating outside it reads a polynomial's extrapolation, not a
     measurement, and an earlier version of this analysis drew a wrong
     conclusion by doing exactly that.

  3. Two dimensions are compared at every Theta_0. This is the claim that
     matters most, because any argument fixing n from geometry alone requires
     the dimensions to differ by 3/2.

  4. The curvature is tested for linearity in mu_g. If the second-order
     coefficient were linear, the Theta_0 dependence of the whole law would
     follow from a single quadratic surface in (ln Theta, mu_g), and the law
     would be derived rather than empirical. It is not.

Usage:  python3 friction_law.py
"""
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nfit

RXM = r'log\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$'

# The rising branch, per cell. Set from the fit-range scan in section 1: 2D
# linearity survives to 0.15 and fails at 0.2, 3D survives to 0.2 and fails at
# 0.3. Both limits coincide with the peak, so the branch is a feature of the
# data rather than a choice.
SWEEPS = (('2D', 'sweep_steady2d', 3.0, 0.15),
          ('3D', 'sweep_steady3d', 4.0, 0.20))
CANON = 8.4664e-04
KK_P = {'2D': 1.0 / 8.0, '3D': 1.0 / 6.0}     # Kim & Kamrin's fitted exponents
KK_MU = 0.1                                   # their lower surface friction


def cells(d):
    out = {}
    for p in glob.glob(f'{d}/log.*'):
        m = re.match(RXM, os.path.basename(p))
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


def series(D, zmin, T0, hi=None):
    """n(mu_g) with the combined error every significance in the paper uses."""
    out = {}
    for m in sorted(D):
        if hi is not None and m > hi:
            continue
        K = jammed(D[m], zmin)
        if len(K) < 4:
            continue
        r = nfit.fit_local_sys(D[m], T0, fn=nfit.fit_local_adaptive,
                               restrict_to=K, z_min=zmin)
        if r:
            out[m] = (r['n'], float(np.hypot(r['tot'], r.get('sys_deg', 0.0))))
    return out


def wlinear(x, y, e):
    """Weighted straight-line fit; returns coefficients, errors, chi2, dof."""
    w = 1.0 / e ** 2
    A = np.column_stack([np.ones_like(x), x])
    W = np.diag(w)
    cov = np.linalg.inv(A.T @ W @ A)
    c = cov @ A.T @ W @ y
    chi2 = float(((y - A @ c) ** 2 * w).sum())
    return c, np.sqrt(np.diag(cov)), chi2, len(x) - 2


def common_window(data):
    """Theta range covered by every cell on the rising branch, both dimensions.

    The Theta_0 scan must stay inside this. Outside it the fit is extrapolating
    and the slope it reports is a property of the polynomial, not of the data.
    """
    los, his = [], []
    for tag, D, zmin, hi in data:
        for m in sorted(D):
            if m > hi:
                continue
            K = jammed(D[m], zmin)
            if len(K) < 4:
                continue
            _, byT = nfit.surviving_setpoints(D[m], z_min=zmin, restrict_to=K)
            th = [r['Theta'] for t in K for r in byT[t] if r['Theta'] > 0]
            if th:
                los.append(min(th))
                his.append(max(th))
    return max(los), min(his)


def main():
    data = []
    for tag, d, zmin, hi in SWEEPS:
        D = cells(d)
        if not D:
            print(f'{d} not unpacked; run this beside the log archive.')
            return 1
        data.append((tag, D, zmin, hi))

    print('=' * 78)
    print('1. WHERE DOES LINEARITY IN GRAIN FRICTION BREAK?')
    print('=' * 78)
    print('   The fit is extended one friction at a time. The range is read off')
    print('   the chi2 column rather than chosen in advance.\n')
    for tag, D, zmin, _ in data:
        full = series(D, zmin, CANON)
        ms = sorted(full)
        print(f'  {tag}')
        print(f'  {"up to mu_g":>11}{"k":>4}{"intercept":>19}{"slope":>17}'
              f'{"chi2/dof":>10}')
        for top in ms[2:]:
            sel = [m for m in ms if m <= top]
            x = np.array(sel)
            y = np.array([full[m][0] for m in sel])
            e = np.array([full[m][1] for m in sel])
            c, s, chi2, dof = wlinear(x, y, e)
            red = chi2 / max(dof, 1)
            print(f'  {top:>11g}{len(sel):>4}{c[0]:>+12.4f}+/-{s[0]:.4f}'
                  f'{c[1]:>10.3f}+/-{s[1]:.3f}{red:>10.2f}'
                  f'{"   <- fails" if red > 3 else ""}')
        print()

    LO, HI = common_window(data)
    print('=' * 78)
    print('2. THE COEFFICIENTS ACROSS Theta_0, AND 3. THE TWO DIMENSIONS')
    print('=' * 78)
    print(f'   Window common to every cell on the rising branch: '
          f'[{LO:.3e}, {HI:.3e}], a factor of {HI/LO:.1f}.')
    print('   The scan stays inside it.\n')
    print(f'  {"Theta_0":>11}{"dim":>5}{"intercept":>19}{"slope":>17}'
          f'{"chi2/dof":>10}{"n(mu_g=0.1)":>13}')
    rows = {}
    for frac in (0.15, 0.30, 0.50, 0.70, 0.85):
        T0 = float(LO * (HI / LO) ** frac)
        got = {}
        for tag, D, zmin, hi in data:
            S = series(D, zmin, T0, hi)
            if len(S) < 3:
                continue
            x = np.array(sorted(S))
            y = np.array([S[m][0] for m in sorted(S)])
            e = np.array([S[m][1] for m in sorted(S)])
            c, s, chi2, dof = wlinear(x, y, e)
            got[tag] = (c, s)
            rows.setdefault(tag, []).append((T0, c, s))
            print(f'  {T0:>11.3e}{tag:>5}{c[0]:>+12.4f}+/-{s[0]:.4f}'
                  f'{c[1]:>10.3f}+/-{s[1]:.3f}{chi2/max(dof,1):>10.2f}'
                  f'{c[0] + c[1]*KK_MU:>13.4f}')
        if len(got) == 2:
            d = got['3D'][0][1] - got['2D'][0][1]
            e = float(np.hypot(got['2D'][1][1], got['3D'][1][1]))
            print(f'  {"":>11}{"":>5}   slope 3D minus 2D = {d:+.3f} +/- {e:.3f}'
                  f'  ->  {abs(d)/e:.2f} sigma')
    for tag in ('2D', '3D'):
        if tag not in rows:
            continue
        b = np.array([r[1][1] for r in rows[tag]])
        pred = np.array([r[1][0] + r[1][1] * KK_MU for r in rows[tag]])
        print(f'\n  {tag}: slope spans {b.min():.3f} to {b.max():.3f} over the '
              f'window ({100*(b.max()-b.min())/b.mean():.0f}% of its mean).')
        print(f'      n(mu_g={KK_MU}) spans {pred.min():.4f} to {pred.max():.4f}; '
              f'Kim & Kamrin fit p = {KK_P[tag]:.4f}.')

    print('\n  DOES THE INTERCEPT PREDICT THE FRICTIONLESS CELL?')
    print('  The branch runs from mu_g = 0, so the frictionless cell is one of')
    print('  the points the law is fitted to and the intercept is not a')
    print('  prediction of it.  Holding that cell out and refitting on the')
    print('  frictional cells alone makes the extrapolation a real one.  Both')
    print('  fits and n(mu_g=0) are taken at the same Theta_0; comparing a fit')
    print('  at one reference against a measurement at another would mix a')
    print('  disagreement with a shift of reference.\n')
    for tag, D, zmin, hi in data:
        S = series(D, zmin, CANON, hi)
        if 0.0 not in S:
            continue
        ms = sorted(S)
        n0, e0 = S[0.0]
        for lab, sel in (('all cells', ms),
                         ('mu_g=0 held out', [m for m in ms if m > 0])):
            x = np.array(sel)
            y = np.array([S[m][0] for m in sel])
            e = np.array([S[m][1] for m in sel])
            c, s, chi2, dof = wlinear(x, y, e)
            g = abs(n0 - c[0]) / float(np.hypot(s[0], e0))
            print(f'    {tag}  {lab:>15}  a = {c[0]:+.4f}+/-{s[0]:.4f}   '
                  f'b = {c[1]:.3f}+/-{s[1]:.3f}   '
                  f'vs n(mu_g=0) = {n0:.4f}+/-{e0:.4f}: {g:.2f} sigma')

    print('\n' + '=' * 78)
    print('4. IS THE CURVATURE LINEAR IN GRAIN FRICTION?')
    print('=' * 78)
    print('   Shifting the reference by delta moves n by -2*c2*delta, so if c2')
    print('   were linear in mu_g the whole Theta_0 dependence would follow from')
    print('   one quadratic surface and the law would be derived, not fitted.\n')
    for tag, D, zmin, hi in data:
        xs, c2s, e2s = [], [], []
        for m in sorted(D):
            if m > hi:
                continue
            K = jammed(D[m], zmin)
            if len(K) < 4:
                continue
            _, byT = nfit.surviving_setpoints(D[m], z_min=zmin, restrict_to=K)
            sel = [r for t in K for r in byT[t]
                   if r['Theta'] > 0 and r['mu'] > 0]
            u = np.log([r['Theta'] for r in sel]) - np.log(CANON)
            y = np.log([r['mu'] for r in sel])
            A = np.column_stack([np.ones_like(u), u, u ** 2])
            c = np.linalg.lstsq(A, y, rcond=None)[0]
            res = y - A @ c
            dof = len(y) - 3
            if dof < 1:
                continue
            cov = float(res @ res) / dof * np.linalg.inv(A.T @ A)
            xs.append(m)
            c2s.append(c[2])
            e2s.append(np.sqrt(cov[2, 2]))
        if len(xs) < 3:
            continue
        c, s, chi2, dof = wlinear(np.array(xs), np.array(c2s), np.array(e2s))
        print(f'  {tag}  c2 = ({c[0]:+.4f} +/- {s[0]:.4f}) + '
              f'({c[1]:+.4f} +/- {s[1]:.4f}) mu_g   chi2/dof = '
              f'{chi2/max(dof,1):.2f}')
        print(f'      c2 by cell: ' +
              ', '.join(f'{m:g}:{v:+.4f}' for m, v in zip(xs, c2s)))
    print('\n  A large chi2 here means the curvature is not linear in mu_g, so')
    print('  the friction law is empirical and does not follow from the shape')
    print('  of mu(Theta).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
