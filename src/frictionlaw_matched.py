"""Does the friction law still agree between dimensions under matched windows?

WHY THIS EXISTS.  Sec. III C reports the coefficients of n = a + b*mu_g
agreeing between dimensions to 0.31 sigma in the slope and 0.01 sigma in the
intercept, and the title rests on that.  Those fits use each cell's own gated
setpoints, so the comparison is UNMATCHED across dimensions.

Sec. III F is the demonstration that unmatched cross-dimension comparisons are
contaminated: per-cell Theta windows differ by more than a factor of four, and
refitting each cell over the window it shares with its partner moves the
three-dimensional values by up to 3.6 times their own errors.  Grouped below
mu_g = 0.2 -- which is where the friction law is fitted -- chi2/dof goes from
1.26 unmatched to 8.51 matched.  The dimensions agree pointwise on unmatched
windows and disagree on matched ones.

So the law's dimension-independence is measured under exactly the treatment the
paper elsewhere argues is wrong, and has never been measured under the
treatment it argues is right.  This runs the fit both ways.

If the coefficients still agree, Sec. III F's inversion is about the pointwise
curve and not about the law, and both statements can stand.  If they do not,
the claim that dimension does not set the exponent rests on unmatched windows
and must be withdrawn or restated.

Usage:  python3 frictionlaw_matched.py
"""
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nfit

RXM = r'log\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$'
SWEEPS = (('2D', 'sweep_steady2d', 3.0, 0.15),
          ('3D', 'sweep_steady3d', 4.0, 0.20))
CANON = 8.4664e-04


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


def gated(runs, zmin):
    Ts, byT = nfit.surviving_setpoints(runs, z_min=zmin)
    return [t for t in Ts
            if not [x['Z'] for x in byT[t] if 'Z' in x]
            or np.mean([x['Z'] for x in byT[t] if 'Z' in x]) >= zmin]


def n_of(runs, keep, zmin):
    r = nfit.fit_local_sys(runs, CANON, fn=nfit.fit_local_adaptive,
                           restrict_to=keep, z_min=zmin)
    if not r:
        return None
    return r['n'], float(np.hypot(r['tot'], r.get('sys_deg', 0.0)))


def wlinear(x, y, e):
    w = 1.0 / np.asarray(e) ** 2
    A = np.column_stack([np.ones_like(x), x])
    cov = np.linalg.inv(A.T @ np.diag(w) @ A)
    c = cov @ A.T @ np.diag(w) @ y
    chi2 = float(((y - A @ c) ** 2 * w).sum())
    return c, np.sqrt(np.diag(cov)), chi2, len(x) - 2


def main():
    D = {tag: (cells(d), zmin, hi) for tag, d, zmin, hi in SWEEPS}
    if any(not v[0] for v in D.values()):
        print('sweeps not unpacked; run this beside the log archive.')
        return 1

    keep = {}
    for tag, (C, zmin, hi) in D.items():
        keep[tag] = {m: gated(C[m], zmin) for m in C
                     if m <= hi and len(gated(C[m], zmin)) >= 4}

    shared = sorted(set(keep['2D']) & set(keep['3D']))
    print('=' * 82)
    print('THE FRICTION LAW, UNMATCHED AND MATCHED ACROSS DIMENSIONS')
    print('=' * 82)
    print(f'  frictions on the branch in both dimensions: {shared}')
    print('  matched means both cells restricted to the setpoints they share.\n')

    out = {}
    for mode in ('unmatched', 'matched'):
        print(f'  {mode}')
        for tag, (C, zmin, hi) in D.items():
            xs, ys, es = [], [], []
            for m in sorted(keep[tag]):
                if mode == 'matched':
                    if m not in shared:
                        continue
                    k = sorted(set(keep['2D'][m]) & set(keep['3D'][m]))
                    if len(k) < 4:
                        continue
                else:
                    k = keep[tag][m]
                got = n_of(C[m], k, zmin)
                if got:
                    xs.append(m); ys.append(got[0]); es.append(got[1])
            if len(xs) < 3:
                print(f'    {tag}: too few cells survive')
                continue
            c, s, chi2, dof = wlinear(np.array(xs), np.array(ys), np.array(es))
            out[(mode, tag)] = (c, s)
            print(f'    {tag}  cells {xs}')
            print(f'        a = {c[0]:+.4f} +/- {s[0]:.4f}   '
                  f'b = {c[1]:.3f} +/- {s[1]:.3f}   '
                  f'chi2/dof = {chi2/max(dof,1):.2f}')
        if (mode, '2D') in out and (mode, '3D') in out:
            (c2, s2), (c3, s3) = out[(mode, '2D')], out[(mode, '3D')]
            for i, nm in ((1, 'slope'), (0, 'intercept')):
                d = c3[i] - c2[i]
                e = float(np.hypot(s2[i], s3[i]))
                print(f'        {nm:>9} 3D minus 2D = {d:+.4f} +/- {e:.4f}'
                      f'  ->  {abs(d)/e:.2f} sigma')
        print()

    print('  The number quoted in Sec. III C is the unmatched slope comparison.')
    print('  If the matched one disagrees, the dimension claim is a property of')
    print('  the fitting windows rather than of the exponent.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
