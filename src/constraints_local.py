"""
The constraint list, rebuilt on the LOCAL exponent n(Theta_0).

The window-averaged n is ambiguous because mu(Theta) is curved: fitting the
low or high half of the same window shifts n by up to 0.11, and that
systematic exceeded the statistical error and sank most quantitative claims
(A_3/A_2, the opposite-signed stiffness derivatives, all nine two-lever
falsifications).

Here n is instead the local logarithmic slope at a stated reference
temperature, from a quadratic log-log fit:

    ln mu = a + b ln(Theta/Theta_0) + c ln^2(Theta/Theta_0)
    n(Theta_0) = -b        curvature = -2c

This is unambiguous once Theta_0 is stated, and -- unlike the window average
-- its error shrinks with data. Theta_0 is chosen as the geometric centre of
the Theta range COMMON to every cell in a given comparison, so no result is
an extrapolation. Comparisons across dimensions are the delicate case: at the
same Tgran, 2D and 3D sit at different Theta, so Theta_0 is taken from their
intersection and reported.

The curvature is reported alongside, because it is a physical result: the
local exponent rises with granular temperature.

Usage:  python3 constraints_local.py
"""
import numpy as np

import nfit


def cells_by_mu(pattern, rx):
    raw = nfit.load_sweep(pattern, rx)
    out = {}
    for key, runs in raw.items():
        out.setdefault(float(dict(key)['mu']), []).extend(runs)
    return out


def report(tag, r):
    if r is None:
        print(f'   {tag:<26} VOID')
        return
    print(f'   {tag:<26} n({r["theta0"]:.2e}) = {r["n"]:+.4f} +/- {r["stat"]:.4f}'
          f'   curv = {r["curv"]:+.4f} +/- {r["curv_err"]:.4f}  [{r["k"]} pts]')


def main():
    m2 = cells_by_mu('sweep_matched2d/log.mu*',
                     r'log\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$')
    d3 = cells_by_mu('sweep_run_3d/log.mu*',
                     r'log\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$')

    core = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
    pool = [m2[m] for m in core if m in m2] + [d3[m] for m in core if m in d3]
    got = nfit.common_theta0(pool)
    if got is None:
        print('no common Theta range across 2D and 3D -- cannot compare')
        return
    T0, (lo, hi) = got
    print(f'common Theta range across 2D+3D core cells: [{lo:.3e}, {hi:.3e}]')
    print(f'reference Theta_0 = {T0:.3e} (geometric centre)\n')

    print('=' * 76)
    print('CONSTRAINT 1 -- frictionless baseline')
    print('=' * 76)
    a = nfit.fit_local(m2[0.0], T0) if 0.0 in m2 else None
    b = nfit.fit_local(d3[0.0], T0) if 0.0 in d3 else None
    report('2D  n(mu_g=0)', a)
    report('3D  n(mu_g=0)', b)
    if a and b:
        d = b['n'] - a['n']; s = np.hypot(a['stat'], b['stat'])
        print(f'   difference {d:+.4f} +/- {s:.4f}  -> {abs(d)/s:.1f} sigma  '
              f'({"consistent" if abs(d)/s < 2 else "DIFFER"})')
        # SIGN. d is n_3D - n_2D, so the prediction is 1/6 - 1/4, NOT 1/4 - 1/6.
        # Coding it the wrong way round compares the measured difference to the
        # negative of the predicted one, which triples the apparent discrepancy
        # and reported 4.0 sigma where the data give 1.3.
        req = 1.0 / 6.0 - 0.25
        print(f'   geometric theory requires n_3D - n_2D = {req:+.3f} -> '
              f'differs at {abs(d - req)/s:.1f} sigma')
        # The difference is the WEAK test: it is a difference of two small
        # numbers. The theory also predicts the magnitudes, and those are what
        # the data actually exclude.
        for nm, r, pred in (('2D', a, 0.25), ('3D', b, 1.0 / 6.0)):
            print(f'   {nm} n = {r["n"]:.4f} +/- {r["stat"]:.4f} vs predicted '
                  f'{pred:.4f} -> {abs(r["n"] - pred)/r["stat"]:.1f} sigma')

    print()
    print('=' * 76)
    print('CONSTRAINT 2 -- n(mu_g) and its peak')
    print('=' * 76)
    for nm, C in (('2D', m2), ('3D', d3)):
        print(f'   {nm}:')
        rows = []
        for mg in sorted(C):
            r = nfit.fit_local(C[mg], T0)
            if r:
                rows.append((mg, r))
                print(f'      mu_g={mg:<5g} n = {r["n"]:+.4f} +/- {r["stat"]:.4f}'
                      f'   curv = {r["curv"]:+.4f} +/- {r["curv_err"]:.4f}')
        if len(rows) >= 3:
            pk = max(rows, key=lambda t: t[1]['n'])
            base = [r for mg, r in rows if mg == 0.0]
            print(f'      peak: n = {pk[1]["n"]:.4f} +/- {pk[1]["stat"]:.4f} '
                  f'at mu_g = {pk[0]:g}')
            if base:
                d = pk[1]['n'] - base[0]['n']
                s = np.hypot(pk[1]['stat'], base[0]['stat'])
                print(f'      rise above the frictionless baseline: {d:+.4f} '
                      f'+/- {s:.4f}  -> {abs(d)/s:.1f} sigma')
            n1 = [r for mg, r in rows if mg == 1.0]
            if n1:
                d = pk[1]['n'] - n1[0]['n']
                s = np.hypot(pk[1]['stat'], n1[0]['stat'])
                print(f'      fall to mu_g=1:                       {d:+.4f} '
                      f'+/- {s:.4f}  -> {abs(d)/s:.1f} sigma')

    print()
    print('=' * 76)
    print('CONSTRAINT 3 -- peak heights, 2D vs 3D  (the A_3/A_2 claim)')
    print('=' * 76)
    pk = {}
    for nm, C in (('2D', m2), ('3D', d3)):
        rows = [(mg, nfit.fit_local(C[mg], T0)) for mg in sorted(C)]
        rows = [(mg, r) for mg, r in rows if r]
        if rows:
            pk[nm] = max(rows, key=lambda t: t[1]['n'])
    if '2D' in pk and '3D' in pk:
        a2, a3 = pk['2D'][1], pk['3D'][1]
        b2 = nfit.fit_local(m2[0.0], T0); b3 = nfit.fit_local(d3[0.0], T0)
        print(f'   2D peak {a2["n"]:.4f} +/- {a2["stat"]:.4f} at mu_g={pk["2D"][0]:g}')
        print(f'   3D peak {a3["n"]:.4f} +/- {a3["stat"]:.4f} at mu_g={pk["3D"][0]:g}')
        d = a3['n'] - a2['n']; s = np.hypot(a2['stat'], a3['stat'])
        print(f'   difference {d:+.4f} +/- {s:.4f} -> {abs(d)/s:.1f} sigma')
        if b2 and b3:
            e2 = a2['n'] - b2['n']; e3 = a3['n'] - b3['n']
            se2 = np.hypot(a2['stat'], b2['stat']); se3 = np.hypot(a3['stat'], b3['stat'])
            if e2 > 0:
                R = e3 / e2
                Rs = abs(R) * np.hypot(se3 / e3, se2 / e2)
                print(f'   enhancement above baseline: 2D {e2:+.4f}+/-{se2:.4f}, '
                      f'3D {e3:+.4f}+/-{se3:.4f}')
                print(f'   A_3/A_2 = {R:.2f} +/- {Rs:.2f}')
                for target, nmm in ((1.0, 'no dimension effect'),
                                    (1.5, 'lambda_2 = 2D geometric ratio'),
                                    (2.0, 'factor 2')):
                    print(f'      vs {target} ({nmm}): {abs(R-target)/Rs:.1f} sigma')

    print()
    print('=' * 76)
    print('CURVATURE -- is it significant cell by cell?')
    print('=' * 76)
    sig = []
    for nm, C in (('2D', m2), ('3D', d3)):
        for mg in sorted(C):
            r = nfit.fit_local(C[mg], T0)
            if r and r['curv_err'] > 0:
                sig.append(r['curv'] / r['curv_err'])
    if sig:
        sig = np.array(sig)
        print(f'   {len(sig)} cells; curvature/error: median {np.median(sig):+.1f}, '
              f'{int((sig > 2).sum())} above +2 sigma, {int((sig < -2).sum())} below -2')
        print(f'   mean {sig.mean():+.2f} sigma -- a coherent sign means the '
              f'curvature is real, not scatter')


if __name__ == '__main__':
    main()
