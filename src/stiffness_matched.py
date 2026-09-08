"""Does n really depend on grain stiffness, or does the fit only look that way?

WHY THIS EXISTS.  analyze_stiffness.py reports n rising from 0.109 to 0.210
across two decades in E at mu_g = 0.3, a 25-sigma trend.  That number is
produced by a single straight-line fit of ln mu against ln Theta over whatever
setpoints happen to be present, with no jamming gate, no common Theta window,
and no degree systematic.  It is not the estimator the paper uses anywhere
else, and Sec. III E of the paper is precisely the demonstration that such a
comparison can invert: with curved mu(Theta), the slope at Theta_0 depends on
the range fitted, so cells whose windows differ will differ in n for reasons
that have nothing to do with the lever being tested.

Two things make that the live worry here rather than a hypothetical.  Z falls
monotonically along the lever, 4.02 to 3.34 at mu_g = 0.3, so the isostatic
gate removes different setpoints at different E.  And the surviving Theta range
therefore moves with E, which is the exact configuration that inverted the
dimension comparison.

WHAT IS TESTED.  n is recomputed with the machinery the paper uses: the
jamming gate at Z >= D+1, the adaptive-degree local fit at a stated Theta_0,
and the calibrated jackknife error with the degree systematic folded in.  Then
the same thing again with every cell restricted to the Theta window common to
all of them, which is the step that removes the range artifact.

If the trend survives window matching it is physics and belongs in the paper.
If it collapses, it was a fitting artifact and the honest report is a bound.

Usage:  python3 stiffness_matched.py
"""
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nfit

RX = (r'log\.E(?P<E>[0-9.eE+-]+)_mu(?P<mu>[0-9.]+)'
      r'_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$')
SWEEPS = (('2D', 'sweep_lev_E', 3.0), ('3D', 'sweep_lev_E3d', 4.0))
CANON = 8.4664e-04


def cells(d):
    out = {}
    for p in glob.glob(f'{d}/log.E*'):
        m = re.match(RX, os.path.basename(p))
        if not m:
            continue
        q = nfit.parse_log(p)
        if not q:
            continue
        q['Tgran'] = float(m.group('T'))
        out.setdefault((float(m.group('mu')), float(m.group('E'))), []).append(q)
    return out


def gated(runs, zmin):
    Ts, byT = nfit.surviving_setpoints(runs, z_min=zmin)
    return [t for t in Ts
            if not [x['Z'] for x in byT[t] if 'Z' in x]
            or np.mean([x['Z'] for x in byT[t] if 'Z' in x]) >= zmin]


def theta_span(runs, keep, zmin):
    _, byT = nfit.surviving_setpoints(runs, z_min=zmin, restrict_to=keep)
    th = [r['Theta'] for t in keep for r in byT[t] if r['Theta'] > 0]
    return (min(th), max(th)) if th else None


def n_of(runs, keep, zmin, T0):
    """Common windows are imposed through restrict_to, per nfit's docstring."""
    r = nfit.fit_local_sys(runs, T0, fn=nfit.fit_local_adaptive,
                           restrict_to=keep, z_min=zmin)
    if not r:
        return None
    return r['n'], float(np.hypot(r['tot'], r.get('sys_deg', 0.0)))


def wslope(x, y, e):
    w = 1.0 / np.asarray(e) ** 2
    A = np.column_stack([np.ones_like(x), x])
    cov = np.linalg.inv(A.T @ np.diag(w) @ A)
    c = cov @ A.T @ np.diag(w) @ y
    return c[1], np.sqrt(cov[1, 1])


def main():
    bydim = {}
    for tag, d, zmin in SWEEPS:
        C = cells(d)
        if not C:
            print(f'{d} not unpacked; skipping {tag}.')
            continue
        mus = sorted({k[0] for k in C})
        print('=' * 84)
        print(f'{tag}:  n ACROSS THE STIFFNESS LEVER, PAPER ESTIMATOR')
        print('=' * 84)
        for mg in mus:
            Es = sorted(E for (m, E) in C if m == mg)
            keep, spans, ok = {}, {}, []
            for E in Es:
                runs = C[(mg, E)]
                k = gated(runs, zmin)
                if len(k) < 4:
                    continue
                sp = theta_span(runs, k, zmin)
                if not sp:
                    continue
                keep[E], spans[E] = k, sp
                ok.append(E)
            if len(ok) < 3:
                print(f'  mu_g={mg:g}: too few fittable stiffnesses\n')
                continue
            lo = max(spans[E][0] for E in ok)
            hi = min(spans[E][1] for E in ok)
            print(f'  mu_g={mg:g}   per-cell Theta spans, and the common window '
                  f'[{lo:.3e}, {hi:.3e}]:')
            for E in ok:
                a, b = spans[E]
                print(f'     E={E:>8.0e}  [{a:.3e}, {b:.3e}]  '
                      f'width x{b/a:>6.1f}  gated setpoints {len(keep[E])}')
            if hi <= lo:
                print('     no common window: the gate leaves these cells '
                      'disjoint in Theta, so no matched comparison exists.\n')
                continue
            common = set(keep[ok[0]])
            for E in ok[1:]:
                common &= set(keep[E])
            print(f'     setpoints common to every stiffness: {len(common)}')
            for label, sets in (('unmatched', {E: keep[E] for E in ok}),
                                ('matched', {E: sorted(common) for E in ok})):
                if label == 'matched' and len(common) < 4:
                    print('     matched: fewer than four common setpoints; '
                          'no matched comparison is possible.')
                    continue
                xs, ys, es = [], [], []
                for E in ok:
                    got = n_of(C[(mg, E)], sets[E], zmin, CANON)
                    if got:
                        xs.append(np.log(E)); ys.append(got[0]); es.append(got[1])
                if len(xs) < 3:
                    print(f'     {label:>9}: too few cells survive')
                    continue
                s, se = wslope(np.array(xs), np.array(ys), np.array(es))
                if label == 'matched':
                    bydim.setdefault(mg, {})[tag] = (s, se)
                vals = '  '.join(f'{v:.3f}' for v in ys)
                print(f'     {label:>9}: n = {vals}')
                print(f'     {"":>9}  dn/dlnE = {s:+.4f} +/- {se:.4f}'
                      f'  ({abs(s)/se:.1f} sigma)')
            print()

    both = {m: v for m, v in bydim.items() if len(v) == 2}
    print('=' * 84)
    print('IS THE STIFFNESS DEPENDENCE ITSELF DIMENSION-INDEPENDENT?')
    print('=' * 84)
    print('  The friction law agrees between dimensions at kappa = 1e4.  If')
    print('  dn/dlnE differed between dimensions, that agreement would hold at')
    print('  one stiffness by coincidence rather than as a property of n.  The')
    print('  frictions below are the ones where both dimensions have a lever.\n')
    if not both:
        print('  no friction has both dimensions; comparison not possible.')
        return 0
    print(f'  {"mu_g":>6}{"2D":>21}{"3D":>21}{"difference":>21}{"":>8}')
    for mg in sorted(both):
        (a, ae), (b, be) = both[mg]['2D'], both[mg]['3D']
        d, de = b - a, float(np.hypot(ae, be))
        print(f'  {mg:>6g}{f"{a:+.4f}+/-{ae:.4f}":>21}'
              f'{f"{b:+.4f}+/-{be:.4f}":>21}'
              f'{f"{d:+.4f}+/-{de:.4f}":>21}{f"{abs(d)/de:.1f}s":>8}')
    print('\n  The 3D lever has three modulus values against five in 2D, so')
    print('  these are the weaker of the two measurements in every row.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
