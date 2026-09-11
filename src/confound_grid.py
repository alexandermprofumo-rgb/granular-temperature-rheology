"""The confound test on the full grid: shear-rate arms at two stiffnesses.

WHY THIS EXISTS.  confound_test.py could not decide whether the shear-rate
dependence of n is a dependence on the inertial number I or on the contact time
relative to the shear time, t_c*gdot, because the existing data was a cross:
arms at one stiffness and a modulus lever at one shear rate, meeting at a single
point.  run_iscan_stiff.sh adds the three arms again at ten times the modulus,
turning the cross into a grid.

WHAT THE GRID CAN AND CANNOT DECIDE.  The driver's header claimed the two
hypotheses predict offsets between the old and new arms of 0 and 0.083, a
four-sigma separation.  That is wrong.  It assumed the alternative to a
contact-time effect is n depending on I alone, but levers_matched.py has
already established that n depends on the stiffness number kappa independently,
confirmed by two opposed levers.  Both live hypotheses therefore predict a
positive offset.  What separates them is its SIZE:

  A.  n depends on one group, u = ln(t_c*gdot) = ln(gdot) - 0.4 ln(E) + const.
      Then dn/dlnE = -0.4 dn/dln(gdot) at every point of the grid, and the
      offset between arms a factor ten apart in E is -0.4 * ln(10) times the
      arm slope.

  B.  n depends on I and kappa as separate variables.  Then dn/dlnE is set by
      the stiffness dependence, measured independently on the modulus lever,
      and has no fixed relation to the arm slope.

With the arm slope near -0.063 and the modulus slope near +0.036, A predicts an
offset near +0.058 and B near +0.083.  The difference is about 0.025, so the
grid discriminates at roughly two sigma, not four.  This script reports that
honestly rather than the number the driver promised.

A second, independent handle: under B the arm slope cannot depend on stiffness,
because I is unchanged between the old and new arms.  Under A with any curvature
in n(u), the new arms sample lower u and their slope can differ.  Equal slopes
are consistent with both; unequal slopes favour A.

Usage:  python3 confound_grid.py
"""
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nfit

MU = 0.3
ZMIN = 3.0
CANON = 8.4664e-04
GAMMAS = (3.162e-4, 1.0e-3, 3.162e-3)
DNDLNE_MOD = (0.0360, 0.0049)          # stiffness_matched.py, 2D, matched

RX_G = r'log\.g(?P<g>[0-9.eE+-]+)_mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$'
RX_M = r'log\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$'


def load(d, rx, gfix=None):
    out = {}
    for p in glob.glob(f'{d}/log.*'):
        m = re.match(rx, os.path.basename(p))
        if not m or abs(float(m.group('mu')) - MU) > 1e-9:
            continue
        q = nfit.parse_log(p)
        if not q:
            continue
        q['Tgran'] = float(m.group('T'))
        g = gfix if gfix is not None else float(m.group('g'))
        out.setdefault(g, []).append(q)
    return out


def gated(runs):
    Ts, byT = nfit.surviving_setpoints(runs, z_min=ZMIN)
    return [t for t in Ts
            if not [x['Z'] for x in byT[t] if 'Z' in x]
            or np.mean([x['Z'] for x in byT[t] if 'Z' in x]) >= ZMIN]


def wline(x, y, e):
    w = 1.0 / np.asarray(e) ** 2
    A = np.column_stack([np.ones_like(x), x])
    cov = np.linalg.inv(A.T @ np.diag(w) @ A)
    c = cov @ A.T @ np.diag(w) @ y
    return c, np.sqrt(np.diag(cov))


def main():
    old = load('sweep_iscan2', RX_G)
    old.update(load('sweep_steady2d', RX_M, gfix=1.0e-3))
    new = load('sweep_iscan_stiff', RX_G)
    grid = {('1e5', g): old.get(g) for g in GAMMAS}
    grid.update({('1e6', g): new.get(g) for g in GAMMAS})
    missing = [k for k, v in grid.items() if not v]
    if missing:
        print('incomplete grid, missing:', missing)
        return 1

    keep = {k: gated(v) for k, v in grid.items()}
    common = sorted(set.intersection(*(set(v) for v in keep.values())))
    print('=' * 86)
    print(f'THE CONFOUND GRID   2D, mu_g = {MU}')
    print('=' * 86)
    print(f'  setpoints surviving the gate in every one of the six cells: '
          f'{len(common)}')
    if len(common) < 4:
        print('  too few to fit on a matched window.')
        return 1
    print('  the middle arm at E = 1e5 comes from the production scan, on')
    print('  different packings from the other five cells.\n')

    n = {}
    print(f'  {"E":>5}{"gdot":>11}{"n":>10}{"+/-":>9}')
    for (E, g), runs in sorted(grid.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        r = nfit.fit_local_sys(runs, CANON, fn=nfit.fit_local_adaptive,
                               restrict_to=common, z_min=ZMIN)
        if not r:
            print(f'  {E:>5}{g:>11.3e}   fit failed')
            continue
        e = float(np.hypot(r['tot'], r.get('sys_deg', 0.0)))
        n[(E, g)] = (r['n'], e)
        print(f'  {E:>5}{g:>11.3e}{r["n"]:>10.4f}{e:>9.4f}')

    slopes = {}
    print('\n  SLOPE ALONG EACH FAMILY OF ARMS')
    for E in ('1e5', '1e6'):
        pts = [(np.log(g), *n[(E, g)]) for g in GAMMAS if (E, g) in n]
        if len(pts) < 2:
            continue
        x, y, e = map(np.array, zip(*pts))
        c, s = wline(x, y, e)
        slopes[E] = (c[1], s[1])
        print(f'    E = {E}:  dn/dln(gdot) = {c[1]:+.4f} +/- {s[1]:.4f}')
    if len(slopes) == 2:
        (a, ae), (b, be) = slopes['1e5'], slopes['1e6']
        d = abs(a - b) / float(np.hypot(ae, be))
        print(f'    the two slopes differ by {d:.1f} sigma '
              f'({"consistent with both" if d < 2 else "favours a single group"})')

    print('\n  OFFSET BETWEEN STIFFNESSES AT EACH SHEAR RATE')
    offs = []
    for g in GAMMAS:
        if ('1e5', g) in n and ('1e6', g) in n:
            (a, ae), (b, be) = n[('1e5', g)], n[('1e6', g)]
            offs.append((b - a, float(np.hypot(ae, be))))
            print(f'    gdot = {g:.3e}:  n(1e6) - n(1e5) = {b-a:+.4f} +/- '
                  f'{np.hypot(ae, be):.4f}')
    w = np.array([1 / e ** 2 for _, e in offs])
    off = float(np.dot(w, [o for o, _ in offs]) / w.sum())
    offe = float(1 / np.sqrt(w.sum()))
    print(f'    weighted mean offset: {off:+.4f} +/- {offe:.4f}')

    arm, arme = slopes['1e5']
    predA, predAe = -0.4 * np.log(10) * arm, 0.4 * np.log(10) * arme
    predB, predBe = DNDLNE_MOD[0] * np.log(10), DNDLNE_MOD[1] * np.log(10)
    gA = abs(off - predA) / float(np.hypot(offe, predAe))
    gB = abs(off - predB) / float(np.hypot(offe, predBe))
    sep = abs(predA - predB) / float(np.hypot(predAe, predBe))
    print('\n  WHICH HYPOTHESIS THE OFFSET FAVOURS')
    print(f'    A, one group t_c*gdot:     predicts {predA:+.4f} +/- {predAe:.4f}'
          f'   measured differs by {gA:.1f} sigma')
    print(f'    B, I and kappa separate:   predicts {predB:+.4f} +/- {predBe:.4f}'
          f'   measured differs by {gB:.1f} sigma')
    print(f'    the two predictions are themselves {sep:.1f} sigma apart, which')
    print('    is the most this test can ever discriminate.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
