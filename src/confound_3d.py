"""Does the shear-rate dependence of n move with stiffness in three dimensions?

WHY THIS EXISTS.  confound_grid.py answers that question in two dimensions over
four moduli, and Sec. IV of the paper ends by conceding that the
three-dimensional rejection of separability has no stiffness control of the same
kind.  These runs are that control: the three 3D shear rates repeated at ten
times the baseline modulus, at mu_g = 0.3.

Two stiffnesses cannot separate the two accounts of the shear-rate dependence
(that needs curvature, and curvature needs three), but they answer the question
that matters here, which is whether the slope moves at all outside two
dimensions.

The middle shear rate of the baseline lives in the steady-state friction scan
rather than in the shear-rate scan, which is why two directories are read for
kappa = 1e4 and one for 1e5.

Usage:  python3 confound_3d.py       (run from the directory holding the sweeps)
"""
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import confound_grid as cg
import nfit

MU = 0.3
ZMIN = 4.0                      # D + 1 in three dimensions
GAMMAS = (3.162e-4, 1.0e-3, 3.162e-3)
P_CONF = 10.0

RX_G = r'log\.g(?P<g>[0-9.eE+-]+)_mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$'
RX_M = r'log\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$'

FAMILIES = (
    ('1e5', 1.0e5, [('sweep_iscan3d', RX_G, None),
                    ('sweep_steady3d', RX_M, 1.0e-3)]),
    ('1e6', 1.0e6, [('sweep_iscan3d_stiff', RX_G, None)]),
)


def load(sources):
    out = {}
    for d, rx, gfix in sources:
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


def main():
    grid = {}
    for tag, E, src in FAMILIES:
        C = load(src)
        for g in GAMMAS:
            if g in C:
                grid[(tag, g)] = C[g]
    missing = [(t, g) for t, _, _ in FAMILIES for g in GAMMAS
               if (t, g) not in grid]
    if missing:
        print('incomplete grid, missing:', missing)
        return 1

    keep = {k: gated(v) for k, v in grid.items()}
    common = sorted(set.intersection(*(set(v) for v in keep.values())))
    print('=' * 88)
    print(f'THE STIFFNESS CONTROL IN THREE DIMENSIONS   mu_g = {MU}')
    print('=' * 88)
    print(f'  targets surviving the cutoffs in all {len(grid)} cases: {len(common)}')
    if len(common) < 4:
        print('  too few for a matched fit.')
        return 1
    theta0 = cg.overlap_theta0(grid, common)
    print(f'  read at Theta = {theta0:.4e}, the centre of the window they share')

    n = {}
    print(f'\n  {"E":>6}{"kappa":>10}{"gdot":>11}{"n":>10}{"+/-":>9}')
    for tag, E, _ in FAMILIES:
        for g in GAMMAS:
            r = nfit.fit_local_sys(grid[(tag, g)], theta0,
                                   fn=nfit.fit_local_adaptive,
                                   restrict_to=common, z_min=ZMIN)
            if not r:
                print(f'  {tag:>6}{E/P_CONF:>10.0f}{g:>11.3e}   fit failed')
                continue
            e = float(np.hypot(r['tot'], r.get('sys_deg', 0.0)))
            n[(tag, g)] = (r['n'], e)
            print(f'  {tag:>6}{E/P_CONF:>10.0f}{g:>11.3e}{r["n"]:>10.4f}{e:>9.4f}')

    print('\n  SLOPE AGAINST SHEAR RATE AT EACH STIFFNESS')
    slopes = {}
    for tag, E, _ in FAMILIES:
        pts = [(np.log(g), *n[(tag, g)]) for g in GAMMAS if (tag, g) in n]
        if len(pts) < 3:
            continue
        x, y, e = map(np.array, zip(*pts))
        c, s, chi2 = cg.wline(x, y, e)
        slopes[tag] = (E, c[1], s[1])
        print(f'    E = {tag}:  dn/dln(gdot) = {c[1]:+.4f} +/- {s[1]:.4f}'
              f'   chi2 = {chi2:.2f} on 1 dof')

    if len(slopes) == 2:
        (_, a, sa), (_, b, sb) = [slopes[t] for t, _, _ in FAMILIES]
        d, sd = b - a, float(np.hypot(sa, sb))
        print(f'\n    change over one decade in kappa: {d:+.4f} +/- {sd:.4f}'
              f'   ({abs(d)/sd:.1f} sigma)')
        print('    2D over the same decade, kappa 1e4 to 1e5: -0.0207 +/- 0.0104')
        print('    (confound_grid.py, read at its own shared window)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
