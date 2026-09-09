"""Is the shear-rate dependence of n a contact-time effect in disguise?

WHY THIS EXISTS.  Two reviewers and one internal check arrived at the same
worry from different directions.  Along a shear-rate arm the modulus and the
pressure are fixed and gdot varies tenfold, so the Hertzian contact time
relative to the shear time, t_c*gdot, varies tenfold.  Along the modulus lever
gdot is fixed and t_c ~ E^(-2/5) varies, so t_c*gdot varies there too.  If n
depends on that single group, then the separability failure of Sec. IV and the
stiffness dependence of Sec. III D are one effect, and the first is a
finite-stiffness artifact rather than a property of the relation.

The two levers happen to cover nearly the same range.  Taking the baseline
(E = 1e5, gdot = 1e-3) as unity:

    modulus lever   t_c*gdot / baseline = (E/1e5)^(-2/5)   ->  0.40 to 2.51
    shear-rate arms t_c*gdot / baseline = gdot/1e-3        ->  0.32 to 3.16

so no extrapolation is needed and no new runs are required.  If n is a function
of t_c*gdot alone, points from the two levers must fall on one curve.  If the
arms sit systematically off the curve the modulus lever traces, the shear-rate
dependence is not a contact-time effect.

This is the same logic as analyze_collapse_pooled.py applied to a different
candidate variable: a correlation between two quantities driven by a common
cause looks exactly like a collapse until an orthogonal lever separates them.

Usage:  python3 confound_test.py
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
E_REF, G_REF = 1.0e5, 1.0e-3

# The middle arm is the baseline production sweep itself, at gdot = 1e-3;
# sweep_iscan2 holds only the slow and fast arms.
ARMS = ('sweep_iscan2',
        r'log\.g(?P<v>[0-9.eE+-]+)_mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$')
MIDARM = ('sweep_steady2d',
          r'log\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$', G_REF)
MODU = ('sweep_lev_E',
        r'log\.E(?P<v>[0-9.eE+-]+)_mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$')


def cells(d, rx):
    out = {}
    for p in glob.glob(f'{d}/log.*'):
        m = re.match(rx, os.path.basename(p))
        if not m or abs(float(m.group('mu')) - MU) > 1e-9:
            continue
        q = nfit.parse_log(p)
        if not q:
            continue
        q['Tgran'] = float(m.group('T'))
        out.setdefault(float(m.group('v')), []).append(q)
    return out


def gated(runs):
    Ts, byT = nfit.surviving_setpoints(runs, z_min=ZMIN)
    return [t for t in Ts
            if not [x['Z'] for x in byT[t] if 'Z' in x]
            or np.mean([x['Z'] for x in byT[t] if 'Z' in x]) >= ZMIN]


def series(C, ratio):
    """n and t_c*gdot, relative to the baseline, for one lever."""
    keep, ok = {}, []
    for v in sorted(C):
        k = gated(C[v])
        if len(k) >= 4:
            keep[v], _ = k, ok.append(v)
    if len(ok) < 3:
        return []
    common = sorted(set.intersection(*(set(keep[v]) for v in ok)))
    if len(common) < 4:
        return []
    out = []
    for v in ok:
        r = nfit.fit_local_sys(C[v], CANON, fn=nfit.fit_local_adaptive,
                               restrict_to=common, z_min=ZMIN)
        if r:
            out.append((ratio(v), r['n'],
                        float(np.hypot(r['tot'], r.get('sys_deg', 0.0))), v))
    return sorted(out)


def wline(x, y, e):
    w = 1.0 / np.asarray(e) ** 2
    A = np.column_stack([np.ones_like(x), x])
    cov = np.linalg.inv(A.T @ np.diag(w) @ A)
    c = cov @ A.T @ np.diag(w) @ y
    r = y - A @ c
    return c, np.sqrt(np.diag(cov)), float((r ** 2 * w).sum())


def main():
    A = cells(*ARMS)
    d, rx, g = MIDARM
    mid = {}
    for q in glob.glob(f'{d}/log.mu*'):
        m = re.match(rx, os.path.basename(q))
        if not m or abs(float(m.group('mu')) - MU) > 1e-9:
            continue
        r = nfit.parse_log(q)
        if r:
            r['Tgran'] = float(m.group('T'))
            mid.setdefault(g, []).append(r)
    A.update(mid)
    M = cells(*MODU)
    if not A or not M:
        print('sweeps not unpacked; run this beside the log archive.')
        return 1

    arms = series(A, lambda g: g / G_REF)
    modu = series(M, lambda E: (E / E_REF) ** -0.4)
    if len(arms) < 3 or len(modu) < 3:
        print('too few cells survive the gate on one lever.')
        return 1

    print('=' * 84)
    print(f'DOES n COLLAPSE ONTO t_c*gdot?   2D, mu_g = {MU}')
    print('=' * 84)
    for tag, S, unit in (('shear-rate arms', arms, 'gdot'),
                         ('modulus lever', modu, 'E')):
        print(f'\n  {tag}')
        print(f'  {"t_c*gdot":>11}{unit:>12}{"n":>10}{"+/-":>9}')
        for t, n, e, v in S:
            print(f'  {t:>11.3f}{v:>12.3g}{n:>10.4f}{e:>9.4f}')

    xm = np.log([t for t, *_ in modu])
    ym = np.array([n for _, n, *_ in modu])
    em = np.array([e for _, _, e, _ in modu])
    c, s, _ = wline(xm, ym, em)
    print(f'\n  modulus-lever curve: n = ({c[0]:+.4f}+/-{s[0]:.4f})'
          f' + ({c[1]:+.4f}+/-{s[1]:.4f}) ln(t_c gdot)')

    print('\n  the arms against that curve, at matched t_c*gdot:')
    gaps = []
    for t, n, e, v in arms:
        pred = c[0] + c[1] * np.log(t)
        pe = float(np.hypot(s[0], s[1] * abs(np.log(t))))
        g = (n - pred) / float(np.hypot(e, pe))
        gaps.append(g)
        print(f'    t_c gdot={t:>6.3f}  n={n:.4f}  curve={pred:.4f}  '
              f'gap={n-pred:+.4f} ({abs(g):.1f} sigma)')
    rms = float(np.sqrt(np.mean(np.square(gaps))))
    # The exponent linking t_c to E is not unambiguous.  A Hertzian collision at
    # fixed impact speed gives t_c ~ E^(-2/5); the oscillation time of a contact
    # already loaded to pressure P gives t_c ~ E^(-1/3) P^(-1/6).  Both are
    # defensible here, and the verdict is reported under each.
    xa = np.log([t for t, *_ in arms])
    ya = np.array([n for _, n, *_ in arms])
    ea = np.array([e for _, _, e, _ in arms])
    ca, sa, _ = wline(xa, ya, ea)
    print(f"\n  the arms have their own slope {ca[1]:+.4f}+/-{sa[1]:.4f};")
    for q in (0.4, 1.0 / 3.0):
        cm, sm, _ = wline(np.log([(v / E_REF) ** -q for *_, v in modu]), ym, em)
        d = abs(ca[1] - cm[1]) / float(np.hypot(sa[1], sm[1]))
        print(f"    t_c ~ E^-{q:.3f}: modulus slope {cm[1]:+.4f}+/-{sm[1]:.4f}"
              f"  ->  slopes differ by {d:.1f} sigma")
    print(f'\n  RMS discrepancy {rms:.1f} sigma over {len(gaps)} arms.')
    print('  VERDICT: ' + ('collapse HOLDS -- the shear-rate dependence is '
                           'consistent with\n           a contact-time effect '
                           'and Sec. IV needs restating.'
                           if rms < 2 else
                           'collapse FAILS -- n is not a function of t_c*gdot,\n'
                           '           so the shear-rate dependence is not the '
                           'stiffness effect.'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
