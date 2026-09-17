"""The confound test on the full grid: shear-rate arms at three stiffnesses.

WHY THIS EXISTS.  Two accounts of the shear-rate dependence of n survive the
existing data, and both are statements about the same function n(ln I, ln kappa),
because the contact time relative to the shear time is itself a combination of
the two: dimensional analysis on rho, d, P and E leaves only one time scale,
d*sqrt(rho/P), times a function of E/P, so t_c*gdot = I * kappa^(-alpha).

  A.  one group.  n = F(ln I - alpha ln kappa).  Changing kappa slides the
      curve n(ln I) SIDEWAYS.
  B.  two variables.  n = G(ln I) + H(ln kappa).  Changing kappa slides it UP.

If n is straight in ln I those two are indistinguishable: shifting a line right
by delta is the same as shifting it down by slope*delta.  That degeneracy, not
the choice of lever, is why two stiffnesses reached only 1.7 sigma.

WHAT BREAKS IT is curvature.  Slide a curved function sideways and its local
slope at fixed I changes; slide it up and the slope does not.  So under B the
arm slope must be the SAME at every stiffness, while under A it drifts with
kappa.  Two stiffnesses give one difference; three give a trend with an error,
which is the point of the E = 1e4 arms.

The offset between stiffnesses is reported as well, but it discriminates weakly:
A predicts -0.4*ln(10) times the arm slope and B predicts ln(10) times the
modulus-lever slope, and those two numbers are close.

The frictionless control runs the same test at mu_g = 0, where only the two
families that scanned it are available.

Usage:  python3 confound_grid.py [mu_g]
"""
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nfit

MU = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
ZMIN = 3.0
CANON = 8.4664e-04
GAMMAS = (3.162e-4, 1.0e-3, 3.162e-3)
P_CONF = 10.0
DNDLNE_MOD = (0.0360, 0.0049)          # stiffness_matched.py, 2D, matched

RX_G = r'log\.g(?P<g>[0-9.eE+-]+)_mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$'
RX_M = r'log\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$'

# label, modulus, and where the three arms live.  E = 1e7 (kappa = 1e6) was
# added last, to test whether the drift of the arm slope continues toward the
# stiffness of real grains or saturates there.  The E = 1e5 middle arm is the
# production scan itself, on different packings from every other cell here.
FAMILIES = (
    ('1e4', 1.0e4, [('sweep_iscan_stiff_E1e4', RX_G, None)]),
    ('1e5', 1.0e5, [('sweep_iscan2', RX_G, None),
                    ('sweep_steady2d', RX_M, 1.0e-3)]),
    ('1e6', 1.0e6, [('sweep_iscan_stiff', RX_G, None)]),
    ('1e7', 1.0e7, [('sweep_iscan_stiff_E1e7', RX_G, None)]),
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


def overlap_theta0(grid, common):
    """Geometric centre of the Theta range every cell covers.

    The setpoint grid is matched by label, not by temperature, and a stiffer
    cell runs colder at the same label: at kappa = 1e6 the slow arm tops out at
    Theta = 8.0e-4, below the canonical Thetaz = 8.47e-4, so reading n there
    would extrapolate past the data.  Reading every cell at the centre of the
    range they share keeps the comparison interpolated.
    """
    lo, hi = [], []
    for runs in grid.values():
        _, byT = nfit.surviving_setpoints(runs, z_min=ZMIN)
        th = [np.mean([x['Theta'] for x in byT[t]]) for t in common]
        lo.append(min(th)); hi.append(max(th))
    return float(np.sqrt(max(lo) * min(hi)))


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
    r = y - A @ c
    return c, np.sqrt(np.diag(cov)), float((r ** 2 * w).sum())


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
        print(f'  mu_g = {MU}: no data for {sorted(set(t for t, _ in missing))},'
              ' those families are dropped')
        for t, g in missing:
            grid.pop((t, g), None)
        if len(grid) < 6:
            print('  too few cells left.')
            return 1

    keep = {k: gated(v) for k, v in grid.items()}
    common = sorted(set.intersection(*(set(v) for v in keep.values())))
    print('=' * 88)
    print(f'THE CONFOUND GRID   2D, mu_g = {MU},  '
          f'{len(grid)//len(GAMMAS)} stiffnesses')
    print('=' * 88)
    print(f'  setpoints surviving the gate in all {len(grid)} cells: {len(common)}')
    if len(common) < 4:
        print('  too few for a matched fit.')
        return 1
    theta0 = overlap_theta0(grid, common)
    print(f'  Theta window shared by every cell centres on {theta0:.4e}')
    print(f'  (canonical Thetaz = {CANON:.4e}; n is read at the shared centre)')

    n = {}
    print(f'\n  {"E":>6}{"kappa":>10}{"gdot":>11}{"n":>10}{"+/-":>9}')
    for tag, E, _ in FAMILIES:
        for g in GAMMAS:
            if (tag, g) not in grid:
                continue
            r = nfit.fit_local_sys(grid[(tag, g)], theta0,
                                   fn=nfit.fit_local_adaptive,
                                   restrict_to=common, z_min=ZMIN)
            if not r:
                print(f'  {tag:>6}{E/P_CONF:>10.0f}{g:>11.3e}   fit failed')
                continue
            e = float(np.hypot(r['tot'], r.get('sys_deg', 0.0)))
            n[(tag, g)] = (r['n'], e)
            print(f'  {tag:>6}{E/P_CONF:>10.0f}{g:>11.3e}{r["n"]:>10.4f}{e:>9.4f}')

    print('\n  ARM SLOPE AT EACH STIFFNESS')
    print('  Under B this must not move.  Under A it drifts with kappa.')
    slopes = {}
    for tag, E, _ in FAMILIES:
        pts = [(np.log(g), *n[(tag, g)]) for g in GAMMAS if (tag, g) in n]
        if len(pts) < 3:
            continue
        x, y, e = map(np.array, zip(*pts))
        c, s, chi2 = wline(x, y, e)
        slopes[tag] = (E, c[1], s[1])
        print(f'    E = {tag}:  dn/dln(gdot) = {c[1]:+.4f} +/- {s[1]:.4f}'
              f'   chi2 = {chi2:.2f} on 1 dof')

    if len(slopes) >= 3:
        x = np.log(np.array([E / P_CONF for E, _, _ in slopes.values()]))
        y = np.array([v for _, v, _ in slopes.values()])
        e = np.array([s for _, _, s in slopes.values()])
        c, s, chi2 = wline(x, y, e)
        print(f'\n    trend of the slope with stiffness:')
        print(f'      d(dn/dln gdot)/dln(kappa) = {c[1]:+.4f} +/- {s[1]:.4f}'
              f'   ({abs(c[1])/s[1]:.1f} sigma from zero)')
        print(f'      flat slope (hypothesis B) fits at chi2 = '
              f'{float(((y - np.average(y, weights=1/e**2))**2/e**2).sum()):.2f}'
              f' on {len(y)-1} dof')
        flat = float(((y - np.average(y, weights=1 / e ** 2)) ** 2
                      / e ** 2).sum())
        dof = len(y) - 1
        # survival of chi2 for small dof, without scipy
        pflat = (np.exp(-flat / 2) if dof == 1 else
                 np.exp(-flat / 2) * (1 + flat / 2) if dof == 3 else
                 np.exp(-flat / 2))
        print(f'      a stiffness-independent slope, which is what B requires,')
        print(f'      fits at chi2 = {flat:.2f} on {dof} dof, p = {pflat:.3f}')
        z = abs(c[1]) / s[1]
        if z < 2:
            print('      READING: no resolved drift.  Consistent with B, and with')
            print('      A if n is straight in ln I over this range.')
        elif z < 3:
            print('      READING: the slope drifts, but at this significance that')
            print('      is a lean and not a rejection.  B is disfavoured; it is')
            print('      not excluded.  A allows the drift, since a curved F seen')
            print('      through a sideways shift changes the local slope.')
        else:
            print('      READING: the slope drifts at a level B cannot absorb.')
            print('      n is not a sum of a term in I and a term in kappa.')

    print('\n  OFFSET BETWEEN ADJACENT STIFFNESSES')
    tags = [t for t, _, _ in FAMILIES]
    for lo, hi in zip(tags, tags[1:]):
        offs = [(n[(hi, g)][0] - n[(lo, g)][0],
                 float(np.hypot(n[(lo, g)][1], n[(hi, g)][1])))
                for g in GAMMAS if (lo, g) in n and (hi, g) in n]
        if not offs:
            continue
        w = np.array([1 / e ** 2 for _, e in offs])
        o = float(np.dot(w, [v for v, _ in offs]) / w.sum())
        oe = float(1 / np.sqrt(w.sum()))
        print(f'    n({hi}) - n({lo}) = {o:+.4f} +/- {oe:.4f}')
        if lo in slopes:
            pA = -0.4 * np.log(10) * slopes[lo][1]
            pB = DNDLNE_MOD[0] * np.log(10)
            print(f'        A predicts {pA:+.4f}, B predicts {pB:+.4f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
