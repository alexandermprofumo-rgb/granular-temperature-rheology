"""The confound grid fitted as one surface n(ln gdot, ln kappa).

WHY THIS EXISTS.  confound_grid.py tests the two accounts one consequence at a
time: the arm slope at each stiffness, then the offset between stiffnesses.  Both
are views of one surface, and fitting all twelve cells jointly uses every
consequence at once, including one the per-arm test ignores: the curvature
WITHIN each arm.

Write x = ln(gdot / 1e-3), which at fixed P is ln I up to a constant, and
y = ln(kappa / 1e4).  Around the centre of the grid:

  full   n = a + b x + c y + e x^2 + f y^2 + d x y                 6 parameters
  B      n = G(x) + H(y)            ->  d = 0                      5 parameters
  A      n = F(x - alpha y)         ->  c = -alpha b,  f = alpha^2 e,
                                        d = -2 alpha e             4 parameters

B allows no interaction.  A allows one, but ties it to the curvature along the
arms: d / e = -2 alpha.  With alpha > 0 (t_c*gdot = I kappa^-alpha, stiffer
grains shorter contacts) the interaction and the within-arm curvature must have
OPPOSITE signs.  The per-arm test sees only d; this one also sees e.

alpha is not pinned by the contact law (0.4 for a Hertzian impact at fixed speed,
1/3 for a contact already loaded to P), so A is fitted at both and with alpha
free.

Every cell is read at the centre of the Theta range they all share, not at the
canonical Thetaz, which lies above the window of the stiffest arms.

Caveat on the errors.  Cells are treated as independent.  The kappa = 1e3, 1e5
and 1e6 families start from the same two packings, so their errors are positively
correlated; ignoring that inflates the error of any difference between them and
makes every rejection below conservative.  The degree systematic is added in
quadrature per cell, which is not conservative if it is shared across cells.

Usage:  python3 confound_joint.py          (run from the directory holding the sweeps)
"""
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import confound_grid as cg
import nfit


def cells():
    grid = {}
    for tag, E, src in cg.FAMILIES:
        C = cg.load(src)
        for g in cg.GAMMAS:
            if g in C:
                grid[(E, g)] = C[g]
    keep = {k: cg.gated(v) for k, v in grid.items()}
    common = sorted(set.intersection(*(set(v) for v in keep.values())))
    theta0 = cg.overlap_theta0(grid, common)
    out = []
    for (E, g), runs in sorted(grid.items()):
        r = nfit.fit_local_sys(runs, theta0, fn=nfit.fit_local_adaptive,
                               restrict_to=common, z_min=cg.ZMIN)
        e = float(np.hypot(r['tot'], r.get('sys_deg', 0.0)))
        out.append((np.log(g / 1e-3), np.log(E / cg.P_CONF / 1e4), r['n'], e))
    return len(grid), len(common), theta0, np.array(out)


def wls(A, y, e):
    w = 1.0 / e ** 2
    cov = np.linalg.inv(A.T @ (A * w[:, None]))
    c = cov @ (A.T @ (w * y))
    return c, cov, float((((y - A @ c) / e) ** 2).sum())


def design(x, y, terms):
    cols = {'1': np.ones_like(x), 'x': x, 'y': y, 'xx': x * x, 'yy': y * y,
            'xy': x * y}
    return np.column_stack([cols[t] for t in terms])


def fit_A(x, y, n, e, alpha):
    u = x - alpha * y
    return wls(design(u, u, ('1', 'x', 'xx')), n, e)


def line(name, chi2, k, N):
    dof = N - k
    p = stats.chi2.sf(chi2, dof) if dof > 0 else float('nan')
    print(f'    {name:<34}{k:>3} par   chi2 = {chi2:6.2f} on {dof} dof'
          f'   p = {p:.3f}')
    return chi2, k


def main():
    ncell, ncommon, theta0, D = cells()
    x, y, n, e = D.T
    N = len(n)
    print('=' * 88)
    print(f'JOINT SURFACE FIT   2D, mu_g = {cg.MU},  {ncell} cells, '
          f'{ncommon} matched setpoints, read at Theta0 = {theta0:.3e}')
    print('=' * 88)

    full_terms = ('1', 'x', 'y', 'xx', 'yy', 'xy')
    cF, covF, chiF = wls(design(x, y, full_terms), n, e)
    sF = np.sqrt(np.diag(covF))
    print('\n  FULL QUADRATIC  (x = ln gdot/1e-3, y = ln kappa/1e4)')
    for t, v, s in zip(full_terms, cF, sF):
        print(f'    {t:>3}: {v:+.4f} +/- {s:.4f}   ({abs(v)/s:.1f} sigma)')

    print('\n  MODELS')
    fits = {}
    fits['full'] = line('full quadratic', chiF, 6, N)
    _, _, chi = wls(design(x, y, ('1', 'x', 'y')), n, e)
    fits['plane'] = line('plane (additive, straight)', chi, 3, N)
    _, _, chi = wls(design(x, y, ('1', 'x', 'y', 'xx', 'yy')), n, e)
    fits['B'] = line('B  additive, curved', chi, 5, N)
    _, _, chi = wls(design(x, y, ('1', 'x', 'y', 'xy')), n, e)
    fits['int'] = line('plane + interaction', chi, 4, N)
    for a in (0.4, 1.0 / 3.0):
        _, _, chi = fit_A(x, y, n, e, a)
        fits[f'A{a:.3f}'] = line(f'A  one group, alpha = {a:.3f}', chi, 3, N)
    grid = np.linspace(-3.0, 3.0, 6001)
    prof = np.array([fit_A(x, y, n, e, a)[2] for a in grid])
    ia = int(np.argmin(prof))
    ahat = grid[ia]
    inside = grid[prof <= prof[ia] + 1.0]
    fits['Afree'] = line('A  one group, alpha free', float(prof[ia]), 4, N)
    print(f'      best alpha = {ahat:+.3f}, 1-sigma range '
          f'[{inside.min():+.3f}, {inside.max():+.3f}]')

    print('\n  NESTED COMPARISONS AGAINST THE FULL SURFACE')
    for key, label in (('B', 'B  (d = 0)'), ('A0.400', 'A  alpha = 0.4'),
                       ('A0.333', 'A  alpha = 1/3'), ('Afree', 'A  alpha free')):
        chi, k = fits[key]
        dchi, ddof = chi - chiF, 6 - k
        p = stats.chi2.sf(dchi, ddof)
        z = stats.norm.isf(p / 2)
        print(f'    {label:<18} delta chi2 = {dchi:6.2f} on {ddof} dof'
              f'   p = {p:.3f}   ({z:.1f} sigma)')

    print('\n  THE SIGN TEST A MUST PASS')
    d, ee = cF[5], cF[3]
    cov_de = covF[np.ix_([5, 3], [5, 3])]
    print(f'    interaction d          = {d:+.4f} +/- {sF[5]:.4f}')
    print(f'    within-arm curvature e = {ee:+.4f} +/- {sF[3]:.4f}')
    print(f'    correlation(d, e)      = '
          f'{cov_de[0, 1] / np.sqrt(cov_de[0, 0] * cov_de[1, 1]):+.2f}')
    print('    A requires d = -2 alpha e, so opposite signs for alpha > 0.')
    for a in (0.4, 1.0 / 3.0):
        g = np.array([1.0, 2 * a])            # d + 2 alpha e
        v = float(g @ cov_de @ g)
        r = d + 2 * a * ee
        print(f'    alpha = {a:.3f}:  d + 2 alpha e = {r:+.4f} +/- {np.sqrt(v):.4f}'
              f'   ({abs(r)/np.sqrt(v):.1f} sigma from A)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
