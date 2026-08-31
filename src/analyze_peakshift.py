"""
AUDIT: does the peak of n(mu_g) actually move with pressure?

This is the claim that resolves the mu_g ~ 0.3 anomaly -- "the peak slides
toward lower friction as pressure rises, by 2-3x over P = 5 -> 50" -- and the
window-systematic audit undercut its evidence. At a single pressure the peak
LOCATION is not determined: at P = 10 the frictions 0.1, 0.15, 0.2 and 0.3 are
all statistically tied for the maximum once the setpoint jackknife is carried.
A claim that the peak moves is built on locating it at each pressure, so it
has to be rechecked rather than assumed.

TWO INDEPENDENT ESTIMATORS, because neither alone is convincing:

  (A) DIRECT. Fit a parabola in ln(mu_g) through the cells bracketing the
      maximum and solve for the vertex. Honest but blunt -- the curve is flat
      near its top, which is exactly why the location is poorly determined.

  (B) VIA b. Under a pure horizontal slide, n(mu_g, P) = g(ln mu_g - d(ln P)),
      so b = dn/dln(kappa) = -g' * dd/dln(kappa), which vanishes EXACTLY at the
      peak and is steepest on the flanks. The zero of b therefore locates the
      peak, and it does so using the flanks -- where the data are informative
      -- instead of the flat top. This is why b gives 0.122 +/- 0.025 where the
      direct maximum gives "somewhere in 0.1-0.3".

      Estimator (B) assumes the slide is rigid. If the peak also changes HEIGHT
      or WIDTH with pressure, b picks up a term that does not vanish at the
      vertex, and (A) and (B) will disagree. So the two estimators together
      test the moving-peak PICTURE, not just the peak position -- agreement
      supports a rigid slide, disagreement means the shape changes too.

Errors: n(Theta_0) carries the setpoint jackknife (fit_local_sys); the vertex
and zero-crossing errors are propagated from the full covariance, never
residual-scaled (see nfit and analyze_crossing for why that matters at low
dof).

Usage:  python3 analyze_peakshift.py
"""
import numpy as np

import nfit
from analyze_crossing import load_piscan, wline, PRESSURES, E_MOD


def reduced_theta0(cells, P, mus):
    """Geometric centre of the reduced-Theta range common to all mu at this P."""
    los, his = [], []
    for m in mus:
        runs = cells.get((P, m))
        if not runs:
            continue
        Ts, byT = nfit.surviving_setpoints(runs)
        if len(Ts) < 4:
            continue
        th = [r['Theta'] / P for t in Ts for r in byT[t] if r['Theta'] > 0]
        if th:
            los.append(min(th)); his.append(max(th))
    if not los:
        return None
    lo, hi = max(los), min(his)
    if hi <= lo:
        return None
    return float(np.sqrt(lo * hi))


def vertex(xs, ys, es):
    """Parabola vertex in x, with the error propagated from the fit covariance."""
    xs = np.asarray(xs, float); ys = np.asarray(ys, float); es = np.asarray(es, float)
    if len(xs) < 4:
        return None
    x0 = float(np.mean(xs))
    X = xs - x0
    A = np.vstack([np.ones_like(X), X, X ** 2]).T
    W = np.diag(1.0 / es ** 2)
    cov = np.linalg.inv(A.T @ W @ A)
    c = cov @ (A.T @ W @ ys)
    if c[2] >= 0:                     # opens upward: no maximum
        return None
    xv = -c[1] / (2 * c[2])
    J = np.array([0.0, -1.0 / (2 * c[2]), c[1] / (2 * c[2] ** 2)])
    err = float(np.sqrt(max(float(J @ cov @ J), 0.0)))
    r = ys - A @ c
    chi2 = float((r / es) @ (r / es)) / max(len(xs) - 3, 1)
    return x0 + xv, err, chi2


def main():
    cells = load_piscan()
    mus = sorted({k[1] for k in cells})

    print('=' * 78)
    print('(A) DIRECT: peak of n(mu_g) at each pressure, from a parabola in ln(mu_g)')
    print('=' * 78)
    peaks = {}
    for P in PRESSURES:
        T0h = reduced_theta0(cells, P, mus)
        if T0h is None:
            print(f'   P = {P:<5g} no common reduced-Theta range')
            continue
        pts = []
        for m in mus:
            runs = cells.get((P, m))
            if not runs:
                continue
            r = nfit.fit_local_sys(runs, T0h * P)
            if r:
                pts.append((m, r['n'], r['tot']))
        if len(pts) < 5:
            print(f'   P = {P:<5g} only {len(pts)} fittable frictions')
            continue
        # restrict to the cells bracketing the maximum: a parabola through the
        # whole 0.03-1.0 range would be fitting the tails, not the peak
        imax = int(np.argmax([p[1] for p in pts]))
        sel = pts[max(0, imax - 3):imax + 4]
        v = vertex([np.log(p[0]) for p in sel], [p[1] for p in sel],
                   [p[2] for p in sel])
        top = pts[imax]
        print(f'   P = {P:<5g} That_0 = {T0h:.2e}  highest cell mu_g = {top[0]:g} '
              f'(n = {top[1]:.4f} +/- {top[2]:.4f})')
        if v is None:
            print('             vertex undetermined (no maximum in the bracket)')
            continue
        xv, ev, chi2 = v
        peaks[P] = (float(np.exp(xv)), float(np.exp(xv) * ev), chi2)
        print(f'             vertex mu_g* = {np.exp(xv):.4f} '
              f'(x{np.exp(ev):.2f}/{np.exp(ev):.2f})   chi2/dof = {chi2:.2f}'
              f'   [{len(sel)} cells]')

    if len(peaks) >= 3:
        Ps = sorted(peaks)
        xs = [np.log(E_MOD / p / 1e4) for p in Ps]
        ys = [np.log(peaks[p][0]) for p in Ps]
        es = [peaks[p][1] / peaks[p][0] for p in Ps]
        s, se, _, chi2, _ = wline(xs, ys, es)
        print(f'\n   trend: dln(mu_g*)/dln(kappa) = {s:+.4f} +/- {se:.4f} '
              f'({abs(s)/se:.1f} sigma), chi2/dof = {chi2:.2f}')
        span = abs(s) * (np.log(E_MOD / min(Ps) / 1e4) - np.log(E_MOD / max(Ps) / 1e4))
        print(f'   implied motion over P = {max(Ps):g} -> {min(Ps):g}: '
              f'x{np.exp(span):.2f}')

    print()
    print('=' * 78)
    print('(B) VIA b: the zero of dn/dln(kappa) locates the peak')
    print('=' * 78)
    print('   from analyze_crossing: b crosses zero at mu_g = 0.1224 +/- 0.0254')
    print('   That is a pressure-AVERAGED peak position. It is sharper than (A)')
    print('   because it uses the flanks rather than the flat top.')
    if peaks:
        Ps = sorted(peaks)
        gm = float(np.exp(np.mean([np.log(peaks[p][0]) for p in Ps])))
        ge = float(np.mean([peaks[p][1] / peaks[p][0] for p in Ps]) / np.sqrt(len(Ps))) * gm
        d = gm - 0.1224
        s = float(np.hypot(ge, 0.0254))
        print(f'\n   (A) geometric-mean peak over pressures: {gm:.4f} +/- {ge:.4f}')
        print(f'   (A) vs (B): {d:+.4f} +/- {s:.4f} -> {abs(d)/s:.1f} sigma')
        print('   Agreement supports a rigid horizontal slide; disagreement would')
        print('   mean the peak changes shape with pressure, not just position.')


if __name__ == '__main__':
    main()
