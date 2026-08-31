"""
Do the two sign changes coincide?

The project has found two places where the response of the exponent changes
sign as friction is increased:

  (i)  the CURVATURE of mu(Theta) -- the local slope n rises with Theta at high
       friction and falls at low friction, crossing zero near mu_g ~ 0.09;
  (ii) the PRESSURE RESPONSE b = dn/dln(kappa), kappa = E/P -- n increases with
       stiffness-to-pressure ratio at one end of the friction range and
       decreases at the other, crossing somewhere in 0.07-0.09.

If these are ONE crossing, a single mechanism governs how n responds to both
temperature and confining pressure. If they are two, the apparent coincidence
is an accident of the coarse grid that measured them.

TWO TRAPS THIS SCRIPT AVOIDS
1. dof<=1 residual-scaled errors. The outer fit (n against ln kappa) had 3
   points and 2 parameters. numpy's residual-scaled covariance then reports an
   error near zero whatever the data, which is how this project once quoted a
   5.7 and a 13.7 sigma "sharp transition" that fell to ~1 sigma the moment
   errors were propagated properly. Here the outer covariance is ALWAYS
   inv(A^T W A) with W = diag(1/e_i^2), propagated from the per-cell errors and
   never rescaled by the residuals. The residual chi2/dof is printed alongside
   as a goodness check, not folded into the error.

2. Comparing pressures at the same DIMENSIONFUL Theta. The pressure lever
   scales Tgran ~ P precisely so the reduced temperature That = Theta/(P/rho)
   is held fixed; comparing cells at a common Theta instead would sweep the
   reduced temperature by the full factor of 10 in P and confound the pressure
   response with the curvature -- i.e. it would manufacture the very
   coincidence being tested. Theta_0 is therefore set per pressure as
   That_0 * P, with That_0 the geometric centre of the reduced range common to
   all four pressures.

Usage:  python3 analyze_crossing.py
"""
import numpy as np

import nfit

E_MOD = 1.0e5
PRESSURES = (5.0, 10.0, 25.0, 50.0)


# weighted straight line with errors propagated from the data, not residuals

def wline(x, y, e):
    """Fit y = c0 + c1*x. Return (c1, err(c1), cov, chi2/dof).

    cov = inv(A^T W A) -- the propagated covariance. It does NOT contain a
    residual scale factor, so it stays meaningful at dof = 1 and 0.
    """
    x = np.asarray(x, float); y = np.asarray(y, float); e = np.asarray(e, float)
    A = np.vstack([np.ones_like(x), x]).T
    W = np.diag(1.0 / e ** 2)
    cov = np.linalg.inv(A.T @ W @ A)
    c = cov @ (A.T @ W @ y)
    r = y - A @ c
    dof = max(len(x) - 2, 1)
    chi2 = float((r / e) @ (r / e)) / dof
    return float(c[1]), float(np.sqrt(cov[1, 1])), cov, chi2, c


def zero_crossing(xs, ys, es, label):
    """Where does y(x) cross zero? Straight-line fit + full error propagation.

    x* = x_ref - c0/c1 with x_ref the weighted centroid (which decorrelates c0
    and c1 and keeps the propagation well conditioned).
    """
    xs = np.asarray(xs, float); ys = np.asarray(ys, float); es = np.asarray(es, float)
    w = 1.0 / es ** 2
    xref = float(np.sum(w * xs) / np.sum(w))
    c1, c1e, cov, chi2, c = wline(xs - xref, ys, es)
    c0 = float(c[0])
    if abs(c1) < 1e-12:
        return None
    xstar = xref - c0 / c1
    J = np.array([-1.0 / c1, c0 / c1 ** 2])
    var = float(J @ cov @ J)
    err = float(np.sqrt(max(var, 0.0)))
    print(f'   {label}: slope {c1:+.4f} +/- {c1e:.4f} ({abs(c1)/c1e:.1f} sigma), '
          f'chi2/dof = {chi2:.2f}')
    print(f'   {label}: crosses zero at mu_g = {xstar:.4f} +/- {err:.4f}')
    return xstar, err, abs(c1) / c1e


# the two levers

def load_piscan():
    raw = nfit.load_sweep(
        'sweep_piscan/log.P*',
        r'log\.P(?P<P>[0-9.]+)_mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$',
        tgran_from=lambda m: float(m.group('T')) * float(m.group('P')) / 10.0)
    out = {}
    for key, runs in raw.items():
        d = dict(key)
        out[(float(d['P']), float(d['mu']))] = runs
    return out


def reduced_window(cells, mu):
    """Intersection of the REDUCED Theta ranges over the USABLE pressures.

    A pressure that cannot support a quadratic (fewer than 4 surviving
    setpoints) is dropped rather than voiding the whole friction. mu_g = 0.12
    was not in the original pressure grid, so it has only the 4 setpoints this
    campaign added, and at P = 5 one of those is barostat-unstable -- demanding
    all four pressures threw the point away, and it sits in the middle of the
    crossing region. The intersection is then taken over what remains, so
    Theta_0 is still interior to every cell that enters the fit.
    """
    los, his, ok = [], [], []
    for P in PRESSURES:
        runs = cells.get((P, mu))
        if not runs:
            continue
        Ts, byT = nfit.surviving_setpoints(runs)
        if len(Ts) < 4:
            continue
        th = [r['Theta'] / P for t in Ts for r in byT[t] if r['Theta'] > 0]
        if not th:
            continue
        los.append(min(th)); his.append(max(th)); ok.append(P)
    if len(ok) < 3:
        return None
    lo, hi = max(los), min(his)
    if hi <= lo:
        return None
    return float(np.sqrt(lo * hi)), lo, hi, ok


def b_of_mu(cells, mu, verbose=False):
    """b = dn/dln(kappa) at fixed reduced temperature. kappa = E/P."""
    got = reduced_window(cells, mu)
    if got is None:
        return None
    That0, lo, hi, ok = got
    xs, ns, es = [], [], []
    for P in ok:
        r = nfit.fit_local(cells[(P, mu)], That0 * P)
        if r is None:
            continue
        xs.append(np.log(E_MOD / P / 1e4)); ns.append(r['n']); es.append(r['stat'])
    if len(xs) < 3:
        return None
    b, be, _, chi2, _ = wline(xs, ns, es)
    if verbose:
        print(f'      That_0 = {That0:.3e} (common reduced range '
              f'[{lo:.2e}, {hi:.2e}]), P = {[int(p) for p in ok]}, '
              f'chi2/dof = {chi2:.2f}')
    return dict(b=b, err=be, npts=len(xs), chi2=chi2, That0=That0, Ps=ok)


def curv_of_mu(mus):
    """Curvature of ln mu vs ln Theta, 2D at Pconf = 10, at a common Theta_0."""
    raw = nfit.load_sweep(
        'sweep_matched2d/log.mu*',
        r'log\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$')
    C = {}
    for key, runs in raw.items():
        C.setdefault(float(dict(key)['mu']), []).extend(runs)
    avail = [m for m in mus if m in C]
    got = nfit.common_theta0([C[m] for m in avail])
    if got is None:
        return {}, None
    T0, _ = got
    out = {}
    for m in avail:
        r = nfit.fit_local(C[m], T0)
        if r:
            out[m] = r
    return out, T0


def main():
    cells = load_piscan()
    # The pressure lever was only ever run at these frictions; the curvature
    # lever has more, and uses all of them -- there is no reason to throw away
    # curvature points just because the pressure scan lacks the matching cell.
    band = [0.05, 0.07, 0.08, 0.09, 0.1, 0.12, 0.15]
    cband = [0.05, 0.06, 0.08, 0.1, 0.11, 0.12, 0.13, 0.15, 0.17]

    print('=' * 78)
    print('LEVER 1 -- pressure response  b = dn/dln(kappa)   [kappa = E/P]')
    print('=' * 78)
    bs = {}
    for mg in band:
        r = b_of_mu(cells, mg, verbose=True)
        if r is None:
            print(f'   mu_g = {mg:<5g} VOID')
            continue
        bs[mg] = r
        print(f'   mu_g = {mg:<5g} b = {r["b"]:+.4f} +/- {r["err"]:.4f}'
              f'   ({abs(r["b"])/r["err"]:.1f} sigma)   [{r["npts"]} P]')

    print()
    print('=' * 78)
    print('LEVER 2 -- curvature of mu(Theta)   [2D, Pconf = 10]')
    print('=' * 78)
    cv, T0 = curv_of_mu(cband)
    if T0:
        print(f'   common Theta_0 = {T0:.3e}')
    for mg in sorted(cv):
        r = cv[mg]
        print(f'   mu_g = {mg:<5g} curv = {r["curv"]:+.4f} +/- {r["curv_err"]:.4f}'
              f'   ({abs(r["curv"])/r["curv_err"]:.1f} sigma)   [{r["k"]} setpoints]')

    print()
    print('=' * 78)
    print('THE TWO CROSSINGS')
    print('=' * 78)
    res = {}
    if len(bs) >= 3:
        mm = sorted(bs)
        res['b'] = zero_crossing(mm, [bs[m]['b'] for m in mm],
                                 [bs[m]['err'] for m in mm], 'pressure b ')
    if len(cv) >= 3:
        mm = sorted(cv)
        res['c'] = zero_crossing(mm, [cv[m]['curv'] for m in mm],
                                 [cv[m]['curv_err'] for m in mm], 'curvature  ')

    if 'b' in res and 'c' in res and res['b'] and res['c']:
        (xb, eb, sb), (xc, ec, sc) = res['b'], res['c']
        d = xb - xc
        s = float(np.hypot(eb, ec))
        print()
        print(f'   separation {d:+.4f} +/- {s:.4f}  ->  {abs(d)/s:.1f} sigma')
        if min(sb, sc) < 3:
            print('   WARNING: at least one lever\'s slope is under 3 sigma, so its '
                  'crossing\n            location is poorly determined -- treat the '
                  'comparison as a bound.')
        if abs(d) / s < 2:
            print('   -> CONSISTENT with a single crossing.')
            print(f'   -> combined location mu_g* = '
                  f'{(xb/eb**2 + xc/ec**2)/(1/eb**2 + 1/ec**2):.4f} '
                  f'+/- {1/np.sqrt(1/eb**2 + 1/ec**2):.4f}')
        else:
            print('   -> DISTINCT crossings: the coincidence was an artefact of the '
                  'coarse grid.')

    robustness(cells, band, cband)


# Robustness. Every quantitative claim this project has retracted looked clean
# until it was pushed on, so the crossing gets pushed on before it is written
# down.

def _cross(xs, ys, es):
    """Crossing location + error, silent."""
    xs = np.asarray(xs, float); ys = np.asarray(ys, float); es = np.asarray(es, float)
    w = 1.0 / es ** 2
    xref = float(np.sum(w * xs) / np.sum(w))
    c1, _, cov, _, c = wline(xs - xref, ys, es)
    if abs(c1) < 1e-12:
        return None
    c0 = float(c[0])
    J = np.array([-1.0 / c1, c0 / c1 ** 2])
    return xref - c0 / c1, float(np.sqrt(max(float(J @ cov @ J), 0.0)))


def robustness(cells, band, cband):
    print()
    print('=' * 78)
    print('ROBUSTNESS')
    print('=' * 78)

    # (a) Does the curvature crossing depend on where in Theta it is measured?
    # If it does, "the same crossing" is a statement about one temperature only.
    raw = nfit.load_sweep(
        'sweep_matched2d/log.mu*',
        r'log\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$')
    C = {}
    for key, runs in raw.items():
        C.setdefault(float(dict(key)['mu']), []).extend(runs)
    avail = [m for m in cband if m in C]
    got = nfit.common_theta0([C[m] for m in avail])
    # NOT by re-evaluating at a different Theta_0: shifting the centre of a
    # quadratic changes b but leaves c invariant, so curv is Theta_0-independent
    # by construction and that test returns the same number five times over. It
    # has to be done by refitting over different Theta SUB-WINDOWS, which is
    # also the honest statement of what curv is -- a property of the window
    # fitted, not a local quantity.
    print('\n  (a) curvature crossing vs the Theta WINDOW fitted')
    if got:
        allT = sorted({round(r['Tgran'], 12) for m in avail for r in C[m]})
        halves = (('low  half', allT[:len(allT) // 2 + 2]),
                  ('high half', allT[-(len(allT) // 2 + 2):]),
                  ('full     ', allT))
        for nm, keep in halves:
            pts = []
            for m in avail:
                r = nfit.fit_local(C[m], 1.0, restrict_to=keep)
                if r:
                    pts.append((m, r))
            if len(pts) < 4:
                print(f'      {nm}: too few cells survive')
                continue
            z = _cross([m for m, _ in pts], [r['curv'] for _, r in pts],
                       [r['curv_err'] for _, r in pts])
            if z:
                print(f'      {nm} ({len(keep)} setpoints, {len(pts)} cells): '
                      f'crossing mu_g = {z[0]:.4f} +/- {z[1]:.4f}')

    # (b) Leave-one-friction-out. A crossing driven by a single cell is not a
    # crossing.
    print('\n  (b) leave-one-friction-out')
    jkres = {}
    for nm, keys, getter in (
            ('curvature', avail,
             lambda m, T: (lambda r: (r['curv'], r['curv_err']) if r else None)(
                 nfit.fit_local(C[m], T))),
            ('pressure b', band,
             lambda m, T: (lambda r: (r['b'], r['err']) if r else None)(
                 b_of_mu(cells, m)))):
        T0 = float(np.sqrt(got[1][0] * got[1][1])) if got else None
        full = [(m, getter(m, T0)) for m in keys]
        full = [(m, v) for m, v in full if v]
        if len(full) < 5:
            print(f'      {nm}: too few points to jackknife')
            continue
        locs = []
        for drop in [m for m, _ in full]:
            sub = [(m, v) for m, v in full if m != drop]
            z = _cross([m for m, _ in sub], [v[0] for _, v in sub],
                       [v[1] for _, v in sub])
            if z:
                locs.append(z[0])
        z0 = _cross([m for m, _ in full], [v[0] for _, v in full],
                    [v[1] for _, v in full])
        k = len(locs)
        jk = float(np.sqrt((k - 1) / k * np.sum((np.array(locs) - z0[0]) ** 2)))
        jkres[nm] = (z0[0], z0[1], jk)
        print(f'      {nm}: full {z0[0]:.4f} +/- {z0[1]:.4f}(stat), '
              f'range over drops [{min(locs):.4f}, {max(locs):.4f}], '
              f'jackknife {jk:.4f}')
        if jk > 2 * z0[1]:
            print('        -> WARNING: one friction dominates the crossing.')

    # The jackknife is a real systematic on the crossing LOCATION -- the choice
    # of friction grid. It belongs in the comparison, exactly as the setpoint
    # jackknife belongs in n.
    if 'curvature' in jkres and 'pressure b' in jkres:
        (xc, ec, jc), (xb, eb, jb) = jkres['curvature'], jkres['pressure b']
        tc, tb = float(np.hypot(ec, jc)), float(np.hypot(eb, jb))
        d, s = xb - xc, float(np.hypot(tc, tb))
        print(f'\n      totals: curvature {xc:.4f} +/- {tc:.4f}, '
              f'pressure b {xb:.4f} +/- {tb:.4f}')
        print(f'      separation with the grid systematic included: '
              f'{d:+.4f} +/- {s:.4f} -> {abs(d)/s:.1f} sigma')

    # (c) The same curvature crossing in 3D. Nothing forces the two dimensions
    # to cross at the same friction; if they do, the mechanism is not
    # dimension-specific.
    print('\n  (c) curvature crossing in 3D')
    r3 = nfit.load_sweep('sweep_run_3d/log.mu*',
                         r'log\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$')
    D3 = {}
    for key, runs in r3.items():
        D3.setdefault(float(dict(key)['mu']), []).extend(runs)
    g3 = nfit.common_theta0(list(D3.values()))
    if not g3:
        print('      no common Theta range in 3D')
        return
    T3, _ = g3
    pts = [(m, nfit.fit_local(D3[m], T3)) for m in sorted(D3)]
    pts = [(m, r) for m, r in pts if r]
    for m, r in pts:
        print(f'      mu_g = {m:<5g} curv = {r["curv"]:+.4f} +/- {r["curv_err"]:.4f}')
    low = [(m, r) for m, r in pts if m <= 0.3]
    if len(low) >= 3:
        z = _cross([m for m, _ in low], [r['curv'] for _, r in low],
                   [r['curv_err'] for _, r in low])
        if z:
            print(f'      3D crossing (mu_g <= 0.3, Theta_0 = {T3:.2e}): '
                  f'{z[0]:.4f} +/- {z[1]:.4f}')


if __name__ == '__main__':
    main()
