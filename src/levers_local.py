"""
Stiffness derivatives and the two-lever tests, on the LOCAL exponent.

Both sets of claims collapsed when the Theta-window systematic was included:
the stiffness derivatives fell from 3.5-3.8 sigma to 0.1-1.0, and all nine
two-lever falsifications fell to 1.0-1.9 sigma. That systematic exists only
because a window-averaged n is ambiguous when mu(Theta) is curved. With
n(Theta_0) -- the local slope at a stated reference temperature -- there is
no window systematic to add, so these are recomputed here.

Theta_0 is the geometric centre of the Theta range COMMON to every cell in
each comparison, so nothing is extrapolated. Because the levers reach
different Theta at the same setpoint, this matters more here than anywhere
else: comparing local slopes at a shared Theta_0 is exactly what removes the
bias that the old comparisons absorbed as an error.

Usage:  python3 levers_local.py
"""
import csv
import re

import numpy as np

import nfit


def load_csv(path):
    rs = list(csv.DictReader(open(path)))
    for r in rs:
        for k in r:
            try:
                r[k] = float(r[k])
            except ValueError:
                pass
    return rs


def cells(pattern, rx):
    return nfit.load_sweep(pattern, rx)


def wslope(xs, ns, es):
    """Weighted straight line, covariance PROPAGATED from the input errors.

    The previous version rescaled the covariance by the fit residuals. With
    4-5 points (dof 2-3) and chi2/dof < 1 -- which is five of the six cells
    here -- that shrinks the error and manufactures significance: it reported
    dn/dln(E) = +0.0313 +/- 0.0034 (9.2 sigma) at mu_g = 0.3 where the
    propagated error is +/- 0.0051 (6.1 sigma). This is exactly the estimator
    must not be used at low dof, and which
    analyze_crossing.wline was written to avoid; levers_local was simply missed.

    chi2/dof is returned alongside as a goodness check, never folded into the
    error.
    """
    x = np.asarray(xs, float); y = np.asarray(ns, float); e = np.asarray(es, float)
    A = np.vstack([x, np.ones(len(x))]).T
    W = np.diag(1.0 / e ** 2)
    cov = np.linalg.inv(A.T @ W @ A)
    c = cov @ A.T @ W @ y
    chi2 = float(((y - A @ c) ** 2 / e ** 2).sum()) / max(len(x) - 2, 1)
    return float(c[0]), float(np.sqrt(cov[0, 0])), chi2


def main():
    st = cells('sweep_stiffness/log.E*',
               r'log\.E(?P<E>[0-9.eE+-]+)_mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$')
    kt = cells('sweep_kt_P10/log.k*',
               r'log\.k(?P<kt>[0-9.eE+-]+)_mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$')

    print('=' * 74)
    print('STIFFNESS DERIVATIVES on the local exponent')
    print('=' * 74)
    for mg in (0.1, 0.3, 1.0):
        Erun = [(float(dict(k)['E']), v) for k, v in st.items()
                if abs(float(dict(k)['mu']) - mg) < 1e-9]
        Krun = [(float(dict(k)['kt']), v) for k, v in kt.items()
                if abs(float(dict(k)['mu']) - mg) < 1e-9]
        Erun.sort(); Krun.sort()
        print(f'\n  mu_g = {mg}')
        out = {}
        for nm, arr, ref in (('dn/dln(E)', Erun, 1e5), ('dn/dln(k_t)', Krun, 7.326e4)):
            if len(arr) < 3:
                print(f'     {nm:<14} too few points')
                continue
            got = nfit.common_theta0([v for _, v in arr])
            if got is None:
                print(f'     {nm:<14} no common Theta range')
                continue
            T0, _ = got
            xs, ns, es = [], [], []
            for x, v in arr:
                # fit_local_sys, not fit_local: n(Theta_0) carries a window
                # systematic wherever the quadratic is inadequate, and quoting
                # stat alone there understates the error by up to 3x.
                r = nfit.fit_local_sys(v, T0)
                if r:
                    xs.append(np.log(x / ref)); ns.append(r['n']); es.append(r['tot'])
            if len(xs) < 3:
                print(f'     {nm:<14} too few fittable')
                continue
            s, se, c2 = wslope(xs, ns, es)
            out[nm] = (s, se)
            print(f'     {nm:<14} {s:+.4f} +/- {se:.4f}   {abs(s)/se:4.1f} sigma'
                  f'   [Theta_0={T0:.2e}, {len(xs)} pts, chi2/dof={c2:.2f}]')
        if 'dn/dln(E)' in out and 'dn/dln(k_t)' in out:
            a, ae = out['dn/dln(E)']; b, be = out['dn/dln(k_t)']
            kn, kne = a - b, float(np.hypot(ae, be))
            print(f'     {"=> dn/dln(k_n)":<14} {kn:+.4f} +/- {kne:.4f}   {abs(kn)/kne:4.1f} sigma')
            print(f'     opposite signs: {(kn > 0) != (b > 0)}'
                  f'   (weaker of the two at {min(abs(kn)/kne, abs(b)/be):.1f} sigma)')

    print()
    print('=' * 74)
    print('TWO-LEVER TESTS on the local exponent')
    print('=' * 74)
    fr = {}
    for k, v in cells('sweep_matched2d/log.mu*',
                      r'log\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$').items():
        fr.setdefault(float(dict(k)['mu']), []).extend(v)
    mf = load_csv('tier1_matched2d_results.csv')
    ms = load_csv('tier1_stiff_results.csv')
    for r in ms:
        r['E'] = float(re.match(r'E([0-9.eE+-]+)_mu', r['label']).group(1))

    stc = {(float(dict(k)['E']), float(dict(k)['mu'])): v for k, v in st.items()}
    pool = [fr[m] for m in sorted(fr) if m > 0] + list(stc.values())
    got = nfit.common_theta0(pool)
    if got is None:
        print('   no common Theta range across both levers')
        return
    T0, (lo, hi) = got
    print(f'   Theta_0 = {T0:.3e}  (common range [{lo:.2e}, {hi:.2e}])\n')
    print(f'   {"variable":<10}{"RMS discrepancy":>18}   verdict')
    print('   ' + '-' * 52)
    # The friction-lever curve and the stiffness-lever points both use
    # fit_local_sys: a two-lever discrepancy is a difference of two n(Theta_0)
    # values, so understating either error inflates the discrepancy. This
    # matters most here because the test only has to clear 2 sigma to count as
    # a falsification -- a candidate that fails on stat alone but survives with
    # the systematic has not been falsified.
    cache = {}

    def N(runs):
        k = id(runs)
        if k not in cache:
            cache[k] = nfit.fit_local_sys(runs, T0)
        return cache[k]

    for q in ('chi', 'Z', 'dZ_eff', 'cv_f', 'pr_f', 'f_mean', 'a_c_w'):
        pts = []
        for mg in sorted(fr):
            if mg == 0:
                continue
            r = N(fr[mg])
            s = [x for x in mf if x['mu_g'] == mg]
            if r and s:
                pts.append((np.mean([x[q] for x in s]), r['n'], r['tot']))
        if len(pts) < 4:
            continue
        pts.sort()
        fx = np.array([p[0] for p in pts]); fy = np.array([p[1] for p in pts])
        fe = np.array([p[2] for p in pts])
        sig = []
        for (E, mg), v in sorted(stc.items()):
            if abs(E - 1e5) < 1:
                continue
            r = N(v)
            s = [x for x in ms if x['E'] == E and x['mu_g'] == mg]
            if not (r and s):
                continue
            x = np.mean([x[q] for x in s])
            if x < fx.min() or x > fx.max():
                continue
            pred = float(np.interp(x, fx, fy))
            perr = float(np.interp(x, fx, fe))
            sig.append((r['n'] - pred) / float(np.hypot(r['tot'], perr)))
        if len(sig) < 3:
            continue
        rms = float(np.sqrt(np.mean(np.array(sig) ** 2)))
        print(f'   {q:<10}{rms:>15.1f} s   '
              f'{"FAILS -- not a state function" if rms >= 2 else "inconclusive"}'
              f'  ({len(sig)} pts)')


if __name__ == '__main__':
    main()
