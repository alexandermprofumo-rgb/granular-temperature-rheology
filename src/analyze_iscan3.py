"""Three arms in I: the FORM test that two arms could not do.

analyze_separability.py showed mu*Theta^n = F(I) is not separable -- n falls
with I -- and that the fall looked multiplicative, n = n_0(mu_g) * I^k with a
common k = -0.267 +/- 0.050. With two arms that was a CONSISTENCY check only:
one k per friction, exactly determined, nothing left over to test the shape.

sweep_iscan2 is now complete, so there are three arms a decade apart:

    slow   gdot = 3.162e-4   I = 1.000e-4   sweep_iscan2
    mid    gdot = 1.0e-3     I = 3.162e-4   sweep_steady2d
    fast   gdot = 3.162e-3   I = 1.000e-3   sweep_iscan2

Three points, two parameters, so each friction now has ONE degree of freedom
against the power law -- the form is falsifiable for the first time.

FOUR TESTS, in increasing strength:

  T1  is n constant in I?          separability of the published relation.
                                   chi^2 on 2 dof per friction.
  T2  is ln n linear in ln I?      the n = n_0 I^k law. chi^2 on 1 dof.
  T3  is k common across friction? multiplicative separation of mu_g and I.
  T4  is ln mu linear in ln I?     whether F(I) is itself a power law, which
                                   the published form does not require but
                                   which every mu(I) rheology assumes.

Usage:  python3 analyze_iscan3.py
"""
import numpy as np

import nfit
from analyze_separability import jk

ZMIN = 3.0
MUS = [0.0, 0.15, 0.3, 1.0]
CANON = 8.466e-4

ARMS = [
    ('slow', 1.000e-4, 'sweep_iscan2/log.g3.162e-4_mu*',
     r'log\.g(?P<g>[0-9.eE+-]+)_mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$'),
    ('mid', 3.162e-4, 'sweep_steady2d/log.mu*',
     r'log\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$'),
    ('fast', 1.000e-3, 'sweep_iscan2/log.g3.162e-3_mu*',
     r'log\.g(?P<g>[0-9.eE+-]+)_mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$'),
]


def load(pattern, rx):
    out = {}
    for key, runs in nfit.load_sweep(pattern, rx).items():
        out.setdefault(float(dict(key)['mu']), []).extend(runs)
    return out


def gated(runs, window=None):
    """Jamming-gated setpoints, optionally restricted to a Theta window.

    THE WINDOW MATTERS. mu(Theta) is curved -- a cubic is required in several
    cells -- so the quadratic's slope at Theta_0 depends on the RANGE fitted,
    not only on Theta_0. Comparing arms fitted over different Theta ranges
    therefore mixes a real I-dependence with a fitting-window artefact. Every
    arm here is fitted over the same shared window.
    """
    Ts, byT = nfit.surviving_setpoints(runs, z_min=ZMIN)
    keep = []
    for t in Ts:
        z = [x['Z'] for x in byT[t] if 'Z' in x]
        if z and np.mean(z) < ZMIN:
            continue
        if window is not None:
            th = [r['Theta'] for r in byT[t] if r['Theta'] > 0]
            if not th or not (window[0] * 0.999 <= np.mean(th) <= window[1] * 1.001):
                continue
        keep.append(t)
    return keep, byT


def wlsq(x, y, e, deg=1):
    """Weighted polynomial fit; returns coefficients, their errors, chi2, dof.

    Errors come from the supplied per-point errors and are NEVER rescaled by
    the residuals -- the project's standing policy, and the reason chi2 here
    is a real goodness-of-fit statistic rather than 1 by construction.
    """
    x = np.asarray(x, float); y = np.asarray(y, float); e = np.asarray(e, float)
    A = np.vander(x, deg + 1, increasing=True)
    W = np.diag(1 / e ** 2)
    cov = np.linalg.inv(A.T @ W @ A)
    c = cov @ A.T @ W @ y
    chi2 = float((((y - A @ c) / e) ** 2).sum())
    return c, np.sqrt(np.diag(cov)), chi2, max(len(x) - deg - 1, 0)


def main():
    data = {tag: (I, load(p, rx)) for tag, I, p, rx in ARMS}
    print('=' * 78)
    print('THREE ARMS IN I   --   the form test')
    print('=' * 78)
    for tag, I, _, _ in ARMS:
        print(f'  {tag:<5} I = {I:.3e}   {len(data[tag][1])} frictions')
    print()

    cells = []
    for mg in MUS:
        pools = [(tag, data[tag][0], data[tag][1][mg])
                 for tag, _, _, _ in ARMS if mg in data[tag][1]]
        if len(pools) < 3:
            print(f'  mu_g = {mg}: only {len(pools)} arms present\n')
            continue
        got = nfit.common_theta0([p[2] for p in pools])
        if got is None:
            print(f'  mu_g = {mg}: no common Theta range\n')
            continue
        T0, (lo, hi) = got
        # prefer the project's canonical Theta_0 when it sits comfortably
        # inside the shared window, so numbers match the constraint list
        use_canon = lo * 1.3 <= CANON <= hi / 1.3
        Tuse = CANON if use_canon else T0

        Is, ns, es, ms, mes, ks = [], [], [], [], [], []
        bad = False
        print(f'  mu_g = {mg}    Theta_0 = {Tuse:.3e}'
              f'{"  (canonical)" if use_canon else "  (window centre)"}'
              f'   shared window [{lo:.2e}, {hi:.2e}]')
        for tag, I, runs in pools:
            ts, byT = gated(runs, window=(lo, hi))
            f = nfit.fit_local_sys(runs, Tuse, restrict_to=ts, z_min=ZMIN)
            if f is None or len(ts) < 4:
                print(f'     {tag:<5} VOID ({len(ts)} jammed setpoints)')
                bad = True
                continue
            # ln mu at Theta_0, for test 4
            sel = [r for t in ts for r in byT[t] if r['Theta'] > 0 and r['mu'] > 0]
            u = np.log([r['Theta'] for r in sel]) - np.log(Tuse)
            lm = np.log([r['mu'] for r in sel])
            A = np.vstack([np.ones_like(u), u, u ** 2]).T
            c, *_ = np.linalg.lstsq(A, lm, rcond=None)
            res = lm - A @ c
            cov = float(np.sum(res ** 2) / max(len(u) - 3, 1)) * np.linalg.inv(A.T @ A)
            Is.append(I); ns.append(f['n']); es.append(f['tot'])
            ms.append(float(c[0])); mes.append(float(np.sqrt(cov[0, 0])))
            ks.append(f['k'])
            print(f'     {tag:<5} I={I:.3e}   n = {f["n"]:+.4f} +/- {f["tot"]:.4f}'
                  f'   ln mu = {c[0]:+.4f}   [{f["k"]} setpoints]')
        if bad or len(Is) < 3:
            print()
            continue
        cells.append(dict(mg=mg, I=np.array(Is), n=np.array(ns),
                          e=np.array(es), lm=np.array(ms), lme=np.array(mes),
                          T0=Tuse, pools=pools, win=(lo, hi)))
        print()

    if not cells:
        print('no cell has all three arms.')
        return

    # --------------------------------------------------------------- T0
    # THE PRE-REGISTERED TEST, and the only Theta_0-FREE one.
    #
    # ANALYSIS_PROTOCOL section 5.3 asks for "a shared exponent against free
    # per-arm quadratics (nested F-test), and the RMS penalty of the ansatz per
    # friction". That was never run; what got reported instead was dn/dlnI
    # evaluated at three positions in the window, which is Theta_0-dependent by
    # construction and invited exactly the objection that the significant
    # entries had been chosen. This is the test that was promised, it is
    # stronger, and it cannot be steered by Theta_0:
    #
    #   M_sep   arm-specific offsets, ONE common shape   (separable: 5 params)
    #   M_full  a free quadratic per arm                 (9 params)
    #
    # and, inside the quadratic family, the joint hypothesis beta = gamma = 0.
    # beta alone is a statement AT a Theta_0; beta and gamma together are not.
    # Both are reported on all points and again on setpoint MEANS, which is
    # cluster-robust against the two seeds sharing a setpoint.
    from scipy import stats

    def _rss(A, y):
        c = np.linalg.lstsq(A, y, rcond=None)[0]
        r = y - A @ c
        return float(r @ r)

    print('=' * 78)
    print('T0   SEPARABILITY, Theta_0-FREE   (ANALYSIS_PROTOCOL section 5.3)')
    print('=' * 78)
    print(f'  {"mu_g":>6} {"nested F":>10} {"p":>10} {"RMS penalty":>12}'
          f' {"beta=gamma=0":>13} {"p":>10} {"on means":>10} {"p":>10}')
    for c in cells:
        rec = []
        for tag, I, runs in c['pools']:
            ts, byT = gated(runs, window=c['win'])
            for t in ts:
                th = [r['Theta'] for r in byT[t] if r['Theta'] > 0 and r['mu'] > 0]
                lm = [np.log(r['mu']) for r in byT[t]
                      if r['Theta'] > 0 and r['mu'] > 0]
                for a, b in zip(th, lm):
                    rec.append((tag, I, np.log(a / c['T0']), b, False))
                if th:
                    rec.append((tag, I, float(np.mean(np.log(np.array(th) / c['T0']))),
                                float(np.mean(lm)), True))
        out = {}
        for lvl in (False, True):
            s = [r for r in rec if r[4] is lvl]
            u = np.array([r[2] for r in s]); y = np.array([r[3] for r in s])
            L = np.array([np.log(r[1] / 3.162e-4) for r in s])
            arm = np.array([r[0] for r in s])
            one = np.ones_like(u)
            n = len(y)
            Af = np.column_stack([one, L, u, L * u, u ** 2, L * u ** 2])
            Ar = np.column_stack([one, L, u, u ** 2])
            F2 = ((_rss(Ar, y) - _rss(Af, y)) / 2) / (_rss(Af, y) / (n - 6))
            out[lvl] = (F2, 1 - stats.f.cdf(F2, 2, n - 6))
            if not lvl:
                d = [(arm == t).astype(float) for t in ('slow', 'mid', 'fast')]
                Asep = np.column_stack(d + [u, u ** 2])
                Afree = np.column_stack([a * b for a in d for b in [one, u, u ** 2]])
                r1, r2 = _rss(Asep, y), _rss(Afree, y)
                p1, p2 = Asep.shape[1], Afree.shape[1]
                Fn = ((r1 - r2) / (p2 - p1)) / (r2 / (n - p2))
                pen = np.sqrt(r1 / max(r2, 1e-300))
                nested = (Fn, 1 - stats.f.cdf(Fn, p2 - p1, n - p2), pen)
        print(f'  {c["mg"]:>6} {nested[0]:>10.2f} {nested[1]:>10.1e}'
              f' {nested[2]:>11.2f}x {out[False][0]:>13.2f} {out[False][1]:>10.1e}'
              f' {out[True][0]:>10.2f} {out[True][1]:>10.1e}')
    print('  Separability requires BOTH beta = 0 and gamma = 0. "RMS penalty" is')
    print('  the factor by which the separable ansatz inflates the residual.')

    # ------------------------------------------------------------ T1 pooled
    # Restricting every arm to the shared window is correct but expensive: it
    # cuts 10-12 setpoints to 6 and inflates the per-arm errors 3-4x. The
    # efficient version fits all three arms JOINTLY on the in-window points,
    #     ln mu = a + s*L + b*u + beta*(L*u) + c*u^2 + gamma*(L*u^2),
    # L = ln(I/I_mid), so beta = -dn/dlnI directly, with errors setpoint-
    # jackknifed and calibrated exactly as in fit_local_sys.
    print('=' * 78)
    print('T1p  POOLED  dn/dlnI  (all in-window points, three arms jointly)')
    print('=' * 78)
    print(f'  {"mu_g":>6} {"dn/dlnI":>18} {"sigma":>7} {"d(curv)/dlnI":>18} {"sigma":>7}')
    for c in cells:
        u, y, L, grp = [], [], [], []
        for tag, I, runs in c['pools']:
            ts, byT = gated(runs, window=c['win'])
            for t in ts:
                for r in byT[t]:
                    if r['Theta'] > 0 and r['mu'] > 0:
                        u.append(np.log(r['Theta'] / c['T0']))
                        y.append(np.log(r['mu']))
                        L.append(np.log(I / 3.162e-4))
                        grp.append(f'{tag}{round(r["Tgran"], 12)}')
        u = np.array(u); y = np.array(y); L = np.array(L); grp = np.array(grp)
        A = np.vstack([np.ones_like(u), L, u, L * u, u ** 2, L * u ** 2]).T
        (beta, be), (gam, ge) = jk(A, y, grp, [3, 5])
        print(f'  {c["mg"]:>6} {-beta:>11.4f}+/-{be:.4f} {abs(beta)/be:>7.1f}'
              f' {-2*gam:>11.4f}+/-{2*ge:.4f} {abs(gam)/ge:>7.1f}')
    print('  separability requires dn/dlnI = 0 at every friction.')

    # ------------------------------------------------------------------ T1
    print()
    print('=' * 78)
    print('T1   is n CONSTANT in I?   (per-arm fits, matched windows)')
    print('=' * 78)
    print(f'  {"mu_g":>6} {"weighted mean n":>20} {"chi2":>8} {"dof":>5} {"verdict":>12}')
    for c in cells:
        w = 1 / c['e'] ** 2
        nb = (w * c['n']).sum() / w.sum()
        chi2 = float((((c['n'] - nb) / c['e']) ** 2).sum())
        print(f'  {c["mg"]:>6} {nb:>13.4f}+/-{1/np.sqrt(w.sum()):.4f}'
              f' {chi2:>8.1f} {len(c["n"])-1:>5} '
              f'{"CONSTANT" if chi2 < 6.0 else "NOT CONSTANT":>12}')

    # ------------------------------------------------------------------ T2
    print()
    print('=' * 78)
    print('T2   is  n = n_0 I^k  ?   3 points, 2 parameters -> 1 dof, falsifiable')
    print('=' * 78)
    print(f'  {"mu_g":>6} {"k":>18} {"n_0 at I=1e-3":>18} {"chi2/1dof":>11} {"verdict":>10}')
    kk, kke = [], []
    for c in cells:
        if (c['n'] <= 0).any():
            print(f'  {c["mg"]:>6}   n <= 0 in some arm, power law undefined')
            continue
        x = np.log(c['I']); y = np.log(c['n']); e = c['e'] / c['n']
        co, ce, chi2, dof = wlsq(x, y, e, deg=1)
        k, ke = co[1], ce[1]
        n0 = np.exp(co[0] + k * np.log(1e-3))
        kk.append(k); kke.append(ke)
        print(f'  {c["mg"]:>6} {k:>11.4f}+/-{ke:.4f} {n0:>18.4f}'
              f' {chi2:>11.2f} {"OK" if chi2 < 3.84 else "REJECTED":>10}')
    print('  (chi2 > 3.84 is p < 0.05 on 1 dof)')

    # ------------------------------------------------------------------ T3
    if len(kk) >= 2:
        kk = np.array(kk); kke = np.array(kke); w = 1 / kke ** 2
        kb = float((w * kk).sum() / w.sum()); kbe = float(1 / np.sqrt(w.sum()))
        c2 = float((((kk - kb) / kke) ** 2).sum()); dof = len(kk) - 1
        print()
        print('=' * 78)
        print('T3   is k COMMON across friction?')
        print('=' * 78)
        print(f'  k = {kb:+.4f} +/- {kbe:.4f}   ({abs(kb)/kbe:.1f} sigma from 0)'
              f'   chi2/dof = {c2/max(dof,1):.2f} on {dof} dof')
        print(f'  -> if common, the friction-only object is n * I^{-kb:+.3f}.')

    # ------------------------------------------------------------------ T4
    print()
    print('=' * 78)
    print('T4   is F(I) itself a power law?   ln mu linear in ln I')
    print('=' * 78)
    print(f'  {"mu_g":>6} {"dlnmu/dlnI":>18} {"chi2/1dof":>11} {"a=(dlnmu/dlnI)/n":>19}')
    for c in cells:
        co, ce, chi2, dof = wlsq(np.log(c['I']), c['lm'], c['lme'], deg=1)
        w = 1 / c['e'] ** 2
        nb = (w * c['n']).sum() / w.sum()
        print(f'  {c["mg"]:>6} {co[1]:>11.4f}+/-{ce[1]:.4f} {chi2:>11.2f}'
              f' {co[1]/nb:>19.2f}')
    print('  a must be constant for any one-variable form mu = H(Theta/I^a).')


if __name__ == '__main__':
    main()
